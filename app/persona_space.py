"""Persona space for the demonstration, following the Assistant Axis recipe.

Roles, system prompts, judge prompts and extraction questions come from the
authors' release in vendor/assistant-axis. A role vector is the mean residual
stream (block output) over response tokens for replies the judge scores fully in
role; the default vector is the mean over replies under neutral prompts; the
axis is the default vector minus the mean role vector, so higher projections
are closer to the Assistant reference.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
except ImportError:  # the llama.cpp runtime reads the space without torch
    torch = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
AXIS_DATA = ROOT / "vendor" / "assistant-axis"


def load_roles() -> dict[str, str]:
    """Role name to short description, in the release order."""
    with open(AXIS_DATA / "data" / "roles" / "role_list.json", encoding="utf-8") as fh:
        return json.load(fh)


def _role_file(role: str) -> dict[str, Any]:
    with open(AXIS_DATA / "data" / "roles" / "instructions" / f"{role}.json", encoding="utf-8") as fh:
        return json.load(fh)


def role_prompts(role: str, model_name: str = "Qwen") -> list[str]:
    """The five system prompts for a role. An empty string means no system prompt."""
    return [item["pos"].replace("{model_name}", model_name) for item in _role_file(role)["instruction"]]


def judge_prompt(role: str) -> str:
    return _role_file(role)["eval_prompt"]


def load_questions() -> list[dict[str, Any]]:
    with open(AXIS_DATA / "data" / "extraction_questions.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_transcript(domain: str) -> dict[str, Any]:
    with open(AXIS_DATA / "transcripts" / "persona_drift" / f"{domain}.json", encoding="utf-8") as fh:
        return json.load(fh)


def _prefer_memory_efficient_attention() -> None:
    """Use the SDPA kernel for grouped query attention to reduce memory use.

    Windows builds of torch ship without flash attention, so the unmasked GQA
    path falls back to the math kernel, which is slower and quadratic in memory.
    Repeating the key/value heads instead lets that kernel run.
    """
    import transformers.integrations.sdpa_attention as sdpa

    if not torch.backends.cuda.is_flash_attention_available():
        sdpa.use_gqa_in_sdpa = lambda *_args, **_kwargs: False


DTYPES = ("bfloat16", "float16", "float32")


def model_device() -> str:
    """PVD_DEVICE if set (cuda or cpu), otherwise the GPU when there is one."""
    return os.environ.get("PVD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def model_dtype() -> torch.dtype:
    """PVD_DTYPE if set; float32 suits processors without bfloat16 support but needs twice the memory."""
    name = os.environ.get("PVD_DTYPE", "bfloat16")
    if name not in DTYPES:
        raise ValueError(f"PVD_DTYPE must be one of {', '.join(DTYPES)}, not {name!r}")
    return getattr(torch, name)


def _no_grad(fn: Any) -> Any:
    """torch.no_grad() as a decorator applied at call time, so this module imports without torch."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            return fn(*args, **kwargs)

    return wrapper


# The persona space is stored as plain arrays so that the live demonstration can
# read it without torch. Strings are saved as unicode arrays.
SPACE_ARRAYS = ("axis", "default_proj", "role_proj", "role_mean", "role_vectors", "role_scale", "trait_vector")
SPACE_SCALARS = ("layer", "n_layers", "trait_pos_proj", "trait_neg_proj")
# Later spaces: topic directions removed at the reading layer, and the Assistant's label threshold.
OPTIONAL_ARRAYS = ("topic_dirs",)
OPTIONAL_SCALARS = ("assistant_lo",)


def save_space(path: Path, space: dict[str, Any]) -> None:
    arrays = {k: np.asarray(space[k].numpy() if hasattr(space[k], "numpy") else space[k]) for k in SPACE_ARRAYS + OPTIONAL_ARRAYS
              if k in space}
    arrays.update({k: np.asarray(space[k]) for k in SPACE_SCALARS + OPTIONAL_SCALARS if k in space})
    arrays["model"] = np.asarray(space["model"])
    arrays["roles"] = np.asarray(list(space["roles"]))
    np.savez(path, **arrays)


def load_space(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        space: dict[str, Any] = {k: data[k] for k in SPACE_ARRAYS}
        space["layer"] = int(data["layer"])
        space["n_layers"] = int(data["n_layers"])
        space["trait_pos_proj"] = float(data["trait_pos_proj"])
        space["trait_neg_proj"] = float(data["trait_neg_proj"])
        space["model"] = str(data["model"])
        space["roles"] = [str(r) for r in data["roles"]]
        for k in OPTIONAL_ARRAYS:
            if k in data:
                space[k] = data[k]
        for k in OPTIONAL_SCALARS:
            if k in data:
                space[k] = float(data[k])
    return space


def load_model(model_id: str, device: str | None = None, dtype: torch.dtype | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or model_device()
    dtype = dtype or model_dtype()
    if device == "cuda":
        _prefer_memory_efficient_attention()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except (ValueError, KeyError):
        # Natively multimodal checkpoints register under the image-text class.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    if device == "cuda":
        cap_gpu_memory()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def find_layers(model: Any) -> list[torch.nn.Module]:
    """The decoder block list: the longest text decoder ModuleList named layers, h or blocks."""
    best: torch.nn.ModuleList | None = None
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList):
            continue
        if name.split(".")[-1] not in ("layers", "h", "blocks"):
            continue
        if "visual" in name or "vision" in name or "audio" in name:
            continue
        if best is None or len(module) > len(best):
            best = module
    if best is None:
        raise RuntimeError("could not locate the decoder layers")
    return list(best)


def eos_ids(model: Any, tokenizer: Any) -> set[int]:
    ids = {tokenizer.eos_token_id}
    configured = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(configured, int):
        ids.add(configured)
    elif configured:
        ids.update(configured)
    return {i for i in ids if i is not None}


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Chat template with the generation prompt. Reasoning is off, as the authors recommend."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def chat_messages(system: str, user: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": user})
    return messages


def release_memory() -> None:
    """Hand cached blocks back to the driver; the display shares this GPU."""
    import gc

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def cap_gpu_memory(headroom_gb: float = 1.0) -> None:
    """Keep the allocator inside the memory that is actually free.

    Under Windows display drivers an allocation past the physical limit is paged
    rather than refused, and the failure surfaces later as a broken CUDA context.
    With a cap, running out of memory raises a clean OutOfMemoryError instead.
    """
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    usable = torch.cuda.memory_reserved() + free - int(headroom_gb * 1e9)
    torch.cuda.set_per_process_memory_fraction(max(0.1, min(0.98, usable / total)))


def _in_chunks(order: list[int], batch_size: int, step: Any) -> None:
    """Run `step` over chunks of `order`, halving a chunk that runs out of memory."""
    pending = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    while pending:
        chunk = pending.pop(0)
        try:
            step(chunk)
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            release_memory()
            half = len(chunk) // 2
            pending[:0] = [chunk[:half], chunk[half:]]
            print(f"out of memory at batch {len(chunk)}, retrying as two of {half}", flush=True)


@_no_grad
def generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 32,
    seed: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Sample one reply per prompt. Returns (prompt ids, response ids) in input order."""
    device = next(model.parameters()).device
    stops = eos_ids(model, tokenizer)
    encoded = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    results: list[tuple[list[int], list[int]]] = [([], [])] * len(prompts)
    torch.manual_seed(seed)

    def step(chunk: list[int]) -> None:
        width = max(len(encoded[i]) for i in chunk)
        pad = tokenizer.pad_token_id
        input_ids = torch.tensor([[pad] * (width - len(encoded[i])) + encoded[i] for i in chunk], device=device)
        attention = torch.tensor(
            [[0] * (width - len(encoded[i])) + [1] * len(encoded[i]) for i in chunk], device=device
        )
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=pad,
        )
        for row, index in enumerate(chunk):
            response: list[int] = []
            for token in output[row, width:].tolist():
                if token in stops:
                    break
                response.append(token)
            results[index] = (encoded[index], response)

    _in_chunks(order, batch_size, step)
    release_memory()
    return results


@_no_grad
def response_means(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[list[int], list[int]]],
    batch_size: int = 16,
) -> torch.Tensor:
    """Mean block output over the response tokens of each (prompt, response) pair.

    Returns a float16 CPU tensor of shape (n, n_layers, hidden). Pairs with an
    empty response give a row of NaN.
    """
    device = next(model.parameters()).device
    layers = find_layers(model)
    hidden = int(getattr(model.config, "hidden_size", 0) or model.config.get_text_config().hidden_size)
    out = torch.full((len(pairs), len(layers), hidden), float("nan"), dtype=torch.float16)
    state: dict[str, Any] = {}

    def make_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            mask = state["mask"]
            summed = (tensor.float() * mask).sum(dim=1)
            state["acts"][:, index, :] = (summed / mask.sum(dim=1).clamp(min=1.0)).half().cpu()

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        order = [i for i in sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1])) if pairs[i][1]]

        def step(chunk: list[int]) -> None:
            seqs = [pairs[i][0] + pairs[i][1] for i in chunk]
            width = max(len(s) for s in seqs)
            pad = tokenizer.pad_token_id
            input_ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=device)
            attention = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=device)
            mask = torch.zeros((len(chunk), width, 1), device=device)
            for row, i in enumerate(chunk):
                begin = len(pairs[i][0])
                mask[row, begin : begin + len(pairs[i][1])] = 1.0
            state["mask"] = mask
            state["acts"] = torch.empty((len(chunk), len(layers), hidden), dtype=torch.float16)
            # Only the hooks are needed; logits for the full sequence over a 150k vocabulary
            # would take several gigabytes.
            model(input_ids=input_ids, attention_mask=attention, use_cache=False, logits_to_keep=1)
            out[chunk] = state["acts"]

        _in_chunks(order, batch_size, step)
    finally:
        for handle in handles:
            handle.remove()
    release_memory()
    return out


def compute_axis(default_vector: torch.Tensor, role_vectors: torch.Tensor) -> torch.Tensor:
    """Default minus mean role vector, per layer: (n_layers, hidden)."""
    return default_vector.float() - role_vectors.float().mean(dim=0)


def project(acts: torch.Tensor, axis: torch.Tensor, layer: int) -> torch.Tensor:
    """Projection of (..., n_layers, hidden) activations on the unit axis at one layer."""
    direction = axis[layer].float()
    direction = direction / direction.norm().clamp(min=1e-8)
    return acts[..., layer, :].float() @ direction
