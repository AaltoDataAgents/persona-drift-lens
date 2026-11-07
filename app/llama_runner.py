"""Run the model with llama.cpp on the CPU and read its residual stream.

The persona space is built with transformers (tools/build_space.py). For the live
demonstration, the same model runs as a GGUF file in llama.cpp, which is several
times faster on a processor. llama.cpp names each block's output `l_out-<layer>`,
which is the block output the build averages; an evaluation callback copies those
tensors out for every generated token.

On the test laptop, Q8_0 files gave block outputs within cosine 0.9997 of
transformers on the same tokens, and a role-versus-Assistant difference within
0.5% of its length (docs/model-test.md).
"""

from __future__ import annotations

import ctypes
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

# GGUF files for models with a persona space. PVD_GGUF overrides with a local path.
GGUF_SOURCES = {
    "Qwen/Qwen3-4B-Instruct-2507": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q8_0.gguf"),
    "Qwen/Qwen3-1.7B": ("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"),
    "Qwen/Qwen3.5-2B": ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q8_0.gguf"),
}


# Smaller files, chosen with PVD_QUANT. For Qwen3-1.7B, Q4_K_M against Q8_0 on the same tokens:
# block outputs within cosine 0.998, axis readings within 0.05 on average (0.13 at most), the same
# label for 85% of replies, about 1.5 times faster on two threads (docs/model-test.md).
GGUF_QUANTS = {
    ("Qwen/Qwen3-1.7B", "Q4_K_M"): ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
}


def gguf_path(model_id: str) -> str:
    if os.environ.get("PVD_GGUF"):
        return os.environ["PVD_GGUF"]
    quant = os.environ.get("PVD_QUANT", "").upper()
    if quant and quant != "Q8_0" and (model_id, quant) not in GGUF_QUANTS:
        raise RuntimeError(f"no {quant} file is listed for {model_id}")
    if model_id not in GGUF_SOURCES:
        raise RuntimeError(f"no GGUF file is listed for {model_id}; set PVD_GGUF to its path")
    from huggingface_hub import hf_hub_download

    repo, filename = GGUF_QUANTS.get((model_id, quant), GGUF_SOURCES[model_id])
    try:
        return hf_hub_download(repo, filename)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"the model file {filename} is not downloaded; run once with a connection: "
            f"HF_HUB_OFFLINE=0 python -m app.llama_runner {model_id}"
        ) from exc


def cpu_threads() -> int:
    """PVD_THREADS if set; otherwise the CPUs this process may use.

    A container usually reports the host's processor count, so the cgroup quota
    (as on JupyterHub) takes precedence. Shared servers such as Noppe set a high
    limit (8 cores) but guarantee much less; when a class generates at once, extra
    llama.cpp threads only wait for each other, so a container uses at most 4.
    Without a quota, use half the logical processors, roughly the physical cores.
    """
    if os.environ.get("PVD_THREADS"):
        return max(1, int(os.environ["PVD_THREADS"]))
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    quota = _cgroup_cpus()
    if quota is not None:
        return max(1, min(available, quota, 4))
    return max(1, available // 2)


def _cgroup_cpus() -> int | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return math.ceil(int(quota) / int(period))
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return math.ceil(quota / period)
    except (OSError, ValueError):
        pass
    return None


def _ggml() -> ctypes.CDLL:
    """The ggml library bundled with llama-cpp-python, for reading tensors in the callback."""
    import llama_cpp

    lib_dir = Path(llama_cpp.__file__).parent / "lib"
    names = ["ggml-base.dll", "libggml-base.so", "libggml-base.dylib", "ggml.dll", "libggml.so", "libggml.dylib",
             "llama.dll", "libllama.so", "libllama.dylib"]
    for name in names:
        path = lib_dir / name
        if not path.exists():
            continue
        lib = ctypes.CDLL(str(path))
        if hasattr(lib, "ggml_get_name") and hasattr(lib, "ggml_backend_tensor_get"):
            lib.ggml_get_name.restype = ctypes.c_char_p
            lib.ggml_get_name.argtypes = [ctypes.c_void_p]
            lib.ggml_nelements.restype = ctypes.c_int64
            lib.ggml_nelements.argtypes = [ctypes.c_void_p]
            lib.ggml_nbytes.restype = ctypes.c_size_t
            lib.ggml_nbytes.argtypes = [ctypes.c_void_p]
            lib.ggml_backend_tensor_get.restype = None
            lib.ggml_backend_tensor_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
            return lib
    raise RuntimeError(f"could not find the ggml library in {lib_dir}")


class LlamaRunner:
    """One llama.cpp context. Generates a token at a time and returns every block output for it."""

    def __init__(self, model_id: str, n_ctx: int = 8192) -> None:
        import llama_cpp
        import llama_cpp.llama_cpp as ll
        from llama_cpp._internals import LlamaSampler
        from transformers import AutoTokenizer

        self._llama_cpp = llama_cpp
        self._sampler_class = LlamaSampler
        # The chat template comes from the transformers tokenizer, as in the build.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.threads = cpu_threads()
        self._ggml = _ggml()
        self._capture = False
        self._rows: dict[int, np.ndarray] = {}
        self._names: dict[bytes, int] = {}

        def on_eval(tensor: int, ask: bool, _user: Any) -> bool:
            if not self._capture:
                return not ask  # ask=True: "do you want this tensor?"; ask=False: "continue?"
            layer = self._names.get(self._ggml.ggml_get_name(tensor))
            if layer is None:
                return not ask
            if ask:
                return True
            n = self._ggml.ggml_nelements(tensor)
            if self._ggml.ggml_nbytes(tensor) != 4 * n:
                return True  # not float32; feed() reports the missing layer
            buf = np.empty(n, dtype=np.float32)
            self._ggml.ggml_backend_tensor_get(tensor, buf.ctypes.data, 0, 4 * n)
            self._rows[layer] = buf.reshape(-1, self.hidden)[-1]
            return True

        # Keep a reference: the C library calls this for as long as the context lives.
        self._callback = ll.ggml_backend_sched_eval_callback(on_eval)
        default_params = ll.llama_context_default_params

        def with_callback() -> Any:
            params = default_params()
            params.cb_eval = self._callback
            params.cb_eval_user_data = None
            return params

        ll.llama_context_default_params = with_callback
        try:
            self.llm = llama_cpp.Llama(
                gguf_path(model_id), n_ctx=n_ctx, n_batch=512, n_threads=self.threads,
                n_threads_batch=self.threads, verbose=False,
            )
        finally:
            ll.llama_context_default_params = default_params
        self.hidden = self.llm.n_embd()
        self.n_layers = int(self.llm.metadata.get(f"{self.llm.metadata['general.architecture']}.block_count"))
        self._names = {f"l_out-{i}".encode(): i for i in range(self.n_layers)}
        self._vocab = self.llm._model.vocab
        self._sampler: Any = None
        self._first: Any = None
        self._stable = 0
        self._snapshot: tuple[list[int], Any] = ([], None)
        self.recurrent = bool(self.llm._is_recurrent or self.llm._is_hybrid)
        self.device_name = f"CPU, llama.cpp, {self.threads} threads"

    def _render(self, messages: list[dict[str, str]], generation: bool) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation, enable_thinking=False
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(self, messages: list[dict[str, str]]) -> list[int]:
        ids = self._render(messages, generation=True)
        # The conversation without the reply header is the start of every later prompt.
        stable = self._render(messages, generation=False)
        self._stable = len(stable) if ids[: len(stable)] == stable and len(stable) < len(ids) else 0
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def is_stop(self, token: int) -> bool:
        return bool(self._llama_cpp.llama_vocab_is_eog(self._vocab, token))

    def start(self, prompt_ids: list[int], temperature: float, top_p: float, seed: int | None,
              top_k: int = 0, presence_penalty: float = 0.0, earlier: list[list[int]] | None = None,
              not_first: list[int] | None = None, dry: float = 0.0) -> None:
        """Evaluate the prompt, reusing the cached start of the conversation where possible."""
        llm = self.llm
        common = 0
        for cached, new in zip(llm.input_ids[: llm.n_tokens], prompt_ids):
            if cached != new:
                break
            common += 1
        common = min(common, len(prompt_ids) - 1)
        if not llm._ctx.kv_cache_seq_rm(-1, common, -1):
            # Recurrent layers (as in Qwen3.5) cannot rewind part of their state, so
            # return to the snapshot taken at the end of the previous user message.
            llm.reset()
            common = 0
            snap_ids, snap = self._snapshot
            if snap is not None and len(snap_ids) < len(prompt_ids) and prompt_ids[: len(snap_ids)] == snap_ids:
                ll = self._llama_cpp.llama_cpp
                if ll.llama_state_seq_set_data(llm._ctx.ctx, snap, len(snap), 0):
                    common = len(snap_ids)
                    llm.input_ids[:common] = snap_ids
                else:
                    llm.reset()
        llm.n_tokens = common
        self._capture = False
        stable = self._stable if self._stable and len(prompt_ids) > self._stable > common else 0
        if self.recurrent and stable:
            llm.eval(prompt_ids[common:stable])
            ll = self._llama_cpp.llama_cpp
            size = ll.llama_state_seq_get_size(llm._ctx.ctx, 0)
            buf = (ctypes.c_uint8 * size)()
            written = ll.llama_state_seq_get_data(llm._ctx.ctx, buf, size, 0)
            self._snapshot = (list(prompt_ids[:stable]), buf if written else None)
            common = stable
        llm.eval(prompt_ids[common:])
        seed = seed if seed is not None else random.getrandbits(32)
        self._sampler = self._chain(temperature, top_p, top_k, presence_penalty, earlier, seed, dry=dry)
        # A separate chain for the first token only, with the listed tokens ruled out.
        self._first = (self._chain(temperature, top_p, top_k, presence_penalty, earlier, seed, not_first, dry)
                       if not_first else None)

    def _chain(self, temperature: float, top_p: float, top_k: int, presence_penalty: float,
               earlier: list[list[int]] | None, seed: int, banned: list[int] | None = None, dry: float = 0.0) -> Any:
        sampler = self._sampler_class()
        if banned:
            sampler.add_logit_bias(self.llm.n_vocab(), {t: float("-inf") for t in banned})
        if presence_penalty:
            sampler.add_penalties(self.llm.n_vocab(), 512, 1.0, 0.0, presence_penalty)
        if dry:
            # DRY penalises continuing a sequence that already occurred, more the longer it runs
            # (settings recommended by its authors), so a reply cannot copy an earlier one.
            # A look-back of -1 ("whole context") silently disables DRY in this build, so use a length.
            sampler.add_dry(self.llm._model, dry, 1.75, 2, 8192, ["\n", ":", "\"", "*"])
        if presence_penalty or dry:
            for ids in earlier or []:  # the model's earlier replies count as already written
                for token in ids:
                    sampler.accept(token)
        if temperature <= 0:
            sampler.add_greedy()
        else:
            if top_k:
                sampler.add_top_k(top_k)
            sampler.add_top_p(top_p, 1)
            sampler.add_temp(temperature)
            sampler.add_dist(seed)
        return sampler

    def sample(self) -> int:
        if self._first is not None:
            token = int(self._first.sample(self.llm._ctx, -1))
            self._first = None
            self._sampler.accept(token)
            return token
        return int(self._sampler.sample(self.llm._ctx, -1))

    def feed(self, token: int) -> np.ndarray:
        """Evaluate one generated token; returns its block outputs, shape (n_layers, hidden)."""
        self._rows = {}
        self._capture = True
        try:
            self.llm.eval([token])
        finally:
            self._capture = False
        if len(self._rows) != self.n_layers:
            raise RuntimeError(f"read {len(self._rows)} of {self.n_layers} block outputs")
        return np.stack([self._rows[i] for i in range(self.n_layers)])


def download(model_id: str) -> None:
    """Fetch what the live demonstration needs: the GGUF file and the tokenizer (no full weights)."""
    from huggingface_hub import snapshot_download

    snapshot_download(model_id, allow_patterns=["*.json", "*.txt", "*.jinja", "tokenizer.model"])
    print(gguf_path(model_id))


if __name__ == "__main__":
    # python -m app.llama_runner Qwen/Qwen3-4B-Instruct-2507
    import sys

    download(sys.argv[1])
