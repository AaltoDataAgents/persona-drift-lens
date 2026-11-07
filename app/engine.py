"""The live model behind the demonstration.

Loads the persona space built by tools/build_space.py and streams each reply of a
conversation token by token, with three readings taken inside the model from the
running mean of the reply so far at the reading layer:

- position on the Assistant Axis, scaled so that 1 is the default Assistant and
  0 is the average character (Lu et al. 2026);
- the closest saved pattern, by cosine similarity to the role vectors and to the
  default Assistant (the Assistant's mean is the average role plus the axis);
- agreement, the projection on the sycophantic persona vector, scaled so that 0
  is the honest side and 1 the flattering side (Chen et al. 2025).

On a processor the model runs in llama.cpp (app/llama_runner.py); with an NVIDIA
GPU and torch installed, in transformers. PVD_BACKEND=llama or torch chooses.
Both runners return every block output for each generated token, which is the
quantity the build averages over response tokens.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import persona_space as ps

Emit = Callable[[str, dict[str, Any]], None]
ASSISTANT = "Assistant"
# Small models tend to open every reply with the same greeting; after the first turn a
# reply may not begin with one of these words (single tokens only).
# Labels come from recognisable personas only. All 275 roles of the paper's release shape the
# axis, but many are deliberate probes (virus, egregore, void, coral reef) that read as noise
# when shown as "the character the reply is closest to".
LABEL_ROLES = (
    "teacher", "tutor", "counselor", "therapist", "coach", "mentor", "consultant", "doctor", "lawyer",
    "scientist", "philosopher", "journalist", "marketer", "chef", "detective", "librarian",
    "parent", "grandparent", "teenager", "caregiver", "romantic", "optimist", "cynic", "rebel", "gossip",
    "narcissist", "celebrity", "influencer",
    "novelist", "poet", "comedian", "actor", "bard", "pirate", "spy", "soldier", "warrior", "criminal",
    "hacker", "mystic", "prophet", "witch", "ghost", "vampire", "angel", "demon", "robot", "alien",
    "trickster", "jester",
)
MIN_LABEL_ROLES = 10  # with fewer of them in a space, every role may be a label

# The voices of the teaching talks (Aalto EE and Data Agents decks): each label persona belongs to
# one. This grouping is a teaching choice, not found in the data; the persona stays visible beneath
# it. The flatterer voice is the Agreement gauge (the sycophancy persona vector), not a label.
TALK_VOICES = {
    "teacher": ("teacher", "tutor", "coach", "mentor", "librarian"),
    "consultant": ("consultant", "doctor", "lawyer", "scientist", "journalist", "chef"),
    "salesperson": ("marketer", "influencer", "celebrity"),
    "confidant": ("counselor", "therapist", "caregiver", "parent", "grandparent", "romantic", "optimist"),
    "novelist": ("novelist", "poet", "bard", "actor", "comedian", "detective", "pirate", "spy", "soldier",
                 "warrior", "robot", "alien"),
    "mystic": ("mystic", "philosopher", "prophet", "witch", "angel", "ghost", "vampire", "demon"),
    "troll": ("trickster", "jester", "cynic", "rebel", "gossip", "narcissist", "criminal", "hacker", "teenager"),
}
VOICE_OF = {persona: voice for voice, personas in TALK_VOICES.items() for persona in personas} | {ASSISTANT: ASSISTANT}
VOICE_DESCRIPTIONS = {
    "teacher": "Explains and guides learning step by step.",
    "consultant": "Gives expert advice on a specific domain.",
    "salesperson": "Promotes, persuades and builds an image.",
    "confidant": "Listens closely and offers personal support.",
    "novelist": "Tells stories and plays characters.",
    "mystic": "Reflects on meaning, spirit and the unknowable.",
    "troll": "Provokes, mocks or plays tricks.",
}
GREETINGS = ("Hello", "Hi", "Hey", "Greetings", "HELLO", "HI", "HEY", "Hola")



class Conversation:
    """One chat: its messages and readings, and a way to stop its reply."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.turns: list[dict[str, Any]] = []
        self.reply_ids: list[list[int]] = []  # token ids of earlier replies, for the repetition penalty
        self.stop = threading.Event()


def _llama_runner_type() -> type:
    from .llama_runner import LlamaRunner

    return LlamaRunner


def _default_backend() -> str:
    """transformers on an NVIDIA GPU; llama.cpp on the processor, and wherever torch is not installed."""
    try:
        import torch
    except ImportError:
        return "llama"
    return "torch" if torch.cuda.is_available() and os.environ.get("PVD_DEVICE", "cuda") == "cuda" else "llama"


class TorchRunner:
    """The model in transformers, with forward hooks on every block."""

    def __init__(self, model_id: str) -> None:
        import torch

        self.torch = torch
        self.model, self.tokenizer = ps.load_model(model_id)
        self.device = next(self.model.parameters()).device
        self.layers = ps.find_layers(self.model)
        self.n_layers = len(self.layers)
        self.stops = ps.eos_ids(self.model, self.tokenizer)
        self._last: list[Any] = [None] * self.n_layers

        def make_hook(index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                self._last[index] = tensor[0, -1]

            return hook

        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(make_hook(i))
        on_gpu = self.device.type == "cuda"
        self.device_name = torch.cuda.get_device_name(0) if on_gpu else f"CPU, transformers, {torch.get_num_threads()} threads"
        self.on_gpu = on_gpu

    def encode(self, messages: list[dict[str, str]]) -> list[int]:
        return self.tokenizer(ps.render_chat(self.tokenizer, messages), add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def is_stop(self, token: int) -> bool:
        return token in self.stops

    def start(self, prompt_ids: list[int], temperature: float, top_p: float, seed: int | None,
              top_k: int = 0, presence_penalty: float = 0.0, earlier: list[list[int]] | None = None,
              not_first: list[int] | None = None) -> None:
        torch = self.torch
        self.not_first = list(not_first or [])
        if seed is not None:
            torch.manual_seed(seed)
        self.temperature, self.top_p, self.top_k, self.presence = temperature, top_p, top_k, presence_penalty
        self.used = {t for ids in earlier or [] for t in ids}
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([prompt_ids], device=self.device), use_cache=True, logits_to_keep=1)
        self.past, self.logits = out.past_key_values, out.logits[0, -1]

    def sample(self) -> int:
        torch = self.torch
        logits = self.logits.float().clone()
        if self.not_first:
            logits[self.not_first] = float("-inf")
            self.not_first = []
        if self.presence and self.used:
            logits[list(self.used)] -= self.presence
        if self.temperature <= 0:
            return int(logits.argmax())
        if self.top_k:
            logits[logits < logits.topk(self.top_k).values[-1]] = float("-inf")
        probs = torch.softmax(logits / self.temperature, dim=-1)
        ordered, index = probs.sort(descending=True)
        ordered[(ordered.cumsum(-1) - ordered) > self.top_p] = 0
        return int(index[torch.multinomial(ordered, 1)])

    def feed(self, token: int) -> np.ndarray:
        torch = self.torch
        self.used.add(token)
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([[token]], device=self.device), past_key_values=self.past, use_cache=True)
        self.past, self.logits = out.past_key_values, out.logits[0, -1]
        return torch.stack(self._last).float().cpu().numpy()


class Engine:
    def __init__(self, space_path: Path) -> None:
        self.space_path = space_path
        self.space: dict[str, Any] = {}
        self.runner: Any = None
        self.state = "starting"
        self.detail = ""
        self.speed = 0.0
        self.device_name = ""
        self.options: dict[str, Any] = {
            # As in Lu et al. (2026), replies may run to 512 tokens and the reading averages the whole
            # reply; most end well before. A shorter cap cut replies mid-sentence and measured a part.
            "max_tokens": 512,
            # Qwen's settings for its non-thinking mode, with the presence penalty also
            # covering the model's earlier replies, so that small models do not repeat
            # the same opening every turn. The persona space was built with T 0.7, top-p 0.9.
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "dry_multiplier": 0.8,  # llama.cpp only: against copying earlier replies
            "system_prompt": "",
            # Tokens of conversation the model may read. 12,000 holds about 15 long turns, the length of
            # the paper's drift conversations; llama.cpp reserves this memory when it loads (about
            # 4.2 GiB in all for Qwen3-1.7B). PVD_MAX_CONTEXT lowers it where memory is tight.
            "max_context": int(os.environ.get("PVD_MAX_CONTEXT", "12000")),
        }
        self._lock = threading.Lock()

    def load_space(self) -> None:
        self.space = ps.load_space(self.space_path)

    def load(self, with_model: bool = True) -> None:
        try:
            self.state = "loading"
            self.load_space()
            if not with_model:
                self.state = "off"
                self.detail = "The live model is switched off; examples only."
                return
            model_id = self.space["model"]
            self.detail = f"loading {model_id}"
            backend = os.environ.get("PVD_BACKEND") or _default_backend()
            if backend == "llama":
                from .llama_runner import LlamaRunner

                self.runner = LlamaRunner(model_id, n_ctx=self.options["max_context"] + 1024)
            elif backend == "torch":
                self.runner = TorchRunner(model_id)
            else:
                raise ValueError(f"PVD_BACKEND must be llama or torch, not {backend!r}")
            if self.runner.n_layers != self.space["n_layers"]:
                raise RuntimeError(f"the model has {self.runner.n_layers} layers but the space was built for {self.space['n_layers']}")
            self.device_name = self.runner.device_name
            self._prepare()
            self.state = "warming"
            self.speed = self._warm_up()
            self.state = "ready"
            self.detail = ""
        except Exception as exc:  # noqa: BLE001
            self.state = "failed"
            self.detail = f"{type(exc).__name__}: {exc}"

    def _prepare(self) -> None:
        sp = self.space
        self.layer = sp["layer"]
        axis = sp["axis"].astype(np.float64)
        self.units = axis / np.linalg.norm(axis, axis=1, keepdims=True)
        self.default_proj = sp["default_proj"].astype(np.float64)
        self.role_proj = sp["role_proj"].astype(np.float64)
        self.role_mean = sp["role_mean"].astype(np.float64)
        centred = sp["role_vectors"].astype(np.float64) - self.role_mean
        self.role_dirs = centred / np.linalg.norm(centred, axis=1, keepdims=True)
        # The default Assistant is a candidate too: centred on the average role, its
        # pattern is the axis itself.
        assistant = axis[self.layer] / np.linalg.norm(axis[self.layer])
        self.role_dirs = np.vstack([self.role_dirs, assistant])
        self.roles = list(sp["roles"]) + [ASSISTANT]
        labels = [i for i, r in enumerate(self.roles[:-1]) if r in LABEL_ROLES]
        self.label_idx = np.array(labels if len(labels) >= MIN_LABEL_ROLES else range(len(self.roles) - 1))
        self.trait = sp["trait_vector"].astype(np.float64)
        self.trait_pos = float(sp["trait_pos_proj"])
        self.trait_neg = float(sp["trait_neg_proj"])
        # Spaces built with topic removal: the same directions are removed from the live reading,
        # and the label is "Assistant" while the reading stays in the Assistant's usual range.
        topic = sp.get("topic_dirs")
        self.topic = topic.astype(np.float64) if topic is not None and len(topic) else None
        self.assistant_lo = sp.get("assistant_lo")
        self.labels = self._load_labels()
        tokenizer = self.runner.tokenizer
        self.greeting_ids = [ids[0] for ids in (tokenizer(w, add_special_tokens=False)["input_ids"] for w in GREETINGS)
                             if len(ids) == 1]

    def _load_labels(self) -> dict[str, Any] | None:
        """The label classifier from tools/build_labels.py, saved beside the space, if there is one."""
        path = self.space_path.with_name(self.space_path.stem + "-labels.npz")
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as d:
            lab = {k: d[k] for k in d.files}
        personas = [str(p) for p in lab["personas"]]
        families = [str(f) for f in lab["families"]]
        names = sorted(set(families))
        member = np.zeros((len(names), len(personas)))
        for i, f in enumerate(families):
            member[names.index(f), i] = 1.0
        return {
            "layers": [int(l) for l in lab["layers"]], "topic": lab["topic"].astype(np.float64),
            "coef": lab["coef"].astype(np.float64), "intercept": lab["intercept"].astype(np.float64),
            "centre": lab["centre"].astype(np.float64), "kind": str(lab["kind"]),
            "personas": personas, "families": families, "names": names, "member": member,
            "uncertain_below": float(lab["uncertain_below"]),
            "temperature": float(lab["temperature"]) if "temperature" in lab else 1.0,
        }

    def _family_label(self, means: np.ndarray) -> tuple[list[str], str, float, list[float]]:
        """Families in order of probability, the nearest persona in the first, the margin, and the probabilities."""
        lab = self.labels
        parts = []
        for k, layer in enumerate(lab["layers"]):
            x, v = means[layer], lab["topic"][k]
            parts.append(x - v.T @ (v @ x))
        feat = np.concatenate(parts)
        if lab["kind"] == "cosine":
            c = feat - lab["centre"]
            scores = 20 * (lab["coef"] @ c) / max(np.linalg.norm(c), 1e-12)
        else:
            # Calibrated: the temperature makes held-out probabilities match how often the label was right.
            scores = (lab["coef"] @ feat + lab["intercept"]) / lab["temperature"]
        p = np.exp(scores - scores.max())
        p /= p.sum()
        fam = lab["member"] @ p
        order = np.argsort(-fam)
        names = [lab["names"][i] for i in order]
        first = lab["names"][order[0]]
        in_first = [i for i, f in enumerate(lab["families"]) if f == first]
        if first == ASSISTANT:  # name the nearest persona all the same
            in_first = [i for i, f in enumerate(lab["families"]) if f != ASSISTANT]
        persona = lab["personas"][max(in_first, key=lambda i: p[i])]
        margin = float(fam[order[0]] - fam[order[1]]) if len(order) > 1 else 1.0
        return names, persona, margin, [float(fam[i]) for i in order]

    def _warm_up(self) -> float:
        # Warm up before timing generation.
        self._generate(Conversation(), "Say hello.", lambda *_: None, max_tokens=8, record=False)
        started = time.time()
        turn = self._generate(Conversation(), "Say hello in one short sentence.", lambda *_: None, max_tokens=24, record=False)
        return round(turn["n_tokens"] / max(1e-6, time.time() - started), 1)

    def _profile(self, mean: np.ndarray) -> np.ndarray:
        """Position on the Assistant Axis at every layer, from (n_layers, hidden) means."""
        return ((mean * self.units).sum(axis=1) - self.role_proj) / (self.default_proj - self.role_proj)

    def _readings(self, means: np.ndarray) -> tuple[float, float, str, list[str], str, list[tuple[str, float]]]:
        """Axis position, agreement, the label, the next labels and the nearest specific persona.

        `means` holds the reply's mean block output at every layer, shape (n_layers, hidden).
        """
        layer = self.layer
        mean = means[layer]
        if self.topic is not None:
            mean = mean - self.topic.T @ (self.topic @ mean)
        s = (mean @ self.units[layer] - self.role_proj[layer]) / (self.default_proj[layer] - self.role_proj[layer])
        t = (mean @ self.trait - self.trait_neg) / (self.trait_pos - self.trait_neg)
        assistant = self.assistant_lo is not None and s >= self.assistant_lo
        if self.labels is not None:
            names, persona, margin, probs = self._family_label(means)
            top3 = [(n, round(p, 3)) for n, p in zip(names[:3], probs[:3])]
            # Labels trained with the Assistant as a class decide on their own; older ones use the axis.
            if assistant and ASSISTANT not in self.labels["names"]:
                return float(s), float(t), ASSISTANT, names[:2], persona, top3
            # The headline is the most likely voice; the bars beneath show how close the others are.
            return float(s), float(t), names[0], names[1:3], persona, top3
        centred = mean - self.role_mean
        candidates = self.label_idx if self.assistant_lo is not None else np.append(self.label_idx, len(self.roles) - 1)
        scores = self.role_dirs[candidates] @ (centred / max(np.linalg.norm(centred), 1e-12))
        top = [self.roles[int(candidates[i])] for i in np.argsort(-scores)[:3]]
        if assistant:
            return float(s), float(t), ASSISTANT, top[:2], top[0], []
        return float(s), float(t), top[0], top[1:3], top[0], []

    def chat(self, conversation: Conversation, message: str, emit: Emit, seed: int | None = None) -> dict[str, Any]:
        if self.state != "ready":
            raise RuntimeError(f"the model is not ready ({self.state})")
        with self._lock:
            return self._generate(conversation, message, emit, seed=seed)

    def _generate(
        self,
        conversation: Conversation,
        message: str,
        emit: Emit,
        max_tokens: int | None = None,
        record: bool = True,
        seed: int | None = None,
    ) -> dict[str, Any]:
        opts = self.options
        runner = self.runner
        stop = conversation.stop
        history = list(conversation.messages)
        if opts["system_prompt"] and not history:
            history.append({"role": "system", "content": opts["system_prompt"]})
        history.append({"role": "user", "content": message})
        prompt_ids = runner.encode(history)
        if len(prompt_ids) > opts["max_context"]:
            raise RuntimeError("The conversation is too long for this computer. Reset to start a new one.")
        started = time.time()
        generated: list[int] = []
        tokens: list[dict[str, Any]] = []
        text = ""
        total: np.ndarray | None = None
        reading = (0.0, 0.0, "", [], "", [])
        dry = {"dry": opts["dry_multiplier"]} if isinstance(runner, _llama_runner_type()) else {}
        runner.start(prompt_ids, opts["temperature"], opts["top_p"], seed, top_k=opts["top_k"],
                     presence_penalty=opts["presence_penalty"], earlier=conversation.reply_ids,
                     not_first=self.greeting_ids if conversation.turns else None, **dry)
        # Every sampled token is fed back, so the readings average over the same
        # response tokens as the offline build.
        for _ in range(max_tokens or opts["max_tokens"]):
            if stop.is_set():
                break
            token = runner.sample()
            if runner.is_stop(token):
                break
            generated.append(token)
            acts = runner.feed(token).astype(np.float64)
            total = acts if total is None else total + acts
            reading = self._readings(total / len(generated))
            decoded = runner.decode(generated)
            if decoded.endswith("�"):
                continue  # wait for the rest of a multibyte character
            delta, text = decoded[len(text):], decoded
            event = {"d": delta, "s": round(reading[0], 4), "t": round(reading[1], 4), "c": reading[2], "k": reading[3],
                     "m": reading[4], "p": reading[5],
                     "ms": int((time.time() - started) * 1000)}
            tokens.append(event)
            emit("token", event)
        text = runner.decode(generated)
        turn = {
            "index": len(conversation.turns) + 1,
            "user": message,
            "reply": text,
            "n_tokens": len(generated),
            "seconds": round(time.time() - started, 2),
            "s": round(reading[0], 4),
            "t": round(reading[1], 4),
            "c": reading[2] if generated else None,
            "closest": reading[2] if generated else None,
            "top": reading[3] if generated else [],
            "persona": reading[4] if generated else None,
            "probs": reading[5] if generated else [],
            "stopped": stop.is_set(),
            # Reached the length limit without ending: the reading covers what was written.
            "cut": bool(generated) and not stop.is_set() and len(generated) >= (max_tokens or opts["max_tokens"]),
            "tokens": tokens,
        }
        if record and generated:
            profile = self._profile(total / len(generated))
            profile[self.layer] = reading[0]  # the reading layer as read live (after topic removal)
            turn["profile"] = [round(float(v), 3) for v in profile]
            conversation.messages = history + [{"role": "assistant", "content": text}]
            conversation.reply_ids.append(list(generated))
            conversation.turns.append(turn)
        return turn

    def status(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "state": self.state,
            "detail": self.detail,
            "model": self.space.get("model", ""),
            "device": self.device_name,
            "speed": self.speed,
            "options": self.options,
            "layer": self.space.get("layer"),
            "n_layers": self.space.get("n_layers"),
            "n_roles": len(self.space.get("roles", [])),
        }
        # Querying memory opens a CUDA context, so ask only when the model is on the GPU.
        if getattr(self.runner, "on_gpu", False):
            free, total = self.runner.torch.cuda.mem_get_info()
            info["gpu_free_gb"] = round(free / 1e9, 1)
            info["gpu_total_gb"] = round(total / 1e9, 1)
        return info
