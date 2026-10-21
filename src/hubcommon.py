"""Shared model, hooks and readouts for the preeclampsia-hub test (Qwen3-32B, MLX).

A neuron is one index of the SwiGLU product silu(gate_proj(x)) * up_proj(x) in a layer's MLP, before
down_proj. Every layer's MLP is wrapped so that each batch element can remove (zero) its own set of neurons
and the activations of chosen neurons can be recorded.

Two ways to hold the model:
- resident (default): all weights in memory (used for the 4-bit model);
- streaming (HUB_STREAM=1): weights are read lazily from the safetensors files one layer at a time and dropped
  after use, so models larger than memory (8-bit, bf16) run in about a layer's worth of memory. Every forward
  pass reads the whole model from disk once; batches therefore pack many sequences per pass.

Batches may mix prompts of different lengths: they are left-padded and the pads are masked out.
Run scripts with `python -I`; each script adds this directory to sys.path itself.
Environment: HUB_MODEL (repo id), HUB_STREAM=1, HUB_RESULTS_TAG (results subfolder), HUB_SMOKE=1 (tiny model).
"""
import os
import glob
import json
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", os.environ.get("HUB_RESULTS_TAG", ""))
STIMULI = os.path.join(ROOT, "stimuli")
PUBLISHED = os.path.join(ROOT, "data", "published")
STIMULI_FILE = os.environ.get("HUB_STIMULI_FILE", os.path.join(STIMULI, "stimuli.json"))
SMOKE = os.environ.get("HUB_SMOKE") == "1"
STREAM = os.environ.get("HUB_STREAM") == "1"
MODEL_ID = os.environ.get("HUB_MODEL", "mlx-community/Qwen3-0.6B-4bit" if SMOKE else "mlx-community/Qwen3-32B-4bit")
os.makedirs(RESULTS, exist_ok=True)

SEED = 20261019

# ---------------------------------------------------------------- published claim
HUB = [(31, 5482), (35, 3336), (35, 20744), (7, 2026)]          # "preeclampsia" group, all positive pole
STANDIN = [(36, 24680), (36, 7272)]                            # "microangiopathic syndromes" (debug stand-in)

CASE_ORIGINAL = ("A 32-year-old female at 30 weeks gestation presents with severe right upper quadrant pain, "
                 "mild headache, and nausea. BP is 162/98 mmHg, and labs show mildly elevated liver enzymes.")
ASK_QUESTION = "If we can only ask about one other symptom, we should ask whether she's experiencing..."
ASK_STEM = "We should ask whether she's experiencing **"
DX_QUESTION = "What is the most likely diagnosis?"
DX_STEM = "The most likely diagnosis is **"

# words whose first token defines "preeclampsia-question mass" in the ask frame
ASK_FAMILY_WORDS = ["visual", "vision", "blurred", "protein", "swelling", "edema", "seizures"]


def ask_prompt(case):
    return f"Human: {case}\n\n{ASK_QUESTION}\n\nAssistant: {ASK_STEM}"


def dx_prompt(case):
    return f"Human: {case}\n\n{DX_QUESTION}\n\nAssistant: {DX_STEM}"


def published_graph():
    return json.load(open(os.path.join(PUBLISHED, "graph-carousel-hover.json")))


def published_knockouts():
    return json.load(open(os.path.join(PUBLISHED, "preeclampsia_knockouts.json")))


def published_group_neurons():
    """All (layer, neuron) pairs in the published preeclampsia graph (excluded from random controls)."""
    out = set()
    for key in published_graph()["neurons"]["preeclampsia"]:
        layer, rest = key.split(":")
        out.add((int(layer), int(rest[:-1])))
    return out


# ---------------------------------------------------------------- hooks
class HookedMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        self.keep = None        # (B, 1, H) array of 0/1, or None
        self.mode = "zero"      # "zero": remove the neuron; "zero_pos": remove only its positive values
        self.record_idx = None  # mx.array of neuron indices to record, or None
        self.recorded = None

    def __call__(self, x):
        h = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.record_idx is not None:
            self.recorded = mx.take(h, self.record_idx, axis=-1)
        if self.keep is not None:
            if self.mode == "zero":
                h = h * self.keep
            else:
                h = h * self.keep + (1 - self.keep) * mx.minimum(h, 0)
        return self.down_proj(h)


def _local_path(repo):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, local_files_only=True))


class Probe:
    def __init__(self):
        t0 = time.time()
        if STREAM:
            from mlx_lm.utils import load_model, load_tokenizer
            path = _local_path(MODEL_ID)
            self.model, _ = load_model(path, lazy=True)          # lazy arrays: nothing is read yet
            self.tok = load_tokenizer(path)
            self.files = sorted(glob.glob(str(path / "model*.safetensors")))
            wmap = json.load(open(path / "model.safetensors.index.json"))["weight_map"]
            self.key_files = {}
            for name, fn in wmap.items():
                self.key_files.setdefault(self._group(name), set()).add(str(path / fn))
        else:
            self.model, self.tok = load(MODEL_ID)
        self.hf = getattr(self.tok, "_tokenizer", self.tok)
        self.layers = self.model.model.layers
        for layer in self.layers:
            layer.mlp = HookedMLP(layer.mlp)
        self.n_layers = len(self.layers)
        w = self.layers[0].mlp.gate_proj.weight
        self.width = w.shape[0]
        self.pad_id = self.hf.pad_token_id if self.hf.pad_token_id is not None else self.hf.eos_token_id
        if STREAM:
            mx.eval(self.model.model.norm.parameters())        # tiny; stays resident
            for g in list(self.key_files):
                if g != "norm":
                    self._unload(g)
            mx.clear_cache()
        self.load_s = time.time() - t0
        self.bytes_read = 0

    # ---- streaming helpers
    @staticmethod
    def _group(name):
        if name.startswith("model.layers."):
            return int(name.split(".")[2])
        if name.startswith("model.embed_tokens"):
            return "embed"
        if name.startswith("lm_head"):
            return "head"
        return "norm"

    def _module(self, g):
        if isinstance(g, int):
            return self.layers[g], f"model.layers.{g}."
        if g == "embed":
            return self.model.model.embed_tokens, "model.embed_tokens."
        if g == "head":
            return self.model.lm_head, "lm_head."
        return self.model.model.norm, "model.norm."

    def _weights(self, g):
        mod, prefix = self._module(g)
        items = []
        for fn in self.key_files[g]:
            for k, v in mx.load(fn).items():            # lazy: only the header is read here
                if self._group(k) == g:
                    items.append((k[len(prefix):], v))
        return mod, items

    def _load(self, g):
        if not STREAM:
            return
        mod, items = self._weights(g)
        if isinstance(g, int):                          # MLP weights live under the hook wrapper, same names
            items = [(k, v) for k, v in items]
        mod.load_weights(items, strict=False)
        self.bytes_read += sum(v.nbytes for _, v in items)

    def _unload(self, g):
        if not STREAM:
            return
        mod, items = self._weights(g)
        mod.load_weights([(k, mx.zeros((1,), dtype=v.dtype)) for k, v in items], strict=False)

    # smoke mode remaps the 32B neuron IDs into the tiny model's range
    def nid(self, layer, idx):
        if not SMOKE:
            return layer, idx
        return layer % self.n_layers, idx % self.width

    def encode(self, text):
        return list(self.hf.encode(text, add_special_tokens=False))

    def pieces(self, ids):
        return [self.hf.decode([i]) for i in ids]

    def offsets(self, text):
        enc = self.hf(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def first_token_after(self, prefix, word):
        a, b = self.encode(prefix), self.encode(prefix + word)
        if b[:len(a)] != a:
            raise ValueError(f"tokenization of {word!r} merges with the prompt")
        return b[len(a)]

    def _set(self, conditions, mode, record):
        """conditions: list (one per batch element) of iterables of (layer, idx) to remove."""
        B = len(conditions)
        by_layer = {}
        for b, cond in enumerate(conditions):
            for (l, i) in cond:
                l, i = self.nid(l, i)
                by_layer.setdefault(l, []).append((b, i))
        rec = {}
        for (l, i) in (record or []):
            l, i = self.nid(l, i)
            rec.setdefault(l, []).append(i)
        for l, layer in enumerate(self.layers):
            m = layer.mlp
            m.mode = mode
            if l in by_layer:
                keep = np.ones((B, 1, self.width), dtype=np.float32)
                for b, i in by_layer[l]:
                    keep[b, 0, i] = 0.0
                m.keep = mx.array(keep).astype(mx.bfloat16)
            else:
                m.keep = None
            m.record_idx = mx.array(rec[l]) if l in rec else None
            m.recorded = None
        return rec

    def _clear(self):
        for layer in self.layers:
            layer.mlp.keep, layer.mlp.record_idx, layer.mlp.recorded = None, None, None

    def _forward(self, x, cache, mask, collect=None):
        """One pass through the model for tokens x (B, L). Returns last-position logits (B, V).
        collect: {layer index: dict} to store the layer's recorded activations right after it runs."""
        self._load("embed")
        h = self.model.model.embed_tokens(x)
        if STREAM:
            mx.eval(h); self._unload("embed")
        for i, layer in enumerate(self.layers):
            self._load(i)
            h = layer(h, mask, cache[i])
            if collect is not None and layer.mlp.recorded is not None:
                collect[i] = layer.mlp.recorded
            if STREAM:
                mx.eval(h, *([collect[i]] if collect is not None and i in collect else []))
                self._unload(i)
                mx.clear_cache()
        h = self.model.model.norm(h[:, -1:, :])[:, 0, :]
        head = "head" if hasattr(self.model, "lm_head") else "embed"       # tied models reuse the embedding
        self._load(head)
        logits = (self.model.lm_head(h) if head == "head" else
                  self.model.model.embed_tokens.as_linear(h)).astype(mx.float32)
        mx.eval(logits)
        if STREAM:
            self._unload(head); mx.clear_cache()
        return logits

    def run(self, ids, conditions, mode="zero", record=None, n_new=0, record_layers_full=None):
        """Run one prompt per condition (batched). `ids` is one token list shared by all conditions, or one list
        per condition (left-padded to a common length). Returns first-token probabilities (B, V) float32,
        recorded activations {(layer, idx): np.array (B, T)} on the padded positions, greedy continuations
        (B lists of ids), full-layer activations {layer: np.array (B, T, H)}, and pads (B list of ints)."""
        B = len(conditions)
        seqs = [ids] * B if (ids and isinstance(ids[0], int)) else ids
        T = max(len(s) for s in seqs)
        pads = [T - len(s) for s in seqs]
        x = mx.array([[self.pad_id] * p + list(s) for p, s in zip(pads, seqs)])
        if max(pads) == 0:
            mask0 = "causal" if T > 1 else None
        else:
            q = np.arange(T)[:, None]; k = np.arange(T)[None, :]
            m = np.zeros((B, 1, T, T), dtype=bool)
            for b, p in enumerate(pads):
                m[b, 0] = (k <= q) & (k >= p)
                m[b, 0, np.arange(p), np.arange(p)] = True     # pad rows see themselves only (avoids NaN)
            mask0 = mx.array(m)
        rec = self._set(conditions, mode, record)
        full_layers = [self.nid(l, 0)[0] for l in (record_layers_full or [])]
        for ll in full_layers:
            self.layers[ll].mlp.record_idx = mx.arange(self.width)
        cache = make_prompt_cache(self.model)
        collect = {}
        logits = self._forward(x, cache, mask0, collect)
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)
        acts = {}
        for (l, i) in (record or []):
            ll, ii = self.nid(l, i)
            arr = collect.get(ll)
            if arr is None:
                continue
            col = ii if ll in full_layers else rec[ll].index(ii)
            acts[(l, i)] = np.array(arr[..., col].astype(mx.float32))
        full = {l: np.array(collect[self.nid(l, 0)[0]].astype(mx.float32)) for l in (record_layers_full or [])}
        for layer in self.layers:                      # decode steps don't record
            layer.mlp.record_idx, layer.mlp.recorded = None, None
        gen = [[] for _ in range(B)]
        tok = mx.argmax(probs, axis=-1)
        S = T
        for _ in range(n_new):
            mx.eval(tok)
            for b, t in enumerate(tok.tolist()):
                gen[b].append(t)
            S += 1
            if max(pads) == 0:
                mstep = None
            else:
                m = np.ones((B, 1, 1, S), dtype=bool)
                for b, p in enumerate(pads):
                    m[b, 0, 0, :p] = False
                mstep = mx.array(m)
            tok = mx.argmax(self._forward(tok[:, None], cache, mstep), axis=-1)
        self._clear()
        del cache
        mx.clear_cache()
        return probs, acts, gen, full, pads


def tv_distance(probs, ref_index=0):
    """Total variation distance of each row of probs from row ref_index."""
    p = np.array(probs)
    return 0.5 * np.abs(p - p[ref_index:ref_index + 1]).sum(-1)


def diagnosis_label(text):
    """PREREGISTRATION §6: preeclampsia-family if the answer contains "eclampsia", hellp if it contains "hellp",
    other otherwise. "eclampsia" is checked first (deviation D2), so "preeclampsia with possible HELLP" is family."""
    t = text.lower()
    if "eclampsia" in t:            # preeclampsia, pre-eclampsia, eclampsia
        return "preeclampsia_family"
    if "hellp" in t:
        return "hellp"
    return "other"


def bootstrap_ci(x, n=10000, seed=SEED, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    return path
