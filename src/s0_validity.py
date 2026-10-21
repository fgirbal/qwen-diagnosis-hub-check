"""Stage 0 (4-bit resident, or 8-bit / bf16 streamed: set HUB_MODEL, HUB_STREAM, HUB_RESULTS_TAG): does the 4-bit model reproduce the essay's preeclampsia results? (Pre-registered gate.)

V0a tokenization = the published 79 tokens; V0b intact P(visual) in [0.75, 0.98] and continuation starts
"visual disturbances"; V0c hub removal gives P(visual) <= 0.05 and a continuation naming itching or pruritus
(ablation modes tried in order: "zero", then "zero_pos"; the first that passes is frozen); V0d >= 18/20
same-layer random removals keep P(visual) >= 0.75; V0e hub activations match the published ones (Spearman >= 0.8,
L31:5482 at token 75 within 30% of 46.5); V0f (fidelity only) removing the 2 "microangiopathic" neurons gives
P(visual) in [0.05, 0.45]. Also: C-act candidate pools and draws, P1 neuron inclusion, timing and K.

Writes results/stage0.json. Exit code 0 if the gate passes, 1 otherwise.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mlx.core as mx

import hubcommon as H

K_MAX = 20
CASES_PER_BATCH = 6   # must match s1_run.py
PLANNED_CASES = 120          # S0-S3, two frames each
PLANNED_S4 = 40
BATCH = 8
N_NEW_ASK, N_NEW_DX = 8, 12


def spearman(a, b):
    def rank(x):
        x = np.asarray(x, float); order = x.argsort(); r = np.empty(len(x)); r[order] = np.arange(len(x))
        for v in np.unique(x):                         # average ties
            m = x == v; r[m] = r[m].mean()
        return r
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    out = {"model": H.MODEL_ID, "smoke": H.SMOKE}
    P = H.Probe(); out["load_s"] = P.load_s
    print("loaded", H.MODEL_ID, f"{P.load_s:.0f}s", "layers", P.n_layers, "width", P.width, flush=True)
    g = H.published_graph(); ko = H.published_knockouts()
    published_tokens = g["prompts"]["preeclampsia"]
    prompt = H.ask_prompt(H.CASE_ORIGINAL)
    ids = P.encode(prompt)

    # ---- V0a
    pieces = P.pieces(ids)
    out["V0a"] = {"text_equal": prompt == "".join(published_tokens), "n_tokens": len(ids),
                  "tokens_equal": pieces == published_tokens}
    out["V0a"]["pass"] = out["V0a"]["text_equal"] and out["V0a"]["tokens_equal"]
    if not out["V0a"]["tokens_equal"]:
        out["V0a"]["first_mismatch"] = next(((i, a, b) for i, (a, b) in enumerate(zip(pieces, published_tokens)) if a != b), None)

    # ---- answer tokens
    vis = P.first_token_after(prompt, "visual"); prot = P.first_token_after(prompt, "protein")
    fam = sorted({P.first_token_after(prompt, w) for w in H.ASK_FAMILY_WORDS})
    out["tokens"] = {"visual": [vis, P.hf.decode([vis])], "protein": [prot, P.hf.decode([prot])],
                     "ask_family": [[t, P.hf.decode([t])] for t in fam]}

    # ---- intact run (records hub, stand-in and full layers for the C-act pools)
    rec = H.HUB + H.STANDIN
    t0 = time.time()
    probs, acts, gen, full, _ = P.run(ids, [[]], record=rec, n_new=16, record_layers_full=sorted({l for l, _ in H.HUB}))
    out["timing_single_s"] = time.time() - t0
    p_vis0, p_prot0 = float(probs[0, vis]), float(probs[0, prot])
    text0 = P.hf.decode(gen[0])
    out["V0b"] = {"p_vis": p_vis0, "p_prot": p_prot0, "text": text0, "published": ko["baseline"],
                  "pass": 0.75 <= p_vis0 <= 0.98 and text0.lstrip().lower().startswith("visual disturbances")}
    print("V0b", out["V0b"], flush=True)

    # ---- V0e
    here = g["neurons"]["preeclampsia"]
    ours, theirs, per = [], [], {}
    for (l, i) in H.HUB:
        key = f"{l}:{i}+"
        a = acts[(l, i)][0]
        per[key] = {"ours": [[p, float(a[p])] for p, _ in here[key]["here"]], "published": here[key]["here"]}
        for p, v in here[key]["here"]:
            ours.append(float(a[p])); theirs.append(v)
    rho = spearman(ours, theirs)
    peak = float(acts[(31, 5482)][0][75])
    out["V0e"] = {"spearman": rho, "L31_5482_tok75": peak, "published_peak": 46.5, "per_neuron": per,
                  "pass": rho >= 0.8 and abs(peak - 46.5) <= 0.3 * 46.5}
    print("V0e", {k: v for k, v in out["V0e"].items() if k != "per_neuron"}, flush=True)

    # ---- V0c with ablation fallback order
    out["V0c"] = {}
    mode_frozen = None
    for mode in ["zero", "zero_pos"]:
        pr, _, gn, _, _ = P.run(ids, [H.HUB], mode=mode, n_new=16)
        txt = P.hf.decode(gn[0]); pv = float(pr[0, vis])
        ok = pv <= 0.05 and ("itch" in txt.lower() or "prurit" in txt.lower())
        out["V0c"][mode] = {"p_vis": pv, "p_prot": float(pr[0, prot]), "text": txt, "pass": ok}
        print("V0c", mode, out["V0c"][mode], flush=True)
        if ok:
            mode_frozen = mode
            break
    out["V0c"]["published"] = ko["groups"][0]["ko"]
    out["V0c"]["pass"] = mode_frozen is not None
    out["ablation_mode"] = mode_frozen or "zero"
    mode = out["ablation_mode"]

    # ---- random controls
    rng = np.random.default_rng(H.SEED)
    exclude = H.published_group_neurons()
    hub_layers = [l for l, _ in H.HUB]

    def pick(layer, taken):
        while True:
            i = int(rng.integers(0, P.width))
            if (layer, i) not in exclude and (layer, i) not in taken:
                return (layer, i)

    c_layer = []
    for _ in range(K_MAX):
        s = []
        for l in hub_layers:
            s.append(pick(l, set(s)))
        c_layer.append(s)

    # C-act: same layer, whole-prompt max positive activation within +-30% (widen to 50%, then adjacent layers)
    pools, pool_rule = {}, {}
    for (l, i) in H.HUB:
        target = float(full[l][0, :, P.nid(l, i)[1]].max())
        for tol, layers in [(0.3, [l]), (0.5, [l]), (0.5, [l - 1, l, l + 1])]:
            cand = []
            for ll in layers:
                if ll not in full:
                    _, _, _, f2, _ = P.run(ids, [[]], record_layers_full=[ll]); full[ll] = f2[ll]
                mx_ = full[ll][0].max(0)
                for j in np.where(np.abs(mx_ - target) <= tol * target)[0]:
                    if (ll, int(j)) not in exclude and (ll, int(j)) != (l, i):
                        cand.append((ll, int(j)))
            if len(cand) >= 5:
                break
        pools[f"{l}:{i}"] = cand; pool_rule[f"{l}:{i}"] = {"target_max": target, "tol": tol, "layers": layers, "n": len(cand)}
    c_act = []
    for _ in range(K_MAX):
        s = []
        for (l, i) in H.HUB:
            pool = [c for c in pools[f"{l}:{i}"] if c not in s]
            s.append(tuple(pool[int(rng.integers(0, len(pool)))]))
        c_act.append(s)
    out["controls"] = {"C_layer": c_layer, "C_act": c_act, "C_act_pools": pool_rule}

    # ---- V0d (C-layer, frozen mode) and C-act on the original prompt, with timing
    def batched(conds, n_new=0):
        res = []
        for k in range(0, len(conds), BATCH):
            pr, _, gn, _, _ = P.run(ids, conds[k:k + BATCH], mode=mode, n_new=n_new)
            res += [(float(pr[b, vis]), P.hf.decode(gn[b])) for b in range(len(conds[k:k + BATCH]))]
        return res
    t0 = time.time()
    r_layer = batched(c_layer)
    t_batch = (time.time() - t0) / int(np.ceil(K_MAX / BATCH))
    r_act = batched(c_act)
    out["V0d"] = {"p_vis": [r[0] for r in r_layer], "n_ge_075": sum(r[0] >= 0.75 for r in r_layer),
                  "published_ctrl_vis": ko["groups"][0]["ko"]["ctrl_vis"]}
    out["V0d"]["pass"] = out["V0d"]["n_ge_075"] >= 18
    out["C_act_original"] = {"p_vis": [r[0] for r in r_act]}
    print("V0d", out["V0d"]["n_ge_075"], "/", K_MAX, "C-act p_vis", np.round(out["C_act_original"]["p_vis"], 3), flush=True)

    # ---- V0f (fidelity only)
    pr, _, gn, _, _ = P.run(ids, [H.STANDIN], mode=mode, n_new=16)
    pv = float(pr[0, vis])
    out["V0f"] = {"p_vis": pv, "text": P.hf.decode(gn[0]), "published": ko["groups"][1]["ko"], "pass": 0.05 <= pv <= 0.45}

    # ---- P1 neuron inclusion: post-case window max >= 20% of whole-prompt max on the original prompt
    case_end = len(P.encode(f"Human: {H.CASE_ORIGINAL}"))
    incl = {}
    for (l, i) in H.HUB:
        a = acts[(l, i)][0]
        whole, post = float(a.max()), float(a[case_end:].max())
        incl[f"{l}:{i}"] = {"whole_max": whole, "post_case_max": post, "included": post >= 0.2 * whole}
    out["P1_inclusion"] = incl

    # ---- timing and K: project the main run (cases packed CASES_PER_BATCH per pass, as in s1_run.py)
    conds42 = [[]] * (2 + 2 * K_MAX)
    t0 = time.time(); P.run(ids, conds42, mode=mode); t_prefill = time.time() - t0
    b0 = P.bytes_read; t0 = time.time(); P.run(ids, conds42[:2], mode=mode); t_small = time.time() - t0; b_pass = P.bytes_read - b0
    per_seq = max(t_prefill - t_small, 0) / (len(conds42) - 2)        # prefill compute per extra sequence
    t_pass = t_small                                                  # roughly one weight pass (+2 tiny sequences)
    def est(K):
        n_seq = PLANNED_CASES * 2 * (2 + 2 * K)
        n_batches = int(np.ceil(PLANNED_CASES / CASES_PER_BATCH))
        passes = n_batches * ((1 + N_NEW_ASK) + (1 + N_NEW_DX))
        return (n_seq * per_seq + passes * t_pass * 1.2 + PLANNED_S4 * t_pass) / 3600
    out["timing"] = {"stream": H.STREAM, "prefill_42_s": t_prefill, "pass_s": t_pass, "per_seq_s": per_seq,
                     "est_hours_K20": est(20), "est_hours_K10": est(10)}
    out["K"] = 20 if est(20) <= 6 else 10
    out["timing"]["GB_read_per_pass"] = b_pass / 1e9

    gate = ["V0a", "V0b", "V0c", "V0d", "V0e"]
    out["gate"] = {k: bool(out[k]["pass"]) for k in gate}
    out["gate_pass"] = all(out["gate"].values())
    path = H.save_json(out, "stage0_smoke.json" if H.SMOKE else "stage0.json")
    print("gate", out["gate"], "->", "PASS" if out["gate_pass"] else "FAIL", "| V0f", out["V0f"]["pass"],
          "| mode", mode, "| K", out["K"], "| est h", round(est(out["K"]), 2), "|", path, flush=True)
    sys.exit(0 if out["gate_pass"] or H.SMOKE else 1)


if __name__ == "__main__":
    main()
