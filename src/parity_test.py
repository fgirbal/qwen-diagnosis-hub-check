"""Parity test for the streaming engine and padded batches (run on the 4-bit model, which fits in memory).

python -I parity_test.py dump resident   # HUB_STREAM unset
HUB_STREAM=1 python -I parity_test.py dump stream
python -I parity_test.py compare
Writes <results>/parity_<name>.npz and <results>/parity.json (all in results/parity/ via HUB_RESULTS_TAG=parity).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import hubcommon as H


def dump(name):
    P = H.Probe()
    stim = json.load(open(H.STIMULI_FILE))
    by = {c["id"]: c for c in stim["cases"]}
    prompts = [H.ask_prompt(H.CASE_ORIGINAL), H.ask_prompt(by["S1-01"]["case"]), H.dx_prompt(by["S3-20"]["case"])]
    ids = [P.encode(p) for p in prompts]
    out = {}
    # (1) original prompt alone: intact + hub removed, greedy 6, hub activations
    pr, acts, gn, _, _ = P.run(ids[0], [[], H.HUB], record=H.HUB, n_new=6)
    out["single_probs"] = np.array(pr)[:, np.argsort(-np.array(pr[0]))[:50]]
    out["single_top_idx"] = np.argsort(-np.array(pr[0]))[:50]
    out["single_gen"] = np.array(gn)
    for (l, i) in H.HUB:
        out[f"single_act_{l}_{i}"] = acts[(l, i)]
    # (2) each prompt alone vs all three padded in one batch (intact + hub each)
    for j, s in enumerate(ids):
        pr, acts, gn, _, _ = P.run(s, [[], H.HUB], record=H.HUB, n_new=4)
        out[f"alone{j}_probs"] = np.array(pr); out[f"alone{j}_gen"] = np.array(gn)
        out[f"alone{j}_act"] = acts[H.HUB[0]]
    seq = [s for s in ids for _ in range(2)]; cond = [[], H.HUB] * len(ids)
    pr, acts, gn, _, pads = P.run(seq, cond, record=H.HUB, n_new=4)
    out["padded_probs"] = np.array(pr); out["padded_gen"] = np.array(gn); out["padded_pads"] = np.array(pads)
    out["padded_act"] = acts[H.HUB[0]]
    out["bytes_read"] = np.array(P.bytes_read)
    np.savez(os.path.join(H.RESULTS, f"parity_{name}.npz"), **out)
    print("dumped", name, "bytes read", P.bytes_read / 1e9, "GB")


def compare():
    st0 = json.load(open(os.path.join(H.ROOT, "results", "stage0.json")))      # the 4-bit Stage 0 run (old code)
    r = dict(np.load(os.path.join(H.RESULTS, "parity_resident.npz")))
    s = dict(np.load(os.path.join(H.RESULTS, "parity_stream.npz")))
    rep = {}
    # A: new resident code reproduces the old Stage 0 numbers
    top = r["single_top_idx"]
    import hubcommon as _  # noqa
    vis_rank = None
    p_int_vis = None
    # visual token id from stage0
    vis = st0["tokens"]["visual"][0]
    k = int(np.where(top == vis)[0][0]) if vis in top else None
    rep["A_old_vs_new_resident"] = {
        "p_vis_old": st0["V0b"]["p_vis"], "p_vis_new": float(r["single_probs"][0, k]) if k is not None else None,
        "p_vis_ko_old": st0["V0c"]["zero"]["p_vis"], "p_vis_ko_new": float(r["single_probs"][1, k]) if k is not None else None,
        "L31_5482_tok75_old": st0["V0e"]["L31_5482_tok75"], "L31_5482_tok75_new": float(r["single_act_31_5482"][0][75])}
    # B: streaming == resident
    rep["B_stream_vs_resident"] = {
        "max_abs_prob_diff_top50": float(np.abs(r["single_probs"] - s["single_probs"]).max()),
        "same_top_idx": bool((r["single_top_idx"] == s["single_top_idx"]).all()),
        "same_greedy": bool((r["single_gen"] == s["single_gen"]).all()),
        "max_abs_act_diff": float(max(np.abs(r[f"single_act_{l}_{i}"] - s[f"single_act_{l}_{i}"]).max() for (l, i) in H.HUB)),
        "padded_max_abs_prob_diff": float(np.abs(r["padded_probs"] - s["padded_probs"]).max()),
        "padded_same_greedy": bool((r["padded_gen"] == s["padded_gen"]).all())}
    # C: padded batch == each prompt alone (both engines)
    for name, d in [("resident", r), ("stream", s)]:
        diffs, gens, acts = [], [], []
        pads = d["padded_pads"]
        for j in range(3):
            diffs.append(float(np.abs(d[f"alone{j}_probs"] - d["padded_probs"][2 * j:2 * j + 2]).max()))
            gens.append(bool((d[f"alone{j}_gen"] == d["padded_gen"][2 * j:2 * j + 2]).all()))
            a_alone = d[f"alone{j}_act"][0]; a_pad = d["padded_act"][2 * j][pads[2 * j]:]
            acts.append(float(np.abs(a_alone - a_pad).max()) if a_alone.shape == a_pad.shape else "shape mismatch")
        rep[f"C_padded_vs_alone_{name}"] = {"max_abs_prob_diff": diffs, "same_greedy": gens, "max_abs_act_diff": acts,
                                            "pads": pads.tolist()}
    rep["bytes_read_stream_GB"] = float(s["bytes_read"]) / 1e9
    json.dump(rep, open(os.path.join(H.RESULTS, "parity.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        compare()
