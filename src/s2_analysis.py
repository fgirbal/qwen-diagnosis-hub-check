"""Pre-registered analysis: P1-P4 and the decision rules (see PREREGISTRATION.md).

python -I s2_analysis.py --tag main
Writes results/decisions_<tag>.json.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import hubcommon as H

FAM = "preeclampsia_family"
MIN_P_VIS = 0.2           # P2 (ask frame) uses cases where the intact model asks about vision at all
MIN_CASES = 10            # fewer included cases than this -> "underpowered"


def manipulation_ok(c, r):
    lab, txt = r["dx"]["intact"]["label"], r["dx"]["intact"]["text"].lower()
    if c["set"] in ("S0", "S1"):
        return lab == FAM
    if c["set"] == "S2":
        return lab not in (FAM, "hellp") and any(k.lower() in txt for k in c.get("dx_keywords", []))
    return lab not in (FAM, "hellp")                      # S3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    sfx = "_smoke" if H.SMOKE else ""
    st0 = json.load(open(os.path.join(H.RESULTS, f"stage0{sfx}.json")))
    stim = json.load(open(H.STIMULI_FILE))
    by_id = {c["id"]: c for c in stim["cases"]}
    rows = [json.loads(l) for l in open(os.path.join(H.RESULTS, f"run_{args.tag}{sfx}.jsonl"))]
    runs = {r["id"]: r for r in rows if r["set"] != "S4"}
    s4 = next((r["rows"] for r in rows if r["set"] == "S4"), [])
    K = sum(k.startswith("C_act_") for k in next(iter(runs.values()))["ask"])
    out = {"tag": args.tag, "K": K}
    for r in runs.values():                       # D2: relabel every dx answer from its stored text
        for row in r["dx"].values():
            row["label"] = H.diagnosis_label(row["text"])

    # ---- inclusion
    incl, excl = {}, {}
    for cid, r in runs.items():
        ok = manipulation_ok(by_id[cid], r)
        (incl if ok else excl)[cid] = r
    out["inclusion"] = {s: {"included": sum(r["set"] == s for r in incl.values()),
                            "excluded": sorted(k for k, r in excl.items() if r["set"] == s)} for s in ["S0", "S1", "S2", "S3"]}

    # ---- P1: hub activation in the post-case window, twins vs their preeclampsia partners
    ref = st0["P1_inclusion"]
    neurons = [k for k, v in ref.items() if v["included"]]

    def hub_level(r):
        a = r["hub_acts"]
        return float(np.mean([max(a[n][r["post_case_start"]:]) / ref[n]["post_case_max"] for n in neurons]))
    ratios, pairs = [], []
    for cid, r in incl.items():
        c = by_id[cid]
        if c["set"] == "S2" and c.get("partner") in incl:
            rt = hub_level(r) / max(hub_level(incl[c["partner"]]), 1e-9)
            ratios.append(rt); pairs.append([cid, c["partner"], rt])
    m, lo, hi = H.bootstrap_ci(ratios)
    p1 = "diagnosis" if (m <= 0.5 and hi < 0.75) else ("wording" if m >= 0.8 else "intermediate")
    pre_bp = {}
    for s in ["S0", "S1", "S2", "S3"]:
        fr = []
        for r in incl.values():
            if r["set"] == s and r.get("bp_token"):
                fr += [max(r["hub_acts"][n][:r["bp_token"]]) / max(max(r["hub_acts"][n]), 1e-9) for n in neurons]
        pre_bp[s] = float(np.mean(fr)) if fr else None
    out["P1"] = {"neurons": neurons, "n_pairs": len(ratios), "ratio_mean": m, "ci95": [lo, hi], "pairs": pairs,
                 "fraction_of_peak_before_BP": pre_bp,
                 "outcome": "underpowered" if len(ratios) < MIN_CASES else p1}

    # ---- P2: generalization on S1
    s1 = [r for r in incl.values() if r["set"] == "S1"]
    use = [r for r in s1 if r["ask"]["intact"]["p_vis"] >= MIN_P_VIS]

    def drop(r, k):
        p0 = r["ask"]["intact"]["p_vis"]
        return (p0 - r["ask"][k]["p_vis"]) / p0
    d_tgt = float(np.mean([drop(r, "target") for r in use])) if use else float("nan")
    d_act = [float(np.mean([drop(r, f"C_act_{k}") for r in use])) for k in range(K)] if use else []
    d_lay = [float(np.mean([drop(r, f"C_layer_{k}") for r in use])) for k in range(K)] if use else []
    sw = lambda key: float(np.mean([r["dx"][key]["label"] != FAM for r in s1])) if s1 else float("nan")
    sw_tgt = sw("target"); sw_act = [sw(f"C_act_{k}") for k in range(K)]; sw_lay = [sw(f"C_layer_{k}") for k in range(K)]
    q = lambda x: float(np.quantile(x, 0.95)) if len(x) else float("nan")
    hub_ok = d_tgt >= 0.5 and d_tgt > q(d_act) and sw_tgt >= 0.5 and sw_tgt > q(sw_act)
    spec_ok = d_tgt < 0.25 or d_tgt <= q(d_act)
    p2 = "hub" if hub_ok else ("prompt_specific" if spec_ok else "intermediate")
    out["P2"] = {"n_S1_included": len(s1), "n_ask_used": len(use),
                 "ask_drop_target": d_tgt, "ask_drop_C_act_q95": q(d_act), "ask_drop_C_act": d_act, "ask_drop_C_layer": d_lay,
                 "dx_switch_target": sw_tgt, "dx_switch_C_act_q95": q(sw_act), "dx_switch_C_act": sw_act, "dx_switch_C_layer": sw_lay,
                 "outcome": "underpowered" if len(use) < MIN_CASES else p2}

    # ---- P3: specificity on S2 and S3
    other = [r for r in incl.values() if r["set"] in ("S2", "S3")]
    tv_tgt = float(np.mean([r["ask"]["target"]["tv"] for r in other])) if other else float("nan")
    tv_act = [float(np.mean([r["ask"][f"C_act_{k}"]["tv"] for r in other])) for k in range(K)] if other else []
    chg = lambda key: float(np.mean([r["dx"][key]["label"] != r["dx"]["intact"]["label"] for r in other])) if other else float("nan")
    out["P3"] = {"n": len(other), "tv_target": tv_tgt, "tv_C_act_q95": q(tv_act), "tv_C_act": tv_act,
                 "dx_label_change_target": chg("target"), "dx_label_change_C_act": [chg(f"C_act_{k}") for k in range(K)],
                 "outcome": "underpowered" if len(other) < MIN_CASES else ("specific" if tv_tgt <= q(tv_act) else "not_specific")}

    # ---- P4: label check on plain sentences
    if s4:
        whole = {k: v["whole_max"] for k, v in ref.items()}
        lvl = lambda row: float(np.mean([row["max"][n] / whole[n] for n in whole]))
        g = {}
        for row in s4:
            g.setdefault(row["group"], []).append(lvl(row))
        pe, op = np.mean(g.get("preeclampsia", [np.nan])), np.mean(g.get("other_pregnancy", [np.nan]))
        out["P4"] = {"group_means": {k: float(np.mean(v)) for k, v in g.items()}, "ratio": float(pe / op),
                     "outcome": "pass" if pe / op >= 3 else "fail"}

    # ---- decision
    if out["P1"]["outcome"] == "diagnosis" and out["P2"]["outcome"] == "hub" and out["P3"]["outcome"] == "specific":
        dec = "hub account supported"
    elif out["P2"]["outcome"] == "prompt_specific" and out["P1"]["outcome"] == "wording":
        dec = "hub account refuted (prompt-specific)"
    else:
        dec = "mixed"
    out["decision"] = dec
    path = H.save_json(out, f"decisions_{args.tag}{sfx}.json")
    print(json.dumps({k: (v["outcome"] if isinstance(v, dict) and "outcome" in v else v) for k, v in out.items()
                      if k in ("P1", "P2", "P3", "P4", "decision")}), "->", path)


if __name__ == "__main__":
    main()
