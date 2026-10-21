"""Apply the pre-registered judge pass rules (J1-J4) and write a per-item report."""
import json, sys

S = sys.argv[1]; rnd = sys.argv[2] if len(sys.argv) > 2 else "0"
d = json.load(open(f"{S}/gen/round{rnd}.json")); cases = {c["id"]: c for c in d["cases"]}
j1 = json.load(open(f"{S}/judge/j1_output{'' if rnd == '0' else '_r' + rnd}.json"))
key = json.load(open(f"{S}/judge/key_round{rnd}.json"))
j2 = json.load(open(f"{S}/judge/j2_output.json")); key2 = json.load(open(f"{S}/judge/key2_round0.json"))
scripted = json.load(open(f"{S}/judge/scripted_round{rnd}.json"))
FAM = ("eclampsia", "hellp")
PREG = ("pregnan", "gestation", "eclampsia", "hellp", "obstetric", "placenta", "hyperemesis", "cholestasis of pregnancy", "fatty liver of pregnancy")

rep = {}
for vid, cid in key.items():
    c, v = cases[cid], j1[vid]
    dx, conf = v["dx"].lower(), int(v["confidence"])
    if c["set"] in ("S0", "S1"):
        ok = any(f in dx for f in FAM)
    elif c["set"] == "S2":
        ok = not any(f in dx for f in FAM) and any(k.lower() in dx for k in c["dx_keywords"])
    else:
        ok = not any(p in dx for p in PREG)
    r = {"set": c["set"], "intended": c["intended"], "judge_dx": v["dx"], "confidence": conf, "note": v.get("note", ""),
         "J1_pass": bool(ok and conf >= 3)}
    if c["set"] == "S1":
        rid = next(k for k, x in key2.items() if x == cid)
        r["J2"] = j2["items"][rid]; r["J2_pass"] = j2["items"][rid]["same_picture"] == "yes"
    if c["set"] == "S2":
        r["J3_pass"] = scripted["J3"][cid]["pass"]
    r["J4_pass"] = scripted["J4"][cid]["pass"]
    r["pass"] = r["J1_pass"] and r.get("J2_pass", True) and r.get("J3_pass", True) and r["J4_pass"]
    rep[cid] = r
fails = {k: v for k, v in sorted(rep.items()) if not v["pass"]}
by_set = {}
for k, v in rep.items():
    s = by_set.setdefault(v["set"], [0, 0]); s[0] += v["pass"]; s[1] += 1
print("pass by set:", by_set)
for k, v in fails.items():
    print(f"  FAIL {k} intended={v['intended']!r} judge={v['judge_dx']!r} conf={v['confidence']} note={v['note']!r}")
json.dump(rep, open(f"{S}/judge/report_round{rnd}.json", "w"), indent=1)
