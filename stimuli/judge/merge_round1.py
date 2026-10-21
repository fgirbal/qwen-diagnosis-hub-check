"""Merge round-1 rewrites into round1.json, rerun scripted checks for them, and write a blind J1 input for the
rewritten items only (fresh neutral IDs, shuffled)."""
import json, sys, random, re

S = sys.argv[1]
d0 = json.load(open(f"{S}/gen/round0.json")); rw = json.load(open(f"{S}/gen/round1_rewrites.json"))["cases"]
new = {c["id"]: c for c in rw}
d1 = {"cases": [new.get(c["id"], c) for c in d0["cases"]], "s4": d0["s4"]}
json.dump(d1, open(f"{S}/gen/round1.json", "w"), indent=1)

diag = r"pre-?eclampsia|eclampsia|toxemia|hellp|gestational hypertension|pregnancy-induced|cholecystitis|appendicitis|pyelonephritis|hepatitis|cholestasis"
vis = r"visual|vision|blurr|seeing spots|scotoma|photophob|\beyes?\b|itch|prurit"
sc = json.load(open(f"{S}/judge/scripted_round0.json"))
for cid, c in new.items():
    t, errs = c["case"], []
    if re.search(diag, t, re.I): errs.append("names a diagnosis")
    if re.search(vis, t, re.I): errs.append("visual/itch word")
    if c["bp_text"] not in t: errs.append("bp_text not a substring")
    if c.get("finding_text") and c["finding_text"] not in t: errs.append("finding_text not a substring")
    n = len(t.split())
    if not 25 <= n <= 45: errs.append(f"length {n} words")
    sc["J4"][cid] = {"pass": not errs, "errors": errs, "round": 1}
json.dump(sc, open(f"{S}/judge/scripted_round1.json", "w"), indent=1)

items = list(new.values()); random.Random(20261020).shuffle(items)
key = {f"W{k + 1:03d}": c["id"] for k, c in enumerate(items)}
key_all = json.load(open(f"{S}/judge/key_round0.json"))
key_all = {k: v for k, v in key_all.items() if v not in new}; key_all.update(key)
json.dump(key_all, open(f"{S}/judge/key_round1.json", "w"), indent=1)
with open(f"{S}/judge/j1_input_r1.txt", "w") as f:
    for k, c in enumerate(items):
        f.write(f"W{k + 1:03d}: {c['case']}\n")
print("rewrites", len(new), "| J4 fails", {k: v["errors"] for k, v in sc["J4"].items() if k in new and not v["pass"]})
for c in items[:3]: print(" ", c["id"], "|", c["case"], "|", c["intended"])
