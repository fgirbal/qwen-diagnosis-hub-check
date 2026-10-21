"""J3 twin minimality (scripted): a twin equals its partner with the BP numbers changed (same count of BP readings,
all in 104-124 / 62-80) and exactly one contiguous span (possibly empty, i.e. an insertion) replaced by finding_text."""
import json, sys, re
BP = re.compile(r"\b\d{2,3}/\d{2,3}\b")


def j3(c, partner):
    t, p = c["case"], partner["case"]
    errs = []
    if c["finding_text"] not in t:
        return ["finding_text not a substring"]
    tn, pn = BP.sub("#BP#", t), BP.sub("#BP#", p)
    if tn.count("#BP#") != pn.count("#BP#") or tn.count("#BP#") == 0:
        errs.append("BP reading count differs")
    pre, suf = tn.split(BP.sub("#BP#", c["finding_text"]), 1)
    pre, suf = pre.rstrip(), suf.lstrip()
    if not (pn.startswith(pre) and pn.endswith(suf) and len(pre) + len(suf) <= len(pn)):
        errs.append("twin is not partner + one replaced span + BP change")
    for b in BP.findall(t):
        s, d = map(int, b.split("/"))
        if not (104 <= s <= 124 and 62 <= d <= 80):
            errs.append(f"twin BP out of range {b}")
    return errs


if __name__ == "__main__":
    d = json.load(open(sys.argv[1])); cases = {c["id"]: c for c in d["cases"]}
    res = {c["id"]: {"pass": not (e := j3(c, cases[c["partner"]])), "errors": e} for c in d["cases"] if c["set"] == "S2"}
    print(json.dumps({k: v for k, v in res.items() if not v["pass"]}), f"| {sum(v['pass'] for v in res.values())}/{len(res)} pass")
    json.dump(res, open(sys.argv[2], "w"), indent=1)
