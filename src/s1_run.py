"""Main run (and the debug run): every case in both frames, under intact, hub removal and the random controls.

python -I s1_run.py --tag main            # the hub, all cases, K from stage0.json
python -I s1_run.py --tag debug           # stand-in group (2 "microangiopathic" neurons), 10 cases per set, K = 5

Reads stimuli/stimuli.json and <results>/stage0.json; appends one line per case to <results>/run_<tag>.jsonl and
skips cases already there, so an interrupted run can be restarted. Several cases are packed into one batch
(about B_TARGET sequences: every case x every condition) so that, when the model is streamed from disk, each
pass over the weights serves many prompts. Prompts of different lengths are left-padded and masked.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import hubcommon as H

B_TARGET = 252            # sequences per batch (6 cases x 42 conditions in the main run)
N_NEW_ASK, N_NEW_DX = 8, 12


def post_case_start(P, prompt, case):
    ids, offs = P.offsets(prompt)
    end = len("Human: ") + len(case)
    return next(k for k, (s, e) in enumerate(offs) if s >= end or e > end)


def char_to_token(P, prompt, char_idx):
    ids, offs = P.offsets(prompt)
    return next(k for k, (s, e) in enumerate(offs) if e > char_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["main", "debug"], required=True)
    args = ap.parse_args()
    st0 = json.load(open(os.path.join(H.RESULTS, "stage0_smoke.json" if H.SMOKE else "stage0.json")))
    if not (st0["gate_pass"] or H.SMOKE):
        sys.exit("stage 0 gate failed: not running")
    stim = json.load(open(H.STIMULI_FILE))
    mode = st0["ablation_mode"]
    if args.tag == "main":
        target, K = H.HUB, st0["K"]
        cases = stim["cases"]
    else:
        target, K = H.STANDIN, 5
        rng = np.random.default_rng(H.SEED)
        cases = []
        for s in ["S0", "S1", "S2", "S3"]:
            pool = [c for c in stim["cases"] if c["set"] == s]
            cases += [pool[i] for i in sorted(rng.choice(len(pool), size=min(10, len(pool)), replace=False))]
    c_layer = [[tuple(x) for x in s] for s in st0["controls"]["C_layer"][:K]]
    c_act = [[tuple(x) for x in s] for s in st0["controls"]["C_act"][:K]]
    conds = [[], target] + c_layer + c_act
    names = ["intact", "target"] + [f"C_layer_{k}" for k in range(K)] + [f"C_act_{k}" for k in range(K)]
    nc = len(conds)
    per_batch = max(1, B_TARGET // nc)

    out_path = os.path.join(H.RESULTS, f"run_{args.tag}{'_smoke' if H.SMOKE else ''}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path)}
    todo = [c for c in cases if c["id"] not in done]
    P = H.Probe()
    print("loaded", f"{P.load_s:.0f}s", "| stream", H.STREAM, "| cases", len(cases), "| done", len(done), "| K", K,
          "| mode", mode, "| cases per batch", per_batch, flush=True)
    t_start = time.time()

    def run_frame(chunk, frame, n_new, record):
        prompts = [H.ask_prompt(c["case"]) if frame == "ask" else H.dx_prompt(c["case"]) for c in chunk]
        ids = [P.encode(p) for p in prompts]
        seq_ids = [i for i in ids for _ in conds]
        seq_conds = [cd for _ in ids for cd in conds]
        pr, acts, gn, _, pads = P.run(seq_ids, seq_conds, mode=mode, record=record, n_new=n_new)
        out = []
        for j, prompt in enumerate(prompts):
            vis = P.first_token_after(prompt, "visual")
            fam = sorted({P.first_token_after(prompt, w) for w in H.ASK_FAMILY_WORDS})
            base = j * nc
            p0 = np.array(pr[base])
            rows = []
            for k in range(nc):
                p = p0 if k == 0 else np.array(pr[base + k])
                rows.append({"p_vis": float(p[vis]), "p_fam": float(p[fam].sum()), "tv": float(0.5 * np.abs(p - p0).sum()),
                             "top5": [[int(t), P.hf.decode([int(t)]), float(p[t])] for t in np.argsort(-p)[:5]],
                             "text": P.hf.decode(gn[base + k])})
            a0 = None
            if record:
                a0 = {f"{l}:{i}": acts[(l, i)][base][pads[base]:].round(3).tolist() for (l, i) in record}
            out.append((rows, a0, len(ids[j])))
        return out

    with open(out_path, "a") as f:
        for b in range(0, len(todo), per_batch):
            chunk = todo[b:b + per_batch]
            t0 = time.time()
            ask = run_frame(chunk, "ask", N_NEW_ASK, H.HUB + H.STANDIN)
            dx = run_frame(chunk, "dx", N_NEW_DX, None)
            for c, (ask_rows, acts, n_tok), (dx_rows, _, _) in zip(chunk, ask, dx):
                for r in dx_rows:
                    r["label"] = H.diagnosis_label(r["text"])
                pa = H.ask_prompt(c["case"])
                rec = {"id": c["id"], "set": c["set"], "n_tokens": n_tok,
                       "post_case_start": post_case_start(P, pa, c["case"]),
                       "bp_token": char_to_token(P, pa, len("Human: ") + c["case"].index(c["bp_text"])) if c.get("bp_text") else None,
                       "hub_acts": acts,
                       "ask": dict(zip(names, ask_rows)), "dx": dict(zip(names, dx_rows))}
                f.write(json.dumps(rec) + "\n")
                print(f"  {c['id']} | p_vis intact {ask_rows[0]['p_vis']:.3f} target {ask_rows[1]['p_vis']:.3f} "
                      f"| dx {dx_rows[0]['label']}->{dx_rows[1]['label']}", flush=True)
            f.flush()
            el = time.time() - t_start
            n_done = b + len(chunk)
            print(f"batch {b // per_batch + 1}: {len(chunk)} cases in {time.time() - t0:.0f}s | {n_done}/{len(todo)} "
                  f"| {el / 60:.1f} min elapsed, ~{el / n_done * (len(todo) - n_done) / 60:.0f} min left", flush=True)

        # S4 plain sentences, intact only (main run only), one batch
        if args.tag == "main" and "s4" in stim and "S4" not in done:
            sents = stim["s4"]
            ids = [P.encode(s["text"]) for s in sents]
            _, acts, _, _, pads = P.run(ids, [[]] * len(ids), record=H.HUB)
            s4 = [{"id": s["id"], "group": s["group"],
                   "max": {f"{l}:{i}": float(acts[(l, i)][j][pads[j]:].max()) for (l, i) in H.HUB}}
                  for j, s in enumerate(sents)]
            f.write(json.dumps({"id": "S4", "set": "S4", "rows": s4}) + "\n")
    print("done", out_path, f"{(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
