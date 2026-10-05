#!/usr/bin/env python3
"""Audit filtered and unfiltered training sets with an independent cross-encoder."""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def audit(model, data_dir, rows_per_split, max_chars=2000):
    import torch
    res = {}
    for f in sorted(glob.glob(f"{data_dir}/train_*.jsonl")):
        split = Path(f).stem.replace("train_", "")
        S = []
        with open(f, encoding="utf-8") as fh, torch.no_grad():
            for i, line in enumerate(fh):
                if i >= rows_per_split:
                    break
                row = json.loads(line)
                keys = sorted((k for k in row if k.startswith("negative_")),
                              key=lambda s: int(s.split("_")[1]))
                docs = [row["positive"][:max_chars]] + [row[k][:max_chars] for k in keys]
                ranked = model.rerank(row["query"][:512], docs)
                sc = [0.0] * len(docs)
                for r in ranked:
                    sc[r["index"]] = float(r["relevance_score"])
                S.append(sc)
        A = np.array(S)
        pos, neg = A[:, 0], A[:, 1:]
        res[split] = {"rows": len(A),
                      "fn_rate": round(float((neg > pos[:, None]).mean()), 4),
                      "pos_at_1": round(float((neg.max(1) < pos).mean()), 4),
                      "margin": round(float((pos - neg.max(1)).mean()), 4)}
        log(f"  {split}: {res[split]}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--model", default="jinaai/jina-reranker-v3.5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModel
    log("loading jina-reranker-v3.5")
    model = AutoModel.from_pretrained(args.model, dtype="auto",
                                      trust_remote_code=True)
    model.eval().cuda()

    out = {}
    for d in args.dirs:
        log(f"=== {d}")
        out[d] = audit(model, d, args.rows)

    print(f"\n{'split':10s}" + "".join(f"{Path(d).name:>24s}" for d in args.dirs))
    splits = sorted({s for v in out.values() for s in v})
    for s in splits:
        line = f"{s:10s}"
        for d in args.dirs:
            v = out[d].get(s)
            line += (f"{100*v['fn_rate']:9.1f}% fn {100*v['pos_at_1']:5.1f}%@1"
                     if v else " " * 24)
        print(line)
    for d in args.dirs:
        v = out[d]
        tot = sum(x["rows"] for x in v.values())
        fn = sum(x["fn_rate"] * x["rows"] for x in v.values()) / tot
        p1 = sum(x["pos_at_1"] * x["rows"] for x in v.values()) / tot
        print(f"WEIGHTED {Path(d).name:24s} fn={100*fn:.1f}%  pos@1={100*p1:.1f}%")
    Path(args.out).write_text(json.dumps(out, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
