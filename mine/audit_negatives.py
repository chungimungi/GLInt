#!/usr/bin/env python3
"""Compare MaxSim-mined and dense-mined candidates with an independent cross-encoder."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

DATA_ROOT = ""
SPLITS = ["fiqa", "squadv2", "hotpotqa", "fever", "msmarco", "nq", "trivia"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class DocStore:
    def __init__(self, split):
        import pyarrow as pa
        import pyarrow.parquet as pq
        tables = [pq.read_table(p, columns=["document_id", "document"], memory_map=True)
                  for p in sorted(glob.glob(f"{DATA_ROOT}/documents/{split}-*.parquet"))]
        t = pa.concat_tables(tables)
        ids = t.column("document_id").to_numpy()
        self.texts = t.column("document")
        o = np.argsort(ids, kind="stable")
        self.ids_sorted, self.order = ids[o], o

    def get(self, doc_id):
        p = np.searchsorted(self.ids_sorted, doc_id)
        if p >= len(self.ids_sorted) or self.ids_sorted[p] != doc_id:
            return None
        return self.texts[int(self.order[p])].as_py()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source", required=True,
                    help="combined output from mine/merge_ablation.py")
    ap.add_argument("--rows", type=int, default=250, help="queries audited per split")
    ap.add_argument("--top", type=int, default=10, help="negatives per arm")
    ap.add_argument("--model", default="jinaai/jina-reranker-v3.5")
    ap.add_argument("--max-doc-chars", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global DATA_ROOT
    DATA_ROOT = str(Path(args.data_root).expanduser().resolve())

    import torch
    from transformers import AutoModel

    by_split = {s: [] for s in SPLITS}
    with open(args.source, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            b = by_split.get(rec["split"])
            if b is not None and len(b) < args.rows:
                b.append(rec)
        # a single pass keeps the first N pairs per split; file order is split-grouped so
        # this is a contiguous sample, not a random one -- fine for a relative comparison
    log("loading jina-reranker-v3.5")
    model = AutoModel.from_pretrained(args.model, dtype="auto", trust_remote_code=True)
    model.eval().cuda()

    results = {}
    print(f"\n{'split':10s} {'n':>4s} | {'MV false':>8s} {'dense false':>11s} | "
          f"{'MV hard':>8s} {'dense hard':>10s} | {'overlap':>7s}")
    print("-" * 74)
    for split in SPLITS:
        rows = by_split[split]
        if not rows:
            continue
        store = DocStore(split)
        mv_false, dn_false, mv_hard, dn_hard, ov = [], [], [], [], []
        with torch.no_grad():
            for rec in rows:
                srcs = rec["negative_sources"]
                ids = rec["negative_ids"]
                mv = [i for i, s in zip(ids, srcs) if s == "mv"][: args.top]
                dn = [i for i, s in zip(ids, srcs) if s == "dense"][: args.top]
                if len(mv) < args.top or len(dn) < args.top:
                    continue
                ptext = store.get(rec["positive_id"])
                if not ptext:
                    continue
                uniq, texts = [], []
                for d in [rec["positive_id"]] + mv + dn:
                    if d in uniq:
                        continue
                    t = store.get(d)
                    if t:
                        uniq.append(d)
                        texts.append(t[: args.max_doc_chars])
                if len(texts) < 3:
                    continue
                ranked = model.rerank(rec["query"][:512], texts)
                sc = {uniq[r["index"]]: float(r["relevance_score"]) for r in ranked}
                ps = sc.get(rec["positive_id"])
                if ps is None:
                    continue
                mvs = [sc[d] for d in mv if d in sc]
                dns = [sc[d] for d in dn if d in sc]
                if not mvs or not dns:
                    continue
                mv_false.append(np.mean([s > ps for s in mvs]))
                dn_false.append(np.mean([s > ps for s in dns]))
                mv_hard.append(np.mean(mvs))
                dn_hard.append(np.mean(dns))
                ov.append(len(set(mv) & set(dn)) / args.top)
        if not mv_false:
            continue
        r = {"n": len(mv_false),
             "mv_false_negative_rate": round(float(np.mean(mv_false)), 4),
             "dense_false_negative_rate": round(float(np.mean(dn_false)), 4),
             "mv_mean_negative_score": round(float(np.mean(mv_hard)), 4),
             "dense_mean_negative_score": round(float(np.mean(dn_hard)), 4),
             "set_overlap": round(float(np.mean(ov)), 4)}
        results[split] = r
        print(f"{split:10s} {r['n']:4d} | {100 * r['mv_false_negative_rate']:7.1f}% "
              f"{100 * r['dense_false_negative_rate']:10.1f}% | "
              f"{r['mv_mean_negative_score']:8.3f} {r['dense_mean_negative_score']:10.3f} | "
              f"{100 * r['set_overlap']:6.1f}%", flush=True)
        del store

    if results:
        w = {s: r["n"] for s, r in results.items()}
        tot = sum(w.values())
        def wavg(k):
            return sum(results[s][k] * w[s] for s in results) / tot
        print("-" * 74)
        print(f"{'WEIGHTED':10s} {tot:4d} | {100 * wavg('mv_false_negative_rate'):7.1f}% "
              f"{100 * wavg('dense_false_negative_rate'):10.1f}% | "
              f"{wavg('mv_mean_negative_score'):8.3f} "
              f"{wavg('dense_mean_negative_score'):10.3f} | "
              f"{100 * wavg('set_overlap'):6.1f}%")
        Path(args.out).write_text(json.dumps(results, indent=2))
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
