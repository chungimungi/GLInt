#!/usr/bin/env python3
"""Build fixed-width, seven-source candidate lists for listwise distillation."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ["fiqa", "squadv2", "hotpotqa", "fever", "msmarco", "nq", "trivia"]
NS = 10 ** 9  # id namespace stride


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pool-dir", required=True,
                    help="directory containing per-query judge score shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ways", type=int, default=32)
    ap.add_argument("--max-queries-per-split", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    pool_dir = os.path.abspath(os.path.expanduser(args.pool_dir))

    os.makedirs(args.out, exist_ok=True)
    for sub in ("queries", "documents", "train"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    all_q, all_d, all_rows = [], [], []
    meta = {}
    for si, split in enumerate(SPLITS):
        t0 = time.time()
        base = si * NS

        # ---- queries and documents text for this split
        qtab = pa.concat_tables([pq.read_table(p) for p in sorted(
            glob.glob(f"{data_root}/queries/{split}-*.parquet"))])
        dtab = pa.concat_tables([pq.read_table(p) for p in sorted(
            glob.glob(f"{data_root}/documents/{split}-*.parquet"))])
        qtext = dict(zip(qtab["query_id"].to_pylist(), qtab["query"].to_pylist()))
        dtext = dict(zip(dtab["document_id"].to_pylist(), dtab["document"].to_pylist()))

        # ---- judged pool: gold + top candidates by judge score
        used_docs: set[int] = set()
        used_qs: list[tuple[int, str]] = []
        n_rows = n_short = 0
        for f in sorted(glob.glob(f"{pool_dir}/{split}_*.npz")):
            b = np.load(f, allow_pickle=True)
            qids = b["qids"]
            gi, gs = b["gold_ids"], b["gold_scores"]
            ci, cs = b["cand_ids"], b["cand_scores"]
            order = np.argsort(-np.where(np.isfinite(cs), cs, -np.inf), axis=1)
            for r in range(len(qids)):
                if args.max_queries_per_split and n_rows >= args.max_queries_per_split:
                    break
                q = int(qids[r])
                if q not in qtext:
                    continue
                docs: list[int] = []
                seen: set[int] = set()
                for g, sc in zip(gi[r], gs[r]):
                    g = int(g)
                    if g >= 0 and np.isfinite(sc) and g not in seen and g in dtext:
                        seen.add(g)
                        docs.append(g)
                for j in order[r]:
                    if len(docs) >= args.n_ways:
                        break
                    c = int(ci[r][j])
                    if c < 0 or not np.isfinite(cs[r][j]) or c in seen or c not in dtext:
                        continue
                    seen.add(c)
                    docs.append(c)
                if len(docs) < args.n_ways:
                    n_short += 1
                    continue
                docs = docs[: args.n_ways]
                used_docs.update(docs)
                used_qs.append((base + q, qtext[q]))
                all_rows.append((base + q, [base + d for d in docs]))
                n_rows += 1

        all_q.extend(used_qs)
        all_d.extend((base + d, dtext[d]) for d in used_docs)
        meta[split] = {"rows": n_rows, "short_dropped": n_short,
                       "docs_used": len(used_docs), "secs": round(time.time() - t0, 1)}
        log(f"{split}: {n_rows:,} rows, {n_short:,} dropped(<{args.n_ways}), "
            f"{len(used_docs):,} docs, {time.time() - t0:.0f}s")

    log(f"TOTAL rows={len(all_rows):,} queries={len(all_q):,} documents={len(all_d):,}")

    pq.write_table(pa.table({
        "query_id": pa.array([q for q, _ in all_q], pa.int64()),
        "text": pa.array([t for _, t in all_q], pa.string()),
    }), os.path.join(args.out, "queries", "train.parquet"))
    pq.write_table(pa.table({
        "document_id": pa.array([d for d, _ in all_d], pa.int64()),
        "text": pa.array([t for _, t in all_d], pa.string()),
    }), os.path.join(args.out, "documents", "train.parquet"))
    zeros = [0.0] * 32
    pq.write_table(pa.table({
        "query_id": pa.array([q for q, _ in all_rows], pa.int64()),
        "document_ids": pa.array([ds for _, ds in all_rows], pa.list_(pa.int64())),
        "scores": pa.array([zeros[: len(ds)] for _, ds in all_rows],
                           pa.list_(pa.float32())),
    }), os.path.join(args.out, "train", "train.parquet"))

    meta["namespace_stride"] = NS
    meta["splits"] = SPLITS
    meta["n_ways"] = args.n_ways
    with open(os.path.join(args.out, "build_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
