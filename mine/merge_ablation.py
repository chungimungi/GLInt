#!/usr/bin/env python3
"""Pair MaxSim and dense candidate sets on the same queries for the mining ablation."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

SPLITS = ["fiqa", "squadv2", "hotpotqa", "fever", "msmarco", "nq", "trivia"]


def load_parts(parts_dir: Path, split: str, top_k: int) -> dict[int, list[int]]:
    merged: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    files = sorted(parts_dir.glob(f"{split}_*.npz"))
    if not files:
        raise FileNotFoundError(f"no candidate shards for {split}")
    for path in files:
        with np.load(path) as data:
            for qid, ids, scores in zip(data["qids"], data["ids"], data["scores"]):
                qid = int(qid)
                if qid in merged:
                    old_ids, old_scores = merged[qid]
                    ids = np.concatenate((old_ids, ids))
                    scores = np.concatenate((old_scores, scores))
                merged[qid] = (ids, scores)

    result = {}
    for qid, (ids, scores) in merged.items():
        valid = (ids >= 0) & np.isfinite(scores)
        ids, scores = ids[valid], scores[valid]
        order = np.argsort(-scores, kind="stable")
        seen: set[int] = set()
        ordered = []
        for index in order:
            doc_id = int(ids[index])
            if doc_id not in seen:
                seen.add(doc_id)
                ordered.append(doc_id)
                if len(ordered) >= top_k:
                    break
        result[qid] = ordered
    return result


def load_labels(data_root: Path, split: str):
    import pyarrow.parquet as pq

    positives: dict[int, list[int]] = {}
    for path in sorted((data_root / "scores").glob(f"{split}-*.parquet")):
        table = pq.read_table(path, columns=["query_id", "document_ids"])
        for qid, ids in zip(table["query_id"].to_pylist(),
                             table["document_ids"].to_pylist()):
            if ids:
                bucket = positives.setdefault(int(qid), [])
                if int(ids[0]) not in bucket:
                    bucket.append(int(ids[0]))

    queries: dict[int, str] = {}
    for path in sorted((data_root / "queries").glob(f"{split}-*.parquet")):
        table = pq.read_table(path, columns=["query_id", "query"])
        queries.update(zip(map(int, table["query_id"].to_pylist()),
                           table["query"].to_pylist()))
    return positives, queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--maxsim-parts", required=True,
                        help="WARP candidate shards from mine/build.py")
    parser.add_argument("--dense-parts", required=True,
                        help="dense candidate shards from mine/mine_dense.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    maxsim_dir = Path(args.maxsim_parts).expanduser().resolve()
    dense_dir = Path(args.dense_parts).expanduser().resolve()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output.open("w", encoding="utf-8") as stream:
        for split in splits:
            maxsim = load_parts(maxsim_dir, split, args.top_k)
            dense = load_parts(dense_dir, split, args.top_k)
            positives, queries = load_labels(data_root, split)
            written = 0
            for qid, positive_ids in positives.items():
                if qid not in maxsim or qid not in dense or qid not in queries:
                    continue
                positive_set = set(positive_ids)
                mv_ids = [doc_id for doc_id in maxsim[qid] if doc_id not in positive_set]
                dense_ids = [doc_id for doc_id in dense[qid] if doc_id not in positive_set]
                if len(mv_ids) < args.top_k or len(dense_ids) < args.top_k:
                    continue
                for positive_id in positive_ids:
                    row = {
                        "split": split,
                        "query_id": qid,
                        "query": queries[qid],
                        "positive_id": positive_id,
                        "negative_ids": mv_ids[:args.top_k] + dense_ids[:args.top_k],
                        "negative_sources": ["mv"] * args.top_k + ["dense"] * args.top_k,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
            total += written
            print(f"{split}: {written:,} paired query-positive rows", flush=True)
    print(f"wrote {total:,} rows", flush=True)


if __name__ == "__main__":
    main()
