#!/usr/bin/env python3
"""Merge sharded candidates into query-positive pairs with known positives excluded."""

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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_positives(split: str) -> dict[int, list[int]]:
    """query_id -> every annotated positive, in first-seen order."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out: dict[int, list[int]] = {}
    for path in sorted(glob.glob(f"{DATA_ROOT}/scores/{split}-*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=8192, columns=["query_id", "document_ids"]):
            qids = batch.column("query_id").to_numpy(zero_copy_only=False)
            firsts = pc.list_element(batch.column("document_ids"), 0).to_numpy(
                zero_copy_only=False)
            for qid, pos in zip(qids, firsts):
                bucket = out.setdefault(int(qid), [])
                if int(pos) not in bucket:
                    bucket.append(int(pos))
    return out


def load_query_text(split: str) -> dict[int, str]:
    import pyarrow.parquet as pq

    out: dict[int, str] = {}
    for path in sorted(glob.glob(f"{DATA_ROOT}/queries/{split}-*.parquet")):
        t = pq.read_table(path, columns=["query_id", "query"])
        for qid, text in zip(t.column("query_id").to_pylist(), t.column("query").to_pylist()):
            out[int(qid)] = text
    return out


def load_candidates(split: str, work: Path) -> dict[int, list[int]]:
    """query_id -> mined candidate ids, best first, merged across shards by score."""
    parts = sorted((work / "parts").glob(f"{split}_*.npz"))
    if not parts:
        return {}
    merged_ids, merged_sc, qids_ref = None, None, None
    for path in parts:
        blob = np.load(path)
        qids, ids, sc = blob["qids"], blob["ids"], blob["scores"]
        if merged_ids is None:
            qids_ref, merged_ids, merged_sc = qids, ids, sc
            continue
        assert np.array_equal(qids, qids_ref), f"{path.name} query order differs"
        merged_ids = np.concatenate([merged_ids, ids], axis=1)
        merged_sc = np.concatenate([merged_sc, sc], axis=1)
    order = np.argsort(-merged_sc, axis=1)
    merged_ids = np.take_along_axis(merged_ids, order, axis=1)
    return {int(q): row.tolist() for q, row in zip(qids_ref, merged_ids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="local cache of the public embeddings-fine-tuning dataset")
    ap.add_argument("--work-dir", default="outputs/mining")
    ap.add_argument("--out", default="outputs/mined_pairs.jsonl")
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--splits", default=",".join(SPLITS))
    args = ap.parse_args()
    global DATA_ROOT
    DATA_ROOT = str(Path(args.data_root).expanduser().resolve())

    work = Path(args.work_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats, written_total = {}, 0
    with out.open("w", encoding="utf-8") as fh:
        for split in [s for s in SPLITS if s in {x.strip() for x in args.splits.split(",")}]:
            log(f"{split}: loading")
            positives = load_positives(split)
            cands = load_candidates(split, work)
            if not cands:
                log(f"  {split}: no mined parts, skipping")
                continue
            qtext = load_query_text(split)

            written = short = no_cand = 0
            multi = 0
            for qid, pos_list in positives.items():
                cand = cands.get(qid)
                if not cand:
                    no_cand += 1
                    continue
                banned = set(pos_list)
                negatives, skipped = [], 0
                for did in cand:
                    if did < 0 or did in banned:
                        continue
                    if skipped < args.skip:
                        skipped += 1
                        continue
                    negatives.append(int(did))
                    if len(negatives) >= args.top_k:
                        break
                if len(negatives) < args.top_k:
                    short += len(pos_list)
                    continue
                if len(pos_list) > 1:
                    multi += 1
                text = qtext.get(qid, "")
                for pos in pos_list:            # one row per (query, positive) pair
                    fh.write(json.dumps({
                        "query_id": qid, "split": split, "query": text,
                        "positive_id": int(pos), "negative_ids": negatives,
                    }) + "\n")
                    written += 1

            stats[split] = {"pairs_written": written, "queries": len(positives),
                            "multi_positive_queries": multi,
                            "pairs_dropped_short_of_k": short,
                            "queries_without_candidates": no_cand}
            written_total += written
            log(f"  {split}: {written:,} pairs from {len(positives):,} queries "
                f"({multi:,} multi-positive); {short:,} pairs dropped, {no_cand:,} without candidates")
            del positives, cands, qtext

    meta = {"out": str(out), "skip": args.skip, "top_k": args.top_k,
            "row_unit": "(query, positive) pair",
            "all_annotated_positives_excluded_from_negatives": True,
            "total_pairs": written_total, "per_split": stats}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {written_total:,} pairs -> {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
