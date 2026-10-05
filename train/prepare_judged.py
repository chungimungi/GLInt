#!/usr/bin/env python3
"""Create the risk-aware supervised fine-tuning mixture from mined and judged pools."""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import numpy as np

SPLITS = ["fiqa", "squadv2", "hotpotqa", "fever", "msmarco", "nq", "trivia"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class DocStore:
    def __init__(self, data_root: Path, split: str):
        import pyarrow as pa
        import pyarrow.parquet as pq

        files = sorted(glob.glob(str(data_root / "documents" / f"{split}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no document shards for {split} in the data root")
        table = pa.concat_tables([
            pq.read_table(path, columns=["document_id", "document"], memory_map=True)
            for path in files
        ])
        ids = table.column("document_id").to_numpy()
        self.texts = table.column("document")
        order = np.argsort(ids, kind="stable")
        self.ids_sorted, self.order = ids[order], order

    def get(self, doc_id: int) -> str | None:
        pos = np.searchsorted(self.ids_sorted, doc_id)
        if pos >= len(self.ids_sorted) or self.ids_sorted[pos] != doc_id:
            return None
        return self.texts[int(self.order[pos])].as_py()


def read_pairs(path: Path, wanted: set[str]) -> dict[str, list[dict]]:
    pairs = {split: [] for split in wanted}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            split = record["split"]
            if split in pairs:
                pairs[split].append(record)
    return pairs


def read_judge_pool(pool_dir: Path, split: str) -> dict[int, tuple[np.ndarray, ...]]:
    rows: dict[int, tuple[np.ndarray, ...]] = {}
    for path in sorted(pool_dir.glob(f"{split}_*.npz")):
        with np.load(path) as data:
            for i, qid in enumerate(data["qids"]):
                rows[int(qid)] = (
                    data["gold_ids"][i], data["gold_scores"][i],
                    data["cand_ids"][i], data["cand_scores"][i],
                )
    if not rows:
        raise FileNotFoundError(f"no judge score shards for {split}")
    return rows


def usable_rows(
    pairs: list[dict],
    judged: dict[int, tuple[np.ndarray, ...]],
    num_negatives: int,
    band_lower: float,
    disable_filter: bool,
) -> tuple[list[tuple[str, int, list[int]]], int, int]:
    rows: list[tuple[str, int, list[int]]] = []
    vetoed = candidates = 0
    for record in pairs:
        item = judged.get(int(record["query_id"]))
        if item is None:
            continue
        gold_ids, gold_scores, candidate_ids, candidate_scores = item
        hit = np.flatnonzero(gold_ids == int(record["positive_id"]))
        if not len(hit) or not np.isfinite(gold_scores[hit[0]]):
            continue
        positive_score = float(gold_scores[hit[0]])
        valid = (candidate_ids >= 0) & np.isfinite(candidate_scores)
        valid &= ~np.isin(candidate_ids, gold_ids[gold_ids >= 0])
        ids, scores = candidate_ids[valid], candidate_scores[valid]
        candidates += len(ids)
        veto = scores >= positive_score
        vetoed += int(veto.sum())

        if disable_filter:
            pool = ids[np.argsort(-scores)]
        else:
            survivors = ~veto
            if int(survivors.sum()) < num_negatives:
                continue
            ratios = scores[survivors] / max(positive_score, 1e-9)
            ids = ids[survivors]
            order = np.argsort(-ratios)
            ratios, ids = ratios[order], ids[order]
            ineligible = int((ratios >= 1.0).sum())
            if len(ids) - ineligible < num_negatives:
                continue
            band_end = int((ratios >= band_lower).sum())
            pool = ids[ineligible:max(band_end, ineligible + num_negatives)]

        if len(pool) >= num_negatives:
            rows.append((record["query"], int(record["positive_id"]), pool.tolist()))
    return rows, vetoed, candidates


def allocate_targets(rows_by_split: dict[str, list], risk: dict[str, float], budget: int):
    weights = {
        split: len(rows) ** 0.5 * (1.0 - risk[split])
        for split, rows in rows_by_split.items()
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("no usable training rows remain after judge filtering")
    exact = {split: budget * weight / total for split, weight in weights.items()}
    targets = {split: int(value) for split, value in exact.items()}
    remainder = budget - sum(targets.values())
    order = sorted(exact, key=lambda split: exact[split] - targets[split], reverse=True)
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="local cache of lightonai/embeddings-fine-tuning")
    parser.add_argument("--pairs", required=True,
                        help="output from mine/merge_pairs.py")
    parser.add_argument("--judge-dir", required=True,
                        help="score shards from mine/judge_pool.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--num-negatives", type=int, default=7)
    parser.add_argument("--budget", type=int, default=1_500_000)
    parser.add_argument("--band-lower", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-filter", action="store_true",
                        help="retain candidates at or above the positive for the filtering ablation")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    pool_dir = Path(args.judge_dir).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    pairs_by_split = read_pairs(Path(args.pairs), set(splits))

    usable_by_split: dict[str, list[tuple[str, int, list[int]]]] = {}
    risk: dict[str, float] = {}
    for split in splits:
        log(f"preparing {split}")
        judged = read_judge_pool(pool_dir, split)
        rows, vetoed, candidates = usable_rows(
            pairs_by_split[split], judged, args.num_negatives,
            args.band_lower, args.disable_filter,
        )
        usable_by_split[split] = rows
        risk[split] = vetoed / max(1, candidates)
        log(f"{split}: {len(rows):,} usable pairs; judge vetoed {vetoed:,}/{candidates:,}")
        del judged

    targets = allocate_targets(usable_by_split, risk, args.budget)
    rng = random.Random(args.seed)
    stats = {}
    for split in splits:
        rows, target = usable_by_split[split], targets[split]
        if not rows and target:
            raise ValueError(f"{split} has a non-zero target but no usable rows")
        chosen = list(rows)
        if target > len(chosen):
            chosen.extend(rng.choices(rows, k=target - len(chosen)))
        else:
            chosen = rng.sample(chosen, target)
        rng.shuffle(chosen)

        store = DocStore(data_root, split)
        written = skipped = 0
        output = out_dir / f"train_{split}.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for query, positive_id, pool in chosen:
                positive = store.get(positive_id)
                if not positive:
                    skipped += 1
                    continue
                negative_ids = rng.sample(pool, args.num_negatives)
                negatives = [store.get(doc_id) for doc_id in negative_ids]
                if any(not text for text in negatives):
                    skipped += 1
                    continue
                row = {"query": query, "positive": positive}
                row.update({f"negative_{i + 1}": text
                            for i, text in enumerate(negatives)})
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        stats[split] = {
            "usable_pairs": len(rows), "target_rows": target,
            "written_rows": written, "skipped_rows": skipped,
            "judge_fn_rate": round(risk[split], 6),
        }
        log(f"{split}: wrote {written:,} rows")
        del store

    metadata = {
        "judge": "lightonai/GTE-ModernColBERT-v1",
        "filter": "disabled" if args.disable_filter else "candidate score < positive score",
        "sampling": "risk-aware square-root mixture; random negatives from the top score band",
        "num_negatives": args.num_negatives,
        "band_lower": args.band_lower,
        "budget": args.budget,
        "risk": risk,
        "targets": targets,
        "per_split": stats,
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
