#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the public BEIR benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

BEIR_HUB = ""
DECONTAMINATED_ROOT = ""

# smallest corpus first; (name, cache_dir_suffix)
BEIR_ORDER = [
    "nfcorpus", "scifact", "arguana", "scidocs", "fiqa",
    "cqadupstack-english", "cqadupstack-gaming", "cqadupstack-gis",
    "cqadupstack-mathematica", "cqadupstack-physics", "cqadupstack-programmers",
    "cqadupstack-stats", "cqadupstack-tex", "cqadupstack-unix",
    "cqadupstack-webmasters", "cqadupstack-wordpress", "CQADupstackAndroidRetrieval",
    "trec-covid", "touche2020", "quora",
    "nq", "dbpedia", "hotpotqa", "fever", "climate-fever", "msmarco",
]

# Shard the corpus to bound index memory; merging keeps retrieval exact.
SHARD_DOCS = int(os.environ.get("SHARD_DOCS", 2_000_000))

# Official BEIR eval split per dataset. Every dataset uses "test" EXCEPT msmarco, which BEIR
# scores on "dev" (6,980 queries); its "test" is the 43-query TREC-DL set and is NOT the
# BEIR number. No silent fallback -- if the official split is absent we raise, so a wrong
# split can never quietly stand in for it.
OFFICIAL_SPLIT: dict[str, str] = {"msmarco": "dev"}


def official_split(name: str) -> str:
    return OFFICIAL_SPLIT.get(name, "test")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snapshot(name: str) -> Path:
    if name.endswith("-decontaminated"):
        if not DECONTAMINATED_ROOT:
            raise ValueError("--decontaminated-root is required for decontaminated datasets")
        snap = Path(DECONTAMINATED_ROOT) / name
        if not snap.is_dir():
            raise FileNotFoundError(f"no decontaminated dataset under {snap}")
        return snap
    root = Path(BEIR_HUB) / f"datasets--mteb--{name}" / "snapshots"
    snaps = sorted(root.iterdir()) if root.is_dir() else []
    if not snaps:
        raise FileNotFoundError(f"no snapshot for {name} under {root}")
    return snaps[-1]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _rows(snap: Path, kind: str, split: str):
    """Yield rows for corpus/queries/qrels from either the jsonl or parquet layout.

    Older mteb mirrors store <kind>.jsonl; newer ones (e.g. CQADupstackAndroidRetrieval)
    store <kind>/<split>-*.parquet with 'id' instead of '_id'. Both layouts appear side by
    side in this mirror, so handle whichever is present rather than assume one.
    """
    # LightOn's decontaminated exports are flat Parquet files:
    # corpus.parquet, queries.parquet, and qrels_<split>.parquet.
    flat = (snap / (f"qrels_{split}.parquet" if kind == "qrels" else f"{kind}.parquet"))
    if flat.exists():
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(flat).iter_batches():
            yield from batch.to_pylist()
        return

    # jsonl layout: corpus/queries are top-level <kind>.jsonl files, but qrels live in a
    # qrels/<split>.jsonl subdir
    jl = (snap / "qrels" / f"{split}.jsonl") if kind == "qrels" else (snap / f"{kind}.jsonl")
    if jl.exists():
        yield from read_jsonl(jl)
        return
    import pyarrow.parquet as pq
    files = sorted((snap / kind).glob(f"{split}-*.parquet")) or \
        sorted((snap / kind).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {kind} (jsonl or parquet) for {snap}")
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches():
            for row in batch.to_pylist():
                yield row


def load_dataset(name: str):
    """-> (doc_ids, doc_texts, query_ids, query_texts, qrels)"""
    snap = snapshot(name)
    split = official_split(name)

    def _id(row):                       # parquet uses 'id', jsonl uses '_id'
        return str(row["_id"] if "_id" in row else row["id"])

    qrels: dict[str, dict[str, int]] = {}
    for row in _rows(snap, "qrels", split):
        rel = int(row.get("score", row.get("relevance", 0)))
        if rel > 0:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = rel
    if not qrels:
        raise FileNotFoundError(f"no '{split}' qrels judgements for {name}")
    log(f"  qrels ({split}): {len(qrels)} queries with judgements")

    query_ids, query_texts = [], []
    for row in _rows(snap, "queries", split):
        qid = _id(row)
        if qid in qrels:                    # only judged queries, as BEIR scores them
            query_ids.append(qid)
            query_texts.append(row.get("text") or "")

    doc_ids, doc_texts = [], []
    for row in _rows(snap, "corpus", split):
        title, text = row.get("title") or "", row.get("text") or ""
        doc_ids.append(_id(row))
        doc_texts.append(f"{title} {text}".strip() if title else text)

    log(f"  {len(doc_ids):,} documents, {len(query_ids):,} queries")
    return doc_ids, doc_texts, query_ids, query_texts, qrels


def run_dataset(name, model, torch, indexes, retrieve, args, work: Path):
    doc_ids, doc_texts, query_ids, query_texts, qrels = load_dataset(name)

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        q_emb = model.encode(query_texts, is_query=True, batch_size=args.query_batch,
                             convert_to_numpy=True, show_progress_bar=False)

    n_shards = max(1, -(-len(doc_ids) // SHARD_DOCS))
    per = -(-len(doc_ids) // n_shards)
    if n_shards > 1:
        log(f"  {n_shards} shards of <= {per:,} docs")

    # doc_id -> best score across shards
    merged: list[dict[str, float]] = [dict() for _ in query_ids]

    for s in range(n_shards):
        lo, hi = s * per, min((s + 1) * per, len(doc_ids))
        index_name = f"{name}_{s:03d}"
        index_dir = work / "indexes" / index_name
        if index_dir.exists():
            shutil.rmtree(index_dir)       # eval indexes are disposable, unlike mining's

        t0 = time.time()
        embeddings = []
        for start in range(0, hi - lo, args.encode_chunk):
            a = lo + start
            b = min(a + args.encode_chunk, hi)
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                part = model.encode(doc_texts[a:b], is_query=False,
                                    batch_size=args.encode_batch,
                                    convert_to_numpy=True, show_progress_bar=False)
            embeddings.extend(e.astype("float16") for e in part)
            done = b - lo
            log(f"  shard {s}: encoded {done:,}/{hi - lo:,} "
                f"({done / (time.time() - t0):.0f} doc/s)")

        index = indexes.PLAID(index_folder=str(work / "indexes"), index_name=index_name,
                             override=True, nbits=args.nbits)
        index.add_documents(documents_ids=doc_ids[lo:hi], documents_embeddings=embeddings)
        del embeddings
        torch.cuda.empty_cache()
        log(f"  shard {s}: indexed in {time.time() - t0:.0f}s")

        retriever = retrieve.ColBERT(index=index)
        t0 = time.time()
        for start in range(0, len(q_emb), args.retrieve_batch):
            hits = retriever.retrieve(
                queries_embeddings=q_emb[start:start + args.retrieve_batch], k=args.k)
            for row, hit_list in enumerate(hits):
                bucket = merged[start + row]
                self_id = query_ids[start + row]
                for h in hit_list:
                    did, sc = str(h["id"]), float(h["score"])
                    # Exclude self-retrieval for datasets whose query IDs match corpus IDs.
                    if did == self_id:
                        continue
                    if sc > bucket.get(did, float("-inf")):
                        bucket[did] = sc
        log(f"  shard {s}: retrieved {len(q_emb):,} queries in {time.time() - t0:.0f}s")

        del index, retriever
        shutil.rmtree(index_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    # keep global top-k after the merge, in pylate's scores format
    scores = []
    for bucket in merged:
        top = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)[:args.k]
        scores.append([{"id": did, "score": sc} for did, sc in top])

    from pylate import evaluation
    return evaluation.evaluate(scores=scores, qrels=qrels, queries=query_ids,
                               metrics=["ndcg@10", "ndcg@100", "recall@10", "recall@100",
                                        "map", "mrr@10", "precision@10"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the trained checkpoint")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", default="outputs/evaluation")
    ap.add_argument("--beir-cache", required=True,
                    help="root of the local Hugging Face MTEB dataset cache")
    ap.add_argument("--decontaminated-root", default="",
                    help="optional root of the decontaminated BEIR export")
    ap.add_argument("--datasets", default="all",
                    help="comma-separated BEIR names, or 'all'")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--encode-batch", type=int, default=512)
    ap.add_argument("--encode-chunk", type=int, default=65536)
    ap.add_argument("--query-batch", type=int, default=1024)
    ap.add_argument("--retrieve-batch", type=int, default=256)
    ap.add_argument("--parts-only", action="store_true",
                    help="write only parts/<dataset>.json; for fanned-out array tasks, "
                         "where several processes share one run directory")
    ap.add_argument("--wandb-project", default="sft-eval")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    global BEIR_HUB, DECONTAMINATED_ROOT
    BEIR_HUB = str(Path(args.beir_cache).expanduser().resolve())
    DECONTAMINATED_ROOT = (str(Path(args.decontaminated_root).expanduser().resolve())
                           if args.decontaminated_root else "")

    import torch
    from pylate import indexes, models, retrieve

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # one file per dataset: concurrent (model, dataset) tasks would otherwise race on a
    # single results.json, each overwriting the others' datasets
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    # Isolate temporary index work so concurrent tasks cannot collide.
    tag = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    work = Path(args.out_dir) / "_work" / f"{args.run_name}_{tag}"
    (work / "indexes").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for part in parts_dir.glob("*.json"):
        results.setdefault(part.stem, json.loads(part.read_text()))

    run = None
    if not args.no_wandb:
        import wandb
        os.environ.setdefault("WANDB_DIR", str(Path(args.out_dir) / "wandb"))
        run = wandb.init(project=args.wandb_project, name=args.run_name,
                         config={**vars(args), "backend": "PLAID (fast-plaid)"})

    log(f"loading {args.model}")
    model = models.ColBERT(args.model, device="cuda")
    model.eval()

    wanted = BEIR_ORDER if args.datasets == "all" else [
        d.strip() for d in args.datasets.split(",") if d.strip()]

    for name in wanted:
        if name in results:
            log(f"{name}: already in results.json, skipping")
            continue
        log(f"=== {name}")
        t0 = time.time()
        try:
            metrics = run_dataset(name, model, torch, indexes, retrieve, args, work)
        except FileNotFoundError as exc:
            log(f"  SKIP {name}: {exc}")
            continue
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["wall_seconds"] = round(time.time() - t0)
        results[name] = metrics
        (parts_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2))
        if not args.parts_only:
            results_path.write_text(json.dumps(results, indent=2))
        log(f"  {name}: ndcg@10={metrics.get('ndcg@10'):.4f} "
            f"recall@100={metrics.get('recall@100'):.4f} "
            f"({metrics['wall_seconds']}s)")
        if run:
            run.log({f"{name}/{k}": v for k, v in metrics.items()})

    full = [n for n in BEIR_ORDER if n in results]
    if full and not args.parts_only:
        avg = sum(results[n]["ndcg@10"] for n in full) / len(full)
        log(f"mean ndcg@10 over {len(full)} datasets: {avg:.4f}")
        results["_summary"] = {"datasets": len(full), "mean_ndcg@10": avg}
        results_path.write_text(json.dumps(results, indent=2))
        if run:
            run.summary["mean_ndcg@10"] = avg
            run.summary["datasets_done"] = len(full)

    shutil.rmtree(work, ignore_errors=True)
    if run:
        run.finish()
    log(f"results -> {results_path}")


if __name__ == "__main__":
    main()
