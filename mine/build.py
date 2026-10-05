#!/usr/bin/env python3
"""Mine one corpus shard with WARP and save each query's top MaxSim candidates."""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np

DATA_ROOT = ""
ENCODE_BATCH = 512
QUERY_BATCH = 1024
ENCODE_CHUNK = 65536        # docs per encode call, keeps the python list churn bounded


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parquet_files(config: str, split: str) -> list[str]:
    files = sorted(glob.glob(f"{DATA_ROOT}/{config}/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet for {config}/{split}")
    return files


def load_documents(split: str):
    import datasets

    return datasets.load_dataset("parquet", data_files=parquet_files("documents", split),
                                 split="train")


def load_queries(split: str):
    import datasets

    return datasets.load_dataset("parquet", data_files=parquet_files("queries", split),
                                 split="train")


def encode_queries_cached(model, split: str, cache_dir: Path, torch, model_name: str = None):
    """Query embeddings are identical for every shard, so compute once and reuse."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"q_{split}.npz"
    if path.exists():
        blob = np.load(path, allow_pickle=True)
        log(f"loaded cached query embeddings {path.name}")
        return blob["qids"], list(blob["emb"])

    if model is None:                       # reuse path with a cold cache
        from pylate import models as _m
        log(f"loading {model_name} for query encoding")
        model = _m.ColBERT(model_name, device="cuda")
        model.eval()
    queries = load_queries(split)
    qids = np.asarray(queries["query_id"], dtype=np.int64)
    log(f"encoding {len(qids)} queries")
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        emb = model.encode(queries["query"], is_query=True, batch_size=QUERY_BATCH,
                           convert_to_numpy=True, show_progress_bar=False)
    emb = [e.astype(np.float16) for e in emb]
    log(f"  encoded in {time.time() - t0:.0f}s ({len(qids) / (time.time() - t0):.0f} q/s)")
    # unique temp per process: concurrent shards of the same split would otherwise all
    # write one shared temp name, and the first os.replace pulls it out from under the rest
    tmp = path.with_suffix(f".tmp{os.getpid()}.{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}.npz")
    np.savez(tmp, qids=qids, emb=np.array(emb, dtype=object))
    os.replace(tmp, path)          # atomic rename, so readers never see a partial file
    return qids, emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--data-root", required=True,
                    help="local cache of the public embeddings-fine-tuning dataset")
    ap.add_argument("--model", default="lightonai/LateOn-unsupervised")
    ap.add_argument("--top-p", type=int, default=2048,
                    help="candidates kept per query per shard; must exceed skip+K plus "
                         "headroom for the positive and cross-shard competition")
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--num-threads", type=int, default=7,
                    help="number of WARP search threads")
    ap.add_argument("--retrieve-batch", type=int, default=512,
                    help="queries per WARP retrieve call; reduced automatically on GPU OOM")
    ap.add_argument("--work-dir", default="outputs/mining")
    ap.add_argument("--parts-dir", default="", help="override parts output; work/indexes still reused")
    args = ap.parse_args()
    global DATA_ROOT
    DATA_ROOT = str(Path(args.data_root).expanduser().resolve())

    import torch
    from pylate import indexes, models, retrieve

    work = Path(args.work_dir)
    # Candidate output can be redirected while reusing the index and query cache.
    parts = Path(args.parts_dir) if args.parts_dir else work / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    out_path = parts / f"{args.split}_{args.shard:03d}.npz"
    if out_path.exists():
        log(f"{out_path.name} exists, nothing to do")
        return

    index_name = f"{args.split}_{args.shard:03d}"
    index_path = work / "indexes" / index_name
    reuse = index_path.is_dir() and any(index_path.iterdir())

    model = None
    if not reuse:
        documents = load_documents(args.split)
        n_docs = len(documents)
        per = -(-n_docs // args.n_shards)
        lo, hi = args.shard * per, min((args.shard + 1) * per, n_docs)
        if lo >= hi:
            log(f"shard {args.shard} empty for {args.split} ({n_docs} docs)")
            return
        log(f"{args.split} shard {args.shard}/{args.n_shards}: docs [{lo},{hi}) of {n_docs}")
        shard = documents.select(range(lo, hi))
        doc_ids = np.asarray(shard["document_id"], dtype=np.int64)
        doc_texts = shard["document"]
        del shard, documents
        log(f"loading {args.model}")
        model = models.ColBERT(args.model, device="cuda")
        model.eval()
    else:
        log(f"reusing existing index {index_name} -- skipping document encode")

    # only needs the model on a cache miss; the cache is written once per split
    qids, q_emb = encode_queries_cached(model, args.split, work / "qcache", torch,
                                       model_name=args.model)

    embeddings = []
    if not reuse:
      log(f"encoding {len(doc_texts)} documents (bf16, batch {ENCODE_BATCH})")
      t0 = time.time()
      for start in range(0, len(doc_texts), ENCODE_CHUNK):
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            part = model.encode(doc_texts[start : start + ENCODE_CHUNK], is_query=False,
                                batch_size=ENCODE_BATCH, convert_to_numpy=True,
                                show_progress_bar=False)
        embeddings.extend(e.astype(np.float16) for e in part)
        done = min(start + ENCODE_CHUNK, len(doc_texts))
        log(f"  encoded {done}/{len(doc_texts)} ({done / (time.time() - t0):.0f} doc/s, "
            f"{sum(e.nbytes for e in embeddings) / 1e9:.1f} GB)")
      del doc_texts

    t0 = time.time()
    index = indexes.WARP(index_folder=str(work / "indexes"), index_name=index_name,
                         override=not reuse, nbits=args.nbits,
                         num_threads=args.num_threads, device="cuda")
    if reuse:
        log(f"  index loaded in {time.time() - t0:.0f}s")
    else:
        index.add_documents(documents_ids=[str(d) for d in doc_ids],
                            documents_embeddings=embeddings)
        log(f"  index built in {time.time() - t0:.0f}s")
        del embeddings
    torch.cuda.empty_cache()

    retriever = retrieve.ColBERT(index=index)
    top_p = args.top_p
    all_ids = np.full((len(qids), top_p), -1, dtype=np.int64)
    all_sc = np.full((len(qids), top_p), -np.inf, dtype=np.float32)

    def retrieve_chunk(chunk, batch):
        """Retrieve, halving the batch on GPU OOM. WARP raises a Rust PanicException
        rather than torch.cuda.OutOfMemoryError, so match on the message."""
        while True:
            try:
                out = []
                for i in range(0, len(chunk), batch):
                    out.extend(retriever.retrieve(queries_embeddings=chunk[i : i + batch],
                                                  k=top_p))
                return out, batch
            except BaseException as exc:
                if batch <= 1 or "out of memory" not in str(exc).lower():
                    raise
                batch = max(1, batch // 2)
                log(f"  retrieval OOM, retrying at batch {batch}")
                torch.cuda.empty_cache()

    log(f"retrieving {len(qids)} queries, top-{top_p} (batch {args.retrieve_batch})")
    t0 = time.time()
    batch = args.retrieve_batch
    for start in range(0, len(qids), args.retrieve_batch):
        chunk = q_emb[start : start + args.retrieve_batch]
        hits, batch = retrieve_chunk(chunk, batch)
        for row, hit_list in enumerate(hits):
            m = min(len(hit_list), top_p)
            if not m:
                continue
            all_ids[start + row, :m] = [int(h["id"]) for h in hit_list[:m]]
            all_sc[start + row, :m] = [float(h["score"]) for h in hit_list[:m]]
        done = min(start + args.retrieve_batch, len(qids))
        log(f"  retrieved {done}/{len(qids)} ({done / (time.time() - t0):.0f} q/s)")

    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, qids=qids, ids=all_ids, scores=all_sc)
    os.replace(tmp, out_path)
    log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")

    # Keep the index for resuming retrieval after an interrupted shard.
    log("done")


if __name__ == "__main__":
    main()
