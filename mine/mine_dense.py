#!/usr/bin/env python3
"""Dense-retrieval baseline for the hard-negative mining ablation."""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np

DATA_ROOT = ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parquet_files(config: str, split: str) -> list[str]:
    files = sorted(glob.glob(f"{DATA_ROOT}/{config}/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet for {config}/{split}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="local cache of the public embeddings-fine-tuning dataset")
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1,
                    help="split the corpus over N GPUs; each shard searches every query "
                         "against its own slice and the merge takes the best per query, "
                         "which is exact because the shards partition the corpus")
    ap.add_argument("--model", default="lightonai/DenseOn")
    ap.add_argument("--top-p", type=int, default=2048)
    ap.add_argument("--encode-batch", type=int, default=1024)
    ap.add_argument("--encode-chunk", type=int, default=262144)
    ap.add_argument("--search-chunk", type=int, default=2_000_000,
                    help="documents per similarity block; caps the [Q, D] score tensor")
    ap.add_argument("--query-chunk", type=int, default=2048,
                    help="number of queries per similarity block")
    ap.add_argument("--work-dir", default="outputs/mining_dense")
    args = ap.parse_args()
    global DATA_ROOT
    DATA_ROOT = str(Path(args.data_root).expanduser().resolve())

    import datasets
    import torch
    from sentence_transformers import SentenceTransformer

    work = Path(args.work_dir)
    parts = work / "parts_dense"
    parts.mkdir(parents=True, exist_ok=True)
    out_path = parts / f"{args.split}_{args.shard:03d}.npz"
    if out_path.exists() or (parts / f"{args.split}.npz").exists():
        log(f"{out_path.name} exists, nothing to do")
        return

    log(f"loading {args.model}")
    model = SentenceTransformer(args.model, device="cuda")
    model.eval()

    queries = datasets.load_dataset("parquet", data_files=parquet_files("queries", args.split),
                                    split="train")
    qids = np.asarray(queries["query_id"], dtype=np.int64)
    log(f"encoding {len(qids):,} queries")
    with torch.no_grad():
        q_emb = model.encode(queries["query"], batch_size=args.encode_batch,
                             convert_to_tensor=True, normalize_embeddings=True,
                             show_progress_bar=False).half()
    del queries

    documents = datasets.load_dataset("parquet", data_files=parquet_files("documents", args.split),
                                      split="train")
    total_docs = len(documents)
    if args.n_shards > 1:
        per = -(-total_docs // args.n_shards)
        lo, hi = args.shard * per, min((args.shard + 1) * per, total_docs)
        if lo >= hi:
            log(f"shard {args.shard} empty for {args.split} ({total_docs} docs)")
            return
        documents = documents.select(range(lo, hi))
        log(f"shard {args.shard}/{args.n_shards}: docs [{lo:,},{hi:,}) of {total_docs:,}")
    n_docs = len(documents)
    doc_ids = np.asarray(documents["document_id"], dtype=np.int64)
    log(f"encoding {n_docs:,} documents (fp16 on GPU, batch {args.encode_batch})")

    dim = q_emb.shape[1]
    d_emb = torch.empty((n_docs, dim), dtype=torch.float16, device="cuda")
    t0 = time.time()
    for start in range(0, n_docs, args.encode_chunk):
        stop = min(start + args.encode_chunk, n_docs)
        with torch.no_grad():
            block = model.encode(documents["document"][start:stop],
                                 batch_size=args.encode_batch, convert_to_tensor=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        d_emb[start:stop] = block.half()
        del block
        log(f"  encoded {stop:,}/{n_docs:,} ({stop / (time.time() - t0):.0f} doc/s, "
            f"{d_emb.element_size() * stop * dim / 1e9:.1f} GB)")
    del documents
    torch.cuda.empty_cache()

    top_p = min(args.top_p, n_docs)
    all_ids = np.full((len(qids), top_p), -1, dtype=np.int64)
    all_sc = np.full((len(qids), top_p), -np.inf, dtype=np.float32)

    log(f"exact search, top-{top_p}")
    t0 = time.time()
    for qs in range(0, len(qids), args.query_chunk):
        qe = min(qs + args.query_chunk, len(qids))
        qblock = q_emb[qs:qe]
        best_sc, best_ix = None, None
        for ds in range(0, n_docs, args.search_chunk):
            de = min(ds + args.search_chunk, n_docs)
            sims = qblock @ d_emb[ds:de].T                    # [q, d] fp16
            k = min(top_p, de - ds)
            # Cast only the top-k scores to fp32 to limit peak memory use.
            sc, ix = torch.topk(sims, k, dim=1)
            sc = sc.float()
            del sims
            ix = ix + ds
            if best_sc is None:
                best_sc, best_ix = sc, ix
            else:                                             # merge running top-k
                best_sc = torch.cat([best_sc, sc], dim=1)
                best_ix = torch.cat([best_ix, ix], dim=1)
                best_sc, sel = torch.topk(best_sc, top_p, dim=1)
                best_ix = torch.gather(best_ix, 1, sel)
        all_sc[qs:qe, :best_sc.shape[1]] = best_sc.cpu().numpy()
        all_ids[qs:qe, :best_ix.shape[1]] = doc_ids[best_ix.cpu().numpy()]
        log(f"  searched {qe:,}/{len(qids):,} ({qe / (time.time() - t0):.0f} q/s)")

    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, qids=qids, ids=all_ids, scores=all_sc)
    os.replace(tmp, out_path)
    log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
