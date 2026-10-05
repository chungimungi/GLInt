#!/usr/bin/env python3
"""Score listwise distillation candidates with the public cross-encoder teacher."""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch

MODEL = "jinaai/jina-reranker-v3.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default="lightonai/ms-marco-en-bge-gemma")
    ap.add_argument("--dataset-dir", default="",
                    help="local parquet dir (queries/documents/train subdirs, "
                         "build_kd_mixture.py layout); overrides --dataset")
    ap.add_argument("--n-ways", type=int, default=32)
    ap.add_argument("--max-doc-tokens", type=int, default=300,
                    help="match the student's document_length so both see the same text")
    ap.add_argument("--max-query-tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2, help="queries per forward pass")
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"jina_scores_{args.shard:03d}.parquet")

    if args.dataset_dir:
        # Read parquet directly so independently scheduled shards do not contend on cache locks.
        d = args.dataset_dir
        train = pq.read_table(f"{d}/train/train.parquet").to_pylist()
        _q = pq.read_table(f"{d}/queries/train.parquet")
        _d = pq.read_table(f"{d}/documents/train.parquet")
        queries = {"query_id": _q["query_id"].to_pylist(), "text": _q["text"].to_pylist()}
        documents = {"document_id": _d["document_id"].to_pylist(),
                     "text": _d["text"].to_pylist()}
        LOCAL = True
    else:
        LOCAL = False
        train = load_dataset(args.dataset, "train", split="train")
        queries = load_dataset(args.dataset, "queries", split="train")
        documents = load_dataset(args.dataset, "documents", split="train")

    qtext = (dict(zip(queries["query_id"], queries["text"])) if LOCAL
             else {q: i for i, q in enumerate(queries["query_id"])})
    dtext = (dict(zip(documents["document_id"], documents["text"])) if LOCAL
             else {d: i for i, d in enumerate(documents["document_id"])})
    qidx, didx = qtext, dtext
    print(f"indexed {len(qidx):,} queries / {len(didx):,} documents", flush=True)

    n = len(train)
    lo = (n * args.shard) // args.n_shards
    hi = (n * (args.shard + 1)) // args.n_shards
    rows = train[lo:hi] if LOCAL else train.select(range(lo, hi))
    print(f"shard {args.shard}/{args.n_shards}: rows [{lo}, {hi}) = {hi - lo:,}", flush=True)

    try:
        model = AutoModel.from_pretrained(
            MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation=args.attn,
        )
    except Exception as exc:  # flash-attn not built for this env
        print(f"attn={args.attn} unavailable ({type(exc).__name__}), falling back to sdpa",
              flush=True)
        model = AutoModel.from_pretrained(
            MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model = model.eval().cuda()
    fmt = sys.modules[type(model).__module__].format_docs_prompts_func
    special = model.special_tokens

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def truncate(text: str, max_tokens: int) -> str:
        ids = tok(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
        return tok.decode(ids["input_ids"])

    def parse(v):
        return v if isinstance(v, list) else ast.literal_eval(v)

    qids: list[int] = []
    all_scores: list[list[float]] = []
    t0 = time.time()
    buf_prompts: list[str] = []
    buf_qids: list[int] = []

    def flush():
        if not buf_prompts:
            return
        batch = tok(buf_prompts, padding=True, padding_side="left",
                    return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**batch)
        s = out.scores.view(len(buf_prompts), -1).float().cpu().numpy()
        assert s.shape[1] == args.n_ways, f"expected {args.n_ways} scores, got {s.shape}"
        for i, q in enumerate(buf_qids):
            qids.append(q)
            all_scores.append([float(v) for v in s[i]])
        buf_prompts.clear()
        buf_qids.clear()

    for k, row in enumerate(rows):
        dids = parse(row["document_ids"])[: args.n_ways]
        if len(dids) < args.n_ways:
            continue  # forward() reshapes on a fixed doc count per row
        _qs = (qidx[row["query_id"]] if LOCAL
               else queries[qidx[row["query_id"]]]["text"])
        qtext_s = truncate(_qs, args.max_query_tokens)
        docs = [truncate(didx[d] if LOCAL else documents[didx[d]]["text"],
                         args.max_doc_tokens) for d in dids]
        buf_prompts.append(fmt(qtext_s, docs, instruction=None, special_tokens=special,
                               no_thinking=True))
        buf_qids.append(row["query_id"])
        if len(buf_prompts) >= args.batch:
            flush()
        if k and k % 200 == 0:
            r = (k + 1) / (time.time() - t0)
            print(f"{k + 1:,}/{hi - lo:,}  {r:.2f} q/s  eta {(hi - lo - k) / r / 60:.1f} min",
                  flush=True)
    flush()

    el = time.time() - t0
    print(f"scored {len(qids):,} queries in {el / 60:.1f} min "
          f"({len(qids) / max(el, 1e-9):.2f} q/s)", flush=True)

    pq.write_table(
        pa.table({"query_id": pa.array(qids, pa.int64()),
                  "scores": pa.array(all_scores, pa.list_(pa.float32()))}),
        out_path,
    )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
