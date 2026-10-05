#!/usr/bin/env python3
"""Train with listwise cross-encoder distillation and an auxiliary InfoNCE loss."""

from __future__ import annotations

import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from losses_mixed import MixedDistillation
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from pylate import models, utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--dataset", default="lightonai/ms-marco-en-bge-gemma")
    ap.add_argument("--dataset-dir", default="",
                    help="local parquet dir (queries/documents/train subdirs, "
                         "build_kd_mixture.py layout); overrides --dataset")
    ap.add_argument("--teacher-scores", default="",
                    help="dir of parquet shards with query_id + scores; replaces the "
                         "dataset's own teacher scores (e.g. jina-reranker-v3.5)")
    ap.add_argument("--n-ways", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="sdpa")
    # loss knobs
    ap.add_argument("--tau-teacher", type=float, default=0.3)
    ap.add_argument("--tau-student", type=float, default=0.3)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--w-nce", type=float, default=0.1)
    ap.add_argument("--tau-nce", type=float, default=0.05)
    ap.add_argument("--fn-mask", type=float, default=0.6,
                    help="mask candidates above this normalised teacher score from the "
                         "InfoNCE denominator; negative disables")
    ap.add_argument("--clip-pct", type=float, default=0.0)
    ap.add_argument("--ddp-find-unused", action="store_true",
                    help="enable DDP unused-parameter detection for backbones with\n"
                         "conditionally unused modules")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    out = os.path.join(args.out_root, args.run_name)
    model = models.ColBERT(
        model_name_or_path=args.init,
        model_kwargs={"attn_implementation": args.attn},
    )
    print(f"init={args.init}  query_len={model.query_length}  doc_len={model.document_length}",
          flush=True)

    if args.dataset_dir:
        d = args.dataset_dir
        train = load_dataset("parquet", data_files=f"{d}/train/train.parquet", split="train")
        queries = load_dataset("parquet", data_files=f"{d}/queries/train.parquet",
                               split="train")
        documents = load_dataset("parquet", data_files=f"{d}/documents/train.parquet",
                                 split="train")
    else:
        train = load_dataset(args.dataset, "train", split="train")
        queries = load_dataset(args.dataset, "queries", split="train")
        documents = load_dataset(args.dataset, "documents", split="train")

    if args.teacher_scores:
        shards = sorted(glob.glob(os.path.join(args.teacher_scores, "*.parquet")))
        if not shards:
            raise SystemExit(f"no parquet shards under {args.teacher_scores}")
        tbl = pa.concat_tables([pq.read_table(p) for p in shards])
        swap = dict(zip(tbl["query_id"].to_pylist(), tbl["scores"].to_pylist()))
        before = len(train)
        train = train.filter(lambda b: [q in swap for q in b["query_id"]], batched=True,
                             num_proc=8)
        train = train.map(lambda b: {"scores": [swap[q] for q in b["query_id"]]},
                          batched=True, num_proc=8)
        print(f"teacher swap: {len(shards)} shards, {len(swap):,} scored queries, "
              f"train {before:,} -> {len(train):,}", flush=True)

    print(f"train={len(train):,} queries  n_ways={args.n_ways}", flush=True)
    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents,
                           n_ways=args.n_ways).transform
    )

    loss = MixedDistillation(
        model=model,
        tau_teacher=args.tau_teacher,
        tau_student=args.tau_student,
        w_kl=args.w_kl,
        w_nce=args.w_nce,
        tau_nce=args.tau_nce,
        fn_mask_thresh=None if args.fn_mask < 0 else args.fn_mask,
        clip_pct=args.clip_pct,
    )
    print(f"loss: w_kl={args.w_kl} tau_t={args.tau_teacher} tau_s={args.tau_student} | "
          f"w_nce={args.w_nce} tau_nce={args.tau_nce} fn_mask={args.fn_mask} "
          f"clip={args.clip_pct}", flush=True)

    targs = SentenceTransformerTrainingArguments(
        output_dir=out,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        bf16=False,
        fp16=False,
        logging_steps=50,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        report_to=[] if args.no_wandb else ["wandb"],
        ddp_find_unused_parameters=args.ddp_find_unused,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train,
        loss=loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    trainer.train()

    final = os.path.join(out, "final")
    model.save_pretrained(final)
    print(f"saved -> {final}", flush=True)


if __name__ == "__main__":
    main()
