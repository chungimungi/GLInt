#!/usr/bin/env python3
"""Supervised fine-tuning for the late-interaction retriever."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="lightonai/LateOn-unsupervised")
    ap.add_argument("--attn", default="flash_attention_2",
                    help="ModernBERT attention impl; flash_attention_2 enables the unpadding "
                         "path (no compute on pad tokens). sdpa to fall back.")
    ap.add_argument("--load-as-is", action="store_true",
                    help="base model is already a PyLate ColBERT (e.g. LateOn-unsupervised): "
                         "keep its own projection head, prefixes and lengths instead of "
                         "rebuilding its architecture")
    ap.add_argument("--data-dir", default="outputs/sft_data")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--run-name", default="sft")
    # ---- ColBERT head
    ap.add_argument("--embedding-size", type=int, default=128)
    ap.add_argument("--query-length", type=int, default=32)
    ap.add_argument("--document-length", type=int, default=300)
    # Zero preserves the checkpoint's configured query and document lengths.
    ap.add_argument("--set-query-length", type=int, default=0)
    ap.add_argument("--set-document-length", type=int, default=0)
    # ---- optimisation
    ap.add_argument("--batch-size", type=int, default=16,
                    help="per-device batch; each row carries 1 positive + N negatives")
    ap.add_argument("--mini-batch-size", type=int, default=8,
                    help="GradCache chunk; lower this, not --batch-size, on OOM")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--score-metric", default="meanmaxsim", choices=["maxsim", "meanmaxsim"])
    ap.add_argument("--temperature", type=float, default=0.001)
    ap.add_argument("--lr-scheduler", default="linear")
    ap.add_argument("--seed", type=int, default=42)
    # ---- logging / eval
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--wandb-project", default="sft")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--gather-across-devices", action="store_true",
                    help="pool in-batch negatives across DDP ranks, so 4x32 behaves like "
                         "the paper's global batch of 128 rather than four batches of 32. "
                         "pylate defaults this to False.")
    ap.add_argument("--no-gather-across-devices", action="store_false",
                    dest="gather_across_devices")
    ap.set_defaults(gather_across_devices=True)
    ap.add_argument("--multi-source", action="store_true",
                    help="load train_<split>.jsonl as a DatasetDict so each batch is drawn "
                         "from a single source (in-domain in-batch negatives)")
    ap.add_argument("--resume-from", default="",
                    help="path to a checkpoint-N dir to continue training from; the dir carries "
                         "optimizer/scheduler/RNG state so the run picks up mid-schedule")
    args = ap.parse_args()

    import glob

    import datasets
    import torch
    from sentence_transformers import (SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.training_args import MultiDatasetBatchSamplers
    from pylate import losses, models, scores

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    report_to = [] if args.no_wandb else ["wandb"]
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(Path(args.out_dir).parent / "wandb"))
        os.environ.setdefault("WANDB_WATCH", "false")

    # ---------------------------------------------------------------- data
    if args.multi_source:
        # one dataset per source; passed as a DatasetDict, sentence-transformers draws each
        # batch from a single source, so in-batch negatives stay in-domain
        files = sorted(glob.glob(str(data_dir / "train_*.jsonl")))
        train_ds = datasets.DatasetDict({
            Path(f).stem.replace("train_", ""):
                datasets.load_dataset("json", data_files=f, split="train")
            for f in files})
        first = next(iter(train_ds.values()))
        n_neg = sum(1 for c in first.column_names if c.startswith("negative_"))
        total = sum(len(d) for d in train_ds.values())
        log(f"train {total:,} rows over {len(train_ds)} sources "
            f"{ {k: len(v) for k, v in train_ds.items()} } | {n_neg} negatives/row")
    else:
        train_ds = datasets.load_dataset("json", data_files=str(data_dir / "train.jsonl"),
                                         split="train")
        n_neg = sum(1 for c in train_ds.column_names if c.startswith("negative_"))
        log(f"train {len(train_ds):,} rows | {n_neg} negatives/row | "
            f"columns {train_ds.column_names}")

    eval_path = data_dir / "eval.jsonl"
    eval_ds = (datasets.load_dataset("json", data_files=str(eval_path), split="train")
               if eval_path.exists() else None)

    # ---------------------------------------------------------------- model
    log(f"building ColBERT from {args.base_model} (load_as_is={args.load_as_is}, attn={args.attn})")
    mk = {"attn_implementation": args.attn}
    if args.load_as_is:
        kw = {}
        if args.set_query_length:
            kw["query_length"] = args.set_query_length
        if args.set_document_length:
            kw["document_length"] = args.set_document_length
        model = models.ColBERT(model_name_or_path=args.base_model, model_kwargs=mk, **kw)
    else:
        model = models.ColBERT(
            model_name_or_path=args.base_model,
            embedding_size=args.embedding_size,
            query_length=args.query_length,
            document_length=args.document_length,
            document_prefix="[D] ",
            query_prefix="[Q] ",
            model_kwargs=mk,
        )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  {n_params / 1e6:.1f}M parameters | query_length={model.query_length} "
        f"document_length={model.document_length}")

    class MeanMaxSim(scores.ColBERTScores):
        """MaxSim normalized by the number of non-padding query tokens."""

        def __call__(self, queries_embeddings, documents_embeddings,
                     queries_mask=None, documents_mask=None, backend=None):
            s = super().__call__(queries_embeddings, documents_embeddings,
                                 queries_mask, documents_mask, backend)
            n = (queries_embeddings.shape[1] if queries_mask is None
                 else queries_mask.sum(dim=-1).clamp(min=1).unsqueeze(-1).to(s.dtype))
            return s / n

    score_metric = MeanMaxSim() if args.score_metric == "meanmaxsim" else None
    log(f"loss: CachedContrastive score_metric={args.score_metric} "
        f"temperature={args.temperature}")
    loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=args.mini_batch_size,
        score_metric=score_metric,
        temperature=args.temperature,
        gather_across_devices=args.gather_across_devices,
        show_progress_bar=False,
    )

    # effective in-batch negatives: every other row's positive and negatives are negatives too
    eff_batch = args.batch_size * args.grad_accum
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.gather_across_devices:
        eff_batch *= world
    in_batch_negs = eff_batch * (1 + n_neg) - 1
    train_rows = (sum(len(d) for d in train_ds.values())
                  if isinstance(train_ds, datasets.DatasetDict) else len(train_ds))

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_ratio,   # transformers v5: float here means a ratio
        lr_scheduler_type=args.lr_scheduler,
        seed=args.seed,
        bf16=False,
        fp16=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=4,
        report_to=report_to,
        gradient_checkpointing=False,
        # PROPORTIONAL over the (already sqrt-smoothed) per-source sizes; each batch stays
        # single-source, which is the point -- in-domain in-batch negatives
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    )

    # Log the resolved method and dataset configuration.
    if not args.no_wandb:
        import wandb

        prep = data_dir / "prepare_meta.json"
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            dir=os.environ["WANDB_DIR"],
            config={
                **vars(args),
                "base_model": args.base_model,
                "negatives_per_row": n_neg,
                "train_rows": train_rows,
                "eval_rows": len(eval_ds) if eval_ds else 0,
                "effective_batch": eff_batch,
                "in_batch_negatives": in_batch_negs,
                "n_params_millions": round(n_params / 1e6, 1),
                "loss": "CachedContrastive",
                "score_metric": args.score_metric,
                "gather_across_devices": args.gather_across_devices,
                "world_size": world,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "negative_source": "LateOn MaxSim (WARP), skip=5, top-50",
                "prepare": json.loads(prep.read_text()) if prep.exists() else None,
            },
        )
        wandb.config.update({"resolved_out_dir": str(out_dir)}, allow_val_change=True)

    log(f"effective batch {eff_batch} -> {in_batch_negs} in-batch negatives per query")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
    )
    trainer.train(resume_from_checkpoint=(args.resume_from or None))

    final = out_dir / "final"
    model.save_pretrained(str(final))
    log(f"saved -> {final}")
    if not args.no_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
