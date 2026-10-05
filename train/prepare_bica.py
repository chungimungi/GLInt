#!/usr/bin/env python3
"""Judge and sample the public BiCA negatives as an additional SFT source."""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

JUDGE = "lightonai/GTE-ModernColBERT-v1"
# Reference statistics used to scale the biomedical source in the SFT mixture.
FIQA_USABLE, FIQA_RISK, FIQA_TARGET = 3672, 0.2075, 31979


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True, help="train_bica.jsonl path")
    ap.add_argument("--meta", default="")
    ap.add_argument("--num-negatives", type=int, default=7)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--target", type=int, default=0, help="0 = mixture formula")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from pylate import models
    from pylate.scores import colbert_scores

    rows = [json.loads(l) for l in open(args.file)]
    log(f"loaded {len(rows):,} rows")

    model = models.ColBERT(JUDGE, device="cuda").eval()

    kept: list[tuple[str, str, list[str]]] = []
    n_neg_in = n_neg_veto = 0
    n_short = 0
    t0 = time.time()
    for s in range(0, len(rows), args.batch):
        chunk = rows[s:s + args.batch]
        qs = [r["query"] for r in chunk]
        docs, spans = [], []
        for r in chunk:
            lo = len(docs)
            docs.append(r["positive"])
            docs.extend(r["negatives"])
            spans.append((lo, len(docs)))
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            q_emb = model.encode(qs, is_query=True, batch_size=128,
                                 convert_to_tensor=True, show_progress_bar=False)
            d_emb = model.encode(docs, is_query=False, batch_size=128,
                                 convert_to_tensor=True, show_progress_bar=False)
        for i, (lo, hi) in enumerate(spans):
            Q = (q_emb[i:i + 1] if not isinstance(q_emb, list)
                 else torch.nn.utils.rnn.pad_sequence(list(q_emb[i:i + 1]), batch_first=True))
            D = (d_emb[lo:hi] if not isinstance(d_emb, list)
                 else torch.nn.utils.rnn.pad_sequence(list(d_emb[lo:hi]), batch_first=True))
            with torch.no_grad():
                sc = colbert_scores(Q, D).float().cpu().numpy()[0]
            pos, negs = sc[0], sc[1:]
            n_neg_in += len(negs)
            survivors = [chunk[i]["negatives"][j] for j in range(len(negs))
                         if negs[j] < pos]
            n_neg_veto += len(negs) - len(survivors)
            if len(survivors) < args.num_negatives:
                n_short += 1
                continue
            kept.append((chunk[i]["query"], chunk[i]["positive"], survivors))
        if s and s % (args.batch * 16) == 0:
            log(f"{s + len(chunk):,}/{len(rows):,}  "
                f"{(s + len(chunk)) / (time.time() - t0):.0f} rows/s")

    usable = len(kept)
    risk = n_neg_veto / max(1, n_neg_in)
    log(f"usable rows={usable:,}  negatives vetoed={n_neg_veto:,}/{n_neg_in:,} "
        f"({risk:.4f})  rows short of {args.num_negatives}={n_short:,}")

    if args.target:
        target = args.target
    else:
        w_bica = (usable ** 0.5) * (1 - risk)
        w_fiqa = (FIQA_USABLE ** 0.5) * (1 - FIQA_RISK)
        target = int(round(FIQA_TARGET * w_bica / w_fiqa))
    log(f"target rows={target:,}  (upsample {target / max(1, usable):.2f}x)")

    rnd = random.Random(args.seed)
    written = 0
    with open(args.out, "w") as fh:
        while written < target:
            for q, p, negs in kept:
                if written >= target:
                    break
                pick = (rnd.sample(negs, args.num_negatives)
                        if len(negs) > args.num_negatives else list(negs))
                row = {"query": q, "positive": p}
                for k, n in enumerate(pick, start=1):
                    row[f"negative_{k}"] = n
                fh.write(json.dumps(row) + "\n")
                written += 1
    log(f"wrote {written:,} rows -> {args.out}")

    meta = {
        "source": "bisectgroup/hard-negatives-traversal (BiCA, arXiv 2511.08029)",
        "raw_rows": len(rows), "usable_rows": usable, "written_rows": written,
        "upsample": round(written / max(1, usable), 3),
        "negatives_in": n_neg_in, "negatives_vetoed": n_neg_veto,
        "negative_fn_rate": round(risk, 4),
        "rows_dropped_short": n_short,
        "num_negatives": args.num_negatives,
        "judge": JUDGE, "veto": "negative >= pair's own positive (per-negative)",
        "target_formula": "usable^0.5 * (1 - fn_risk), calibrated on fiqa",
        "seed": args.seed,
    }
    if args.meta:
        with open(args.meta, "w") as fh:
            json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
