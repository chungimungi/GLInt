#!/usr/bin/env python3
"""Score mined candidates and annotated positives with an independent MaxSim judge."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

DATA_ROOT = ""


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class DocStore:
    def __init__(self, split):
        import pyarrow as pa
        import pyarrow.parquet as pq
        tables = [pq.read_table(p, columns=["document_id", "document"], memory_map=True)
                  for p in sorted(glob.glob(f"{DATA_ROOT}/documents/{split}-*.parquet"))]
        t = pa.concat_tables(tables)
        ids = t.column("document_id").to_numpy()
        self.texts = t.column("document")
        o = np.argsort(ids, kind="stable")
        self.ids_sorted, self.order = ids[o], o

    def get(self, doc_id):
        p = np.searchsorted(self.ids_sorted, doc_id)
        if p >= len(self.ids_sorted) or self.ids_sorted[p] != doc_id:
            return None
        return self.texts[int(self.order[p])].as_py()


def load_pool(split, parts_dir):
    acc = {}
    pd = str(Path(parts_dir).expanduser().resolve())
    for p in sorted(glob.glob(f"{pd}/{split}_*.npz")) or \
            sorted(glob.glob(f"{pd}/{split}.npz")):
        b = np.load(p)
        for q, i, s in zip(b["qids"], b["ids"], b["scores"]):
            qid = int(q)
            if qid in acc:
                pi, ps = acc[qid]
                i = np.concatenate([pi, i]); s = np.concatenate([ps, s])
            acc[qid] = (i, s)
    for qid, (i, s) in acc.items():
        o = np.argsort(-s)
        acc[qid] = (i[o], s[o])
    return acc


def gold_and_queries(split):
    import pyarrow.parquet as pq
    gold, qtext = {}, {}
    for path in sorted(glob.glob(f"{DATA_ROOT}/scores/{split}-*.parquet")):
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=8192, columns=["query_id", "document_ids"]):
            import pyarrow.compute as pc
            q = b.column("query_id").to_pylist()
            d = pc.list_element(b.column("document_ids"), 0).to_pylist()
            for qi, di in zip(q, d):
                gold.setdefault(int(qi), []).append(int(di))
    for path in sorted(glob.glob(f"{DATA_ROOT}/queries/{split}-*.parquet")):
        import pyarrow.parquet as pq2
        t = pq2.read_table(path, columns=["query_id", "query"])
        for qid, txt in zip(t.column("query_id").to_pylist(), t.column("query").to_pylist()):
            qtext[int(qid)] = txt
    return gold, qtext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="local cache of the public embeddings-fine-tuning dataset")
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--judge", default="lightonai/GTE-ModernColBERT-v1")
    ap.add_argument("--top-n", type=int, default=256)
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--out-dir", default="outputs/judge_pool")
    ap.add_argument("--encode-batch", type=int, default=256)
    args = ap.parse_args()
    global DATA_ROOT
    DATA_ROOT = str(Path(args.data_root).expanduser().resolve())

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / f"{args.split}_{args.shard:03d}.npz"
    if part.exists():
        log(f"{part.name} exists, done"); return

    import torch
    from pylate import models
    from pylate.scores import colbert_scores

    pool = load_pool(args.split, args.parts_dir)
    gold, qtext = gold_and_queries(args.split)
    store = DocStore(args.split)
    qids = sorted(q for q in pool if q in gold and q in qtext)[args.shard::args.n_shards]
    log(f"{args.split} shard {args.shard}/{args.n_shards}: {len(qids):,} queries")

    model = models.ColBERT(args.judge, device="cuda")
    model.eval()
    log(f"judge {args.judge} q={model.query_length} d={model.document_length}")

    N = args.top_n
    R_qid, R_gids, R_gsc, R_cids, R_csc = [], [], [], [], []
    t0 = time.time(); done = 0

    def flush(batch):
        """batch: list of (qid, gold_ids, gold_texts, cand_ids, cand_texts)"""
        if not batch:
            return
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            q_emb = model.encode([qtext[b[0]] for b in batch], is_query=True,
                                 batch_size=args.encode_batch, convert_to_tensor=True,
                                 show_progress_bar=False)
            docs = [t for b in batch for t in (b[2] + b[4])]
            d_emb = model.encode(docs, is_query=False, batch_size=args.encode_batch,
                                 convert_to_tensor=True, show_progress_bar=False)
        k = 0
        for j, (qid, gids, gtexts, cids, ctexts) in enumerate(batch):
            n = len(gtexts) + len(ctexts)
            de = d_emb[k:k + n]; k += n
            qe = q_emb[j]
            if isinstance(de, list):
                import torch as _t
                from torch.nn.utils.rnn import pad_sequence
                de = pad_sequence(de, batch_first=True)
            sc = colbert_scores(qe.unsqueeze(0), de).squeeze(0).float().cpu().numpy()
            g, c = sc[:len(gtexts)], sc[len(gtexts):]
            R_qid.append(qid)
            R_gids.append(np.pad(np.asarray(gids, np.int64), (0, 12 - len(gids)),
                                 constant_values=-1)[:12])
            R_gsc.append(np.pad(g.astype(np.float32), (0, 12 - len(g)),
                                constant_values=-np.inf)[:12])
            R_cids.append(np.pad(np.asarray(cids, np.int64), (0, N - len(cids)),
                                 constant_values=-1)[:N])
            R_csc.append(np.pad(c.astype(np.float32), (0, N - len(c)),
                                constant_values=-np.inf)[:N])

    batch = []
    for qid in qids:
        ids, _ = pool[qid]
        gset = set(gold[qid][:12])
        cand = [int(i) for i in ids if int(i) not in gset][:N]
        gtexts, gids = [], []
        for g in list(gset)[:12]:
            t = store.get(g)
            if t: gids.append(g); gtexts.append(t)
        ctexts, cids = [], []
        for c in cand:
            t = store.get(c)
            if t: cids.append(c); ctexts.append(t)
        if not gids or len(cids) < 5:
            continue
        batch.append((qid, gids, gtexts, cids, ctexts))
        if len(batch) >= 64:
            flush(batch); done += len(batch); batch = []
            if done % 1920 == 0:
                log(f"  {done:,}/{len(qids):,} ({done / (time.time() - t0):.0f} q/s)")
    flush(batch); done += len(batch)

    np.savez_compressed(part, qids=np.asarray(R_qid, np.int64),
                        gold_ids=np.stack(R_gids), gold_scores=np.stack(R_gsc),
                        cand_ids=np.stack(R_cids), cand_scores=np.stack(R_csc))
    log(f"wrote {part.name}: {done:,} queries in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
