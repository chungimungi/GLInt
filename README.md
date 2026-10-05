# GLInt reproducibility code

## Public resources

| Role | Public identifier |
|---|---|
| Student initialization and MaxSim miner | [`lightonai/LateOn-unsupervised`](https://huggingface.co/lightonai/LateOn-unsupervised) |
| Independent late-interaction judge | [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) |
| Listwise teacher and audit model | [`jinaai/jina-reranker-v3.5`](https://huggingface.co/jinaai/jina-reranker-v3.5) |
| Dense-mining comparison | [`lightonai/DenseOn`](https://huggingface.co/lightonai/DenseOn) |
| General retrieval data | [`lightonai/embeddings-fine-tuning`](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning) |
| Biomedical SFT source | [`bisectgroup/hard-negatives-traversal`](https://huggingface.co/datasets/bisectgroup/hard-negatives-traversal) |

## Method

1. `mine/build.py` partitions each corpus and retrieves 2,048 candidates per query with WARP using MaxSim.
2. `mine/judge_pool.py` scores candidates and annotated positives with GTE-ModernColBERT. Candidates at or above the positive score are vetoed.
3. `train/prepare_judged.py` samples seven negatives from the judge-cleared score band and allocates rows in proportion to `sqrt(usable queries) * (1 - false-negative rate)`. `train/prepare_bica.py` applies the same per-negative veto and mixture weighting to BiCA.
4. `train/train.py` fine-tunes the student with MeanMaxSim InfoNCE.
5. `train/build_kd_mixture.py`, `train/score_jina.py`, and `train/train_kd_mixed.py` build 32-document lists, score them with the cross-encoder, and train with listwise KL plus weighted InfoNCE.

The dense-mining comparison uses `mine/mine_dense.py`, `mine/merge_ablation.py`, and `mine/audit_negatives.py`. The filtering comparison uses `--disable-filter` with `prepare_judged.py` and audits both sets with `train/gate_audit.py`.

## Setup

Install the packages in `requirements.txt` in an environment with GPU support:

```bash
python -m pip install -r requirements.txt
```

Obtain the public datasets and make their parquet shards available in the expected `queries/`, `documents/`, and `scores/` layout. Set `DATA_ROOT` to that local data root. All working and output locations are command-line arguments or relative output directories; no machine-specific paths are embedded in the code.

Mining and judging are sharded. Run each command for every corpus split and shard; the example below shows one task. Use the same split and shard count for all mining and judge jobs.

```bash
python mine/build.py \
  --data-root "$DATA_ROOT" --split msmarco --shard 0 --n-shards 1 \
  --top-p 2048 --work-dir outputs/mining

python mine/judge_pool.py \
  --data-root "$DATA_ROOT" --split msmarco --shard 0 --n-shards 1 \
  --parts-dir outputs/mining/parts --top-n 256 --out-dir outputs/judge

python mine/merge_pairs.py \
  --data-root "$DATA_ROOT" --work-dir outputs/mining \
  --out outputs/mined_pairs.jsonl

python train/prepare_judged.py \
  --data-root "$DATA_ROOT" --pairs outputs/mined_pairs.jsonl \
  --judge-dir outputs/judge --out-dir outputs/sft_data
```

For the filtering ablation, repeat preparation with `--disable-filter` and a separate output directory, then compare the filtered and unfiltered sets:

```bash
python train/prepare_judged.py \
  --data-root "$DATA_ROOT" --pairs outputs/mined_pairs.jsonl \
  --judge-dir outputs/judge --out-dir outputs/sft_data_unfiltered \
  --disable-filter

python train/gate_audit.py \
  --dirs outputs/sft_data outputs/sft_data_unfiltered \
  --out outputs/filter_audit.json
```

For the mining ablation, run `mine/mine_dense.py` over the same splits and shards as `mine/build.py`, then pair the candidate outputs and score them:

```bash
python mine/mine_dense.py --data-root "$DATA_ROOT" --split msmarco --shard 0 --n-shards 1 --top-p 2048 --work-dir outputs/mining_dense

python mine/merge_ablation.py \
  --data-root "$DATA_ROOT" --maxsim-parts outputs/mining/parts \
  --dense-parts outputs/mining_dense/parts_dense \
  --out outputs/mining_ablation.jsonl

python mine/audit_negatives.py \
  --data-root "$DATA_ROOT" --source outputs/mining_ablation.jsonl \
  --out outputs/mining_audit.json
```

## Training and evaluation

The paper configuration uses eight GPUs. SFT uses one epoch, per-device batch size 16, learning rate `3e-6`, MeanMaxSim temperature `0.001`, and cross-device gathering. The distillation stage uses 32 candidates, per-device batch size 8, accumulation 2, learning rate `1e-5`, teacher/student temperatures `0.3`, and InfoNCE weight `0.1` with temperature `0.05`.

```bash
torchrun --standalone --nproc_per_node=8 train/train.py \
  --base-model lightonai/LateOn-unsupervised --load-as-is \
  --data-dir outputs/sft_data --multi-source --run-name sft \
  --batch-size 16 --lr 3e-6 --score-metric meanmaxsim \
  --temperature 0.001 --gather-across-devices --no-wandb
```

Build the seven-source KD lists from the judged pool, score every list shard with `score_jina.py`, then train the KD student:

```bash
python train/build_kd_mixture.py \
  --data-root "$DATA_ROOT" --pool-dir outputs/judge \
  --out outputs/kd_mixture

python train/score_jina.py \
  --shard 0 --n-shards 1 --dataset-dir outputs/kd_mixture \
  --out-dir outputs/kd_scores

torchrun --standalone --nproc_per_node=8 train/train_kd_mixed.py \
  --init runs/sft/final --run-name kd --dataset-dir outputs/kd_mixture \
  --teacher-scores outputs/kd_scores --n-ways 32 --batch-size 8 --accum 2 \
  --lr 1e-5 --tau-teacher 0.3 --tau-student 0.3 \
  --w-kl 1.0 --w-nce 0.1 --tau-nce 0.05 --no-wandb
```

`train/eval_beir.py` evaluates a trained checkpoint using a local Hugging Face MTEB cache. Pass the cache root with `--beir-cache`; pass `--decontaminated-root` when evaluating the decontaminated export.
