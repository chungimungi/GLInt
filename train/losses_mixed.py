"""Temperature-scaled listwise KL distillation with an auxiliary InfoNCE term."""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from pylate.losses.contrastive import extract_skiplist_mask
from pylate.models import ColBERT
from pylate.scores import colbert_kd_scores


def _minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    lo = x.min(dim=1, keepdim=True).values
    hi = x.max(dim=1, keepdim=True).values
    return (x - lo) / (hi - lo + eps)


class MixedDistillation(torch.nn.Module):
    """KL(teacher || student) with temperatures, plus an optional InfoNCE anchor.

    Parameters
    ----------
    tau_teacher, tau_student
        Temperatures applied to the min-max-normalised teacher / student score vectors before
        log_softmax. 1.0 for both reproduces pylate's `Distillation`.
    w_kl, w_nce
        Weights on the two terms. The paper's tuned pair is 1.0 / 0.1.
    tau_nce
        Temperature for the contrastive logits (min-max-normalised student scores).
    fn_mask_thresh
        Candidates whose normalised teacher score is above this are dropped from the InfoNCE
        denominator as probable false negatives. None disables. At 0.6 this removes ~5.9 of 32
        candidates per query on the bge-gemma data.
    clip_pct
        Row-wise percentile clip on teacher scores before normalisation. 0.0 disables.
    normalize_scores
        Min-max normalise the student score vector, as pylate does.
    """

    def __init__(
        self,
        model: ColBERT,
        score_metric: Callable = colbert_kd_scores,
        tau_teacher: float = 0.3,
        tau_student: float = 1.0,
        w_kl: float = 1.0,
        w_nce: float = 0.1,
        tau_nce: float = 0.05,
        fn_mask_thresh: float | None = 0.6,
        clip_pct: float = 0.0,
        normalize_scores: bool = True,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.score_metric = score_metric
        self.tau_teacher = tau_teacher
        self.tau_student = tau_student
        self.w_kl = w_kl
        self.w_nce = w_nce
        self.tau_nce = tau_nce
        self.fn_mask_thresh = fn_mask_thresh
        self.clip_pct = clip_pct
        self.normalize_scores = normalize_scores
        self.kl = torch.nn.KLDivLoss(
            reduction="batchmean" if size_average else "sum", log_target=True
        )

    def _student_scores(
        self, sentence_features: Iterable[dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Byte-for-byte the score path of pylate.losses.Distillation."""
        queries_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[0])["token_embeddings"], p=2, dim=-1
        )
        documents_embeddings = torch.nn.functional.normalize(
            self.model(sentence_features[1])["token_embeddings"], p=2, dim=-1
        )
        documents_embeddings = documents_embeddings.view(
            queries_embeddings.size(0), -1, *documents_embeddings.shape[1:]
        )
        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )
        documents_embeddings_mask = masks[1].view(
            queries_embeddings.size(0), -1, *masks[1].shape[1:]
        )
        return self.score_metric(
            queries_embeddings,
            documents_embeddings,
            queries_mask=masks[0] if not do_query_expansion else None,
            documents_mask=documents_embeddings_mask,
        )

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor
    ) -> torch.Tensor:
        # float32 for the loss arithmetic: min-max followed by division by a temperature as
        # small as 0.05 amplifies bf16's ~3 significant digits into visible quantisation of the
        # target distribution. pylate's Distillation leaves this in bf16.
        scores = self._student_scores(sentence_features).float()
        if self.normalize_scores:
            scores = _minmax(scores)

        teacher = labels.float()
        if self.clip_pct > 0.0:
            lo = torch.quantile(teacher, self.clip_pct, dim=1, keepdim=True)
            hi = torch.quantile(teacher, 1.0 - self.clip_pct, dim=1, keepdim=True)
            teacher = teacher.clamp(min=lo, max=hi)
        # Idempotent when the source dataset already ships per-row min-max scores, and
        # necessary when it does not (raw logits, or cosines in [-1, 1] from a listwise
        # teacher), so that tau_teacher means the same thing across teachers.
        teacher = _minmax(teacher)

        loss = self.w_kl * self.kl(
            torch.nn.functional.log_softmax(scores / self.tau_student, dim=-1),
            torch.nn.functional.log_softmax(teacher / self.tau_teacher, dim=-1),
        )

        if self.w_nce > 0.0:
            positive = teacher.argmax(dim=1)
            logits = scores / self.tau_nce
            if self.fn_mask_thresh is not None:
                drop = teacher > self.fn_mask_thresh
                drop.scatter_(1, positive.unsqueeze(1), False)  # never mask the positive
                logits = logits.masked_fill(drop, float("-inf"))
            loss = loss + self.w_nce * torch.nn.functional.cross_entropy(logits, positive)

        return loss
