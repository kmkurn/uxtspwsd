# Copyright (c) 2021 Kemal Kurniawan

from typing import Mapping, Optional, Sequence
import logging
import math

from sacred.run import Run
from torch import Tensor
from tqdm import tqdm
import numpy as np
import torch

from crf import DepTreeCRF


logger = logging.getLogger(__name__)


def extend_word_embedding(
    weight: Tensor,
    words: Sequence[str],
    kv: Optional[Mapping[str, np.ndarray]] = None,
    unk_id: Optional[int] = None,
) -> Tensor:
    assert weight.dim() == 2
    if kv is None:
        kv = {}

    new_weight = torch.randn(len(words), weight.size(1))
    new_weight[: weight.size(0)] = weight
    cnt_pre, cnt_unk = 0, 0
    for w in words:
        wid = words.index(w)
        if wid < weight.size(0):
            continue
        if w in kv:
            new_weight[wid] = torch.from_numpy(kv[w])
            cnt_pre += 1
        elif w.lower() in kv:
            new_weight[wid] = torch.from_numpy(kv[w.lower()])
            cnt_pre += 1
        else:
            cnt_unk += 1
            if unk_id is not None:
                new_weight[wid] = weight[unk_id]

    logger.info("Initialized %d target words with pre-trained embedding", cnt_pre)
    logger.info("Found %d unknown words", cnt_unk)

    return new_weight


def report_median_ntrees(
    samples: Sequence[dict],
    aa_mask_field: str,
    batch_size: int = 1,
    projective: bool = False,
    multiroot: bool = False,
) -> float:
    log_ntrees: list = []
    for s in tqdm(samples, unit="sample", leave=False):
        mask = torch.tensor(s[aa_mask_field]).unsqueeze(0)
        cnt_scores = torch.zeros_like(mask).float().masked_fill(~mask, -1e9)
        log_ntrees.append(
            DepTreeCRF(cnt_scores, projective=projective, multiroot=multiroot)
            .log_partitions()
            .item()
        )
    log_ntrees.sort()
    mid = len(log_ntrees) // 2
    if len(log_ntrees) % 2:
        log_median = log_ntrees[mid]
    else:
        max_ = max(log_ntrees[mid - 1], log_ntrees[mid])
        log_median = (
            max_
            + math.log(math.exp(log_ntrees[mid - 1] - max_) + math.exp(log_ntrees[mid] - max_))
            - math.log(2)
        )
    logger.info("Median number of trees in chart: %.1e", math.exp(log_median))
    return math.exp(log_median)


def print_accs(
    accs: Mapping[str, float],
    on: str = "dev",
    run: Optional[Run] = None,
    step: Optional[int] = None,
) -> None:
    for key, acc in accs.items():
        logger.info(f"{on}_{key}: {acc:.2%}")
        if run is not None:
            run.log_scalar(f"{on}_{key}", acc, step=step)
