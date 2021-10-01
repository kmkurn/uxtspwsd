# Copyright (c) 2021 Kemal Kurniawan

from typing import Optional
import math

from einops import rearrange
from torch import BoolTensor, Tensor

from crf import DepTreeCRF, LinearCRF


def compute_aatrn_loss(
    scores: Tensor,
    aa_mask: BoolTensor,
    mask: Optional[BoolTensor] = None,
    projective: bool = False,
    multiroot: bool = True,
) -> Tensor:
    assert aa_mask.shape == scores.shape
    masked_scores = scores.masked_fill(~aa_mask, -1e9)
    crf = DepTreeCRF(masked_scores, mask, projective, multiroot)
    crf_z = DepTreeCRF(scores, mask, projective, multiroot)
    return -crf.log_partitions().sum() + crf_z.log_partitions().sum()


def compute_ambiguous_arcs_mask(
    scores: Tensor,
    threshold: float = 0.95,
    projective: bool = False,
    multiroot: bool = True,
    is_log_marginals: bool = False,
) -> BoolTensor:
    """If is_log_marginals then scores are assumed to be the log marginals."""
    assert scores.dim() == 4
    assert 0 <= threshold <= 1

    if is_log_marginals:
        return _compute_ambiguous_arcs_mask_from_log_marginals(
            scores, threshold, projective, multiroot
        )
    return _compute_ambiguous_arcs_mask(scores, threshold, projective, multiroot)


def compute_ambiguous_tag_pairs_mask(
    scores: Tensor, threshold: float = 0.95, is_log_marginals: bool = False
) -> BoolTensor:
    if is_log_marginals:
        return _compute_ambiguous_tag_pairs_mask_from_log_marginals(scores, threshold)
    return _compute_ambiguous_tag_pairs_mask(scores, threshold)


def _compute_ambiguous_arcs_mask(
    scores, threshold, projective, multiroot, include_max_tree=True
):
    _, slen, _, n_types = scores.shape
    crf = DepTreeCRF(scores, projective=projective, multiroot=multiroot)
    marginals = crf.marginals()

    # select high-prob arcs until their cumulative probability exceeds threshold
    marginals = rearrange(marginals, "bsz hlen dlen ntypes -> bsz dlen (hlen ntypes)")
    marginals, orig_indices = marginals.sort(dim=2, descending=True)
    arc_mask = marginals.cumsum(dim=2) < threshold

    # mark the arc that makes the cum sum exceeds threshold
    last_idx = arc_mask.long().sum(dim=2, keepdim=True).clamp(max=slen * n_types - 1)
    arc_mask = arc_mask.scatter(2, last_idx, True)

    # restore the arc_mask order and shape
    _, restore_indices = orig_indices.sort(dim=2)
    arc_mask = arc_mask.gather(2, restore_indices)

    if include_max_tree:
        # ensure maximum scoring tree is selected
        # each shape: (bsz, slen)
        best_heads, best_types = crf.argmax()
        best_idx = best_heads * n_types + best_types
        arc_mask = arc_mask.scatter(2, best_idx.unsqueeze(2), True)

    arc_mask = rearrange(arc_mask, "bsz dlen (hlen ntypes) -> bsz hlen dlen ntypes", hlen=slen)
    return arc_mask


def _compute_ambiguous_arcs_mask_from_log_marginals(
    log_marginals, threshold, projective, multiroot
):
    _, slen, _, n_types = log_marginals.shape

    # select high-prob arcs until their cumulative probability exceeds threshold
    log_marginals = rearrange(log_marginals, "bsz hlen dlen ntypes -> bsz dlen (hlen ntypes)")
    log_marginals, orig_indices = log_marginals.sort(dim=2, descending=True)
    arc_mask = _logcumsumexp(log_marginals, dim=2) < math.log(threshold)

    # mark the arc that makes the cum sum exceeds threshold
    last_idx = arc_mask.long().sum(dim=2, keepdim=True).clamp(max=slen * n_types - 1)
    arc_mask = arc_mask.scatter(2, last_idx, True)

    # restore the arc_mask order and shape
    _, restore_indices = orig_indices.sort(dim=2)
    arc_mask = arc_mask.gather(2, restore_indices)

    arc_mask = rearrange(arc_mask, "bsz dlen (hlen ntypes) -> bsz hlen dlen ntypes", hlen=slen)
    return arc_mask


def _compute_ambiguous_tag_pairs_mask(
    scores: Tensor, threshold: float = 0.95, include_max_tags: bool = True
) -> BoolTensor:
    bsz, slen, n_next_tags, n_tags = scores.shape

    crf = LinearCRF(scores)
    margs = crf.marginals()

    # select high prob tag pairs until their cumulative probability exceeds threshold
    margs = rearrange(margs, "bsz slen nntags ntags -> bsz slen (nntags ntags)")
    margs, orig_indices = margs.sort(dim=2, descending=True)
    tp_mask = margs.cumsum(dim=2) < threshold

    # select the tag pairs that make the cum sum exceeds threshold
    last_idx = tp_mask.long().sum(dim=2, keepdim=True).clamp(max=n_next_tags * n_tags - 1)
    tp_mask = tp_mask.scatter(2, last_idx, True)

    # restore the order and shape
    _, restore_indices = orig_indices.sort(dim=2)
    tp_mask = tp_mask.gather(2, restore_indices)

    if include_max_tags:
        best_tags = crf.argmax()
        assert best_tags.shape == (bsz, slen + 1)
        best_idx = best_tags[:, 1:] * n_tags + best_tags[:, :-1]
        assert best_idx.shape == (bsz, slen)
        tp_mask = tp_mask.scatter(2, best_idx.unsqueeze(2), True)

    tp_mask = rearrange(
        tp_mask, "bsz slen (nntags ntags) -> bsz slen nntags ntags", nntags=n_next_tags
    )
    return tp_mask  # type: ignore


def _compute_ambiguous_tag_pairs_mask_from_log_marginals(
    log_marginals: Tensor, threshold: float = 0.95
) -> BoolTensor:
    _, _, n_next_tags, n_tags = log_marginals.shape

    # select high prob tag pairs until their cumulative probability exceeds threshold
    log_margs = rearrange(log_marginals, "bsz slen nntags ntags -> bsz slen (nntags ntags)")
    log_margs, orig_indices = log_margs.sort(dim=2, descending=True)
    tp_mask = _logcumsumexp(log_margs, dim=2) < math.log(threshold)

    # select the tag pairs that make the cum sum exceeds threshold
    last_idx = tp_mask.long().sum(dim=2, keepdim=True).clamp(max=n_next_tags * n_tags - 1)
    tp_mask = tp_mask.scatter(2, last_idx, True)

    # restore the order and shape
    _, restore_indices = orig_indices.sort(dim=2)
    tp_mask = tp_mask.gather(2, restore_indices)

    tp_mask = rearrange(
        tp_mask, "bsz slen (nntags ntags) -> bsz slen nntags ntags", nntags=n_next_tags
    )
    return tp_mask  # type: ignore


def _logcumsumexp(x: Tensor, dim: int = -1) -> Tensor:
    max = x.max(dim, keepdim=True)[0]
    return (x - max).exp().cumsum(dim).log() + max
