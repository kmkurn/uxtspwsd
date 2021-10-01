#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from argparse import ArgumentParser
from pathlib import Path
import math
import pickle
import sys

from tqdm import tqdm
import torch

from crf import DepTreeCRF
from serialization import load
from utils import report_median_ntrees


def main(
    samples_file: Path, projective: bool = False, multiroot: bool = False, device: str = "cpu"
) -> None:
    print(f"Loading samples from {samples_file}", file=sys.stderr)
    with open(samples_file, "rb") as f:
        trn_samples = pickle.load(f)["train"]

    print(f"Loading vocab from {samples_file.parent / 'vocab.yml'}", file=sys.stderr)
    vocab = load((samples_file.parent / "vocab.yml").read_text("utf8"))
    print(f"Found {len(vocab['types'])} types", file=sys.stderr)

    med_ntrees = report_median_ntrees(trn_samples, aa_mask_field="pptx_mask", projective=projective, multiroot=multiroot)
    print(f"Median number of trees in chart is {med_ntrees:.1e}")

    log_n_prec_arcs, log_n_chart_arcs = [], []
    n_rec_arcs = n_gold_arcs = 0
    for s_id, s in enumerate(tqdm(trn_samples, desc="Computing chart quality on train set")):
        mask = torch.tensor(s["pptx_mask"]).bool().to(device)
        if mask.shape != (len(s["words"]), len(s["words"]), len(vocab["types"])):
            raise ValueError(f"train sample {s_id} has invalid mask shape")
        scores = torch.zeros_like(mask).float()
        scores.masked_fill_(~mask, -1e9)
        crf = DepTreeCRF(scores.unsqueeze(0), projective=projective, multiroot=multiroot)
        log_part = crf.log_partitions().item()
        margs = crf.marginals().squeeze(0)
        assert margs.shape == mask.shape

        tot_arc_p = 1e-9  # prevent zero for log later
        if len(s["heads"]) != len(s["types"]):
            raise ValueError(f"train sample {s_id} has mismatched number of heads and types")
        for d, (h, y) in enumerate(zip(s["heads"], s["types"])):
            if d == 0 and h != 0:
                raise ValueError(f"train sample {s_id} has no root in gold tree")
            if d != 0:
                tot_arc_p += margs[h, d, vocab["types"].index(y)].item()
                n_rec_arcs += 1 if margs[h, d, vocab["types"].index(y)].item() > 1e-9 else 0
        n_gold_arcs += len(s["heads"]) - 1
        log_n_prec_arcs.append(log_part + math.log(tot_arc_p))
        log_n_chart_arcs.append(log_part + math.log(len(s["heads"]) - 1))
    log_prec = torch.tensor(log_n_prec_arcs).logsumexp(0) - torch.tensor(
        log_n_chart_arcs
    ).logsumexp(0)
    print(f"Chart precision is {log_prec.exp().item():.1%}")
    print(f"Chart recall is {n_rec_arcs / n_gold_arcs:.1%}")


if __name__ == "__main__":
    p = ArgumentParser(description="Analyze chart quality")
    p.add_argument("samples_file", metavar="FILE", type=Path, help="path to samples.pkl")
    p.add_argument("--projective", action="store_true")
    p.add_argument("--multiroot", action="store_true")
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()
    main(args.samples_file, args.projective, args.multiroot, args.device)
