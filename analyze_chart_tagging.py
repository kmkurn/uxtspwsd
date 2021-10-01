#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from argparse import ArgumentParser
from pathlib import Path
import math
import pickle
import sys

from tqdm import tqdm
import torch

from crf import LinearCRF
from serialization import load


def main(samples_file: Path, device: str = "cpu") -> None:
    print(f"Loading samples from {samples_file}", file=sys.stderr)
    with open(samples_file, "rb") as f:
        trn_samples = pickle.load(f)["train"]

    print(f"Loading vocab from {samples_file.parent / 'vocab.yml'}", file=sys.stderr)
    vocab = load((samples_file.parent / "vocab.yml").read_text("utf8"))
    print(f"Found {len(vocab['tags'])} tags", file=sys.stderr)

    log_n_prec_tps, log_n_chart_tps = [], []
    n_rec_tps = n_gold_tps = 0
    for s_id, s in enumerate(tqdm(trn_samples, desc="Computing chart quality on train set")):
        mask = torch.tensor(s["pptx_mask"]).bool().to(device)
        if mask.shape != (len(s["words"]) - 1, len(vocab["tags"]), len(vocab["tags"])):
            raise ValueError(f"train sample {s_id} has invalid mask shape")
        scores = torch.zeros_like(mask).float()
        scores.masked_fill_(~mask, -1e9)
        crf = LinearCRF(scores.unsqueeze(0))
        log_part = crf.log_partitions().item()
        margs = crf.marginals().squeeze(0)
        assert margs.shape == mask.shape

        tot_tp_p = 1e-9  # prevent zero for log later
        for i, (nt, t) in enumerate(zip(s["tags"][1:], s["tags"])):
            tot_tp_p += margs[i, vocab["tags"].index(nt), vocab["tags"].index(t)].item()
            n_rec_tps += (
                1 if margs[i, vocab["tags"].index(nt), vocab["tags"].index(t)].item() > 1e-9 else 0
            )
        n_gold_tps += len(s["tags"]) - 1
        log_n_prec_tps.append(log_part + math.log(tot_tp_p))
        log_n_chart_tps.append(log_part + math.log(len(s["tags"]) - 1))
    log_prec = torch.tensor(log_n_prec_tps).logsumexp(0) - torch.tensor(
        log_n_chart_tps
    ).logsumexp(0)
    print(f"Chart precision is {log_prec.exp().item():.1%}")
    print(f"Chart recall is {n_rec_tps / n_gold_tps:.1%}")


if __name__ == "__main__":
    p = ArgumentParser(description="Analyze chart quality")
    p.add_argument("samples_file", metavar="FILE", type=Path, help="path to samples.pkl")
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()
    main(args.samples_file, args.device)
