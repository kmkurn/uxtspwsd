#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from itertools import chain
from pathlib import Path
import os

from gensim.models.keyedvectors import KeyedVectors
from rnnr import Event, Runner
from rnnr.attachments import ProgressBar
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import BucketIterator
from tqdm import tqdm
import torch

from callbacks import (
    batch2tensors,
    compute_total_arc_type_scores,
    get_n_items,
    predict_batch,
    set_train_mode,
)
from crf import DepTreeCRF
from evaluation import count_correct
from ingredients.corpus import ing, read_samples
from serialization import load
from utils import extend_word_embedding, print_accs

ex = Experiment("xduft-majority-testrun", ingredients=[ing])
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # load source models from these directories and parameters {key: (load_from, load_params)}
    load_src = {}
    # path to word embedding in word2vec format
    word_emb_path = "wiki.en.vec"
    # discard train/dev/test samples with length greater than these numbers
    max_length = {"train": 30, "dev": 150, "test": 150}
    # whether to treat keys in load_src as lang codes
    src_key_as_lang = False
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # whether to operate in the space of projective trees
    projective = False
    # whether to consider multi-root trees (otherwise only single-root trees)
    multiroot = False
    # batch size
    batch_size = 80


@ex.named_config
def testrun():
    seed = 12345
    corpus = dict(portion=0.05)


@ex.automain
def evaluate(
    load_src,
    word_emb_path,
    corpus,
    _log,
    _run,
    max_length=None,
    src_key_as_lang=False,
    device="cpu",
    projective=False,
    multiroot=False,
    batch_size=16,
):
    """Evaluate majority vote ensemble baseline."""
    if max_length is None:
        max_length = {}
    if load_src is None:
        load_src = {"src": ("artifacts", "model.pth")}

    samples = {
        wh: list(read_samples(which=wh, max_length=max_length.get(wh)))
        for wh in ["dev", "test"]
    }
    for wh in samples:
        n_toks = sum(len(s["words"]) for s in samples[wh])
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    kv = KeyedVectors.load_word2vec_format(word_emb_path)

    srcs = list(load_src)
    if src_key_as_lang and corpus["lang"] in srcs:
        _log.info("Removing %s from src parsers because it's the tgt", corpus["lang"])
        srcs.remove(corpus["lang"])

    prev_type_vocab = None
    for src_i, src in enumerate(srcs):
        _log.info("Processing src %s [%d/%d]", src, src_i + 1, len(srcs))
        load_from, load_params = load_src[src]
        path = Path(load_from) / "vocab.yml"
        _log.info("Loading %s vocabulary from %s", src, path)
        vocab = load(path.read_text(encoding="utf8"))
        for name in vocab:
            _log.info("Found %d %s", len(vocab[name]), name)
        if prev_type_vocab is not None and vocab["types"] != prev_type_vocab:
            _log.warn("type vocabulary for %s is different from ones seen so far", src)
        prev_type_vocab = vocab["types"]

        _log.info("Extending %s vocabulary with target words", src)
        vocab.extend(chain(*samples.values()), ["words"])
        _log.info("Found %d words now", len(vocab["words"]))

        samples_ = {wh: list(vocab.stoi(samples[wh])) for wh in samples}

        path = Path(load_from) / "model.yml"
        _log.info("Loading %s model from metadata %s", src, path)
        model = load(path.read_text(encoding="utf8"))

        path = Path(load_from) / load_params
        _log.info("Loading %s model parameters from %s", src, path)
        model.load_state_dict(torch.load(path, "cpu"))

        _log.info("Creating %s extended word embedding layer", src)
        assert model.word_emb.embedding_dim == kv.vector_size
        with torch.no_grad():
            model.word_emb = torch.nn.Embedding.from_pretrained(
                extend_word_embedding(
                    model.word_emb.weight,
                    vocab["words"],
                    kv,
                    vocab["words"].index(vocab.UNK_TOKEN),
                )
            )
        model.to(device)

        for wh in samples_:
            for i, s in enumerate(samples_[wh]):
                s["_id"] = i

            runner = Runner()
            ProgressBar(
                total=sum(len(s["words"]) for s in samples_[wh]), unit="tok", desc="Predicting"
            ).attach_on(runner)
            runner.state.update({"_ids": [], "pheads": [], "ptypes": []})
            runner.on(
                Event.BATCH,
                [
                    batch2tensors(device, vocab),
                    set_train_mode(model, training=False),
                    compute_total_arc_type_scores(model, vocab),
                    predict_batch(projective, multiroot),
                    get_n_items(),
                ],
            )

            @runner.on(Event.BATCH)
            def accumulate(state):
                state["_ids"].extend(state["batch"]["_id"].tolist())
                state["pheads"].extend(state["pred_heads"].tolist())
                state["ptypes"].extend(state["pred_types"].tolist())

            with torch.no_grad():
                runner.run(BucketIterator(samples_[wh], lambda s: len(s["words"]), batch_size))

            for i, pheads, ptypes in zip(
                runner.state["_ids"], runner.state["pheads"], runner.state["ptypes"]
            ):
                x = samples[wh][i].get("pheads", [])
                x.append(pheads)
                samples[wh][i]["pheads"] = x
                x = samples[wh][i].get("ptypes", [])
                x.append(ptypes)
                samples[wh][i]["ptypes"] = x

    res = None
    for wh in samples:
        c = None
        for s in tqdm(
            vocab.stoi(samples[wh]),
            total=len(samples[wh]),
            unit="sample",
            desc=f"Evaluating on {wh}",
        ):
            scores = torch.zeros(len(s["words"]), len(s["words"]), len(vocab["types"]))
            assert len(s["pheads"]) == len(s["ptypes"]) == len(srcs)
            for phs, pys in zip(s["pheads"], s["ptypes"]):
                assert len(phs) == len(pys) == len(s["words"])
                for d, (h, y) in enumerate(zip(phs, pys)):
                    scores[h, d, y] += 1
            pred_heads, pred_types = DepTreeCRF(
                scores.unsqueeze(0), projective=projective, multiroot=multiroot
            ).argmax()
            heads = torch.tensor(s["heads"]).unsqueeze(0)
            types = torch.tensor(s["types"]).unsqueeze(0)
            mask = torch.ones_like(heads).bool()
            is_punct = torch.tensor(s["punct?"]).unsqueeze(0)
            is_proj = torch.tensor(s["proj?"]).unsqueeze(0)

            if (heads[:, 0] == 0).all():  # remove root
                heads, types, pred_heads, pred_types = heads[:, 1:], types[:, 1:], pred_heads[:, 1:], pred_types[:, 1:]
                mask, is_punct, is_proj = mask[:, 1:], is_punct[:, 1:], is_proj[:, 1:]

            if c is None:
                c = count_correct(
                    heads, types, pred_heads, pred_types, mask, ~is_punct, is_proj
                )
            else:
                c += count_correct(
                    heads, types, pred_heads, pred_types, mask, ~is_punct, is_proj
                )

        print_accs(c.accs, on=wh, run=_run)
        if wh == "dev":
            res = c.accs["las_nopunct"]

    assert res is not None
    return res
