#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from collections import Counter
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

from crf import LinearCRF
from ingredients.corpus import ing, read_tagging_samples
from serialization import load
from utils import extend_word_embedding

ex = Experiment("xduft-majority-tagging-testrun", ingredients=[ing])
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
    max_length = {"train": 60}
    # whether to treat keys in load_src as lang codes
    src_key_as_lang = False
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    batch_size=16,
):
    """Evaluate majority vote ensemble baseline for POS tagging."""
    if max_length is None:
        max_length = {}
    if load_src is None:
        load_src = {"src": ("artifacts", "model.pth")}

    samples = {wh: list(read_tagging_samples(wh, max_length.get(wh))) for wh in ["dev", "test"]}
    for wh in samples:
        n_toks = sum(len(s["words"]) - 2 for s in samples[wh])  # don't count BOS/EOS tokens
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    kv = KeyedVectors.load_word2vec_format(word_emb_path)

    srcs = list(load_src)
    if src_key_as_lang and corpus["lang"] in srcs:
        _log.info("Removing %s from src parsers because it's the tgt", corpus["lang"])
        srcs.remove(corpus["lang"])

    prev_tag_vocab = None
    for src_i, src in enumerate(srcs):
        _log.info("Processing src %s [%d/%d]", src, src_i + 1, len(srcs))
        load_from, load_params = load_src[src]
        path = Path(load_from) / "vocab.yml"
        _log.info("Loading %s vocabulary from %s", src, path)
        vocab = load(path.read_text(encoding="utf8"))
        for name in vocab:
            _log.info("Found %d %s", len(vocab[name]), name)
        if prev_tag_vocab is not None and vocab["tags"] != prev_tag_vocab:
            _log.warn("tag vocabulary for %s is different from ones seen so far", src)
        prev_tag_vocab = vocab["tags"]

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
            runner.state.update({"_ids": [], "ptags": []})

            @runner.on(Event.BATCH)
            def accumulate(state):
                batch = state["batch"].to_array()
                words = torch.from_numpy(batch["words"]).to(device)
                mask = words != vocab["words"].index(vocab.PAD_TOKEN)
                assert mask.all(), "must not have masking at test time"

                model.eval()
                scores = model(words)
                state["_ids"].extend(batch["_id"].tolist())
                state["ptags"].extend(LinearCRF(scores).argmax().tolist())
                state["n_items"] = words.numel()

            with torch.no_grad():
                runner.run(BucketIterator(samples_[wh], lambda s: len(s["words"]), batch_size))

            for i, ptags in zip(runner.state["_ids"], runner.state["ptags"]):
                x = samples[wh][i].get("ptags", [])
                x.append(ptags)
                samples[wh][i]["ptags"] = x

    res = None
    for wh in samples:
        corr = total = 0
        for s in tqdm(
            samples[wh], total=len(samples[wh]), unit="sample", desc=f"Evaluating on {wh}",
        ):
            assert len(s["ptags"]) == len(srcs)
            assert not s["ptags"] or all(len(ts) == len(s["ptags"][0]) for ts in s["ptags"])
            ptags = []
            for j in range(len(s["ptags"][0])):
                pt, _ = Counter(ts[j] for ts in s["ptags"]).most_common(1)[0]
                ptags.append(vocab["tags"][pt])

            tags = s["tags"]
            assert len(tags) == len(ptags)
            if tags[0] == "<s>":
                tags, ptags = tags[1:], ptags[1:]
            if tags[-1] == "</s>":
                tags, ptags = tags[:-1], ptags[:-1]

            corr += sum(pt == t for pt, t in zip(ptags, tags))
            total += len(tags)

        acc = corr / total
        if wh == "dev":
            res = acc
        _log.info("%s_acc: %.1f%%", wh, 100 * acc)
        _run.log_scalar(f"{wh}_acc", acc)

    assert res is not None
    return res
