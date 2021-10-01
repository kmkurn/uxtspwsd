#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from itertools import chain
from pathlib import Path
import os

from einops import rearrange
from gensim.models.keyedvectors import KeyedVectors
from rnnr import Event, Runner
from rnnr.attachments import ProgressBar
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import BucketIterator
from torch.distributions import Categorical, kl_divergence
from tqdm import tqdm
import torch

from crf import LinearCRF
from ingredients.corpus import ing as corpus_ing, read_tagging_samples
from serialization import load
from utils import extend_word_embedding

ex = Experiment("xduft-divergences-tagging-testrun", ingredients=[corpus_ing])
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
    # discard train/dev/test samples with length greater than these numbers
    max_length = {}
    # path to word embedding in word2vec format
    word_emb_path = "wiki.en.vec"
    # whether to treat keys in load_src as lang codes
    src_key_as_lang = False
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # batch size
    batch_size = 80


@ex.named_config
def normalize_digits():
    corpus = dict(normalize_digits=True)


@ex.named_config
def testrun():
    seed = 12345
    corpus = dict(portion=0.05)


@ex.automain
def compute_divergences(
    corpus,
    load_src,
    _log,
    _run,
    max_length=None,
    word_emb_path="wiki.id.vec",
    src_key_as_lang=False,
    device="cpu",
    batch_size=16,
):
    """Compute divergences of source taggers a la Heskes (1998)."""
    if max_length is None:
        max_length = {}

    samples = {
        wh: list(read_tagging_samples(wh, max_length.get(wh))) for wh in ["train", "dev"]
    }

    for wh in samples:
        n_toks = sum(len(s["words"]) - 2 for s in samples[wh])  # don't count BOS/EOS tokens
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    kv = KeyedVectors.load_word2vec_format(word_emb_path)

    srcs = list(load_src.keys())
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
        if prev_tag_vocab is not None and vocab["tags"] != prev_tag_vocab:
            raise ValueError(f"tag vocab for src {src} isn't compatible")
        prev_tag_vocab = vocab["tags"]
        for name in vocab:
            _log.info("Found %d %s", len(vocab[name]), name)

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

        for wh in ["train", "dev"]:
            for i, s in enumerate(samples_[wh]):
                s["_id"] = i

            runner = Runner()
            runner.state["_ids"] = []
            runner.state.update({"_ids": [], "log_marginals": []})

            @runner.on(Event.BATCH)
            def compute_marginals(state):
                batch = state["batch"].to_array()
                words = torch.from_numpy(batch["words"]).to(device)
                mask = words != vocab["words"].index(vocab.PAD_TOKEN)
                assert mask.all(), "must not have masking at test time"

                model.eval()
                scores = model(words)
                crf = LinearCRF(scores)
                lm = (crf.marginals() + 1e-9).log()
                assert not torch.isnan(lm).any()
                state["log_marginals"].extend(lm)
                state["_ids"].extend(batch["_id"].tolist())
                state["n_items"] = words.numel()

            n_toks = sum(len(s["words"]) for s in samples_[wh])
            ProgressBar(total=n_toks, unit="tok").attach_on(runner)

            _log.info("Computing marginals for %s set with source %s", wh, src)
            with torch.no_grad():
                runner.run(BucketIterator(samples_[wh], lambda s: len(s["words"]), batch_size))

            assert len(runner.state["log_marginals"]) == len(samples_[wh])
            assert len(runner.state["_ids"]) == len(samples_[wh])
            for i, lms in zip(runner.state["_ids"], runner.state["log_marginals"]):
                samples_[wh][i]["log_marginals"] = lms

            assert len(samples_[wh]) == len(samples[wh])
            _log.info("Combining the marginals")
            for i in tqdm(range(len(samples_[wh])), unit="sample", leave=False):
                lms = samples[wh][i].get("log_marginals", 0)
                lms = torch.tensor(lms, device=device) + samples_[wh][i]["log_marginals"]
                samples[wh][i]["log_marginals"] = lms.tolist()
                lmss = samples[wh][i].get("log_marginals_ls", [])
                lmss.append(samples_[wh][i]["log_marginals"].tolist())
                assert len(lmss) == src_i + 1
                samples[wh][i]["log_marginals_ls"] = lmss

    for wh in ["train", "dev"]:
        _log.info("Computing the LOP on %s set", wh)
        for s in tqdm(samples[wh], unit="sample", leave=False):
            lms = torch.tensor(s["log_marginals"])
            lms /= len(srcs)
            assert lms.dim() == 3 and lms.size(1) == lms.size(2)
            # Renormalise the marginal probabilities
            lms = rearrange(lms, "slen nntags ntags -> slen (nntags ntags)")
            lms = lms.log_softmax(dim=1)
            lms = rearrange(
                lms, "slen (nntags ntags) -> slen nntags ntags", ntags=len(vocab["tags"])
            )
            s["lop"] = lms
            s.pop("log_marginals")

    for wh in ["train", "dev"]:
        _log.info("Computing error and diversity on %s set", wh)
        error = diversity = 0
        for src_i in range(len(srcs)):
            n_words = total_q_kl = total_kl = 0
            for s in tqdm(samples[wh], unit="sample", leave=False):
                lms = torch.tensor(s["log_marginals_ls"][src_i])
                assert lms.dim() == 3 and lms.size(1) == lms.size(2)

                for j in range(1, len(s["words"])):
                    if s["words"][j - 1] in ("<s>", "</s>"):
                        continue
                    if s["words"][j] in ("<s>", "</s>"):
                        continue
                    n_words += 1
                    total_q_kl += -lms[
                        j - 1,
                        vocab["tags"].index(s["tags"][j]),
                        vocab["tags"].index(s["tags"][j - 1]),
                    ]

                lop = s["lop"]
                assert lms.shape == lop.shape
                lms = rearrange(lms, "slen nntags ntags -> slen (nntags ntags)")
                lop = rearrange(lop, "slen nntags ntags -> slen (nntags ntags)")
                if s["words"] and s["words"][0] == "<s>":
                    lms, lop = lms[1:], lop[1:]  # remove BOS
                if s["words"] and s["words"][-1] == "</s>":
                    lms, lop = lms[:-1], lop[:-1]  # remove EOS
                kl = kl_divergence(Categorical(logits=lop), Categorical(logits=lms))
                assert kl.dim() == 1
                total_kl += kl.sum()

            q_kl = total_q_kl / n_words
            src_kl = total_kl / n_words
            error += q_kl
            diversity += src_kl
        error /= len(srcs)
        diversity /= len(srcs)
        _log.info("Error is %.4f", error)
        _log.info("Diversity is %.4f", diversity)
        _run.log_scalar(f"error_on_{wh}", float(error))
        _run.log_scalar(f"diversity_on_{wh}", float(diversity))
