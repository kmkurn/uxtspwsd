#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from pathlib import Path
import math
import os

from einops import rearrange
from gensim.models.keyedvectors import KeyedVectors
from rnnr import Event, Runner
from rnnr.attachments import ProgressBar
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import BucketIterator, ShuffleIterator
from tqdm import tqdm
import torch

from callbacks import log_stats, update_params
from crf import LinearCRF
from ingredients.corpus import ing as corpus_ing, read_tagging_samples
from serialization import dump, load
from utils import extend_word_embedding

ex = Experiment("xduft-learn-weighting-tagging-testrun", ingredients=[corpus_ing])
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # directory to save artifacts
    artifacts_dir = "lopw_tagging_artifacts"
    # path to word embedding in word2vec format
    word_emb_path = "wiki.en.vec"
    # load source models from these directories and parameters {key: (load_from, load_params)}
    load_src = {}
    # discard train/dev/test samples with length greater than these numbers
    max_length = {"train": 60}
    # whether to overwrite existing artifacts directory
    overwrite = False
    # number of sampled data to be labeled data
    n_samples = 50
    # whether to treat keys in load_src as lang codes
    src_key_as_lang = False
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # batch size
    batch_size = 16
    # learning rate
    lr = 0.1
    # max number of epochs
    max_epoch = 100


@ex.named_config
def testrun():
    seed = 12345
    corpus = dict(portion=0.05)


@ex.automain
def learn(
    artifacts_dir,
    word_emb_path,
    load_src,
    corpus,
    _log,
    _rnd,
    _run,
    max_length=None,
    overwrite=False,
    n_samples=50,
    src_key_as_lang=False,
    device="cpu",
    batch_size=16,
    lr=0.1,
    max_epoch=100,
):
    """Learn LOP weight factors."""
    if max_length is None:
        max_length = {}

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=overwrite)

    trn_samples = list(read_tagging_samples("train", max_length.get("train")))
    for i, s in enumerate(trn_samples):
        assert "id" not in s
        s["id"] = i

    try:
        _log.info("Reading picked IDs from %s", artifacts_dir / "picked_ids.txt")
        with open(artifacts_dir / "picked_ids.txt", encoding="utf8") as f:
            picked_ids = {int(line.strip()) for line in f}
        _log.info("Picked IDs hash: %d", hash(tuple(sorted(picked_ids))))
    except FileNotFoundError:
        _log.info("File not found")
        picked_ids = set(_rnd.randint(len(trn_samples), size=n_samples))
        _log.info("Picking %d samples randomly", len(picked_ids))
        _log.info("Picked IDs hash: %d", hash(tuple(sorted(picked_ids))))
        _log.info("Writing picked IDs to %s", artifacts_dir / "picked_ids.txt")
        with open(artifacts_dir / "picked_ids.txt", "w", encoding="utf8") as f:
            for id_ in sorted(picked_ids):
                print(id_, file=f)

    trn_samples = [s for s in trn_samples if s["id"] in picked_ids]
    n_toks = sum(len(s["words"]) for s in trn_samples)
    _log.info("Found %d tokens", n_toks)

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
        vocab.extend(trn_samples, ["words"])
        _log.info("Found %d words now", len(vocab["words"]))

        trn_samples_ = list(vocab.stoi(trn_samples))

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

        for i, s in enumerate(trn_samples_):
            s["_id"] = i

        runner = Runner()
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

        n_toks = sum(len(s["words"]) for s in trn_samples_)
        ProgressBar(total=n_toks, unit="tok").attach_on(runner)

        _log.info("Computing marginals for train set with source %s", src)
        with torch.no_grad():
            runner.run(BucketIterator(trn_samples_, lambda s: len(s["words"]), batch_size))

        assert len(runner.state["_ids"]) == len(trn_samples_)
        assert len(runner.state["log_marginals"]) == len(trn_samples_)
        for i, lms in zip(runner.state["_ids"], runner.state["log_marginals"]):
            trn_samples_[i]["log_marginals"] = lms

        assert len(trn_samples_) == len(trn_samples)
        _log.info("Combining the marginals")
        for i in tqdm(range(len(trn_samples_)), unit="sample", leave=False):
            lmss = trn_samples[i].get("log_marginals_ls", [])
            lmss.append(trn_samples_[i]["log_marginals"].tolist())
            assert len(lmss) == src_i + 1
            trn_samples[i]["log_marginals_ls"] = lmss

    src_logws = torch.full(
        [len(srcs)], math.log(1 / len(srcs)), device=device, requires_grad=True
    )
    opt = torch.optim.SGD([src_logws], lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lambda _: 0.9)

    learner = Runner()
    learner.state["norm"] = src_logws.norm().item()

    @learner.on(Event.BATCH)
    def train_on_batch(state):
        batch = state["batch"].to_array(pad_with=vocab["words"].index(vocab.PAD_TOKEN))
        tags = torch.tensor(batch["tags"]).long().to(device)
        mask = torch.tensor(batch["words"]).long().to(device) != vocab["words"].index(
            vocab.PAD_TOKEN
        )
        assert tags.dim() == 2 and tags.shape == mask.shape
        bsz, slen = tags.shape

        lmss = torch.tensor(batch["log_marginals_ls"]).float().to(device)
        assert lmss.shape == (bsz, len(srcs), slen - 1, len(vocab["tags"]), len(vocab["tags"]))
        lmss = rearrange(lmss, "bsz nsrc slen nntags ntags -> bsz nsrc slen (nntags ntags)")
        lmss = (src_logws.softmax(dim=0).reshape(1, -1, 1, 1) * lmss).sum(dim=1)
        assert lmss.shape == (bsz, slen - 1, len(vocab["tags"]) * len(vocab["tags"]))
        lmss = lmss.log_softmax(dim=2)  # renormalise

        llhs = rearrange(
            lmss, "bsz slen (nntags ntags) -> bsz slen nntags ntags", ntags=len(vocab["tags"])
        )
        llhs = llhs.gather(
            2, tags[:, 1:].reshape(bsz, -1, 1, 1).expand(bsz, -1, 1, len(vocab["tags"]))
        ).squeeze(2)
        assert llhs.shape == (bsz, slen - 1, len(vocab["tags"]))
        llhs = llhs.gather(2, tags[:, :-1].unsqueeze(2)).squeeze(2)
        assert llhs.shape == (bsz, slen - 1)

        length = mask.long().sum(dim=1)
        assert length.shape == (bsz,)
        mask.scatter_(1, length.unsqueeze(1) - 1, False)
        assert not mask[:, -1].any()
        mask = mask[:, :-1]
        state["loss"] = -llhs.masked_fill(~mask, 0).sum() / mask.float().sum()
        state["stats"] = {"loss": state["loss"].item(), "ppl": state["loss"].exp().item()}

    learner.on(Event.BATCH, [update_params(opt), log_stats(_run)])

    @learner.on(Event.BATCH)
    def maybe_report_stats(state):
        if state["n_iters"] % 5 != 0:
            return
        _log.info(
            "Iter %d: loss %.4f | ppl %.4f | grad %.4f",
            state["n_iters"],
            state["stats"]["loss"],
            state["stats"]["ppl"],
            src_logws.grad.norm().item(),
        )

    @learner.on(Event.BATCH)
    def maybe_log_grads(state):
        if state["n_iters"] % 5 != 0:
            return
        _run.log_scalar("grad", src_logws.grad.norm().item(), state["n_iters"])

    @learner.on(Event.BATCH)
    def maybe_stop(state):
        if abs(src_logws.norm().item() - state["norm"]) < 0.001:
            state["running"] = False
        else:
            state["norm"] = src_logws.norm().item()

    learner.on(Event.EPOCH_FINISHED, lambda _: scheduler.step())

    iter_ = ShuffleIterator(
        BucketIterator(
            list(vocab.stoi(trn_samples)), lambda s: (len(s["words"]) - 1) // 10, batch_size,
        ),
        rng=_rnd,
    )
    _log.info("Found %d batches", len(iter_))
    _log.info("Learning the weights")
    try:
        learner.run(iter_, max_epoch)
    except KeyboardInterrupt:
        _log.info("Interrupt detected; aborting training")

    _log.info("Sources: %s", srcs)
    _log.info("Weights: %s", src_logws.softmax(dim=0).tolist())
    _log.info("Saving weights to %s", artifacts_dir / "src2ws.yml")
    src2ws = dict(zip(srcs, src_logws.softmax(dim=0).tolist()))
    (artifacts_dir / "src2ws.yml").write_text(dump(src2ws), encoding="utf8")

    loss = learner.state.get("loss")
    if loss is not None:
        loss = loss.item()
    return loss
