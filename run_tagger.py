#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from itertools import chain
from pathlib import Path
import os

from gensim.models.keyedvectors import KeyedVectors
from rnnr import Event, Runner
from rnnr.attachments import EpochTimer, ProgressBar, SumReducer
from rnnr.callbacks import maybe_stop_early
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import BucketIterator, ShuffleIterator, Vocab
import torch

from callbacks import log_grads, log_stats, save_state_dict, update_params
from crf import LinearCRF
from ingredients.corpus import ing as corpus_ing, read_tagging_samples
from models import POSTagger
from serialization import dump, load
from utils import extend_word_embedding

ex = Experiment("xduft-tagger-testrun", ingredients=[corpus_ing])
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # discard train/dev/test samples with length greater than these numbers
    max_length = {"train": 100, "dev": 999, "test": 999}
    # load tag vocabulary from this file
    load_tags_vocab_from = ""
    # directory to save training artifacts
    artifacts_dir = ""
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # learning rate
    lr = 4.9e-4
    # how many epochs to wait before early stopping
    patience = 50
    # batch size
    batch_size = 80
    # max number of epochs
    max_epoch = 1000
    # path to word embedding in word2vec format
    word_emb_path = "wiki.en.vec"
    # word dropout rate
    word_dropout = 0.039
    # number of heads in transformer encoder
    n_heads = 8
    # size of feedforward hidden layer in transformer encoder
    ff_size = 512
    # size of keys and values in the transformer encoder
    kv_size = 64
    # number of layers in transformer encoder
    n_layers = 6
    # whether to evaluate on train set at every epoch end
    eval_on_train = False
    # load model parameters from this file under artifacts directory (only for evaluate)
    load_from = ""


@ex.named_config
def testrun():
    seed = 12345
    n_heads = 2
    n_layers = 2
    ff_size = 7
    kv_size = 6
    max_epoch = 3
    corpus = dict(portion=0.05)


@ex.capture
def make_model(
    vocab,
    word_emb_path,
    _log,
    artifacts_dir=None,
    word_dropout=0.1,
    n_heads=10,
    ff_size=2048,
    kv_size=64,
    n_layers=6,
):
    kv = KeyedVectors.load_word2vec_format(word_emb_path)
    model = POSTagger(
        len(vocab["words"]),
        len(vocab["tags"]),
        kv.vector_size,
        word_dropout,
        n_heads,
        ff_size,
        kv_size,
        n_layers,
    )
    _log.info("Model created with %d parameters", sum(p.numel() for p in model.parameters()))

    weight = torch.randn(len(vocab["words"]), kv.vector_size)
    cnt_pre, cnt_unk = 0, 0
    for w in vocab["words"]:
        wid = vocab["words"].index(w)
        if w in kv:
            weight[wid] = torch.from_numpy(kv[w])
            cnt_pre += 1
        elif w.lower() in kv:
            weight[wid] = torch.from_numpy(kv[w.lower()])
            cnt_pre += 1
        else:
            cnt_unk += 1

    with torch.no_grad():
        # freeze embedding to preserve alignment
        model.word_emb = torch.nn.Embedding.from_pretrained(weight, freeze=True)
    _log.info("Initialized %d words with pre-trained embedding", cnt_pre)
    _log.info("Found %d unknown words", cnt_unk)

    if artifacts_dir:
        path = Path(artifacts_dir) / "model.yml"
        _log.info("Saving model metadata to %s", path)
        path.write_text(dump(model), encoding="utf8")

    return model


@ex.capture
def run_eval(model, vocab, samples, device="cpu", batch_size=16):
    runner = Runner()
    SumReducer("corr", value="bcorr").attach_on(runner)
    SumReducer("total", value="btotal").attach_on(runner)
    ProgressBar(total=sum(len(s["words"]) for s in samples), unit="tok", leave=False).attach_on(
        runner
    )

    @runner.on(Event.BATCH)
    def evaluate_batch(state):
        batch = state["batch"].to_array()
        words = torch.from_numpy(batch["words"]).to(device)
        tags = torch.from_numpy(batch["tags"]).to(device)
        mask = words != vocab["words"].index(vocab.PAD_TOKEN)
        assert mask.all(), "must not have masking at test time"

        model.eval()
        scores = model(words)
        ptags = LinearCRF(scores).argmax()
        assert ptags.shape == tags.shape

        if (tags[:, 0] == vocab["tags"].index("<s>")).all():
            tags, ptags = tags[:, 1:], ptags[:, 1:]
        if (tags[:, -1] == vocab["tags"].index("</s>")).all():
            tags, ptags = tags[:, :-1], ptags[:, :-1]

        state["bcorr"] = (ptags == tags).long().sum().item()
        state["btotal"] = tags.numel()
        state["n_items"] = words.numel()

    with torch.no_grad():
        runner.run(BucketIterator(samples, lambda s: len(s["words"]), batch_size))

    return runner.state["corr"] / runner.state["total"]


@ex.command
def evaluate(artifacts_dir, load_from, _log, _run, max_length=None, device="cpu", word_emb_path="wiki.id.vec"):
    """Evaluate a trained POS tagger."""
    if max_length is None:
        max_length = {}

    samples = {wh: list(read_tagging_samples(wh, max_length.get(wh))) for wh in ["dev", "test"]}
    for wh in samples:
        n_toks = sum(len(s["words"]) - 2 for s in samples[wh])  # don't count BOS/EOS tokens
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    artifacts_dir = Path(artifacts_dir)
    _log.info("Loading vocabulary from %s", artifacts_dir / "vocab.yml")
    vocab = load((artifacts_dir / "vocab.yml").read_text(encoding="utf8"))
    for name in vocab:
        _log.info(
            "Found %d %s, top 10: %s", len(vocab[name]), name, list(vocab[name])[2:12]
        )  # skip pad and unk tokens

    _log.info("Extending vocab with target words")
    old_n_words = len(vocab["words"])
    vocab.extend(chain(*samples.values()), ["words"])
    _log.info("Found %d words now", len(vocab["words"]))

    _log.info("Loading model metadata from %s", artifacts_dir / "model.yml")
    model = load((artifacts_dir / "model.yml").read_text(encoding="utf8"))

    _log.info("Loading model parameters from %s", artifacts_dir / load_from)
    model.load_state_dict(torch.load(artifacts_dir / load_from, "cpu"))

    if len(vocab["words"]) > old_n_words:
        _log.info("Creating extended word embedding layer")
        if word_emb_path:
            kv = KeyedVectors.load_word2vec_format(word_emb_path)
            assert model.word_emb.embedding_dim == kv.vector_size
        else:
            _log.warning(
                "Word embedding file not specified; any extra target words will be treated as unks"
            )
            kv = None
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

    samples = {wh: list(vocab.stoi(samples[wh])) for wh in samples}
    dev_acc = None
    for wh in ["dev", "test"]:
        _log.info("Evaluating on %s", wh)
        acc = run_eval(model, vocab, samples[wh])
        _log.info("%s_acc: %.1f%%", wh, 100 * acc)
        _run.log_scalar(f"{wh}_acc", acc)
        if wh == "dev":
            dev_acc = acc

    assert dev_acc is not None
    return dev_acc


@ex.automain
def train(
    _log,
    _rnd,
    _run,
    max_length=None,
    load_tags_vocab_from=None,
    artifacts_dir=None,
    device="cpu",
    lr=1e-3,
    eval_on_train=False,
    patience=10,
    batch_size=16,
    max_epoch=10,
):
    """Train a POS tagger."""
    if max_length is None:
        max_length = {}
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)

    samples = {
        wh: list(read_tagging_samples(wh, max_length.get(wh)))
        for wh in ["train", "dev", "test"]
    }
    for wh in samples:
        n_toks = sum(len(s["words"]) - 2 for s in samples[wh])  # don't count BOS/EOS tokens
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    _log.info("Creating vocabulary")
    vocab = Vocab.from_samples(chain(*samples.values()))
    if load_tags_vocab_from:
        path = Path(load_tags_vocab_from)
        _log.info("Loading tags vocab from %s", path)
        vocab["tags"] = load(path.read_text(encoding="utf8"))["tags"]

    _log.info("Vocabulary created")
    for name in vocab:
        _log.info(
            "Found %d %s, top 10: %s", len(vocab[name]), name, list(vocab[name])[2:12]
        )  # skip pad and unk tokens

    if artifacts_dir:
        path = artifacts_dir / "vocab.yml"
        _log.info("Saving vocabulary to %s", path)
        path.write_text(dump(vocab), encoding="utf8")

    model = make_model(vocab)
    model.to(device)

    _log.info("Creating optimizer")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5)

    trainer = Runner()
    EpochTimer().attach_on(trainer)
    ProgressBar(
        stats="stats", total=sum(len(s["words"]) for s in samples["train"]), unit="tok"
    ).attach_on(trainer)

    @trainer.on(Event.BATCH)
    def compute_loss(state):
        batch = state["batch"].to_array()
        words = torch.from_numpy(batch["words"]).to(device)
        tags = torch.from_numpy(batch["tags"]).to(device)
        mask = words != vocab["words"].index(Vocab.PAD_TOKEN)

        model.train()
        scores = model(words, mask)
        bsz, slen = words.shape
        assert scores.shape == (bsz, slen - 1, len(vocab["tags"]), len(vocab["tags"]))
        lengths = mask.long().sum(dim=1)
        mask[torch.arange(bsz).to(mask.device), lengths - 1] = False  # exclude last position
        crf = LinearCRF(scores.contiguous(), mask[:, :-1])  # exclude last position from mask
        loss = -crf.log_probs(tags).sum() / lengths.sum()

        state["loss"] = loss
        state["stats"] = {"loss": loss.item(), "ppl": loss.exp().item()}
        state["n_items"] = lengths.sum().item()

    trainer.on(Event.BATCH, [update_params(opt), log_grads(_run, model), log_stats(_run)])

    @trainer.on(Event.EPOCH_FINISHED)
    def maybe_eval_on_train(state):
        if not eval_on_train:
            return

        _log.info("Evaluating on train")
        acc = run_eval(model, vocab, samples["train"])
        _log.info("train_acc: %.1f%%", 100 * acc)
        _run.log_scalar("train_acc", acc, step=state["n_iters"])

    @trainer.on(Event.EPOCH_FINISHED)
    def eval_on_dev(state):
        _log.info("Evaluating on dev")
        acc = run_eval(model, vocab, samples["dev"])
        _log.info("dev_acc: %.1f%%", 100 * acc)
        _run.log_scalar("dev_acc", acc, step=state["n_iters"])

        scheduler.step(acc)

        if acc > state.get("dev_acc", 0.0):
            state["better"] = True
            _log.info("Found new best result on dev!")
            state["dev_acc"] = acc
            state["dev_epoch"] = state["epoch"]
        else:
            state["better"] = False
            _log.info("Not better, the best so far is epoch %d:", state["dev_epoch"])
            _log.info("dev_acc: %.1f%%", 100 * state["dev_acc"])
            _log.info("test_acc: %.1f%%", 100 * state["test_acc"])

    @trainer.on(Event.EPOCH_FINISHED)
    def maybe_eval_on_test(state):
        if not state["better"]:
            return

        _log.info("Evaluating on test")
        acc = run_eval(model, vocab, samples["test"])
        _log.info("test_acc: %.1f%%", 100 * acc)
        _run.log_scalar("test_acc", acc, step=state["n_iters"])
        state["test_acc"] = acc

    trainer.on(Event.EPOCH_FINISHED, maybe_stop_early(patience=patience))
    if artifacts_dir:
        trainer.on(
            Event.EPOCH_FINISHED,
            save_state_dict("model", model, under=artifacts_dir, when="better"),
        )

    samples = {wh: list(vocab.stoi(samples[wh])) for wh in samples}
    bucket_key = lambda s: (len(s["words"]) - 1) // 10
    trn_iter = ShuffleIterator(
        BucketIterator(samples["train"], bucket_key, batch_size, shuffle_bucket=True, rng=_rnd),
        rng=_rnd,
    )
    _log.info("Starting training")
    try:
        trainer.run(trn_iter, max_epoch)
    except KeyboardInterrupt:
        _log.info("Interrupt detected, training will abort")
    else:
        return trainer.state.get("dev_acc")
