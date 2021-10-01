#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from itertools import chain
from pathlib import Path
import math
import os
import pickle

from einops import rearrange
from gensim.models.keyedvectors import KeyedVectors
from rnnr import Event, Runner
from rnnr.attachments import EpochTimer, MeanReducer, ProgressBar, SumReducer
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import BucketIterator, ShuffleIterator
from tqdm import tqdm
import torch

from aatrn import compute_ambiguous_tag_pairs_mask
from callbacks import compute_l2_loss, log_grads, log_stats, save_state_dict, update_params
from crf import LinearCRF
from ingredients.corpus import ing as corpus_ing, read_tagging_samples
from serialization import dump, load
from utils import extend_word_embedding

ex = Experiment("xduft-pptx-tagging-testrun", ingredients=[corpus_ing])
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # directory to save finetuning artifacts
    artifacts_dir = ""
    # discard train/dev/test samples with length greater than these numbers
    max_length = {"train": 60}
    # load source models from these directories and parameters {key: (load_from, load_params)}
    load_src = {}
    # whether to treat keys in load_src as lang codes
    src_key_as_lang = False
    # the main source to start finetuning from
    main_src = ""
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # path to word embedding in word2vec format
    word_emb_path = "wiki.en.vec"
    # cumulative prob threshold
    thresh = 0.95
    # batch size
    batch_size = 80
    # learning rate
    lr = 1e-5
    # coefficient of L2 regularization against initial parameters
    l2_coef = 1.0
    # max number of epochs
    max_epoch = 10
    # whether to save the final samples as an artifact
    save_samples = False
    # load samples from this file (*.pkl)
    load_samples_from = ""
    # how to combine PPTX charts
    combine = "union"
    # whether to evaluate on train set at every epoch end
    eval_on_train = False
    # load src2ws from this path
    load_src2ws_from = ""


@ex.named_config
def prag():
    l2_coef = 0.1
    lr = 5.9e-5
    combine = "union"


@ex.named_config
def prag_gmean():
    l2_coef = 4.7e-3
    lr = 2.6e-4
    combine = "geom_mean"


@ex.named_config
def prag_lopw():
    l2_coef = 0.062
    lr = 4.7e-4
    combine = "geom_mean"


@ex.named_config
def testrun():
    seed = 12345
    max_epoch = 2
    corpus = dict(portion=0.05)


@ex.capture
def run_eval(model, vocab, samples, device="cpu", batch_size=16, compute_loss=True):
    runner = Runner()
    SumReducer("corr", value="bcorr").attach_on(runner)
    SumReducer("total", value="btotal").attach_on(runner)
    ProgressBar(total=sum(len(s["words"]) for s in samples), unit="tok", leave=False).attach_on(
        runner
    )
    if compute_loss:
        MeanReducer("mean_pptx_loss", value="pptx_loss").attach_on(runner)

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
        if compute_loss:
            state["scores"] = scores
            state["pptx_mask"] = torch.from_numpy(batch["pptx_mask"]).to(device).bool()

    @runner.on(Event.BATCH)
    def maybe_compute_loss(state):
        if not compute_loss:
            return

        masked_scores = state["scores"].masked_fill(~state["pptx_mask"], -1e9)
        crf = LinearCRF(masked_scores.contiguous())
        crf_z = LinearCRF(state["scores"].contiguous())
        pptx_loss = -crf.log_partitions().sum() + crf_z.log_partitions().sum()

        state["pptx_loss"] = pptx_loss.item()
        state["size"] = masked_scores.size(0)

    with torch.no_grad():
        runner.run(BucketIterator(samples, lambda s: len(s["words"]), batch_size))

    return runner.state["corr"] / runner.state["total"], runner.state.get("mean_pptx_loss")


@ex.automain
def finetune(
    corpus,
    _log,
    _run,
    _rnd,
    max_length=None,
    load_src=None,
    main_src=None,
    load_src2ws_from=None,
    artifacts_dir=None,
    load_samples_from=None,
    word_emb_path="wiki.id.vec",
    src_key_as_lang=False,
    device="cpu",
    combine="union",
    thresh=0.95,
    batch_size=16,
    save_samples=False,
    lr=1e-5,
    l2_coef=1.0,
    eval_on_train=False,
    max_epoch=10,
):
    """Finetune/adapt a trained tagger with PPTX."""
    if max_length is None:
        max_length = {}
    if load_src is None:
        load_src = {"src": ("artifacts", "model.pth")}
        main_src = "src"
    elif main_src not in load_src:
        raise ValueError(f"{main_src} not found in load_src")

    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)

    if load_samples_from:
        _log.info("Loading samples from %s", load_samples_from)
        with open(load_samples_from, "rb") as f:
            samples = pickle.load(f)
    else:
        samples = {
            wh: list(read_tagging_samples(wh, max_length.get(wh)))
            for wh in ["train", "dev", "test"]
        }

    for wh in samples:
        n_toks = sum(len(s["words"]) - 2 for s in samples[wh])  # don't count BOS/EOS tokens
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    kv = KeyedVectors.load_word2vec_format(word_emb_path)

    if load_samples_from:
        _log.info("Skipping non-main src because samples are processed and loaded")
        srcs = []
    else:
        srcs = [src for src in load_src if src != main_src]
        if src_key_as_lang and corpus["lang"] in srcs:
            _log.info("Removing %s from src parsers because it's the tgt", corpus["lang"])
            srcs.remove(corpus["lang"])
    srcs.append(main_src)

    if load_src2ws_from:
        _log.info("Loading src weights from %s", load_src2ws_from)
        src2ws = load(Path(load_src2ws_from).read_text(encoding="utf8"))
        if any(src not in src2ws for src in srcs):
            _log.warning("Some srcs have no weights, will be set to zero")
        if any(src not in srcs for src in src2ws):
            _log.warning("Too many srcs in src2ws, weights won't sum to one")
        _log.info("Sources: %s", list(srcs))
        _log.info("Weights: %s", [src2ws[src] for src in srcs])
    else:
        src2ws = {src: 1 / len(srcs) for src in srcs}

    for src_i, src in enumerate(srcs):
        _log.info("Processing src %s [%d/%d]", src, src_i + 1, len(srcs))
        load_from, load_params = load_src[src]
        path = Path(load_from) / "vocab.yml"
        _log.info("Loading %s vocabulary from %s", src, path)
        vocab = load(path.read_text(encoding="utf8"))
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
            if load_samples_from:
                assert all("pptx_mask" in s for s in samples[wh])
                continue

            for i, s in enumerate(samples_[wh]):
                s["_id"] = i

            runner = Runner()
            runner.state["_ids"] = []
            if combine == "geom_mean":
                runner.state.update({"log_marginals": [], "pred_tags": []})
            elif combine == "union":
                runner.state["pptx_masks"] = []
            else:
                raise ValueError(f"unknown value for combine: {combine}")

            @runner.on(Event.BATCH)
            def compute_pptx_ambiguous_tag_pairs_mask(state):
                batch = state["batch"].to_array()
                words = torch.from_numpy(batch["words"]).to(device)
                mask = words != vocab["words"].index(vocab.PAD_TOKEN)
                assert mask.all(), "must not have masking at test time"

                model.eval()
                scores = model(words)
                if combine == "geom_mean":
                    crf = LinearCRF(scores)
                    state["log_marginals"].extend((crf.marginals() + 1e-9).log())
                    state["pred_tags"].extend(crf.argmax())
                else:
                    pptx_mask = compute_ambiguous_tag_pairs_mask(scores, thresh)
                    state["pptx_masks"].extend(pptx_mask)
                state["_ids"].extend(batch["_id"].tolist())
                state["n_items"] = words.numel()

            n_toks = sum(len(s["words"]) for s in samples_[wh])
            ProgressBar(total=n_toks, unit="tok").attach_on(runner)

            if combine == "geom_mean":
                _log.info(
                    "Computing marginals and best tags for %s set with source %s", wh, src
                )
            else:
                _log.info(
                    "Computing PPTX ambiguous tag pairs mask for %s set with source %s", wh, src
                )
            with torch.no_grad():
                runner.run(BucketIterator(samples_[wh], lambda s: len(s["words"]), batch_size))

            for x in "pptx_masks log_marginals pred_tags".split():
                assert x not in runner.state or len(runner.state[x]) == len(samples_[wh])
            assert len(runner.state["_ids"]) == len(samples_[wh])
            if "pptx_masks" in runner.state:
                for i, pptx_mask in zip(runner.state["_ids"], runner.state["pptx_masks"]):
                    samples_[wh][i]["pptx_mask"] = pptx_mask.tolist()
            else:
                zips = [runner.state[x] for x in "log_marginals pred_tags".split()]
                for i, lms, pts in zip(runner.state["_ids"], *zips):
                    samples_[wh][i]["log_marginals"] = lms
                    samples_[wh][i]["pred_tags"] = pts

            if combine != "geom_mean":
                _log.info("Computing median number of tag sequences in chart on %s set", wh)
                log_ntags = []
                for s in tqdm(samples_[wh], unit="sample", leave=False):
                    mask = torch.tensor(s["pptx_mask"]).unsqueeze(0)
                    cnt_scores = torch.zeros_like(mask).float().masked_fill(~mask, -1e9)
                    log_ntags.append(LinearCRF(cnt_scores).log_partitions().item())
                log_ntags.sort()
                mid = len(log_ntags) // 2
                if len(log_ntags) % 2:
                    log_med = log_ntags[mid]
                else:
                    max_ = max(log_ntags[mid - 1], log_ntags[mid])
                    log_med = (
                        max_
                        + math.log(
                            math.exp(log_ntags[mid - 1] - max_)
                            + math.exp(log_ntags[mid] - max_)
                        )
                        - math.log(2)
                    )
                _log.info("Median number of tag sequences in chart: %.1e", math.exp(log_med))

            assert len(samples_[wh]) == len(samples[wh])
            if combine == "geom_mean":
                _log.info("Combining the marginals")
                for i in tqdm(range(len(samples_[wh])), unit="sample", leave=False):
                    lms = samples[wh][i].get("log_marginals", 0)
                    pts = samples[wh][i].get("pred_tags", [])
                    lms = torch.tensor(lms, device=device) + src2ws[src] * samples_[wh][i]["log_marginals"]
                    pts.append(samples_[wh][i]["pred_tags"].tolist())
                    samples[wh][i]["log_marginals"] = lms.tolist()
                    samples[wh][i]["pred_tags"] = pts
            else:
                _log.info("Combining the ambiguous tag pairs mask")
                for i in tqdm(range(len(samples_[wh])), unit="sample", leave=False):
                    pptx_mask = torch.tensor(samples_[wh][i]["pptx_mask"])
                    assert pptx_mask.dim() == 3
                    if "pptx_mask" in samples[wh][i]:
                        old_mask = torch.tensor(samples[wh][i]["pptx_mask"])
                    else:
                        old_mask = torch.zeros(1, 1, 1).bool()
                    samples[wh][i]["pptx_mask"] = (old_mask | pptx_mask).tolist()

    if not load_samples_from and combine == "geom_mean":
        for wh in ["train", "dev"]:
            _log.info("Computing the ambiguous tag pairs mask on %s set", wh)
            for s in tqdm(samples[wh], unit="sample", leave=False):
                lms = torch.tensor(s["log_marginals"])
                assert lms.dim() == 3 and lms.size(1) == lms.size(2)
                # Renormalise the marginal probabilities
                lms = rearrange(lms, "slen nntags ntags -> slen (nntags ntags)")
                lms = lms.log_softmax(dim=1)
                lms = rearrange(
                    lms, "slen (nntags ntags) -> slen nntags ntags", ntags=len(vocab["tags"])
                )
                lms = lms.unsqueeze(0)
                mask = compute_ambiguous_tag_pairs_mask(lms, thresh, is_log_marginals=True)
                assert mask.shape == lms.shape
                mask = mask.squeeze(0)
                for pts in s["pred_tags"]:
                    for j in range(1, len(pts)):
                        mask[j - 1, pts[j], pts[j - 1]] = True
                s["pptx_mask"] = mask.tolist()
                for k in "log_marginals pred_tags".split():
                    s.pop(k)

    assert src == main_src
    _log.info("Main source is %s", src)

    if artifacts_dir:
        path = artifacts_dir / "vocab.yml"
        _log.info("Saving vocabulary to %s", path)
        path.write_text(dump(vocab), encoding="utf8")

        path = artifacts_dir / "model.yml"
        _log.info("Saving model metadata to %s", path)
        path.write_text(dump(model), encoding="utf8")

    if artifacts_dir and save_samples:
        path = artifacts_dir / "samples.pkl"
        _log.info("Saving samples to %s", path)
        with open(path, "wb") as f:
            pickle.dump(samples, f)

    samples = {wh: list(vocab.stoi(samples[wh])) for wh in samples}

    for wh in ["train", "dev"]:
        _log.info("Computing median number of tag sequences in chart on %s set", wh)
        log_ntags = []
        for s in tqdm(samples[wh], unit="sample", leave=False):
            mask = torch.tensor(s["pptx_mask"]).unsqueeze(0)
            cnt_scores = torch.zeros_like(mask).float().masked_fill(~mask, -1e9)
            log_ntags.append(LinearCRF(cnt_scores).log_partitions().item())
        log_ntags.sort()
        mid = len(log_ntags) // 2
        if len(log_ntags) % 2:
            log_med = log_ntags[mid]
        else:
            max_ = max(log_ntags[mid - 1], log_ntags[mid])
            log_med = (
                max_
                + math.log(
                    math.exp(log_ntags[mid - 1] - max_) + math.exp(log_ntags[mid] - max_)
                )
                - math.log(2)
            )
        _log.info("Median number of tag sequences in chart: %.1e", math.exp(log_med))

    _log.info("Creating optimizer")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    finetuner = Runner()
    EpochTimer().attach_on(finetuner)
    ProgressBar(
        stats="stats", total=sum(len(s["words"]) for s in samples["train"]), unit="tok"
    ).attach_on(finetuner)

    origin_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    finetuner.on(Event.BATCH, compute_l2_loss(model, origin_params))

    @finetuner.on(Event.BATCH)
    def compute_loss(state):
        batch = state["batch"].to_array()
        words = torch.from_numpy(batch["words"]).to(device)
        pptx_mask = torch.from_numpy(batch["pptx_mask"]).to(device).bool()
        mask = words != vocab["words"].index(vocab.PAD_TOKEN)

        model.train()
        scores = model(words, mask)
        bsz, slen = words.shape
        assert scores.shape == (bsz, slen - 1, len(vocab["tags"]), len(vocab["tags"]))
        assert pptx_mask.shape == scores.shape
        masked_scores = scores.masked_fill(~pptx_mask, -1e9)
        lengths = mask.long().sum(dim=1)
        mask[torch.arange(bsz).to(mask.device), lengths - 1] = False  # exclude last position
        crf = LinearCRF(
            masked_scores.contiguous(), mask[:, :-1]
        )  # exclude last position from mask
        crf_z = LinearCRF(scores.contiguous(), mask[:, :-1])  # exclude last position from mask
        pptx_loss = (-crf.log_partitions().sum() + crf_z.log_partitions().sum()) / bsz
        loss = pptx_loss + l2_coef * state["l2_loss"]

        state["loss"] = loss
        state["stats"] = {
            "pptx_loss": pptx_loss.item(),
            "l2_loss": state["l2_loss"].item(),
        }
        state["extra_stats"] = {"loss": loss.item()}
        state["n_items"] = lengths.sum().item()

    finetuner.on(Event.BATCH, [update_params(opt), log_grads(_run, model), log_stats(_run)])

    @finetuner.on(Event.EPOCH_FINISHED)
    def maybe_eval_on_train(state):
        if not eval_on_train:
            return

        _log.info("Evaluating on train")
        acc, pptx_loss = run_eval(model, vocab, samples["train"])
        _log.info("train_acc: %.1f%%", 100 * acc)
        _run.log_scalar("train_acc", acc, step=state["n_iters"])
        assert pptx_loss is not None
        _log.info("train_pptx_loss: %.4f", pptx_loss)
        _run.log_scalar("train_pptx_loss", pptx_loss, step=state["n_iters"])

    @finetuner.on(Event.EPOCH_FINISHED)
    def eval_on_dev(state):
        _log.info("Evaluating on dev")
        acc, pptx_loss = run_eval(model, vocab, samples["dev"])
        _log.info("dev_acc: %.1f%%", 100 * acc)
        _run.log_scalar("dev_acc", acc, step=state["n_iters"])
        assert pptx_loss is not None
        _log.info("dev_pptx_loss: %.4f", pptx_loss)
        _run.log_scalar("dev_pptx_loss", pptx_loss, step=state["n_iters"])
        state["dev_acc"] = acc

    @finetuner.on(Event.EPOCH_FINISHED)
    def maybe_eval_on_test(state):
        if state["epoch"] != max_epoch:
            return

        _log.info("Evaluating on test")
        acc, _ = run_eval(model, vocab, samples["test"], compute_loss=False)
        _log.info("test_acc: %.1f%%", 100 * acc)
        _run.log_scalar("test_acc", acc, step=state["n_iters"])

    if artifacts_dir:
        finetuner.on(Event.EPOCH_FINISHED, save_state_dict("model", model, under=artifacts_dir))

    samples["train"].sort(key=lambda s: len(s["words"]))
    trn_iter = ShuffleIterator(
        BucketIterator(samples["train"], lambda s: len(s["words"]) // 10, batch_size), rng=_rnd
    )
    _log.info("Starting finetuning")
    try:
        finetuner.run(trn_iter, max_epoch)
    except KeyboardInterrupt:
        _log.info("Interrupt detected, training will abort")
    else:
        return finetuner.state.get("dev_acc")
