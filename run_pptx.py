#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from itertools import chain
from pathlib import Path
import os
import pickle

from einops import rearrange
from gensim.models.keyedvectors import KeyedVectors
from rnnr import Event, Runner
from rnnr.attachments import EpochTimer, MeanReducer, ProgressBar, SumReducer
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import ShuffleIterator
from tqdm import tqdm
import text2array
import torch

from aatrn import compute_aatrn_loss, compute_ambiguous_arcs_mask
from callbacks import (
    batch2tensors,
    compute_l2_loss,
    compute_total_arc_type_scores,
    evaluate_batch,
    get_n_items,
    log_grads,
    log_stats,
    predict_batch,
    save_state_dict,
    set_train_mode,
    update_params,
)
from crf import DepTreeCRF
from ingredients.corpus import ing as corpus_ing, read_samples
from serialization import dump, load
from utils import extend_word_embedding, print_accs, report_median_ntrees

ex = Experiment("xduft-pptx-testrun", ingredients=[corpus_ing])
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # directory to save finetuning artifacts
    artifacts_dir = "ft_artifacts"
    # whether to overwrite existing artifacts directory
    overwrite = False
    # discard train/dev/test samples with length greater than these numbers
    max_length = {"train": 30, "dev": 150, "test": 150}
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
    # whether to freeze word and tag embedding
    freeze = False
    # cumulative prob threshold
    thresh = 0.95
    # whether to operate in the space of projective trees
    projective = False
    # whether to consider multi-root trees (otherwise only single-root trees)
    multiroot = False
    # batch size
    batch_size = 80
    # learning rate
    lr = 1e-5
    # coefficient of L2 regularization against initial parameters
    l2_coef = 1.0
    # max number of epochs
    max_epoch = 5
    # whether to save the final samples as an artifact
    save_samples = False
    # load samples from this file (*.pkl)
    load_samples_from = ""
    # how to combine PPTX charts
    combine = "union"
    # load src2ws from this path
    load_src2ws_from = ""


@ex.named_config
def prag():
    l2_coef = 2.8e-5
    lr = 8.5e-5
    combine = "union"


@ex.named_config
def prag_gmean():
    l2_coef = 1.6e-4
    lr = 9.4e-5
    combine = "geom_mean"


@ex.named_config
def prag_lopw():
    l2_coef = 5.1e-4
    lr = 9.1e-5
    combine = "geom_mean"


@ex.named_config
def testrun():
    seed = 12345
    max_epoch = 2
    corpus = dict(portion=0.05)


class BucketIterator(text2array.BucketIterator):
    def __iter__(self):
        for ss in self._buckets:
            if self._shuf and len(ss) > 1:
                ss = ShuffleIterator(ss, key=lambda s: len(s["words"]), rng=self._rng)
            yield from text2array.BatchIterator(ss, self._bsz)


@ex.capture
def run_eval(
    model,
    vocab,
    samples,
    compute_loss=True,
    device="cpu",
    projective=False,
    multiroot=True,
    batch_size=32,
):
    runner = Runner()
    runner.on(
        Event.BATCH,
        [
            batch2tensors(device, vocab),
            set_train_mode(model, training=False),
            compute_total_arc_type_scores(model, vocab),
            predict_batch(projective, multiroot),
            evaluate_batch(),
            get_n_items(),
        ],
    )

    @runner.on(Event.BATCH)
    def maybe_compute_loss(state):
        if not compute_loss:
            return

        pptx_loss = compute_aatrn_loss(
            state["total_arc_type_scores"],
            state["batch"]["pptx_mask"].bool(),
            projective=projective,
            multiroot=multiroot,
        )
        state["pptx_loss"] = pptx_loss.item()
        state["size"] = state["batch"]["words"].size(0)

    n_tokens = sum(len(s["words"]) for s in samples)
    ProgressBar(leave=False, total=n_tokens, unit="tok").attach_on(runner)
    SumReducer("counts").attach_on(runner)
    if compute_loss:
        MeanReducer("mean_pptx_loss", value="pptx_loss").attach_on(runner)

    with torch.no_grad():
        runner.run(BucketIterator(samples, lambda s: len(s["words"]), batch_size))

    return runner.state


@ex.command(unobserved=True)
def compute_charts(
    load_src,
    artifacts_dir,
    word_emb_path,
    corpus,
    _log,
    overwrite=False,
    max_length=None,
    main_src=None,
    src_key_as_lang=False,
    device="cpu",
    combine="union",
    projective=False,
    multiroot=False,
    thresh=0.95,
    batch_size=16,
):
    """Compute and save charts."""
    if max_length is None:
        max_length = {}
    if main_src not in load_src:
        raise ValueError(f"{main_src} not found in load_src")

    artifacts_dir = Path(artifacts_dir)
    _log.info("Creating artifacts directory %s", artifacts_dir)
    artifacts_dir.mkdir(exist_ok=overwrite)

    samples = {
        wh: list(read_samples(which=wh, max_length=max_length.get(wh)))
        for wh in ["train", "dev", "test"]
    }
    for wh in samples:
        n_toks = sum(len(s["words"]) for s in samples[wh])
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    kv = KeyedVectors.load_word2vec_format(word_emb_path)

    srcs = [src for src in load_src if src != main_src]
    if src_key_as_lang and corpus["lang"] in srcs:
        _log.info("Removing %s from src parsers because it's the tgt", corpus["lang"])
        srcs.remove(corpus["lang"])
    srcs.append(main_src)

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
            for i, s in enumerate(samples_[wh]):
                s["_id"] = i

            runner = Runner()
            runner.state["_ids"] = []
            if combine == "geom_mean":
                runner.state.update({"log_marginals": [], "pred_heads": [], "pred_types": []})
            elif combine == "union":
                runner.state["pptx_masks"] = []
            else:
                raise ValueError(f"unknown value for combine: {combine}")
            runner.on(
                Event.BATCH,
                [
                    batch2tensors(device, vocab),
                    set_train_mode(model, training=False),
                    compute_total_arc_type_scores(model, vocab),
                ],
            )

            @runner.on(Event.BATCH)
            def compute_pptx_ambiguous_arcs_mask(state):
                assert state["batch"]["mask"].all()
                scores = state["total_arc_type_scores"]
                if combine == "geom_mean":
                    crf = DepTreeCRF(scores, projective=projective, multiroot=multiroot)
                    pred_heads, pred_types = crf.argmax()
                    state["log_marginals"].extend((crf.marginals() + 1e-9).log())
                    state["pred_heads"].extend(pred_heads)
                    state["pred_types"].extend(pred_types)
                else:
                    pptx_mask = compute_ambiguous_arcs_mask(
                        scores, thresh, projective, multiroot
                    )
                    state["pptx_masks"].extend(pptx_mask)
                state["_ids"].extend(state["batch"]["_id"].tolist())
                state["n_items"] = state["batch"]["words"].numel()

            n_toks = sum(len(s["words"]) for s in samples_[wh])
            ProgressBar(total=n_toks, unit="tok").attach_on(runner)

            if combine == "geom_mean":
                _log.info(
                    "Computing marginals and best trees for %s set with source %s", wh, src
                )
            else:
                _log.info(
                    "Computing PPTX ambiguous arcs mask for %s set with source %s", wh, src
                )
            with torch.no_grad():
                runner.run(BucketIterator(samples_[wh], lambda s: len(s["words"]), batch_size))

            for x in "pptx_masks log_marginals pred_heads pred_types".split():
                assert x not in runner.state or len(runner.state[x]) == len(samples_[wh])
            assert len(runner.state["_ids"]) == len(samples_[wh])
            if "pptx_masks" in runner.state:
                for i, pptx_mask in zip(runner.state["_ids"], runner.state["pptx_masks"]):
                    samples_[wh][i]["pptx_mask"] = pptx_mask.tolist()
            else:
                zips = [runner.state[x] for x in "log_marginals pred_heads pred_types".split()]
                for i, lms, phs, pys in zip(runner.state["_ids"], *zips):
                    samples_[wh][i]["log_marginals"] = lms
                    samples_[wh][i]["pred_heads"] = phs
                    samples_[wh][i]["pred_types"] = pys

            if combine != "geom_mean":
                _log.info("Computing median number of trees in chart on %s set", wh)
                report_median_ntrees(
                    samples_[wh], "pptx_mask", batch_size, projective, multiroot
                )

            assert len(samples_[wh]) == len(samples[wh])
            if combine == "geom_mean":
                _log.info("Combining the marginals")
                for i in tqdm(range(len(samples_[wh])), unit="sample", leave=False):
                    lms = samples[wh][i].get("log_marginals", 0)
                    phs = samples[wh][i].get("pred_heads", [])
                    pys = samples[wh][i].get("pred_types", [])
                    lms = torch.tensor(lms, device=device) + samples_[wh][i]["log_marginals"]
                    phs.append(samples_[wh][i]["pred_heads"].tolist())
                    pys.append(samples_[wh][i]["pred_types"].tolist())
                    samples[wh][i]["log_marginals"] = lms.tolist()
                    samples[wh][i]["pred_heads"] = phs
                    samples[wh][i]["pred_types"] = pys
            else:
                _log.info("Combining the ambiguous arcs mask")
                for i in tqdm(range(len(samples_[wh])), unit="sample", leave=False):
                    pptx_mask = torch.tensor(samples_[wh][i]["pptx_mask"])
                    assert pptx_mask.dim() == 3
                    if "pptx_mask" in samples[wh][i]:
                        old_mask = torch.tensor(samples[wh][i]["pptx_mask"])
                    else:
                        old_mask = torch.zeros(1, 1, 1).bool()
                    samples[wh][i]["pptx_mask"] = (old_mask | pptx_mask).tolist()

    if combine == "geom_mean":
        for wh in ["train", "dev"]:
            _log.info("Computing the ambiguous arcs mask on %s set", wh)
            for s in tqdm(samples[wh], unit="sample", leave=False):
                lms = torch.tensor(s["log_marginals"])
                lms /= len(srcs)
                assert lms.dim() == 3 and lms.size(0) == lms.size(1)
                # Renormalise the marginal probabilities
                lms = rearrange(lms, "hlen dlen ntypes -> dlen (hlen ntypes)")
                lms = lms.log_softmax(dim=1)
                lms = rearrange(lms, "dlen (hlen ntypes) -> hlen dlen ntypes", hlen=lms.size(0))
                lms = lms.unsqueeze(0)
                mask = compute_ambiguous_arcs_mask(
                    lms, thresh, projective, multiroot, is_log_marginals=True
                )
                assert mask.shape == (1, len(s["words"]), len(s["words"]), len(vocab["types"]))
                mask = mask.squeeze(0)
                for phs, pys in zip(s["pred_heads"], s["pred_types"]):
                    for d, (h, y) in enumerate(zip(phs, pys)):
                        mask[h, d, y] = True
                s["pptx_mask"] = mask.tolist()
                for k in "log_marginals pred_heads pred_types".split():
                    s.pop(k)

    path = artifacts_dir / "samples.pkl"
    _log.info("Saving samples to %s", path)
    with open(path, "wb") as f:
        pickle.dump(samples, f)


@ex.automain
def finetune(
    corpus,
    _log,
    _run,
    _rnd,
    max_length=None,
    artifacts_dir="ft_artifacts",
    load_samples_from=None,
    overwrite=False,
    load_src=None,
    src_key_as_lang=False,
    main_src=None,
    load_src2ws_from=None,
    device="cpu",
    word_emb_path="wiki.id.vec",
    freeze=False,
    thresh=0.95,
    projective=False,
    multiroot=True,
    batch_size=32,
    save_samples=False,
    lr=1e-5,
    l2_coef=1.0,
    max_epoch=5,
    combine="union",
):
    """Finetune a trained model with PPTX of Kurniawan et al. (2021)."""
    if max_length is None:
        max_length = {}
    if load_src is None:
        load_src = {"src": ("artifacts", "model.pth")}
        main_src = "src"
    elif main_src not in load_src:
        raise ValueError(f"{main_src} not found in load_src")

    artifacts_dir = Path(artifacts_dir)
    _log.info("Creating artifacts directory %s", artifacts_dir)
    artifacts_dir.mkdir(exist_ok=overwrite)

    if load_samples_from:
        _log.info("Loading samples from %s", load_samples_from)
        with open(load_samples_from, "rb") as f:
            samples = pickle.load(f)
    else:
        samples = {
            wh: list(read_samples(which=wh, max_length=max_length.get(wh)))
            for wh in ["train", "dev", "test"]
        }

    for wh in samples:
        n_toks = sum(len(s["words"]) for s in samples[wh])
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
                runner.state.update({"log_marginals": [], "pred_heads": [], "pred_types": []})
            elif combine == "union":
                runner.state["pptx_masks"] = []
            else:
                raise ValueError(f"unknown value for combine: {combine}")
            runner.on(
                Event.BATCH,
                [
                    batch2tensors(device, vocab),
                    set_train_mode(model, training=False),
                    compute_total_arc_type_scores(model, vocab),
                ],
            )

            @runner.on(Event.BATCH)
            def compute_pptx_ambiguous_arcs_mask(state):
                assert state["batch"]["mask"].all()
                scores = state["total_arc_type_scores"]
                if combine == "geom_mean":
                    crf = DepTreeCRF(scores, projective=projective, multiroot=multiroot)
                    pred_heads, pred_types = crf.argmax()
                    state["log_marginals"].extend((crf.marginals() + 1e-9).log())
                    state["pred_heads"].extend(pred_heads)
                    state["pred_types"].extend(pred_types)
                else:
                    pptx_mask = compute_ambiguous_arcs_mask(
                        scores, thresh, projective, multiroot
                    )
                    state["pptx_masks"].extend(pptx_mask)
                state["_ids"].extend(state["batch"]["_id"].tolist())
                state["n_items"] = state["batch"]["words"].numel()

            n_toks = sum(len(s["words"]) for s in samples_[wh])
            ProgressBar(total=n_toks, unit="tok").attach_on(runner)

            if combine == "geom_mean":
                _log.info(
                    "Computing marginals and best trees for %s set with source %s", wh, src
                )
            else:
                _log.info(
                    "Computing PPTX ambiguous arcs mask for %s set with source %s", wh, src
                )
            with torch.no_grad():
                runner.run(BucketIterator(samples_[wh], lambda s: len(s["words"]), batch_size))

            for x in "pptx_masks log_marginals pred_heads pred_types".split():
                assert x not in runner.state or len(runner.state[x]) == len(samples_[wh])
            assert len(runner.state["_ids"]) == len(samples_[wh])
            if "pptx_masks" in runner.state:
                for i, pptx_mask in zip(runner.state["_ids"], runner.state["pptx_masks"]):
                    samples_[wh][i]["pptx_mask"] = pptx_mask.tolist()
            else:
                zips = [runner.state[x] for x in "log_marginals pred_heads pred_types".split()]
                for i, lms, phs, pys in zip(runner.state["_ids"], *zips):
                    samples_[wh][i]["log_marginals"] = lms
                    samples_[wh][i]["pred_heads"] = phs
                    samples_[wh][i]["pred_types"] = pys

            if combine != "geom_mean":
                _log.info("Computing median number of trees in chart on %s set", wh)
                report_median_ntrees(
                    samples_[wh], "pptx_mask", batch_size, projective, multiroot
                )

            assert len(samples_[wh]) == len(samples[wh])
            if combine == "geom_mean":
                _log.info("Combining the marginals")
                for i in tqdm(range(len(samples_[wh])), unit="sample", leave=False):
                    lms = samples[wh][i].get("log_marginals", 0)
                    phs = samples[wh][i].get("pred_heads", [])
                    pys = samples[wh][i].get("pred_types", [])
                    lms = (
                        torch.tensor(lms, device=device)
                        + src2ws[src] * samples_[wh][i]["log_marginals"]
                    )
                    phs.append(samples_[wh][i]["pred_heads"].tolist())
                    pys.append(samples_[wh][i]["pred_types"].tolist())
                    samples[wh][i]["log_marginals"] = lms.tolist()
                    samples[wh][i]["pred_heads"] = phs
                    samples[wh][i]["pred_types"] = pys
            else:
                _log.info("Combining the ambiguous arcs mask")
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
            _log.info("Computing the ambiguous arcs mask on %s set", wh)
            for s in tqdm(samples[wh], unit="sample", leave=False):
                lms = torch.tensor(s["log_marginals"])
                assert lms.dim() == 3 and lms.size(0) == lms.size(1)
                # Renormalise the marginal probabilities
                lms = rearrange(lms, "hlen dlen ntypes -> dlen (hlen ntypes)")
                lms = lms.log_softmax(dim=1)
                lms = rearrange(lms, "dlen (hlen ntypes) -> hlen dlen ntypes", hlen=lms.size(0))
                lms = lms.unsqueeze(0)
                mask = compute_ambiguous_arcs_mask(
                    lms, thresh, projective, multiroot, is_log_marginals=True
                )
                assert mask.shape == (1, len(s["words"]), len(s["words"]), len(vocab["types"]))
                mask = mask.squeeze(0)
                for phs, pys in zip(s["pred_heads"], s["pred_types"]):
                    for d, (h, y) in enumerate(zip(phs, pys)):
                        mask[h, d, y] = True
                s["pptx_mask"] = mask.tolist()
                for k in "log_marginals pred_heads pred_types".split():
                    s.pop(k)

    assert src == main_src
    _log.info("Main source is %s", src)

    path = artifacts_dir / "vocab.yml"
    _log.info("Saving vocabulary to %s", path)
    path.write_text(dump(vocab), encoding="utf8")

    path = artifacts_dir / "model.yml"
    _log.info("Saving model metadata to %s", path)
    path.write_text(dump(model), encoding="utf8")

    if save_samples:
        path = artifacts_dir / "samples.pkl"
        _log.info("Saving samples to %s", path)
        with open(path, "wb") as f:
            pickle.dump(samples, f)

    samples = {wh: list(vocab.stoi(samples[wh])) for wh in samples}

    for wh in ["train", "dev"]:
        _log.info("Computing median number of trees in chart on %s set", wh)
        report_median_ntrees(samples[wh], "pptx_mask", batch_size, projective, multiroot)

        _log.info("Computing coverage of gold trees on %s set", wh)
        n_cov = 0
        for s in samples[wh]:
            assert len(s["heads"]) == len(s["types"])
            for d, (h, y) in enumerate(zip(s["heads"], s["types"])):
                assert d != 0 or h == 0, "root's head must be itself"
                if d != 0 and not s["pptx_mask"][h][d][y]:
                    break
            else:
                n_cov += 1
        _log.info("Gold tree coverage is %.1f%%", 100.0 * n_cov / len(samples[wh]))

    model.word_emb.requires_grad_(not freeze)
    model.tag_emb.requires_grad_(not freeze)

    _log.info("Creating optimizer")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    finetuner = Runner()
    finetuner.state["dev_accs"] = {}
    origin_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    finetuner.on(
        Event.BATCH,
        [
            batch2tensors(device, vocab),
            set_train_mode(model),
            compute_l2_loss(model, origin_params),
            compute_total_arc_type_scores(model, vocab),
        ],
    )

    @finetuner.on(Event.BATCH)
    def compute_loss(state):
        mask = state["batch"]["mask"]
        pptx_mask = state["batch"]["pptx_mask"].bool()
        scores = state["total_arc_type_scores"]

        pptx_loss = compute_aatrn_loss(scores, pptx_mask, mask, projective, multiroot)
        pptx_loss /= mask.size(0)
        loss = pptx_loss + l2_coef * state["l2_loss"]

        state["loss"] = loss
        state["stats"] = {
            "pptx_loss": pptx_loss.item(),
            "l2_loss": state["l2_loss"].item(),
        }
        state["extra_stats"] = {"loss": loss.item()}
        state["n_items"] = mask.long().sum().item()

    finetuner.on(Event.BATCH, [update_params(opt), log_grads(_run, model), log_stats(_run)])

    @finetuner.on(Event.EPOCH_FINISHED)
    def eval_on_dev(state):
        _log.info("Evaluating on dev")
        eval_state = run_eval(model, vocab, samples["dev"])
        accs = eval_state["counts"].accs
        print_accs(accs, run=_run, step=state["n_iters"])

        pptx_loss = eval_state["mean_pptx_loss"]
        _log.info("dev_pptx_loss: %.4f", pptx_loss)
        _run.log_scalar("dev_pptx_loss", pptx_loss, step=state["n_iters"])

        state["dev_accs"] = accs

    @finetuner.on(Event.EPOCH_FINISHED)
    def maybe_eval_on_test(state):
        if state["epoch"] != max_epoch:
            return

        _log.info("Evaluating on test")
        eval_state = run_eval(model, vocab, samples["test"], compute_loss=False)
        print_accs(eval_state["counts"].accs, on="test", run=_run, step=state["n_iters"])

    finetuner.on(Event.EPOCH_FINISHED, save_state_dict("model", model, under=artifacts_dir))

    EpochTimer().attach_on(finetuner)
    n_tokens = sum(len(s["words"]) for s in samples["train"])
    ProgressBar(stats="stats", total=n_tokens, unit="tok").attach_on(finetuner)

    bucket_key = lambda s: (len(s["words"]) - 1) // 10
    trn_iter = ShuffleIterator(
        BucketIterator(samples["train"], bucket_key, batch_size, shuffle_bucket=True, rng=_rnd),
        rng=_rnd,
    )
    _log.info("Starting finetuning")
    try:
        finetuner.run(trn_iter, max_epoch)
    except KeyboardInterrupt:
        _log.info("Interrupt detected, training will abort")
    else:
        return finetuner.state["dev_accs"].get("las_nopunct")
