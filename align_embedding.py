#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from pathlib import Path
import os

from fastText_multilingual.fasttext import FastVector
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("xduft-multilingual-alignment-testrun")

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # path to directory containing fasttext *.vec files
    fasttext_dir = "fasttext"
    # comma-separated list of languages to align
    langs = "fa,ar,id,ko,tr,hi,hr,he,bg,it,pt,fr,es,no,da,sv,nl,de,en"
    # output directory
    output_dir = "aligned_fasttext"


@ex.automain
def align(_log, fasttext_dir="fasttext", langs="en,id", output_dir="aligned_fasttext"):
    """Align fasttext embeddings with the method of Smith et al. (2017)."""
    output_dir = Path(output_dir)

    for lang in langs.split(","):
        _log.info("Aligning embedding for %s", lang)
        output_path = Path(output_dir) / f"wiki.multi.{lang}.vec"
        if output_path.exists():
            _log.info("Aligned embedding already exists, skipping")
            continue
        dictionary = FastVector(vector_file=Path(fasttext_dir) / f"wiki.{lang}.vec")
        dictionary.apply_transform(
            str(Path("fastText_multilingual") / "alignment_matrices" / f"{lang}.txt")
        )
        dictionary.export(output_path)
