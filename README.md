# Unsupervised Cross-Lingual Transfer of Structured Predictors without Source Data

This repository contains the code for our paper: https://arxiv.org/abs/2110.03866v1. If
you use this code, please cite:

```
@article{kurniawan2021b,
  title = {Unsupervised {{Cross}}-{{Lingual Transfer}} of {{Structured Predictors}} without {{Source Data}}},
  author = {Kurniawan, Kemal and Frermann, Lea and Schulz, Philip and Cohn, Trevor},
  year = {2021},
  month = oct,
  url = {https://arxiv.org/abs/2110.03866v1},
}
```

## Installing requirements

Using conda package manager:

    conda env create -n {env} -f environment.yml

and replace `{env}` with the desired environment name. This command creates the enviroment
and install all the dependencies. Once created, activate the environment. The command above also
installs the CPU version of PyTorch. If you need the GPU version, follow the corresponding
PyTorch installation docs afterwards. If you're using other package manager (e.g., pip),
you can look at the `environment.yml` file to see what the requirements are.

## Downloading dataset

Download UD treebanks v2.2 from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2837

## Preparing word embeddings

Next, download FastText's Wiki word embeddings from
[this page](https://fasttext.cc/docs/en/pretrained-vectors.html). You need to download the
text format (`.vec`). Suppose you put the word embedding files in `fasttext` directory. Next,
clone https://github.com/babylonhealth/fastText_multilingual under this directory. Then, perform
the word embedding alignment to get the multilingual embeddings:

    ./align_embedding.py

Lastly, minimise the word embedding files so they contain only words that actually occur in
the UD data. Assuming the UD data is stored in ud-treebanks-v2.2, then run

    ./minimize_vectors_file.py with vectors_path=aligned_fasttext/wiki.multi.{lang}.vec \
      output_path=aligned_fasttext/wiki.multi.min.{lang}.vec corpus.lang={lang}

The command above minimises the word vector file for language `{lang}`. You can set it to language
codes mentioned in the paper, e.g., ar for Arabic, es for Spanish, etc. hereinafter.

## Training source models

### Parsers

To train the English parser, run:

    ./run_parser.py with word_emb_path=aligned_fasttext/wiki.multi.min.en.vec

This command saves the model under `artifacts` directory. To train the other parsers, run:

    ./run_parser.py with artifacts_dir=artifacts_{lang} corpus.lang={lang} \
      word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec \
      load_types_vocab_from=artifacts/vocab.yml

Once finished, to make it easier for running further experiments, create a JSON file named
`prag.json` with the content:

```
{
  "load_src": {
    "en": ["artifacts", "{epoch_en}_model.pth"],
    "ar": ["artifacts_ar", "{epoch_ar}_model.pth"],
    "de": ["artifacts_de", "{epoch_de}_model.pth"],
    "es": ["artifacts_es", "{epoch_es}_model.pth"],
    "fr": ["artifacts_fr", "{epoch_fr}_model.pth"]
  },
  "main_src": "en",
  "src_key_as_lang": true
}
```

Replace `{epoch_en/ar/de/es/fr}` with the actual value of the model parameter file under the
corresponding artifacts directory.

### Taggers

To train the English tagger, run:

    ./run_tagger.py with artifacts_dir=tagger_artifacts word_emb_path=aligned_fasttext/wiki.multi.min.en.vec

To train the other taggers, run:

    ./run_tagger.py with artifacts_dir=tagger_artifacts_{lang} corpus.lang={lang} \
      word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec \
      load_tags_vocab_from=tagger_artifacts/vocab.yml

As before, create a JSON file named `prag_tagging.json` with a similar content, replacing the
paths to the taggers' artifacts directories and model parameter files accordingly.

## Running majority voting baseline

### Parsing

    ./run_majority.py with prag.json corpus.lang={lang} word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec -f

### Tagging

    ./run_majority_tagging.py with prag_tagging.json corpus.lang={lang} \
      word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec -f

## Running PPTX baseline

### Parsing

    ./run_pptx.py with prag prag.json artifacts_dir=pptx_artifacts corpus.lang={lang} \
      word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec

### Tagging

    ./run_pptx_tagging.py with prag prag_tagging.json artifacts_dir=pptx_tagging_artifacts corpus.lang={lang} \
      word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec

## Running our method (uniform LOP weights)

### Parsing

    ./run_pptx.py with prag_gmean prag.json artifacts_dir=gmean_artifacts corpus.lang={lang} \
      word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec

### Tagging

    ./run_pptx_tagging.py with prag_gmean prag_tagging.json artifacts_dir=gmean_tagging_artifacts corpus.lang={lang} \
      word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec

## Learning the LOP weights

### Parsing

    ./learn_weighting.py with prag.json artifacts_dir=lopw_artifacts corpus.lang={lang} \
      word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec -f

### Tagging

    ./learn_weighting_tagging.py with prag_tagging.json artifacts_dir=lopw_tagging_artifacts corpus.lang={lang} \
      word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec -f

## Running with learned LOP weights

### Parsing

    ./run_pptx.py with prag_lopw prag.json artifacts_dir=lopw_artifacts overwrite=True \
      corpus.lang={lang} word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec \
      load_src2ws_from=lopw_artifacts/src2ws.yml

### Tagging

    ./run_pptx_tagging.py with prag_lopw prag_tagging.json artifacts_dir=lopw_tagging_artifacts overwrite=True \
      corpus.lang={lang} word_emb_path=aligned_fasttext/wiki.multi.min.{lang}.vec \
      load_src2ws_from=lopw_tagging_artifacts/src2ws.yml

## (Optional) Sacred: an experiment manager

Almost all scripts in this repository use [Sacred](https://github.com/IDSIA/sacred/). The scripts
are written so that you can store all about an experiment run in a MongoDB database. Simply set
environment variables `SACRED_MONGO_URL` to point to a MongoDB instance and `SACRED_DB_NAME` to a
database name to activate it. Also, invoke the `help` command of any such script to print its usage,
e.g., `./run_parser.py help`.
