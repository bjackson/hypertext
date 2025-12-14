import pathlib
import re
import unicodedata

import spacy
from datasets import load_dataset
from tqdm import tqdm

# Lazy-load spaCy to avoid heavy import side-effects.
_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


DATASETS = [
    # (dataset_name, config_name or None, text_field, splits, streaming)
    ("mnemoraorg/tweetfeels-1m6", None, " text", ["train"], True),  # fixed stray space in field name
]

SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
OUT = pathlib.Path("../data/raw_corpus.txt")  # overwrite, or use corpus_large.txt
MAX_SENTENCES = 3_000_000
MIN_TOK, MAX_TOK = 3, 60


def clean(s: str) -> str:
    """Clean a raw tweet sentence.

    Steps (in order):

    * Unicode NFKC normalization and lowercasing.
    * Remove whole Twitter @mentions and #hashtags (the leading symbol plus the token).
    * Remove characters outside an allowed set (alphanumerics + limited punctuation).
    * Collapse consecutive whitespace and trim.

    Parameters
    ----------
    s : str
        Raw input sentence.

    Returns
    -------
    str
        Cleaned sentence suitable for corpus building.
    """
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[@#][a-z0-9_]+", " ", s)
    s = re.sub(r"[^a-z0-9 ,;:'\"().!?-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iter_sents(name, cfg, field, splits, streaming):
    ds = load_dataset(name, cfg, streaming=streaming) if cfg else load_dataset(name, streaming=streaming)
    nlp = get_nlp()
    for split in splits:
        part = ds[split]
        for rec in part:
            txt = rec.get(field, "")
            if not txt or not isinstance(txt, str):
                continue
            for sent in nlp(txt).sents:
                s = clean(sent.text)
                n_tok = len(s.split())
                if MIN_TOK <= n_tok <= MAX_TOK:
                    yield s


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for name, cfg, field, splits, streaming in DATASETS:
            for s in tqdm(iter_sents(name, cfg, field, splits, streaming), desc=f"Processing {name}", unit="sent"):
                if s in seen:
                    continue
                seen.add(s)
                f.write(s + "\n")
                n += 1
                if n >= MAX_SENTENCES:
                    break
            if n >= MAX_SENTENCES:
                break
    print(f"Wrote {n} sentences to {OUT}")


if __name__ == "__main__":
    main()
