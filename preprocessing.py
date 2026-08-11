"""NLP text normalization helpers."""

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize


def ensure_nltk_resources() -> None:
    """Download the NLTK corpora required by the preprocessing pipeline."""
    resources = (("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"))
    for lookup, package in resources:
        try:
            nltk.data.find(lookup)
        except LookupError:
            nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def _stopwords() -> set[str]:
    """Return English stop words once per process."""
    return set(stopwords.words("english"))


def clean_text(text: object) -> str:
    """Lowercase, remove punctuation/stopwords, tokenize, and lemmatize text."""
    normalized = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    lemmatizer = WordNetLemmatizer()
    tokens = wordpunct_tokenize(normalized)
    meaningful = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in _stopwords()
    ]
    return " ".join(meaningful)


def raw_tokens(text: object) -> list[str]:
    """Extract lowercase word tokens while retaining stop words for heuristics."""
    return re.findall(r"\b\w+\b", str(text).lower())
