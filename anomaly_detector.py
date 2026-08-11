"""Local, explainable review anomaly detection using text and date signals."""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import ANOMALY_SIMILARITY_THRESHOLD
from preprocessing import raw_tokens


def detect_review_anomalies(reviews: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious reviews without declaring that any review is fake.

    TF-IDF cosine similarity identifies near duplicates; transparent textual
    signals identify unusually short, emphatic, emoji-heavy, or repetitive reviews.
    """
    output = reviews.copy()
    texts = output["Review_Text"].fillna("").astype(str)
    similar = np.zeros(len(output), dtype=bool)
    if len(output) > 1 and texts.str.strip().ne("").any():
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
        similarity = cosine_similarity(matrix)
        np.fill_diagonal(similarity, 0)
        similar = similarity.max(axis=1) >= ANOMALY_SIMILARITY_THRESHOLD
    duplicate = texts.str.lower().str.replace(r"\W+", "", regex=True).duplicated(keep=False).to_numpy()
    flags, reasons = [], []
    for position, text in enumerate(texts):
        tokens = raw_tokens(text)
        trigger = []
        if len(tokens) <= 2: trigger.append("very short")
        if duplicate[position]: trigger.append("duplicate text")
        if similar[position] and not duplicate[position]: trigger.append("highly similar to another review")
        if sum(ord(char) > 10000 for char in text) >= 3: trigger.append("many emojis")
        if text.count("!") >= 4: trigger.append("many exclamation marks")
        if any(count >= 3 for count in Counter(tokens).values()): trigger.append("repeated words")
        if any(char * 5 in text.lower() for char in set(text.lower()) if char.isalpha()): trigger.append("repeated characters")
        flags.append(bool(trigger)); reasons.append(", ".join(trigger))
    output["Is_Anomalous"] = flags
    output["Anomaly_Reason"] = reasons
    output["Anomaly_Weight"] = np.where(output["Is_Anomalous"], 0.55, 1.0)
    return output
