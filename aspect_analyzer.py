"""Explainable, local aspect-based sentiment analysis."""

import json
import re

import numpy as np
import pandas as pd
from textblob import TextBlob

ASPECT_KEYWORDS = {
    "Battery": ("battery", "charge", "charging", "drain", "power"),
    "Camera": ("camera", "photo", "picture", "selfie", "lens"),
    "Performance": ("performance", "fast", "slow", "lag", "heating", "heat", "speed"),
    "Display": ("display", "screen", "brightness", "colour", "color", "resolution"),
    "Design": ("design", "look", "style", "stylish", "appearance"),
    "Build Quality": ("build", "quality", "material", "finish", "sturdy"),
    "Durability": ("durable", "durability", "broke", "broken", "lasting"),
    "Audio": ("sound", "audio", "speaker", "music", "volume"),
    "Comfort": ("comfort", "comfortable", "fit", "ergonomic", "lightweight"),
    "Software": ("software", "app", "update", "interface", "bug", "crash"),
    "Features": ("feature", "function", "option", "mode"),
    "Price": ("price", "value", "cost", "expensive", "cheap", "money"),
    "Delivery": ("delivery", "shipping", "courier", "dispatch", "late"),
    "Packaging": ("packaging", "package", "box", "packed"),
    "Seller": ("seller", "amazon", "vendor", "store"),
    "Customer Service": ("customer service", "support", "refund", "return", "service"),
}


def analyze_aspects(text: object) -> dict[str, str]:
    """Extract mentioned aspects and score their local sentence context.

    Each keyword's containing sentence is scored independently, allowing mixed
    feedback (for example, a positive camera but negative battery) to remain visible.
    """
    content = str(text).lower()
    sentences = [item.strip() for item in re.split(r"[.!?;]+", content) if item.strip()] or [content]
    findings: dict[str, dict[str, float | str]] = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        matches = [sentence for sentence in sentences if any(keyword in sentence for keyword in keywords)]
        if matches:
            polarity = float(np.mean([TextBlob(sentence).sentiment.polarity for sentence in matches]))
            label = "Positive" if polarity > .10 else "Negative" if polarity < -.10 else "Neutral"
            findings[aspect] = {"sentiment": label, "polarity": round(polarity, 4)}
    return {
        "Detected_Aspects": ", ".join(findings) if findings else "",
        "Aspect_Sentiments": json.dumps({key: value["sentiment"] for key, value in findings.items()}),
        "Aspect_Polarities": json.dumps({key: value["polarity"] for key, value in findings.items()}),
    }


def calculate_aspect_health(reviews: pd.DataFrame) -> pd.DataFrame:
    """Calculate 0-100 health for every detected aspect, excluding service from product health."""
    rows: list[dict[str, object]] = []
    for aspect in ASPECT_KEYWORDS:
        matching = reviews[reviews["Detected_Aspects"].fillna("").str.contains(re.escape(aspect), regex=True)]
        if matching.empty:
            continue
        polarities = matching["Aspect_Polarities"].map(lambda value: json.loads(value).get(aspect)).dropna()
        if polarities.empty:
            continue
        weights = matching.loc[polarities.index, "Effective_Weight"].astype(float)
        score = float(np.average((polarities.astype(float) + 1) * 50, weights=weights))
        rows.append({"Aspect": aspect, "Aspect_Health_Score": round(np.clip(score, 0, 100), 2), "Mentions": len(polarities), "Mention_Ratio": round(len(polarities) / len(reviews) * 100, 2)})
    return pd.DataFrame(rows).sort_values("Aspect_Health_Score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["Aspect", "Aspect_Health_Score", "Mentions", "Mention_Ratio"])
