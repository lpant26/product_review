"""Weighted Buy Score and recommendation calculation."""

import numpy as np
import pandas as pd

from config import HEALTH_WEIGHTS
from recommendation import recommendation_for


def calculate_product_score(reviews: pd.DataFrame) -> dict[str, float | str]:
    """Calculate transparent Product Health Score, down-weighting service and anomalies."""
    weights = reviews["Effective_Weight"].astype(float) if "Effective_Weight" in reviews else reviews["Relevance_Weight"].astype(float)
    sentiment_score = np.average((reviews["Polarity"] + 1) * 50, weights=weights)
    rating_score = np.average(reviews["Star_Rating"] * 20, weights=weights)
    relevant_ratio = reviews["Is_Relevant"].mean() * 100
    verified_ratio = reviews["Verified_Purchase"].mean() * 100
    confidence_score = np.average(reviews["Sentiment_Confidence"] * 100, weights=weights)
    health_score = sum((sentiment_score, rating_score, relevant_ratio, verified_ratio, confidence_score)[index] * weight for index, weight in enumerate(HEALTH_WEIGHTS.values()))
    health_score = round(float(np.clip(health_score, 0, 100)), 2)
    return {
        "Product Health Score": health_score, "Recommendation": recommendation_for(health_score),
        "Sentiment Score": round(float(sentiment_score), 2), "Rating Score": round(float(rating_score), 2),
        "Relevant Review Ratio": round(float(relevant_ratio), 2), "Verified Purchase Ratio": round(float(verified_ratio), 2), "Confidence Score": round(float(confidence_score), 2),
    }


def add_review_weighted_scores(reviews: pd.DataFrame) -> pd.DataFrame:
    """Add a per-review contribution score for transparent exports and charts."""
    output = reviews.copy()
    output["Effective_Weight"] = output["Relevance_Weight"] * output.get("Anomaly_Weight", 1.0)
    output["Weighted_Score"] = ((output["Polarity"] + 1) * 50 * output["Effective_Weight"]).round(2)
    return output
