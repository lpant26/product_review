"""TextBlob-based sentiment scoring and confidence estimation."""

import numpy as np
from textblob import TextBlob


def analyze_sentiment(text: object) -> dict[str, float | str]:
    """Return polarity, subjectivity, class label, and a transparent confidence score."""
    result = TextBlob(str(text)).sentiment
    polarity, subjectivity = float(result.polarity), float(result.subjectivity)
    if polarity > 0.10:
        label = "Positive"
    elif polarity < -0.10:
        label = "Negative"
    else:
        label = "Neutral"
    # Strength and opinionatedness jointly express confidence, bounded to [0, 1].
    confidence = float(np.clip(0.45 + 0.40 * abs(polarity) + 0.15 * subjectivity, 0, 1))
    return {"Polarity": polarity, "Subjectivity": subjectivity, "Sentiment": label, "Sentiment_Confidence": confidence}
