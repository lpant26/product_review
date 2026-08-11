"""General reporting and fake-review heuristics."""

from collections import Counter

import pandas as pd

from preprocessing import raw_tokens


def flag_suspicious_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Flag short, duplicated, emoji-heavy, exclamation-heavy, or repetitive reviews."""
    result = reviews.copy()
    normalized = result["Review_Text"].astype(str).str.lower().str.replace(r"\W+", "", regex=True)
    duplicate_text = normalized.duplicated(keep=False)

    flags, reasons = [], []
    for index, text in result["Review_Text"].items():
        tokens = raw_tokens(text)
        repeated = any(count >= 3 for count in Counter(tokens).values())
        emoji_count = sum(ord(char) > 10000 for char in str(text))
        triggers = []
        if len(tokens) <= 2:
            triggers.append("very short")
        if duplicate_text.loc[index]:
            triggers.append("repeated text")
        if emoji_count >= 3:
            triggers.append("many emojis")
        if str(text).count("!") >= 4:
            triggers.append("many exclamation marks")
        if repeated:
            triggers.append("repeated words")
        flags.append(bool(triggers))
        reasons.append(", ".join(triggers))
    result["Is_Suspicious"] = flags
    result["Suspicion_Reasons"] = reasons
    return result


def build_console_report(reviews: pd.DataFrame, summary: dict[str, object]) -> str:
    """Create a concise, interview-ready product intelligence report."""
    counts = reviews["Sentiment"].value_counts()
    return "\n".join([
        "AI PRODUCT INTELLIGENCE SYSTEM",
        "=" * 42,
        f"Total Reviews: {len(reviews)}",
        f"Positive Reviews: {counts.get('Positive', 0)}",
        f"Negative Reviews: {counts.get('Negative', 0)}",
        f"Neutral Reviews: {counts.get('Neutral', 0)}",
        f"Relevant Reviews: {int(reviews['Is_Relevant'].sum())}",
        f"Irrelevant Reviews: {int((~reviews['Is_Relevant']).sum())}",
        f"Suspicious Reviews: {int(reviews['Is_Anomalous'].sum())}",
        f"Average Rating: {reviews['Star_Rating'].mean():.2f}/5",
        f"Average Sentiment: {reviews['Polarity'].mean():.3f}",
        f"Product Health Score: {summary['Product Health Score']:.2f}/100",
        f"Recommendation: {summary['Recommendation']}",
        f"Top Strength: {summary['Explanation']['Top Strengths'][0]['Aspect'] if summary.get('Explanation', {}).get('Top Strengths') else 'Insufficient aspect evidence'}",
        f"Top Concern: {summary['Explanation']['Top Issues'][0]['Aspect'] if summary.get('Explanation', {}).get('Top Issues') else 'Insufficient aspect evidence'}",
    ])
