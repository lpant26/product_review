"""Rule-assisted tone detection for product-review language."""

from textblob import TextBlob

TONE_KEYWORDS = {
    "Happy": {"happy", "love", "great", "wonderful", "pleased", "delighted"},
    "Excited": {"excited", "amazing", "awesome", "fantastic", "wow", "superb"},
    "Satisfied": {"satisfied", "good", "nice", "decent", "worth", "recommend"},
    "Angry": {"angry", "worst", "hate", "fraud", "cheated", "useless"},
    "Frustrated": {"frustrated", "issue", "problem", "not working", "stuck", "annoying"},
    "Disappointed": {"disappointed", "poor", "bad", "terrible", "broken", "regret"},
}


def detect_tone(text: object) -> str:
    """Infer the strongest tone using keywords first and TextBlob polarity as fallback."""
    content = str(text).lower()
    matches = {tone: sum(keyword in content for keyword in words) for tone, words in TONE_KEYWORDS.items()}
    strongest, count = max(matches.items(), key=lambda item: item[1])
    if count:
        return strongest
    polarity = TextBlob(content).sentiment.polarity
    return "Satisfied" if polarity > 0.10 else "Disappointed" if polarity < -0.10 else "Neutral"
