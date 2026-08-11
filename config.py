"""Central, explainable configuration for decision-support scoring."""

HEALTH_WEIGHTS = {
    "Sentiment Score": 0.40,
    "Rating Score": 0.30,
    "Relevance Score": 0.15,
    "Verified Purchase Score": 0.10,
    "Confidence Score": 0.05,
}
RECOMMENDATION_BANDS = ((85, "STRONG BUY"), (70, "BUY"), (55, "CONSIDER"), (40, "CAUTION"), (0, "AVOID"))
ANOMALY_SIMILARITY_THRESHOLD = 0.88
EMERGING_ISSUE_INCREASE = 0.15
SERVICE_WEIGHT = 0.10
