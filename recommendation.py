"""Traceable product and personalized recommendation generation."""

import pandas as pd

from config import RECOMMENDATION_BANDS


def recommendation_for(score: float) -> str:
    """Return the configurable recommendation label for a 0-100 score."""
    return next(label for threshold, label in RECOMMENDATION_BANDS if score >= threshold)


def build_explanation(reviews: pd.DataFrame, summary: dict[str, object], aspect_health: pd.DataFrame) -> dict[str, object]:
    """Build evidence-based recommendation reasons and concerns from actual metrics."""
    relevant = reviews[reviews["Is_Relevant"]]
    positive_pct = (relevant["Sentiment"].eq("Positive").mean() * 100) if len(relevant) else 0
    product_aspects = aspect_health[~aspect_health["Aspect"].isin({"Delivery", "Packaging", "Seller", "Customer Service"})]
    strengths = product_aspects.head(3).to_dict("records")
    issues = product_aspects.sort_values("Aspect_Health_Score").head(3).to_dict("records")
    reasons = [f"{positive_pct:.0f}% positive sentiment among relevant reviews", f"{summary['Relevant Review Ratio']:.0f}% of reviews discuss the product"]
    reasons.extend(f"{item['Aspect']} health: {item['Aspect_Health_Score']:.0f}/100" for item in strengths[:2])
    concerns = [f"{item['Aspect']} health: {item['Aspect_Health_Score']:.0f}/100" for item in issues if item["Aspect_Health_Score"] < 75]
    anomaly_pct = reviews["Is_Anomalous"].mean() * 100
    if anomaly_pct:
        concerns.append(f"{anomaly_pct:.0f}% of reviews are marked suspicious and down-weighted")
    service_count = reviews["Review_Category"].str.contains("DELIVERY|PACKAGING|SELLER|CUSTOMER_SERVICE", regex=True).sum()
    if service_count:
        reasons.append(f"{service_count} service/delivery comments were retained but down-weighted for product health")
    return {"Recommendation": summary["Recommendation"], "Score": summary["Product Health Score"], "Reasons": reasons, "Concerns": concerns, "Top Strengths": strengths, "Top Issues": issues}


def personalized_score(aspect_health: pd.DataFrame, priorities: dict[str, float]) -> dict[str, object]:
    """Calculate a priority-weighted score using available aspect health scores."""
    selected = {aspect: weight for aspect, weight in priorities.items() if float(weight) > 0}
    lookup = aspect_health.set_index("Aspect")["Aspect_Health_Score"].to_dict() if not aspect_health.empty else {}
    available = {aspect: float(weight) for aspect, weight in selected.items() if aspect in lookup}
    if not available:
        return {"Personalized Product Score": None, "Personalized Recommendation": "No selected aspects have review evidence."}
    total = sum(available.values())
    score = sum(lookup[aspect] * weight / total for aspect, weight in available.items())
    return {"Personalized Product Score": round(score, 2), "Personalized Recommendation": recommendation_for(score)}
