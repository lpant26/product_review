"""Trend, competitor, and value analysis derived from reviewed products."""

import pandas as pd

from score_calculator import calculate_product_score


def detect_emerging_issues(reviews: pd.DataFrame) -> list[str]:
    """Alert when an aspect complaint share rises by at least 15 points month-to-month."""
    dated = reviews.dropna(subset=["Date"]).copy()
    if dated.empty or dated["Date"].dt.to_period("M").nunique() < 2:
        return []
    dated["Month"] = dated["Date"].dt.to_period("M")
    alerts = []
    for aspect in ("Battery", "Performance", "Software", "Durability", "Display"):
        mentions = dated[dated["Detected_Aspects"].str.contains(aspect, na=False)].copy()
        if mentions.empty: continue
        negatives = mentions[mentions["Aspect_Sentiments"].str.contains(f'"{aspect}": "Negative"', regex=False, na=False)]
        monthly = negatives.groupby("Month").size().reindex(dated.groupby("Month").size().index, fill_value=0) / dated.groupby("Month").size()
        if len(monthly) >= 2 and monthly.iloc[-1] - monthly.iloc[-2] >= .15:
            alerts.append(f"Potential emerging issue: {aspect.lower()} complaints increased significantly over time.")
    return alerts


def build_product_comparison(reviews: pd.DataFrame, aspect_health: pd.DataFrame) -> pd.DataFrame:
    """Compare products fairly using per-product percentages and health, not review volume."""
    rows = []
    for product, group in reviews.groupby("Product_Name", dropna=False):
        summary = calculate_product_score(group)
        product_aspects = aspect_health[aspect_health.get("Product_Name", pd.Series(dtype=str)).eq(product)] if "Product_Name" in aspect_health else pd.DataFrame()
        if product_aspects.empty:
            top_strength = top_weakness = "Insufficient aspect evidence"
        else:
            top_strength = product_aspects.iloc[0]["Aspect"]
            top_weakness = product_aspects.sort_values("Aspect_Health_Score").iloc[0]["Aspect"]
        rows.append({"Product": product, "Product Health Score": summary["Product Health Score"], "Average Rating": round(group["Star_Rating"].mean(), 2), "Positive %": round(group["Sentiment"].eq("Positive").mean()*100, 2), "Negative %": round(group["Sentiment"].eq("Negative").mean()*100, 2), "Relevant Review %": round(group["Is_Relevant"].mean()*100, 2), "Verified Purchase %": round(group["Verified_Purchase"].mean()*100, 2), "Top Strength": top_strength, "Top Weakness": top_weakness})
    return pd.DataFrame(rows).sort_values("Product Health Score", ascending=False).reset_index(drop=True)


def add_value_scores(comparison: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Add relative value only when a valid Price column is available."""
    if "Price" not in reviews.columns:
        return comparison
    prices = pd.to_numeric(reviews["Price"], errors="coerce").groupby(reviews["Product_Name"]).mean()
    if prices.dropna().empty:
        return comparison
    output = comparison.copy(); output["Average Price"] = output["Product"].map(prices)
    price_component = (1 - (output["Average Price"] - prices.min()) / max(prices.max() - prices.min(), 1)) * 100
    output["Value Score"] = (0.7 * output["Product Health Score"] + 0.3 * price_component).round(2)
    return output
