"""Professional, file-based visualizations for product intelligence outputs."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig: plt.Figure, destination: Path, filename: str) -> None:
    """Save a compact figure without retaining memory between charts."""
    fig.tight_layout(); fig.savefig(destination / filename, dpi=170, bbox_inches="tight"); plt.close(fig)


def _bar(frame: pd.DataFrame, x: str, y: str, title: str, destination: Path, filename: str, color: str = "#2878B5") -> None:
    """Write a sorted horizontal bar chart or an informative empty state."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if frame.empty: ax.text(.5, .5, "Insufficient data", ha="center", va="center"); ax.axis("off")
    else: sns.barplot(data=frame, x=x, y=y, ax=ax, color=color); ax.set_title(title)
    _save(fig, destination, filename)


def create_visualizations(reviews: pd.DataFrame, summary: dict[str, object], graph_dir: str | Path, aspect_health: pd.DataFrame, comparison: pd.DataFrame) -> None:
    """Generate only meaningful sentiment, quality, trend, anomaly, and comparison charts."""
    destination = Path(graph_dir); destination.mkdir(parents=True, exist_ok=True)
    for column, title, filename, color in (("Sentiment", "Sentiment Distribution", "01_sentiment_distribution.png", "#2878B5"), ("Star_Rating", "Star Rating Distribution", "02_star_rating_distribution.png", "#5F9E6E"), ("Tone", "Tone Distribution", "03_tone_distribution.png", "#7A5195"), ("Review_Category", "Product and Service Review Categories", "04_review_categories.png", "#E67E22")):
        counts = reviews[column].astype(str).value_counts().rename_axis(column).reset_index(name="Reviews")
        _bar(counts, "Reviews", column, title, destination, filename, color)
    _bar(aspect_health, "Aspect_Health_Score", "Aspect", "Aspect Health Scores", destination, "05_aspect_health.png", "#2E8B57")
    _bar(aspect_health.head(5), "Aspect_Health_Score", "Aspect", "Top Positive Aspects", destination, "06_top_positive_aspects.png", "#2E8B57")
    _bar(aspect_health.sort_values("Aspect_Health_Score").head(5), "Aspect_Health_Score", "Aspect", "Top Negative Aspects", destination, "07_top_negative_aspects.png", "#C0392B")
    dated = reviews.dropna(subset=["Date"]).copy(); fig, ax = plt.subplots(figsize=(9,4.5))
    if dated.empty: ax.text(.5,.5,"No valid dates available",ha="center",va="center"); ax.axis("off")
    else:
        trend = dated.set_index("Date").resample("ME")["Polarity"].mean(); ax.plot(trend.index, trend.values, marker="o"); ax.set(title="Review Sentiment Trend", xlabel="Month", ylabel="Average polarity")
    _save(fig,destination,"08_sentiment_trend.png")
    counts = reviews["Is_Anomalous"].map({True:"Suspicious",False:"Normal"}).value_counts().rename_axis("Status").reset_index(name="Reviews"); _bar(counts,"Reviews","Status","Suspicious vs Normal Reviews",destination,"09_anomaly_distribution.png","#C0392B")
    _bar(comparison,"Product Health Score","Product","Product Comparison",destination,"10_product_comparison.png","#2878B5")
    if "Value Score" in comparison: _bar(comparison,"Value Score","Product","Price-to-Value Score",destination,"11_value_score.png","#7A5195")
    words = " ".join(reviews["Cleaned_Text"].dropna()) or "no reviews"; fig, ax = plt.subplots(figsize=(10,5)); ax.imshow(WordCloud(width=1200,height=600,background_color="white",colormap="viridis").generate(words)); ax.axis("off"); ax.set_title("Review Word Cloud"); _save(fig,destination,"12_word_cloud.png")
    corr = reviews.assign(Relevance=reviews["Is_Relevant"].astype(int), Verified=reviews["Verified_Purchase"].astype(int), Anomalous=reviews["Is_Anomalous"].astype(int))[["Polarity","Star_Rating","Relevance","Verified","Anomalous"]].corr(); fig, ax = plt.subplots(figsize=(6,5)); sns.heatmap(corr,annot=True,fmt=".2f",cmap="RdYlGn",center=0,ax=ax); ax.set_title("Review Signal Correlation"); _save(fig,destination,"13_correlation_heatmap.png")
