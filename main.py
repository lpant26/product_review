"""Command-line entry point for Smart Product Review Analyzer."""

import argparse
from pathlib import Path

import pandas as pd

from data_loader import load_reviews
from anomaly_detector import detect_review_anomalies
from aspect_analyzer import analyze_aspects, calculate_aspect_health
from intelligence import add_value_scores, build_product_comparison, detect_emerging_issues
from ml_sentiment import add_ml_predictions
from preprocessing import clean_text, ensure_nltk_resources
from relevance_detector import assess_relevance
from score_calculator import add_review_weighted_scores, calculate_product_score
from sentiment import analyze_sentiment
from tone_detector import detect_tone
from recommendation import build_explanation
from utils import build_console_report
from visualization import create_visualizations


def analyze_reviews(csv_path: str | Path, output_dir: str | Path = "output", graph_dir: str | Path = "graphs") -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the complete review-analysis workflow and write all requested outputs."""
    ensure_nltk_resources()
    reviews = load_reviews(csv_path)
    reviews["Cleaned_Text"] = reviews["Review_Text"].map(clean_text)
    reviews = add_ml_predictions(reviews)
    # Legacy sample labels are consumed by the optional ML model; retaining a
    # lowercase ``sentiment`` beside generated ``Sentiment`` breaks Excel and
    # PowerShell readers, which treat headers case-insensitively.
    if "sentiment" in reviews.columns:
        reviews = reviews.drop(columns="sentiment")
    sentiment_data = reviews["Review_Text"].map(analyze_sentiment).apply(pd.Series)
    relevance_data = reviews["Review_Text"].map(assess_relevance).apply(pd.Series)
    reviews = pd.concat([reviews, sentiment_data, relevance_data], axis=1)
    reviews["Tone"] = reviews["Review_Text"].map(detect_tone)
    aspect_data = reviews["Review_Text"].map(analyze_aspects).apply(pd.Series)
    reviews = pd.concat([reviews, aspect_data], axis=1)
    reviews = detect_review_anomalies(reviews)
    reviews = add_review_weighted_scores(reviews)
    summary = calculate_product_score(reviews)
    aspect_health = calculate_aspect_health(reviews)
    product_aspects = []
    for product, group in reviews.groupby("Product_Name"):
        item = calculate_aspect_health(group); item["Product_Name"] = product; product_aspects.append(item)
    product_aspect_health = pd.concat(product_aspects, ignore_index=True) if product_aspects else pd.DataFrame()
    comparison = add_value_scores(build_product_comparison(reviews, product_aspect_health), reviews)
    summary["Emerging Alerts"] = " | ".join(detect_emerging_issues(reviews))
    explanation = build_explanation(reviews, summary, aspect_health)
    reviews["Product_Health_Score"] = summary["Product Health Score"]
    reviews["Final_Recommendation"] = summary["Recommendation"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(output_path / "processed_reviews.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_path / "product_summary.csv", index=False)
    aspect_health.to_csv(output_path / "aspect_health_scores.csv", index=False)
    comparison.to_csv(output_path / "product_comparison.csv", index=False)
    explanation_rows = ([{"Type": "Reason", "Detail": item} for item in explanation["Reasons"]] + [{"Type": "Concern", "Detail": item} for item in explanation["Concerns"]])
    pd.DataFrame(explanation_rows).to_csv(output_path / "recommendation_explanation.csv", index=False)
    create_visualizations(reviews, summary, graph_dir, aspect_health, comparison)
    summary.update({"Aspect Health": aspect_health, "Comparison": comparison, "Explanation": explanation})
    return reviews, summary


def main() -> None:
    """Parse command-line options, execute the pipeline, and print its report."""
    parser = argparse.ArgumentParser(description="AI Product Intelligence & Decision Support System")
    parser.add_argument("csv_path", nargs="?", default="sample_product_reviews.csv", help="Path to the review CSV")
    parser.add_argument("--output-dir", default="output", help="Directory for processed CSV files")
    parser.add_argument("--graph-dir", default="graphs", help="Directory for PNG visualizations")
    args = parser.parse_args()
    reviews, summary = analyze_reviews(args.csv_path, args.output_dir, args.graph_dir)
    print(build_console_report(reviews, summary))
    print(f"\nProcessed data: {Path(args.output_dir).resolve() / 'processed_reviews.csv'}")
    print(f"Charts: {Path(args.graph_dir).resolve()}")


if __name__ == "__main__":
    main()
