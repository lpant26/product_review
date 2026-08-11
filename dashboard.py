"""Streamlit interface for the AI Product Intelligence & Decision Support System."""

from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st

from main import analyze_reviews
from recommendation import personalized_score


def run() -> None:
    """Render a local, upload-driven dashboard with filters and decision explanations."""
    st.set_page_config(page_title="AI Product Intelligence", page_icon="📊", layout="wide")
    st.title("AI Product Intelligence & Decision Support System")
    st.caption("Explainable local NLP analytics for product-review decisions.")
    uploaded = st.sidebar.file_uploader("Upload reviews CSV", type="csv")
    if not uploaded:
        st.info("Upload a CSV. Required: Review_Text and Star_Rating. Optional: Product_Name, Date, Verified_Purchase, Price.")
        return
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); csv_path = root / "reviews.csv"; csv_path.write_bytes(uploaded.getvalue())
            reviews, summary = analyze_reviews(csv_path, root / "output", root / "graphs")
            products = sorted(reviews["Product_Name"].dropna().unique())
            product = st.sidebar.selectbox("Product", ["All products", *products])
            data = reviews if product == "All products" else reviews[reviews["Product_Name"].eq(product)]
            sentiments = st.sidebar.multiselect("Sentiment", sorted(reviews["Sentiment"].unique()), default=sorted(reviews["Sentiment"].unique()))
            tones = st.sidebar.multiselect("Tone", sorted(reviews["Tone"].unique()), default=sorted(reviews["Tone"].unique()))
            aspects = st.sidebar.multiselect("Aspect", sorted({item.strip() for value in reviews["Detected_Aspects"].dropna() for item in value.split(",") if item.strip()}))
            data = data[data["Sentiment"].isin(sentiments) & data["Tone"].isin(tones)]
            if aspects: data = data[data["Detected_Aspects"].str.contains("|".join(aspects), na=False)]
            st.sidebar.header("Your priorities")
            priorities = {name: st.sidebar.slider(name, 0, 100, 0, 5) for name in ["Battery", "Camera", "Performance", "Display", "Build Quality", "Price", "Design", "Software"]}
            aspect_health = summary["Aspect Health"]
            personal = personalized_score(aspect_health, priorities)
            positive = data["Sentiment"].eq("Positive").mean()*100 if len(data) else 0
            cols = st.columns(6)
            for column, label, value in zip(cols, ["Product Health", "Recommendation", "Average Rating", "Positive Reviews", "Relevant Reviews", "Suspicious Reviews"], [f"{summary['Product Health Score']}/100", summary["Recommendation"], f"{data['Star_Rating'].mean():.1f}/5" if len(data) else "—", f"{positive:.0f}%", f"{data['Is_Relevant'].mean()*100:.0f}%" if len(data) else "—", f"{data['Is_Anomalous'].mean()*100:.0f}%" if len(data) else "—"]): column.metric(label, value)
            st.header("Overall Product Health")
            st.dataframe(pd.DataFrame([{key:value for key,value in summary.items() if isinstance(value,(str,int,float))}]), use_container_width=True)
            st.header("Aspect Analysis")
            st.dataframe(aspect_health, use_container_width=True)
            strength_col, issue_col = st.columns(2)
            strength_col.subheader("Top Strengths"); strength_col.dataframe(aspect_health.head(3), hide_index=True)
            issue_col.subheader("Top Problems"); issue_col.dataframe(aspect_health.sort_values("Aspect_Health_Score").head(3), hide_index=True)
            st.header("Explainable Recommendation")
            for reason in summary["Explanation"]["Reasons"]: st.success(reason)
            for concern in summary["Explanation"]["Concerns"]: st.warning(concern)
            st.header("Personalized Recommendation")
            if personal["Personalized Product Score"] is None: st.info(personal["Personalized Recommendation"])
            else: st.metric("Personalized Product Score", f"{personal['Personalized Product Score']}/100", personal["Personalized Recommendation"])
            st.header("Competitor Comparison"); st.dataframe(summary["Comparison"], use_container_width=True)
            if summary["Emerging Alerts"]: st.warning(summary["Emerging Alerts"])
            st.header("Visualizations")
            for image in sorted((root / "graphs").glob("*.png")): st.image(str(image), caption=image.stem.replace("_", " ").title(), use_container_width=True)
            st.header("Raw Reviews")
            search = st.text_input("Search review text")
            if search: data = data[data["Review_Text"].str.contains(search, case=False, na=False)]
            st.dataframe(data, use_container_width=True)
            st.download_button("Download processed CSV", reviews.to_csv(index=False), "processed_reviews.csv", "text/csv")
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))


if __name__ == "__main__": run()
