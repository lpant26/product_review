"""Optional supervised sentiment model for CSV files that include sentiment labels."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def add_ml_predictions(reviews: pd.DataFrame) -> pd.DataFrame:
    """Train a TF-IDF/Logistic Regression model when labeled reviews are supplied.

    The core workflow always uses TextBlob so every valid input works without
    labels. If a CSV additionally provides ``Sentiment_Label`` or legacy
    ``sentiment`` labels, this optional supervised model adds its predictions
    and probability confidence for comparison.
    """
    output = reviews.copy()
    label_column = next((column for column in ("Sentiment_Label", "sentiment") if column in output.columns), None)
    output["ML_Sentiment"] = "Not trained (labels unavailable)"
    output["ML_Confidence"] = pd.NA
    if label_column is None:
        return output

    labels = output[label_column].astype(str).str.strip().str.title()
    valid = labels.isin({"Positive", "Negative", "Neutral"})
    if valid.sum() < 6 or labels[valid].nunique() < 2 or labels[valid].value_counts().min() < 2:
        return output

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(output.loc[valid, "Cleaned_Text"], labels[valid])
    probabilities = model.predict_proba(output["Cleaned_Text"])
    output["ML_Sentiment"] = model.classes_[probabilities.argmax(axis=1)]
    output["ML_Confidence"] = probabilities.max(axis=1).round(4)
    return output
