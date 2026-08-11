"""CSV loading and schema validation for product-review analysis."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"Review_Text", "Star_Rating"}


def load_reviews(csv_path: str | Path) -> pd.DataFrame:
    """Load reviews, remove blank records/duplicates, and validate the input schema.

    A small compatibility mapping is provided for the repository's original
    ``review,sentiment`` sample file; production inputs should use the documented
    required columns.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Review CSV was not found: {path.resolve()}")

    frame = pd.read_csv(path)
    if {"review", "sentiment"}.issubset(frame.columns) and not REQUIRED_COLUMNS.issubset(frame.columns):
        frame = frame.rename(columns={"review": "Review_Text"})
        frame["Review_ID"] = range(1, len(frame) + 1)
        frame["Product_Name"] = "Sample Product"
        frame["Star_Rating"] = frame["sentiment"].astype(str).str.lower().map(
            {"positive": 5, "neutral": 3, "negative": 1}
        )
        frame["Date"] = pd.Timestamp.today().normalize()
        frame["Verified_Purchase"] = True

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))

    frame = frame.copy()
    defaults = {"Review_ID": range(1, len(frame) + 1), "Product_Name": "Unspecified Product", "Date": pd.NaT, "Verified_Purchase": False}
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    frame["Review_Text"] = frame["Review_Text"].fillna("").astype(str).str.strip()
    frame = frame[frame["Review_Text"].ne("")].drop_duplicates(
        subset=["Product_Name", "Review_Text"], keep="first"
    )
    frame["Star_Rating"] = pd.to_numeric(frame["Star_Rating"], errors="coerce").clip(1, 5)
    frame = frame.dropna(subset=["Star_Rating"])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Verified_Purchase"] = frame["Verified_Purchase"].apply(_to_bool)
    return frame.reset_index(drop=True)


def _to_bool(value: object) -> bool:
    """Normalize common CSV representations of boolean values."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
