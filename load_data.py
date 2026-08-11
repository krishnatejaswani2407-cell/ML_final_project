import pandas as pd

# Dataset Path
DATASET_PATH = r"D:\ML PPTS\Project\WA_Fn-UseC_-HR-Employee-Attrition.csv"


def load_dataset():
    """Load the HR dataset."""
    return pd.read_csv(DATASET_PATH)


def get_dataset_info():
    """Return all information required for dashboard."""

    df = load_dataset()

    return {
        "dataset_name": DATASET_PATH,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_info": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum())
            }
            for col in df.columns
        ],
        "preview": df.head(10).to_html(
            classes="table preview-table",
            index=False
        )
    }