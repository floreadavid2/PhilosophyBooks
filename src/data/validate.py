import os
from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/philosophy_books.csv")
PROCESSED_DATA_PATH = Path("data/processed/philosophy_books_clean.csv")

EXPECTED_COLUMNS = {"id", "title", "author", "school", "summary", "key_concepts"}

def validate_and_clean_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"File doesn't exist at: {input_path}")

    df = pd.read_csv(input_path)

    # 1. Verifies the existence of all columns
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    assert not missing_cols, f"Missing these columns: {missing_cols}"

    # 2. Verifies duplicates
    assert df["id"].is_unique, "Exists duplicate ID's!"

    # 3. Checks for nulls
    assert df["title"].notnull().all(), "There are null titles!"
    assert df["summary"].notnull().all(), "There are null summaries!"

    # 4. Checks lenght of the summary
    too_short = df[df["summary"].str.strip().str.len() < 20]
    assert len(too_short) == 0, f"Too short summaries: {too_short['title'].tolist()}"

    #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data validation passed. Clean dataset saved to {output_path} ({len(df)} rows).")
    return df

if __name__ == "__main__":
    validate_and_clean_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)