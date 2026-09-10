from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/philosophy_books_clean.csv")
OUTPUT_PATH = Path("data/processed/philosophy_books_featured.csv")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build unified representation text for semantic embedding."""
    df_feat = df.copy()

    # Combine metadata and text fields into a single search representation
    df_feat["content_to_embed"] = (
            "Title: "
            + df_feat["title"].astype(str)
            + ". Author: "
            + df_feat["author"].astype(str)
            + ". School: "
            + df_feat["school"].astype(str)
            + ". Summary: "
            + df_feat["summary"].astype(str)
            + ". Key Concepts: "
            + df_feat["key_concepts"].astype(str)
    )

    return df_feat


def run_feature_pipeline(
        input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH
) -> pd.DataFrame:
    print(f"[1/2] Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)

    print("[2/2] Generating content_to_embed feature...")
    featured_df = build_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(output_path, index=False)
    print(f"Feature dataset saved to {output_path}. Shape: {featured_df.shape}")

    return featured_df


if __name__ == "__main__":
    run_feature_pipeline()