"""
Handles:
- Loading dataset from local data directory if present
- Otherwise downloading via KaggleHub
- Copying into assignment data folder
- Feature engineering
- Train / Test split
"""

import os
import shutil
from pathlib import Path
from pyspark.ml.feature import VectorAssembler

# ============================================================
# RESOLVE PROJECT ROOT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_data(spark):
    """
    Loads the credit card fraud dataset into Spark.

    Priority:
        1) mlso_assignment_2/data/creditcard.csv
        2) Download from KaggleHub if missing

    Returns:
        train_df : Spark DataFrame
        test_df  : Spark DataFrame
    """

    print(">>> ENTER load_data()")

    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    csv_path = DATA_DIR / "creditcard.csv"

    # --------------------------------------------------------
    # Download dataset ONLY if missing locally
    # --------------------------------------------------------
    if not csv_path.exists():

        print(">>> Dataset not found locally — downloading from KaggleHub...")

        # Lazy import to avoid blocking at module load time
        import kagglehub

        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

        downloaded_csv = Path(path) / "creditcard.csv"

        print(">>> Copying dataset into data directory...")

        shutil.copy(downloaded_csv, csv_path)

    else:
        print(">>> Using local dataset:", csv_path)

    # --------------------------------------------------------
    # Load CSV into Spark DataFrame
    # --------------------------------------------------------
    print(">>> Reading CSV into Spark...")

    df = spark.read.csv(
        str(csv_path),
        header=True,
        inferSchema=True
    )

    # --------------------------------------------------------
    # Assemble feature vector
    # --------------------------------------------------------
    feature_cols = [c for c in df.columns if c != "Class"]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    df = assembler.transform(df).select("features", "Class")

    # --------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print(">>> load_data() finished")

    return train_df, test_df
