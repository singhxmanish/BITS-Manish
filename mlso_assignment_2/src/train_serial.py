"""
Trains Spark-XGBoost in serial mode
by forcing a single partition.
"""

import time


def train_serial(df):
    """
    Serial training by repartitioning data into ONE partition.

    Args:
        df : training DataFrame

    Returns:
        trained_model
        elapsed_time (seconds)
    """

    print(">>> ENTER train_serial()")

    # Import XGBoost lazily to avoid blocking at module import time
    from xgboost.spark import SparkXGBClassifier

    # One partition => only one Spark task => serial execution
    df = df.repartition(1).cache()

    model = SparkXGBClassifier(
        features_col="features",
        label_col="Class",
        tree_method="hist",
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        eval_metric="auc"
    )

    start = time.time()
    trained_model = model.fit(df)
    end = time.time()

    print(">>> train_serial() finished")

    return trained_model, end - start
