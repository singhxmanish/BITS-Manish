"""
Parallel Spark-XGBoost training
using multiple data partitions.
"""

import time


def train_parallel(df, partitions=8):
    """
    Parallel training by repartitioning data
    across multiple Spark tasks.
    """

    print(">>> ENTER train_parallel()")

    # Lazy import
    from xgboost.spark import SparkXGBClassifier

    df = df.repartition(partitions).cache()

    model = SparkXGBClassifier(
        features_col="features",
        label_col="Class",
        num_workers=1,
        tree_method="hist",
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        eval_metric="auc"
    )

    start = time.time()
    trained_model = model.fit(df)
    end = time.time()

    print(">>> train_parallel() finished")

    return trained_model, end - start
