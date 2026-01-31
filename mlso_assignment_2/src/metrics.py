"""
Manual classification metrics computation from Spark DataFrame.
"""
from pyspark.sql.functions import col


def classification_metrics(preds):
    print(">>> ENTER METRICS - classification_metrics()")
    TP = preds.filter((col("Class") == 1) &
                      (col("prediction") == 1)).count()

    TN = preds.filter((col("Class") == 0) &
                      (col("prediction") == 0)).count()

    FP = preds.filter((col("Class") == 0) &
                      (col("prediction") == 1)).count()

    FN = preds.filter((col("Class") == 1) &
                      (col("prediction") == 0)).count()

    acc = (TP + TN) / (TP + TN + FP + FN)

    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0

    f1 = (2 * prec * rec) / (prec + rec) if prec + rec else 0

    print(">>> METRICS - classification_metrics() finished")

    return acc, prec, rec, f1
