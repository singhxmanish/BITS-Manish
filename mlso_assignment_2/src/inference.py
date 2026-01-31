"""
Runs inference on test data
and computes metrics + AUC.
"""

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from metrics import classification_metrics


def evaluate_model(model, test_df):
    print(">>> ENTER INFERENCE - evaluate_model()")
    preds = model.transform(test_df)

    acc, prec, rec, f1 = classification_metrics(preds)

    evaluator = BinaryClassificationEvaluator(
        labelCol="Class",
        rawPredictionCol="probability",
        metricName="areaUnderROC"
    )

    auc = evaluator.evaluate(preds)
    print(">>> INFERENCE - evaluate_model() finished")
    return acc, prec, rec, f1, auc
