"""
Evaluation step
"""
import json
import logging

import joblib
import pandas as pd
import yaml
from sklearn.metrics import fbeta_score, precision_score, recall_score

from train import Pipeline  # noqa F401

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s - %(levelname)s - %(message)s"
)


def compute_model_metrics(y, preds):
    """Validates the trained machine learning model
    using precision, recall, and F1.

    Args:
        y: Known labels, binarized.
        preds: Predicted labels, binarized.

    Returns:
        precision: float
        recall: float
        fbeta: float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_slice_metrics(test_data, cat_features, model, slice_metrics):
    """Validates the trained machine learning model
    with slice metrics and write in text file.

    Args:
        test_data: Test data for computes model slice metrics.
        cat_features: List with categorical features.
        model: Artifact of trained model.
        report: Path to reporting.

    Returns:
        None
    """
    slice_metrics_data = {}
    logging.info("Run compute slice metrics")
    for feature in cat_features:
        slice_metrics_data.update({feature: {}})
        for value in test_data[feature].unique():
            df_temp = test_data[test_data[feature] == value]
            X_test = df_temp.drop(["salary"], axis=1)
            y_test = df_temp["salary"]
            y_test = model.lb.transform(y_test)
            preds = model.inference(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            scores = {value: {"precision": precision, "recall": recall, "fbeta": fbeta}}
            slice_metrics_data[feature].update(scores)

    slice_metrics_json = json.dumps(slice_metrics_data)
    with open(slice_metrics, "w") as file:
        file.write(slice_metrics_json)


def main(params):
    test_data = pd.read_csv(params["segregation"]["test_data"])
    cat_features = params["train"]["categorical_features"]
    model = joblib.load(params["train"]["model"])
    slice_metrics = params["evaluate"]["slice_metrics"]
    metrics = params["evaluate"]["metrics"]

    X_test = test_data.drop(["salary"], axis=1)
    y_test = test_data["salary"]
    y_test = model.lb.transform(y_test)

    preds = model.inference(X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    logging.info("fbeta: %s | precision: %s | recall: %s", fbeta, precision, recall)

    metrics_data = {"precision": precision, "recall": recall, "fbeta": fbeta}
    metrics_json = json.dumps(metrics_data)

    with open(metrics, "w") as file:
        file.write(metrics_json)

    compute_model_slice_metrics(test_data, cat_features, model, slice_metrics)


if __name__ == "__main__":
    with open("params.yaml", mode="r", encoding="utf8") as conf_file:
        config = yaml.safe_load(conf_file)
    main(config)
