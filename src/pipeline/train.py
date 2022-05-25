"""
Training step
"""
import logging

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s - %(levelname)s - %(message)s"
)


class Pipeline:
    """Census ML Pipeline.

    Processing data, train model and inference.
    """

    def __init__(self, params):
        """Initialize object."""
        logging.info("Initialize ML Pipeline with params: %s", params)
        self.categorical_features = params["categorical_features"]
        self.label = params["label"]
        self.loss = params["loss"]
        self.learning_rate = params["learning_rate"]
        self.max_depth = params["max_depth"]
        self.random_state = params["random_state"]
        self.lb = None
        self.encoder = None
        self.model = None

    def process_data(self, X, categorical_features, label=None, training=True):
        """Process the data used in the machine learning pipeline.

        Processes the data using one hot encoding for the categorical
        features and a  label binarizer for the labels. This can be
        used in either training or
        inference/validation.

        Args:
            X: Dataframe containing the features and label.
               Columns in `categorical_features`
            categorical_features: List containing the names of
                                  the categorical features (default=[])
            label: Name of the label column in `X`. If None,
                   then an empty array will be returned
                   for y (default=None)
            training: Indicator if training mode or inference/validation mode.

        Returns:
            X: Processed data.
            y: Processed labels if labeled=True, otherwise empty np.array.
        """
        if label is not None:
            y = X[label]
            X = X.drop([label], axis=1)
        else:
            y = np.array([])

        X_categorical = X[categorical_features].values
        X_continuous = X.drop(*[categorical_features], axis=1)

        if training is True:
            self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.lb = LabelBinarizer()
            X_categorical = self.encoder.fit_transform(X_categorical)
            y = self.lb.fit_transform(y.values).ravel()
        else:
            X_categorical = self.encoder.transform(X_categorical)
            try:
                y = self.lb.transform(y.values).ravel()
            # Catch the case where y is None because we're doing inference.
            except AttributeError:
                pass

        X = np.concatenate([X_continuous, X_categorical], axis=1)
        return X, y

    def train_model(self, X, y):
        """Trains a machine learning model and returns it.

        Args:
            X: Training data.
            y: Labels.

        Returns:
            model: Trained machine learning model.
        """
        logging.info("Training model ...")
        self.model = HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

    def inference(self, X):
        """Run model inferences and return the predictions.

        Args:
            X: Data used for prediction.

        Returns:
            preds: Predictions from the model.
        """

        X, _ = self.process_data(X, self.categorical_features, training=False)

        predictions = self.model.predict(X)
        return predictions

    def fit(self, X):
        """Build ML Pipeline.

        Args:
            X: Train data.
        """
        logging.info("Train step - Running")
        logging.info("Process data input shape: %s", X.shape)
        X, y = self.process_data(X, self.categorical_features, self.label)
        logging.info("Process data output shape: %s", X.shape)
        self.train_model(X, y)
        logging.info("Train step - completed")


def main(params):
    """Create object of Pipeline and run building model."""
    census_pipeline = Pipeline(params["train"])
    train_data = pd.read_csv(params["segregation"]["train_data"])
    census_pipeline.fit(train_data)
    joblib.dump(census_pipeline, params["train"]["model"])


if __name__ == "__main__":
    with open("params.yaml", mode="r", encoding="utf8") as conf_file:
        config = yaml.safe_load(conf_file)
    main(config)
