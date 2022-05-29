import joblib

from api import controllers


def init_model():
    if controllers.model is None:
        controllers.model = joblib.load("../data/models/census_model_v1")
    else:
        pass