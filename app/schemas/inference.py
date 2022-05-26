from typing import Any, List, Optional

from pydantic import BaseModel


class Inference(BaseModel):
    """Result of model inference"""

    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]


class Person(BaseModel):
    """Input model data schema"""

    age: Optional[int]
    workclass: Optional[str]
    fnight: Optional[str]
    education: Optional[str]
    education_num: Optional[int]
    marital_status: Optional[str]
    occupation: Optional[str]
    relationship: Optional[str]
    race: Optional[str]
    sex: Optional[str]
    capital_gain: Optional[int]
    capital_loss: Optional[int]
    hours_per_week: Optional[int]
    native_country: Optional[str]


class MultiplePerson(BaseModel):
    """Input model multiple data schema"""

    persons: List[Person]

    class Config:
        schema_extra = {
            "example": {
                "persons": [
                    {
                        "age": 39,
                        "workclass": "State-gov",
                        "fnlgt": 77516,
                        "education": "Bachelors",
                        "education-num": 13,
                        "marital-status": "Never-married",
                        "occupation": "Adm-clerical",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "Male",
                        "capital-gain": 2174,
                        "capital-loss": 0,
                        "hours-per-week": 40,
                        "native-country": "United-States",
                    }
                ]
            }
        }
