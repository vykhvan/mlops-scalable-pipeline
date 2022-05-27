"""Schemas for API responses."""
from typing import List, Optional

from pydantic import BaseModel


class Person(BaseModel):
    """Person data model"""

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


class Persons(BaseModel):
    """Persons data model"""

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
                        "education_num": 13,
                        "marital_status": "Never-married",
                        "occupation": "Adm-clerical",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "Male",
                        "capital_gain": 2174,
                        "capital_loss": 0,
                        "hours_per_week": 40,
                        "native_country": "United-States",
                    }
                ]
            }
        }
