from typing import Any, List, Optional

from pydantic import BaseModel


class Prediction(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]
