# src/api/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict


class FlowInput(BaseModel):
    """A single network flow's features, keyed by the same column names
    used everywhere else in the project (feature_cols.pkl). Sending a dict
    instead of fixed fields means this works with whatever features your
    pipeline actually selected, without needing to keep two schemas in sync."""
    features: Dict[str, float]


class PredictionOut(BaseModel):
    label: str
    confidence: float
    risk_level: str
    top_features: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None