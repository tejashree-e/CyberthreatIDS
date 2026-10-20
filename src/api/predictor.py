# src/api/predictor.py
"""
Thin wrapper around the Orchestrator (Detection + Knowledge + Explanation
agents) so the API layer has one simple call: predict_one(flow_dict).
"""

from src.agents.orchestrator import Orchestrator


class Predictor:
    def __init__(self):
        self.orchestrator = Orchestrator()

    def predict_one(self, flow_dict: dict) -> dict:
        return self.orchestrator.process_flow(flow_dict)