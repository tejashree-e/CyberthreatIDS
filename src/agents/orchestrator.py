import numpy as np
import joblib

from src.models.ensemble import EnsemblePredictor
from src.models.explain import explain_single
from src.rag.knowledge_base import KnowledgeRetriever, generate_explanation


class DetectionAgent:
    def __init__(self):
        self.ensemble = EnsemblePredictor()
        self.scaler = joblib.load("data/models/scaler.pkl")
        self.feature_cols = joblib.load("data/models/feature_cols.pkl")
        self.le = joblib.load("data/models/label_encoder.pkl")

    def detect(self, feature_dict: dict) -> dict:
        row = np.array([feature_dict.get(c, 0.0) for c in self.feature_cols])
        row_scaled = self.scaler.transform(row.reshape(1, -1))
        result = self.ensemble.predict(row_scaled)[0]
        result["_row_scaled"] = row_scaled[0]  # internal, used by ExplanationAgent
        result["_predicted_idx"] = int(self.le.transform([result["label"]])[0])
        return result


class KnowledgeAgent:
    def __init__(self):
        self.retriever = KnowledgeRetriever()

    def lookup(self, predicted_label: str) -> list:
        return self.retriever.retrieve(predicted_label)


class ExplanationAgent:
    def __init__(self, knowledge_agent: KnowledgeAgent):
        self.knowledge_agent = knowledge_agent
        self.feature_cols = joblib.load("data/models/feature_cols.pkl")

    def explain(self, detection_result: dict) -> dict:
        try:
            shap_top = explain_single(
                detection_result["_row_scaled"],
                self.feature_cols,
                detection_result["_predicted_idx"],
            )
        except Exception:
            shap_top = {}

        explanation_text = generate_explanation(
            detection_result["label"],
            detection_result["confidence"],
            shap_top,
            self.knowledge_agent.retriever,
        )
        return {"top_features": shap_top, "explanation": explanation_text}


class Orchestrator:
    """The coordinator that a demo/report can point at as 'the multi-agent
    system'. Swap CaptureAgent for live_capture.py or wireshark_import.py
    depending on where the flow feature dict came from."""

    def __init__(self):
        self.detection_agent = DetectionAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.explanation_agent = ExplanationAgent(self.knowledge_agent)

    def process_flow(self, feature_dict: dict) -> dict:
        detection = self.detection_agent.detect(feature_dict)
        explanation = self.explanation_agent.explain(detection)

        detection.pop("_row_scaled", None)
        detection.pop("_predicted_idx", None)

        return {**detection, **explanation}


if __name__ == "__main__":
    # Smoke test with a dummy all-zero flow — replace with a real captured flow.
    orch = Orchestrator()
    dummy_flow = {c: 0.0 for c in joblib.load("data/models/feature_cols.pkl")}
    print(orch.process_flow(dummy_flow))