# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import FlowInput, PredictionOut
from src.api.predictor import Predictor
from src.capture.live_capture import result_queue

app = FastAPI(title="Cyber Threat Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor()
import threading
from src.capture.live_capture import start_capture

@app.on_event("startup")
def _start_capture():
    threading.Thread(target=start_capture, kwargs={"interface": "lo0"}, daemon=True).start()
    

@app.post("/predict", response_model=PredictionOut)
async def predict(flow: FlowInput):
    """Accept a single flow's features and return a classification
    with SHAP top features and a RAG-grounded explanation."""
    try:
        result = predictor.predict_one(flow.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live-alerts")
async def live_alerts():
    """Return all pending real-time alerts from the live capture queue."""
    alerts = []
    while not result_queue.empty():
        alerts.append(result_queue.get_nowait())
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/health")
def health():
    return {"status": "ok", "models": ["rf", "xgb", "lstm"]}


# To run: uvicorn src.api.main:app --reload --port 8000