# server.py (no auth) - Water Requirement Predictor API
from __future__ import annotations
import numpy as np, joblib, os, tensorflow as tf
from typing import List, Annotated
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# ===== CONFIG =====
MODEL_PATH = "export/cnn_lstm_best.keras"
X_SCALER_PATH = "export/X_scaler.pkl"
Y_SCALER_PATH = "export/y_scaler.pkl"
LOOKBACK = 14
FEATURES = ["day", "temperature_C", "humidity_pct"]
N_FEATS = len(FEATURES)

# ===== LOAD ARTIFACTS =====
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    Xsc = joblib.load(X_SCALER_PATH)
    Ysc = joblib.load(Y_SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scalers: {e}")

# ===== APP =====
app = FastAPI(
    title="Green Gram Water Requirement Predictor",
    description=(
        "Predict daily irrigation water requirement (L/m²/day) from "
        "[day, temperature_C, humidity_pct].\n"
        "Auth is DISABLED for simplicity."
    ),
    version="1.0.1",
)

# CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ===== Schemas =====
WindowRow = Annotated[List[float], Field(min_length=N_FEATS, max_length=N_FEATS)]
Window = Annotated[List[WindowRow], Field(min_length=LOOKBACK, max_length=LOOKBACK)]

class PredictIn(BaseModel):
    window: Window = Field(..., description=f"{LOOKBACK} rows of [day, temperature_C, humidity_pct]")
    area_m2: float | None = Field(None, ge=0, description="Optional area to get pump seconds")
    flow_lpm: float | None = Field(None, gt=0, description="Optional pump flow to get seconds")
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "window": [
                [34,30.1,69],[35,29.9,70],[36,30.3,68],[37,30.0,71],
                [38,30.6,67],[39,30.4,69],[40,30.2,70],[41,30.1,69],
                [42,29.8,72],[43,30.0,70],[44,30.5,68],[45,30.7,66],
                [46,31.0,65],[47,31.2,64]
            ],
            "area_m2": 1.0, "flow_lpm": 3.5
        }]
    })

class PredictSingleIn(BaseModel):
    day: float
    temperature_C: float
    humidity_pct: float
    area_m2: float | None = Field(None, ge=0)
    flow_lpm: float | None = Field(None, gt=0)
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"day":47,"temperature_C":30.5,"humidity_pct":68,"area_m2":1.0,"flow_lpm":3.5}]
    })

class PredictOut(BaseModel):
    predicted_lpm2: float
    seconds_for_area: float | None = None
    model: str = "CNN_LSTM"
    lookback: int = LOOKBACK

class MetaOut(BaseModel):
    model: str
    lookback: int
    n_features: int
    features: List[str]
    y_unit: str = "L/m²/day"

# ===== Helpers =====
def _predict_from_window(window: np.ndarray) -> float:
    Xs = Xsc.transform(window.astype(np.float32))          # (LOOKBACK, 3)
    Xs = Xs.reshape(1, LOOKBACK, N_FEATS)                  # (1, L, 3)
    y_scaled = model.predict(Xs, verbose=0).ravel()[0]
    y = float(Ysc.inverse_transform([[y_scaled]]).ravel()[0])
    return max(0.0, min(y, 20.0))  # safety clamp

def _seconds_for_area(lpm2: float, area_m2: float, flow_lpm: float) -> float:
    total_L = lpm2 * area_m2
    secs = (total_L / max(1e-6, flow_lpm)) * 60.0
    return float(max(0.0, secs))

# ===== Routes =====
@app.get("/", tags=["info"])
def root():
    return {"service": "Green Gram Predictor", "docs": "/docs"}

@app.get("/health", tags=["info"])
def health():
    return {"status": "ok", "tensorflow": tf.__version__}

@app.get("/metadata", response_model=MetaOut, tags=["info"])
def metadata():
    return MetaOut(model="CNN_LSTM", lookback=LOOKBACK, n_features=N_FEATS, features=FEATURES)

@app.post("/predict", response_model=PredictOut, tags=["predict"])
def predict(payload: PredictIn):
    try:
        window = np.array(payload.window, dtype=np.float32)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid window format")
    if window.shape != (LOOKBACK, N_FEATS):
        raise HTTPException(status_code=422, detail=f"window must have shape ({LOOKBACK}, {N_FEATS})")

    y = _predict_from_window(window)
    secs = None
    if payload.area_m2 is not None and payload.flow_lpm is not None:
        secs = _seconds_for_area(y, payload.area_m2, payload.flow_lpm)
    return PredictOut(predicted_lpm2=round(y,3),
                      seconds_for_area=None if secs is None else round(secs,1))

@app.post("/predict_single", response_model=PredictOut, tags=["predict"])
def predict_single(payload: PredictSingleIn):
    row = np.array([[payload.day, payload.temperature_C, payload.humidity_pct]], dtype=np.float32)
    window = np.repeat(row, LOOKBACK, axis=0)
    y = _predict_from_window(window)
    secs = None
    if payload.area_m2 is not None and payload.flow_lpm is not None:
        secs = _seconds_for_area(y, payload.area_m2, payload.flow_lpm)
    return PredictOut(predicted_lpm2=round(y,3),
                      seconds_for_area=None if secs is None else round(secs,1))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
