import os
import socket
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "./model"))
POD_NAME = socket.gethostname()
CHAOS_ENABLED = True
state = {"model": None, "version": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state['model'] = mlflow.pyfunc.load_model(MODEL_PATH)
    version_file = MODEL_PATH / "VERSION"
    if version_file.exists():
        with open(version_file, "r") as f:
            state['version'] = f.read().strip()
    else:
        state['version'] = "unknown"
    yield

app = FastAPI(title = "iris_classifier",lifespan=lifespan)


class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float



@app.get("/")
def info():
    return {"model_version": state["version"], "pod_name": POD_NAME}

@app.get("/ready")
def ready():
    if state["model"] is not None:
        return {"status": "ready"}
    return {"status": "not ready"}

@app.get("/healthz")
def healthz():
    return {"status": "alive"}

@app.post("/predict")
def predict(input_data: InputData):
    input_array = np.array([[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]])
    prediction = state["model"].predict(input_array)
    return {"prediction": int(prediction[0])}
def require_chaos() -> None:
    if not CHAOS_ENABLED:
        raise HTTPException(status_code=404, detail="Not Found")


@app.post("/chaos/crash")
def chaos_crash():
    """Kill the process after replying, like an unhandled fatal error."""
    require_chaos()
    threading.Timer(0.5, os._exit, args=[1]).start()
    return {"action": "crashing", "pod": POD_NAME}


@app.post("/chaos/hang")
def chaos_hang():
    """Start failing liveness checks; Kubernetes restarts the container in ~30s."""
    require_chaos()
    state["healthy"] = False
    return {"action": "failing liveness", "pod": POD_NAME}


@app.post("/chaos/unready")
def chaos_unready():
    """Start failing readiness; the Service stops routing here, no restart."""
    require_chaos()
    state["ready"] = False
    return {"action": "marked unready", "pod": POD_NAME}


@app.post("/chaos/recover")
def chaos_recover():
    require_chaos()
    state["healthy"] = state["ready"] = True
    return {"action": "recovered", "pod": POD_NAME}

if __name__ =="__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
