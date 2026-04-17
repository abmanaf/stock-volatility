import sqlite3

from config import settings
from data import SQLRepository
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import GarchModel
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional
import psutil
import time
import os


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# `FitIn` class
class FitIn(BaseModel):
    ticker: str
    use_new_data: bool
    n_observations: int
    p: int
    q: int
    arima_order: tuple = (1, 0, 1)


# `FitOut` class
class FitOut(FitIn):
    success: bool
    message: str


# `PredictIn` class
class PredictIn(BaseModel):
    ticker: str
    n_days: int
    predict_type: str = "volatility"


# `PredictOut` class
class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str


def build_model(ticker, use_new_data):
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)
    repo = SQLRepository(connection=connection)
    model = GarchModel(ticker=ticker, use_new_data=use_new_data, repo=repo)
    return model


@app.get("/hello", status_code=200)
def hello():
    return {"message":"Hello world"}


@app.post("/fit", response_model=FitOut)
def fit_model(request: FitIn):
    response = request.dict()
    try:
        model = build_model(ticker=request.ticker, use_new_data=request.use_new_data)
        model.wrangle_data(n_observations=request.n_observations)
        model.fit(p=request.p, q=request.q)
        model.fit_arima(order=request.arima_order)
        filepath = model.dump()
        response["success"] = True
        response["message"] = f"Trained and saved '{filepath}'. Metrics: AIC {model.aic}, BIC {model.bic}."
    except Exception as e:
        response["success"] = False
        response["message"] = str(e)
    return response


@app.post("/predict", status_code=200, response_model=PredictOut)
def get_prediction(request: PredictIn):
    response = request.dict()
    try:
        model = build_model(ticker=request.ticker, use_new_data=False)
        model.load()
        if request.predict_type == "returns":
            prediction = model.predict_returns(horizon=request.n_days)
        else:
            prediction = model.predict_volatility(horizon=request.n_days)
        response["success"] = True
        response["forecast"] = prediction
        response["message"] = ""
    except Exception as e:
        response["success"] = False
        response["forecast"] = {}
        response["message"] = str(e)
    return response

@app.get("/api/system/status")
async def get_system_status():
    """Check actual system health"""
    
    # 1. Check API health (measure actual latency)
    start_time = time.time()
    api_healthy = True
    latency = 0
    try:
        # Quick DB check to verify connectivity
        connection = sqlite3.connect(settings.db_name, check_same_thread=False)
        connection.execute("SELECT 1")
        connection.close()
        latency = round((time.time() - start_time) * 1000, 2)  # ms
    except Exception:
        api_healthy = False
    
    # 2. Check if models exist and are valid
    model_status = check_model_status()
    
    # 3. Check system resources
    engine_status = {
        "status": "running" if api_healthy else "degraded",
        "currentModel": get_current_model_info(),
        # will later read this from a version file or env
        "version": "1.0.0",  
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    }
    
    return {
        "api": {
            "status": "online" if api_healthy else "offline",
            "latency": latency,
            "lastChecked": datetime.now().isoformat()
        },
        "model": model_status,
        "engine": engine_status,
        "timestamp": datetime.now().isoformat()
    }


def check_model_status() -> Dict[str, Any]:
    """Actually check if we have trained models"""
    try:
        # Check if models directory exists
        models_dir = "models"
        if not os.path.exists(models_dir):
            return {
                "status": "no_models",
                "modelType": None,
                "lastTrained": None,
                "availableModels": []
            }
        
        # Find all trained models
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            return {
                "status": "no_models",
                "modelType": None,
                "lastTrained": None,
                "availableModels": []
            }
        
        # Get most recent model info
        latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
        model_path = os.path.join(models_dir, latest_model)
        last_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
        
        # Try to extract ticker from filename (assuming naming convention like "AAPL_model.pkl")
        ticker = latest_model.split('_')[0] if '_' in latest_model else "Unknown"
        
        return {
            "status": "ready",
            "modelType": f"GARCH for {ticker}",
            "lastTrained": last_modified.isoformat(),
            "availableModels": [f.replace('.pkl', '') for f in model_files],
            "totalModels": len(model_files)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "modelType": None,
            "lastTrained": None,
            "error": str(e)
        }


def get_current_model_info() -> Optional[str]:
    """Get info about the currently active model"""
    try:
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if model_files:
                latest = max(model_files, key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
                ticker = latest.split('_')[0]
                return f"GARCH - {ticker}"
    except:
        pass
    return "No active model"

@app.get("/health")
async def health_check():
    """Simple health check for monitoring"""
    try:
        # Check database connection
        connection = sqlite3.connect(settings.db_name, check_same_thread=False)
        connection.execute("SELECT 1")
        connection.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }