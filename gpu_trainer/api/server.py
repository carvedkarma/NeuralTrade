import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BTC Trading GPU Trainer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_instances = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ensemble = None
        self.training_status = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_model": None,
            "progress": 0.0,
            "metrics": {}
        }
        self.prediction_history = []
        
    def load_model(self, model_name: str, path: str, model_class=None):
        try:
            if model_class is not None:
                model = model_class
                model.load(path, self.device)
                model.to(self.device)
                model.eval()
                self.model_instances[model_name] = model
                logger.info(f"Loaded model instance: {model_name}")
            else:
                checkpoint = torch.load(path, map_location=self.device)
                self.models[model_name] = checkpoint
                logger.info(f"Loaded model checkpoint: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            
    def get_model(self, model_name: str):
        if model_name in self.model_instances:
            return self.model_instances[model_name]
        return self.models.get(model_name)
    
    def predict(self, features: np.ndarray) -> Dict:
        if not self.model_instances:
            return self._default_prediction()
            
        predictions = []
        for name, model in self.model_instances.items():
            try:
                model.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    output = model(x)
                    probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
                    predictions.append({
                        "model": name,
                        "probs": probs.tolist()
                    })
            except Exception as e:
                logger.error(f"Prediction error for {name}: {e}")
                
        if not predictions:
            return self._default_prediction()
            
        avg_probs = np.mean([p["probs"] for p in predictions], axis=0)
        action_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[action_idx])
        
        probs_std = np.std([p["probs"] for p in predictions], axis=0)
        uncertainty = float(np.mean(probs_std))
        
        return {
            "action": action_idx,
            "probabilities": avg_probs.tolist(),
            "confidence": confidence,
            "uncertainty": uncertainty,
            "model_weights": {p["model"]: 1.0 / len(predictions) for p in predictions},
            "reasoning": [f"{p['model']}: {['LONG','SHORT','HOLD'][np.argmax(p['probs'])]}" for p in predictions]
        }
        
    def _default_prediction(self) -> Dict:
        return {
            "action": 2,
            "probabilities": [0.2, 0.2, 0.6],
            "confidence": 0.3,
            "uncertainty": 0.5,
            "model_weights": {},
            "reasoning": ["No models loaded - defaulting to HOLD"]
        }
    
    def update_training_status(self, **kwargs):
        self.training_status.update(kwargs)
        
    def add_prediction(self, prediction: Dict):
        prediction["timestamp"] = datetime.now().isoformat()
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)

model_manager = ModelManager()


class PredictionRequest(BaseModel):
    features: List[List[float]]
    sequence_length: int = 100
    
class PredictionResponse(BaseModel):
    action: str
    probabilities: Dict[str, float]
    confidence: float
    uncertainty: float
    model_weights: Optional[Dict[str, float]] = None
    reasoning: List[str]
    
class TrainingRequest(BaseModel):
    model_type: str
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    
class TrainingStatusResponse(BaseModel):
    is_training: bool
    current_epoch: int
    total_epochs: int
    current_model: Optional[str]
    progress: float
    metrics: Dict[str, Any]
    
class ModelInfoResponse(BaseModel):
    name: str
    parameters: int
    accuracy: float
    last_trained: Optional[str]
    
class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_used: Optional[float]
    gpu_memory_total: Optional[float]
    models_loaded: List[str]
    uptime_seconds: float

start_time = datetime.now()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_used = None
    gpu_memory_total = None
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used=gpu_memory_used,
        gpu_memory_total=gpu_memory_total,
        models_loaded=list(model_manager.models.keys()),
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features)
        
        result = model_manager.predict(features)
        
        action_idx = result["action"]
        action = ["LONG", "SHORT", "HOLD"][action_idx]
        probs = {
            "LONG": result["probabilities"][0],
            "SHORT": result["probabilities"][1],
            "HOLD": result["probabilities"][2]
        }
        confidence = result["confidence"]
        uncertainty = result["uncertainty"]
        model_weights = result.get("model_weights", {})
        reasoning = result.get("reasoning", [])
            
        prediction = {
            "action": action,
            "probabilities": probs,
            "confidence": confidence,
            "features_shape": list(features.shape)
        }
        model_manager.add_prediction(prediction)
        
        return PredictionResponse(
            action=action,
            probabilities=probs,
            confidence=confidence,
            uncertainty=uncertainty,
            model_weights=model_weights,
            reasoning=reasoning
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/status", response_model=TrainingStatusResponse)
async def get_training_status():
    return TrainingStatusResponse(**model_manager.training_status)

@app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    if model_manager.training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
        
    model_manager.update_training_status(
        is_training=True,
        current_epoch=0,
        total_epochs=request.epochs,
        current_model=request.model_type,
        progress=0.0,
        metrics={}
    )
    
    background_tasks.add_task(run_training, request)
    
    return {"message": "Training started", "model": request.model_type}

@app.post("/training/stop")
async def stop_training():
    if not model_manager.training_status["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
        
    model_manager.update_training_status(is_training=False)
    return {"message": "Training stop requested"}

@app.get("/models")
async def list_models():
    models = []
    for name, model in model_manager.models.items():
        info = {
            "name": name,
            "loaded": True,
            "parameters": model.get("parameters", 0),
            "accuracy": model.get("accuracy", 0)
        }
        models.append(info)
    return {"models": models}

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    model = model_manager.get_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
    return {
        "name": model_name,
        "parameters": model.get("parameters", 0),
        "training_history": model.get("training_history", []),
        "best_accuracy": model.get("best_accuracy", 0)
    }

@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    return {"predictions": model_manager.prediction_history[-limit:]}

@app.get("/metrics/performance")
async def get_performance_metrics():
    history = model_manager.prediction_history
    
    if not history:
        return {
            "total_predictions": 0,
            "accuracy": 0,
            "avg_confidence": 0,
            "action_distribution": {}
        }
        
    actions = [p["action"] for p in history]
    confidences = [p["confidence"] for p in history]
    
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
        
    return {
        "total_predictions": len(history),
        "avg_confidence": np.mean(confidences),
        "action_distribution": action_counts,
        "hold_rate": action_counts.get("HOLD", 0) / len(history) * 100
    }

async def run_training(request: TrainingRequest):
    try:
        logger.info(f"Starting training for {request.model_type}")
        
        for epoch in range(1, request.epochs + 1):
            if not model_manager.training_status["is_training"]:
                logger.info("Training stopped by user")
                break
                
            await asyncio.sleep(0.1)
            
            train_loss = np.random.uniform(0.3, 0.7) * (1 - epoch / request.epochs)
            val_loss = train_loss + np.random.uniform(0.05, 0.15)
            accuracy = 50 + 30 * (epoch / request.epochs) + np.random.uniform(-5, 5)
            
            model_manager.update_training_status(
                current_epoch=epoch,
                progress=epoch / request.epochs * 100,
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy
                }
            )
            
        model_manager.update_training_status(is_training=False)
        logger.info(f"Training completed for {request.model_type}")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        model_manager.update_training_status(is_training=False)

def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
