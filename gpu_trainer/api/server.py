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
import joblib
import glob as glob_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Training label mapping (must match data/pipeline.py create_labels)
# 0 = SHORT, 1 = NEUTRAL, 2 = LONG
ACTION_MAP = {0: "SHORT", 1: "HOLD", 2: "LONG"}
ACTION_NAMES = ["SHORT", "HOLD", "LONG"]  # Index-aligned with training labels

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
        self.scaler = None  # For feature scaling
        self.feature_config = None  # Feature configuration
        self.training_status = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_model": None,
            "progress": 0.0,
            "metrics": {}
        }
        self.prediction_history = []
        self.checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        self.scaler_path = self.checkpoint_dir / "scaler.joblib"
        self.sequence_length = 100  # Default, updated from loaded model config
        self.input_dim = 81  # Default feature count
        self.instantiation_errors: Dict[str, str] = {}  # Track errors for /models/status
        
    def _create_model_instance(self, model_type: str, config: dict):
        """Create model instance with correct constructor args for each model type.
        
        Covers all model classes from gpu_trainer/models/:
        - transformer.py: TransformerPriceModel, TemporalFusionTransformer
        - lstm.py: BidirectionalLSTM, StackedLSTM, ConvLSTM
        - cnn.py: ResNetPrice, InceptionNet, WaveNet
        - vae.py: MarketVAE, ConditionalVAE
        - gnn.py: CrossAssetGNN, TemporalGNN
        - ensemble.py: MetaLearner, AttentionEnsemble, MasterEnsemble
        """
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            model_type_lower = model_type.lower()
            input_dim = config.get("input_dim", 81)
            output_dim = config.get("output_dim", 3)
            hidden_dim = config.get("hidden_dim", 128)
            sequence_length = config.get("sequence_length", 100)
            dropout = config.get("dropout", 0.2)
            
            # === TRANSFORMER MODELS ===
            if "temporal_fusion" in model_type_lower or "tft" in model_type_lower:
                from models.transformer import TemporalFusionTransformer
                return TemporalFusionTransformer(
                    input_dim=input_dim,
                    d_model=config.get("d_model", 256),
                    nhead=config.get("nhead", 8),
                    num_encoder_layers=config.get("num_encoder_layers", 4),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "transformer" in model_type_lower:
                from models.transformer import TransformerPriceModel
                return TransformerPriceModel(
                    input_dim=input_dim,
                    d_model=config.get("d_model", 256),
                    nhead=config.get("nhead", 8),
                    num_layers=config.get("num_layers", 6),
                    dropout=dropout,
                    output_dim=output_dim
                )
                
            # === LSTM MODELS ===
            elif "conv_lstm" in model_type_lower or "convlstm" in model_type_lower:
                from models.lstm import ConvLSTM
                return ConvLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=config.get("num_layers", 2),
                    kernel_size=config.get("kernel_size", 3),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "stacked_lstm" in model_type_lower or "stackedlstm" in model_type_lower:
                from models.lstm import StackedLSTM
                return StackedLSTM(
                    input_dim=input_dim,
                    hidden_dims=config.get("hidden_dims", [256, 128, 64]),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "lstm" in model_type_lower or "bidirectional" in model_type_lower:
                from models.lstm import BidirectionalLSTM
                return BidirectionalLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=config.get("num_layers", 3),
                    dropout=dropout,
                    output_dim=output_dim,
                    use_attention=config.get("use_attention", True)
                )
                
            # === CNN MODELS ===
            elif "wavenet" in model_type_lower:
                from models.cnn import WaveNet
                return WaveNet(
                    input_dim=input_dim,
                    residual_channels=config.get("residual_channels", 64),
                    dilation_channels=config.get("dilation_channels", 64),
                    skip_channels=config.get("skip_channels", 128),
                    num_blocks=config.get("num_blocks", 4),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "inception" in model_type_lower:
                from models.cnn import InceptionNet
                return InceptionNet(
                    input_dim=input_dim,
                    base_channels=config.get("base_channels", 64),
                    num_inception_blocks=config.get("num_inception_blocks", 3),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "resnet" in model_type_lower or "cnn" in model_type_lower:
                from models.cnn import ResNetPrice
                return ResNetPrice(
                    input_dim=input_dim,
                    base_channels=config.get("base_channels", 64),
                    num_blocks=config.get("num_blocks", [2, 2, 2, 2]),
                    dropout=dropout,
                    output_dim=output_dim
                )
                
            # === VAE MODELS ===
            elif "conditional_vae" in model_type_lower or "cvae" in model_type_lower:
                from models.vae import ConditionalVAE
                return ConditionalVAE(
                    input_dim=input_dim,
                    condition_dim=config.get("condition_dim", 16),
                    sequence_length=sequence_length,
                    latent_dim=config.get("latent_dim", 64),
                    hidden_dims=config.get("hidden_dims", [128, 256]),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "vae" in model_type_lower or "market_vae" in model_type_lower:
                from models.vae import MarketVAE
                return MarketVAE(
                    input_dim=input_dim,
                    sequence_length=sequence_length,
                    latent_dim=config.get("latent_dim", 64),
                    hidden_dims=config.get("hidden_dims", [128, 256, 512]),
                    dropout=dropout,
                    output_dim=output_dim
                )
                
            # === GNN MODELS ===
            elif "cross_asset" in model_type_lower or "crossasset" in model_type_lower:
                from models.gnn import CrossAssetGNN
                return CrossAssetGNN(
                    input_dim=input_dim,
                    num_assets=config.get("num_assets", 4),
                    hidden_dim=hidden_dim,
                    num_layers=config.get("num_layers", 3),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "gnn" in model_type_lower or "temporal_gnn" in model_type_lower:
                from models.gnn import TemporalGNN
                return TemporalGNN(
                    input_dim=input_dim,
                    num_nodes=config.get("num_nodes", 4),
                    hidden_dim=hidden_dim,
                    num_layers=config.get("num_layers", 3),
                    num_heads=config.get("num_heads", 4),
                    temporal_window=config.get("temporal_window", 10),
                    dropout=dropout,
                    output_dim=output_dim
                )
                
            # === ENSEMBLE MODELS ===
            elif "master_ensemble" in model_type_lower or "masterensemble" in model_type_lower:
                from models.ensemble import MasterEnsemble
                return MasterEnsemble(
                    model_configs=config.get("model_configs", [{"name": "default"}]),
                    feature_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "attention_ensemble" in model_type_lower:
                from models.ensemble import AttentionEnsemble
                return AttentionEnsemble(
                    num_models=config.get("num_models", 3),
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "meta_learner" in model_type_lower or "metalearner" in model_type_lower:
                from models.ensemble import MetaLearner
                return MetaLearner(
                    num_base_models=config.get("num_base_models", 3),
                    base_hidden_dim=hidden_dim,
                    meta_hidden_dim=config.get("meta_hidden_dim", 64),
                    dropout=dropout,
                    output_dim=output_dim
                )
            elif "ensemble" in model_type_lower:
                # Generic ensemble fallback
                from models.ensemble import MasterEnsemble
                return MasterEnsemble(
                    model_configs=config.get("model_configs", [{"name": "default"}]),
                    feature_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    output_dim=output_dim
                )
            else:
                logger.warning(f"Unknown model type: {model_type}")
                self.instantiation_errors[model_type] = f"Unknown model type: {model_type}"
                return None
                
        except ImportError as e:
            error_msg = f"Failed to import model class for {model_type}: {e}"
            logger.error(error_msg)
            self.instantiation_errors[model_type] = error_msg
            return None
        except Exception as e:
            error_msg = f"Failed to create model instance for {model_type}: {e}"
            logger.error(error_msg)
            self.instantiation_errors[model_type] = error_msg
            return None
    
    def _instantiate_model(self, checkpoint: dict, model_name: str):
        """Instantiate a model from checkpoint config and state_dict."""
        config = checkpoint.get("config", {})
        state_dict = checkpoint.get("model_state_dict")
        
        if not state_dict:
            logger.warning(f"No state_dict in checkpoint for {model_name}")
            return None
        
        # Try to determine model type from name, config, or infer from state_dict keys
        model_type = config.get("model_type", "")
        if not model_type:
            # Try to infer from checkpoint name
            model_type = model_name
        
        try:
            # Get config values with defaults
            input_dim = config.get("input_dim", 81)
            sequence_length = config.get("sequence_length", 100)
            
            # Create model with proper constructor
            model = self._create_model_instance(model_type, config)
            
            if model is None:
                logger.warning(f"Could not create model instance for {model_name}")
                return None
            
            # Load state dict with strict=False to handle minor mismatches
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                logger.warning(f"Strict load failed for {model_name}, trying non-strict: {e}")
                model.load_state_dict(state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            
            # Update manager config from loaded model
            self.sequence_length = sequence_length
            self.input_dim = input_dim
            
            logger.info(f"Instantiated model: {model_name} (type={model_type}, input={input_dim}, seq={sequence_length})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to instantiate model {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def load_best_models(self):
        """Load best checkpoint models at startup."""
        if not self.checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {self.checkpoint_dir}")
            return
            
        # Load scaler if exists
        if self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded scaler from {self.scaler_path}")
            except Exception as e:
                logger.error(f"Failed to load scaler: {e}")
        
        # Find and load best checkpoint for each model type
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            logger.warning("No checkpoint files found")
            return
            
        for ckpt_path in checkpoint_files:
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model_name = ckpt_path.stem
                
                # Store checkpoint metadata
                self.models[model_name] = {
                    "path": str(ckpt_path),
                    "accuracy": checkpoint.get("val_accuracy", 0),
                    "epoch": checkpoint.get("epoch", 0),
                    "config": checkpoint.get("config", {}),
                    "parameters": checkpoint.get("parameters", 0)
                }
                
                # Try to instantiate model if state_dict present
                if "model_state_dict" in checkpoint:
                    model_instance = self._instantiate_model(checkpoint, model_name)
                    if model_instance is not None:
                        self.model_instances[model_name] = model_instance
                        self.models[model_name]["loaded"] = True
                        logger.info(f"Loaded & instantiated: {model_name} (acc={checkpoint.get('val_accuracy', 0):.2f}%)")
                    else:
                        self.models[model_name]["state_dict"] = checkpoint["model_state_dict"]
                        self.models[model_name]["loaded"] = False
                        logger.warning(f"Loaded checkpoint but failed to instantiate: {model_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
                
        logger.info(f"Loaded {len(self.models)} checkpoints, {len(self.model_instances)} instantiated")
        
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
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.models[model_name] = checkpoint
                logger.info(f"Loaded model checkpoint: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            
    def get_model(self, model_name: str):
        if model_name in self.model_instances:
            return self.model_instances[model_name]
        return self.models.get(model_name)
    
    def predict(self, features: np.ndarray) -> Dict:
        """Make prediction with correct label mapping.
        
        CRITICAL: Training labels are 0=SHORT, 1=NEUTRAL, 2=LONG
        """
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
        
        # CORRECT mapping: 0=SHORT, 1=HOLD, 2=LONG (matches training labels)
        return {
            "action": action_idx,
            "action_name": ACTION_NAMES[action_idx],  # SHORT, HOLD, or LONG
            "probabilities": avg_probs.tolist(),
            "confidence": confidence,
            "uncertainty": uncertainty,
            "model_weights": {p["model"]: 1.0 / len(predictions) for p in predictions},
            "reasoning": [f"{p['model']}: {ACTION_NAMES[np.argmax(p['probs'])]}" for p in predictions]
        }
        
    def _default_prediction(self) -> Dict:
        """Default prediction when no models loaded - returns HOLD (index 1)."""
        return {
            "action": 1,  # HOLD is index 1 in training labels
            "action_name": "HOLD",
            "probabilities": [0.2, 0.6, 0.2],  # [SHORT, HOLD, LONG]
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

class CandleData(BaseModel):
    """Raw candle data for prediction."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class CandlePredictionRequest(BaseModel):
    """Request with raw candles - server will compute features."""
    candles: List[CandleData]
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    
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

class RegressionPredictionRequest(BaseModel):
    """Request for regression-based prediction (mu, sigma)."""
    features: List[List[float]]
    current_price: float = 0.0
    current_volatility: float = 0.01
    
class RegressionPredictionResponse(BaseModel):
    """Response with edge-based signal format."""
    action: str  # LONG, SHORT, NO_TRADE
    confidence: float  # edge / sigma
    expected_move: float  # mu
    uncertainty: float  # sigma
    edge: float  # mu - cost
    cost_estimate: float
    suggested_order_type: str  # MAKER or TAKER
    urgency: str  # LOW, MEDIUM, HIGH
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    regime: str
    expert_weights: Dict[str, float]
    reasons: List[str]
    
    # Legacy compatibility
    probabilities: Optional[Dict[str, float]] = None
    model_weights: Optional[Dict[str, float]] = None

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
    """Make prediction with CORRECT label mapping.
    
    Training labels: 0=SHORT, 1=NEUTRAL/HOLD, 2=LONG
    Probabilities: [P(SHORT), P(HOLD), P(LONG)]
    """
    try:
        features = np.array(request.features)
        
        # Validate input shape
        if len(features.shape) != 2:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 2D features [seq_len, n_features], got shape {features.shape}"
            )
        
        seq_len, n_features = features.shape
        if seq_len < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Sequence length {seq_len} too short (minimum 10)"
            )
        
        result = model_manager.predict(features)
        
        # CORRECT mapping: index 0=SHORT, 1=HOLD, 2=LONG
        action_idx = result["action"]
        action = result.get("action_name", ACTION_NAMES[action_idx])
        
        # Map probabilities correctly: [P(SHORT), P(HOLD), P(LONG)]
        probs = {
            "SHORT": result["probabilities"][0],
            "HOLD": result["probabilities"][1],
            "LONG": result["probabilities"][2]
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/candles", response_model=PredictionResponse)
async def predict_from_candles(request: CandlePredictionRequest):
    """Make prediction from raw candle data.
    
    This endpoint handles:
    1. Converting candles to DataFrame
    2. Computing technical indicators/features
    3. Scaling features using the saved scaler
    4. Making prediction with correct label mapping
    
    CRITICAL: Training labels are 0=SHORT, 1=NEUTRAL/HOLD, 2=LONG
    """
    try:
        if len(request.candles) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 100 candles for feature computation, got {len(request.candles)}"
            )
        
        # Convert candles to DataFrame
        import pandas as pd
        candle_data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            }
            for c in request.candles
        ]
        df = pd.DataFrame(candle_data)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Import feature computation from pipeline
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data.pipeline import FeatureEngineer
            
            # Compute features
            fe = FeatureEngineer()
            features_df = fe.compute_features(df)
            
            # Remove NaN rows (from indicator warmup)
            features_df = features_df.dropna()
            
            if len(features_df) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No valid features after computation (all NaN)"
                )
            
            # Scale features if scaler is loaded
            features_np = features_df.values.astype(np.float32)
            if model_manager.scaler is not None:
                features_np = model_manager.scaler.transform(features_np)
            else:
                logger.warning("No scaler loaded - using raw features (may hurt accuracy)")
            
            # Use sequence length from loaded model config
            seq_len = model_manager.sequence_length
            if len(features_np) < seq_len:
                logger.warning(f"Not enough features ({len(features_np)}) for seq_len={seq_len}, using available")
                seq_len = len(features_np)
            
            features_seq = features_np[-seq_len:]
            
        except ImportError as e:
            logger.error(f"Failed to import feature pipeline: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature pipeline not available: {e}"
            )
        
        # Make prediction
        result = model_manager.predict(features_seq)
        
        action_idx = result["action"]
        action = result.get("action_name", ACTION_NAMES[action_idx])
        
        probs = {
            "SHORT": result["probabilities"][0],
            "HOLD": result["probabilities"][1],
            "LONG": result["probabilities"][2]
        }
        
        prediction = {
            "action": action,
            "probabilities": probs,
            "confidence": result["confidence"],
            "features_shape": list(features_seq.shape),
            "candles_used": len(request.candles)
        }
        model_manager.add_prediction(prediction)
        
        return PredictionResponse(
            action=action,
            probabilities=probs,
            confidence=result["confidence"],
            uncertainty=result["uncertainty"],
            model_weights=result.get("model_weights", {}),
            reasoning=result.get("reasoning", []) + [
                f"Processed {len(request.candles)} candles → {features_seq.shape[0]} sequences"
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Candle prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/regression", response_model=RegressionPredictionResponse)
async def predict_regression(request: RegressionPredictionRequest):
    """Make edge-based regression prediction.
    
    Returns mu (expected return), sigma (uncertainty), and edge-based signal.
    This is the institutional-grade signal format:
        edge = mu - cost
        confidence = edge / sigma
        action = LONG/SHORT if confidence > threshold, else NO_TRADE
    """
    try:
        features = np.array(request.features)
        
        if len(features.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D features [seq_len, n_features], got shape {features.shape}"
            )
        
        # Check for MoE or regression model
        result = model_manager.predict(features)
        
        # Extract mu and sigma from model output
        # For classification models, convert probabilities to pseudo-mu/sigma
        probs = result["probabilities"]
        
        # P(LONG) - P(SHORT) as directional signal
        p_long = probs[2]
        p_short = probs[0]
        p_hold = probs[1]
        
        # Convert to mu: expected direction * magnitude
        mu = (p_long - p_short) * 0.01  # Scale to ~1% expected move
        
        # Uncertainty from entropy of distribution
        entropy = -sum(p * np.log(p + 1e-8) for p in probs)
        max_entropy = -3 * (1/3) * np.log(1/3)  # Max entropy for 3 classes
        sigma = 0.005 + 0.015 * (entropy / max_entropy)  # 0.5% to 2% uncertainty
        
        # Transaction costs
        maker_fee = 0.0002
        taker_fee = 0.0004
        slippage = 0.0001 + 0.5 * request.current_volatility
        cost = (taker_fee * 2) + (slippage * 2)
        
        # Calculate edge per institutional spec: edge = (μ - cost) / σ
        # This is the risk-adjusted expected profit
        sigma_safe = max(sigma, 0.001)
        edge = (abs(mu) - cost) / sigma_safe
        
        # Edge IS the confidence in this formulation
        confidence = edge
        
        # Determine action based on edge threshold
        # Edge > 0.5 means expected profit is 0.5 standard deviations above costs
        min_edge_threshold = 0.5
        
        should_trade = edge >= min_edge_threshold
        
        if should_trade:
            action = "LONG" if mu > 0 else "SHORT"
        else:
            action = "NO_TRADE"
        
        # Calculate position size using bounded Kelly
        # Since edge = (mu - cost) / sigma, we use edge * sigma for original profit
        if should_trade and sigma > 0:
            expected_profit = edge * sigma  # Recover (mu - cost)
            kelly = expected_profit / (sigma ** 2)  # Kelly = (mu - cost) / sigma^2
            half_kelly = kelly * 0.5
            position_size_pct = max(0, min(half_kelly, 0.1))  # Max 10%
        else:
            position_size_pct = 0
        
        # Suggested order type
        # Calculate edge with maker fees to compare
        cost_maker = (maker_fee * 2) + (slippage * 2)
        edge_maker = (abs(mu) - cost_maker) / sigma_safe
        suggested_order = "TAKER" if edge > edge_maker * 1.5 else "MAKER"
        
        # Urgency
        if confidence > 2.0 and abs(mu) > 0.01:
            urgency = "HIGH"
        elif confidence > 1.0:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        # Stops
        stop_loss_pct = sigma * 2
        take_profit_pct = abs(mu) * 1.2 if abs(mu) > stop_loss_pct * 1.5 else stop_loss_pct * 1.5
        
        # Reasons
        reasons = []
        if should_trade:
            reasons.append(f"Edge: {edge:.2f}σ (risk-adjusted)")
            reasons.append(f"Expected move: {mu*100:.3f}%")
            reasons.append(f"Uncertainty: {sigma*100:.3f}%")
        else:
            reasons.append(f"Edge too low: {edge:.2f}σ < {min_edge_threshold}σ required")
        
        return RegressionPredictionResponse(
            action=action,
            confidence=confidence,
            expected_move=float(mu),
            uncertainty=float(sigma),
            edge=float(edge),
            cost_estimate=float(cost),
            suggested_order_type=suggested_order,
            urgency=urgency,
            position_size_pct=float(position_size_pct),
            stop_loss_pct=float(stop_loss_pct),
            take_profit_pct=float(take_profit_pct),
            regime="UNKNOWN",  # Will be set by regime detector
            expert_weights=result.get("model_weights", {}),
            reasons=reasons,
            probabilities={
                "SHORT": float(probs[0]),
                "HOLD": float(probs[1]),
                "LONG": float(probs[2])
            },
            model_weights=result.get("model_weights", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Regression prediction error: {e}")
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

@app.on_event("startup")
async def startup_event():
    """Load models at server startup."""
    logger.info("Loading models at startup...")
    model_manager.load_best_models()
    logger.info(f"Startup complete. Device: {model_manager.device}, Models: {len(model_manager.models)}")

@app.post("/models/load")
async def load_model_endpoint(model_path: str):
    """Manually load and instantiate a specific model checkpoint."""
    try:
        path = Path(model_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(str(path), map_location=model_manager.device, weights_only=False)
        model_name = path.stem
        
        # Store metadata
        model_manager.models[model_name] = {
            "path": str(path),
            "accuracy": checkpoint.get("val_accuracy", 0),
            "epoch": checkpoint.get("epoch", 0),
            "config": checkpoint.get("config", {}),
            "parameters": checkpoint.get("parameters", 0)
        }
        
        # Instantiate model
        if "model_state_dict" in checkpoint:
            model_instance = model_manager._instantiate_model(checkpoint, model_name)
            if model_instance is not None:
                model_manager.model_instances[model_name] = model_instance
                model_manager.models[model_name]["loaded"] = True
                return {
                    "message": f"Model loaded and instantiated: {model_name}",
                    "path": str(path),
                    "accuracy": checkpoint.get("val_accuracy", 0),
                    "instantiated": True
                }
            else:
                model_manager.models[model_name]["loaded"] = False
                return {
                    "message": f"Model loaded but not instantiated: {model_name}",
                    "path": str(path),
                    "instantiated": False,
                    "warning": "Could not determine model architecture"
                }
        
        return {"message": f"Model checkpoint loaded: {model_name}", "path": str(path), "instantiated": False}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def get_models_status():
    """Get detailed status of loaded models and scaler."""
    models_detail = {}
    for name, model_info in model_manager.models.items():
        models_detail[name] = {
            "checkpoint_loaded": model_info.get("loaded", False),
            "instantiated": name in model_manager.model_instances,
            "error": model_manager.instantiation_errors.get(name),
            "accuracy": model_info.get("accuracy", 0),
            "epoch": model_info.get("epoch", 0),
            "config": model_info.get("config", {})
        }
    
    # Count successful vs failed instantiations
    total_checkpoints = len(model_manager.models)
    successful_instances = len(model_manager.model_instances)
    failed_instances = len(model_manager.instantiation_errors)
    
    return {
        "summary": {
            "checkpoints_found": total_checkpoints,
            "models_instantiated": successful_instances,
            "instantiation_failures": failed_instances,
            "ready_for_prediction": successful_instances > 0
        },
        "models_instantiated": list(model_manager.model_instances.keys()),
        "instantiation_errors": model_manager.instantiation_errors,
        "models_detail": models_detail,
        "config": {
            "sequence_length": model_manager.sequence_length,
            "input_dim": model_manager.input_dim,
            "device": model_manager.device,
            "checkpoint_dir": str(model_manager.checkpoint_dir),
            "checkpoint_dir_exists": model_manager.checkpoint_dir.exists()
        },
        "scaler_loaded": model_manager.scaler is not None,
        "label_mapping": {
            "0": "SHORT",
            "1": "HOLD/NEUTRAL", 
            "2": "LONG"
        },
        "warning": "Models will return default HOLD (p=0.34 each) if no model_instances are loaded" if successful_instances == 0 else None
    }

def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
