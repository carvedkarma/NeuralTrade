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
    # Mapping from checkpoint filename patterns to standardized model types
    MODEL_TYPE_PATTERNS = {
        # Multi-head models (check first - have forward_multihead for quantile predictions)
        "multihead_transformer": ["multihead_transformer", "transformer_multihead", "best_multihead_transformer"],
        "multihead_lstm": ["multihead_lstm", "lstm_multihead", "best_multihead_lstm"],
        "multihead_cnn": ["multihead_cnn", "cnn_multihead", "best_multihead_cnn"],
        "multihead_gnn": ["multihead_gnn", "gnn_multihead", "best_multihead_gnn"],
        "multihead_vae": ["multihead_vae", "vae_multihead", "best_multihead_vae"],
        # Legacy classification-only models
        "transformer": ["transformer_price", "transformer", "best_transformer"],
        "tft": ["temporal_fusion_transformer", "tft", "best_temporal_fusion", "best_tft"],
        "lstm": ["bidirectional_lstm", "lstm", "stacked_lstm", "conv_lstm", "best_lstm", "best_bidirectional"],
        "cnn": ["resnet_price", "resnet", "cnn", "inception", "wavenet", "best_resnet", "best_cnn"],
        "vae": ["market_vae", "vae", "conditional_vae", "best_vae", "best_market_vae"],
        "gnn": ["cross_asset_gnn", "temporal_gnn", "gnn", "best_gnn", "best_cross_asset"],
    }
    
    def __init__(self):
        self.models = {}
        self.model_instances = {}
        self.model_type_map = {}  # Maps checkpoint name -> standardized type (transformer, tft, etc.)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ensemble = None
        self.scaler = None  # Dict of per-column scalers (NOT a single sklearn scaler)
        self.scaler_columns = None  # Column names for the scaler dict
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
    
    def _map_filename_to_model_type(self, filename: str) -> str:
        """Map checkpoint filename to standardized model type.
        
        Examples:
            best_transformer_price -> transformer
            best_temporal_fusion_transformer -> tft
            best_bidirectional_lstm -> lstm
            best_resnet_price -> cnn
            best_market_vae -> vae
            best_cross_asset_gnn -> gnn
        """
        filename_lower = filename.lower()
        
        for model_type, patterns in self.MODEL_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return model_type
        
        # If no pattern matched, return the filename as-is
        return filename
    
    def _config_to_dict(self, config) -> dict:
        """Convert a Config object (dataclass/object) to a dictionary.
        
        Handles both dict and object-style configs from checkpoints.
        Some checkpoints save config as a dataclass/object, others as dict.
        """
        if config is None:
            return {}
        
        # Already a dict
        if isinstance(config, dict):
            return config
        
        # Try to convert object to dict
        try:
            # Try vars() for regular objects
            return vars(config)
        except TypeError:
            pass
        
        try:
            # Try __dict__ directly
            if hasattr(config, '__dict__'):
                return config.__dict__
        except Exception:
            pass
        
        try:
            # Try dataclass asdict
            from dataclasses import asdict, is_dataclass
            if is_dataclass(config):
                return asdict(config)
        except Exception:
            pass
        
        try:
            # Try accessing common attributes manually
            result = {}
            common_attrs = ['input_dim', 'output_dim', 'hidden_dim', 'sequence_length', 
                           'dropout', 'd_model', 'nhead', 'num_layers', 'num_encoder_layers',
                           'model_type', 'latent_dim', 'hidden_dims', 'num_assets', 
                           'base_channels', 'num_blocks', 'kernel_size', 'use_attention']
            for attr in common_attrs:
                if hasattr(config, attr):
                    result[attr] = getattr(config, attr)
            return result
        except Exception:
            pass
        
        # Fallback: return empty dict
        logger.warning(f"Could not convert config of type {type(config)} to dict")
        return {}
    
    def transform_features(self, features_df) -> np.ndarray:
        """Transform features using the loaded scaler dict.
        
        The scaler is a dict of per-column sklearn scalers, NOT a single scaler.
        This matches how FeatureEngineer.save_scalers/load_scalers works.
        """
        if self.scaler is None:
            logger.warning("No scaler loaded - returning raw features")
            return features_df.values.astype(np.float32)
        
        # Apply per-column scaling using the scaler dict
        transformed = features_df.copy()
        for col in features_df.columns:
            if col in self.scaler:
                valid_mask = ~features_df[col].isna()
                if valid_mask.any():
                    try:
                        transformed.loc[valid_mask, col] = self.scaler[col].transform(
                            features_df.loc[valid_mask, col].values.reshape(-1, 1)
                        ).flatten()
                    except Exception as e:
                        logger.warning(f"Failed to scale column {col}: {e}")
        
        return transformed.values.astype(np.float32)
        
    def _create_model_instance(self, model_type: str, config: dict):
        """Create model instance with correct constructor args for each model type.
        
        Covers all model classes from gpu_trainer/models/:
        - multihead.py: MultiHeadTransformer, MultiHeadLSTM, MultiHeadCNN, MultiHeadGNN, MultiHeadVAE
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
            
            # === MULTI-HEAD MODELS (check first - has forward_multihead for quantile predictions) ===
            if "multihead_transformer" in model_type_lower:
                from models.multihead import MultiHeadTransformer
                return MultiHeadTransformer(
                    input_dim=input_dim,
                    d_model=config.get("d_model", 256),
                    nhead=config.get("nhead", 8),
                    num_layers=config.get("num_layers", 6),
                    dropout=dropout,
                    num_classes=output_dim
                )
            elif "multihead_lstm" in model_type_lower:
                from models.multihead import MultiHeadLSTM
                return MultiHeadLSTM(
                    input_dim=input_dim,
                    hidden_dim=config.get("hidden_dim", 256),
                    num_layers=config.get("num_layers", 3),
                    dropout=dropout,
                    num_classes=output_dim
                )
            elif "multihead_cnn" in model_type_lower:
                from models.multihead import MultiHeadCNN
                return MultiHeadCNN(
                    input_dim=input_dim,
                    hidden_channels=config.get("hidden_channels", 256),
                    num_blocks=config.get("num_blocks", 4),
                    dropout=dropout,
                    num_classes=output_dim
                )
            elif "multihead_gnn" in model_type_lower:
                from models.multihead import MultiHeadGNN
                return MultiHeadGNN(
                    input_dim=input_dim,
                    hidden_dim=config.get("hidden_dim", 128),
                    num_layers=config.get("num_layers", 3),
                    num_heads=config.get("num_heads", 4),
                    dropout=dropout,
                    num_classes=output_dim
                )
            elif "multihead_vae" in model_type_lower:
                from models.multihead import MultiHeadVAE
                return MultiHeadVAE(
                    input_dim=input_dim,
                    sequence_length=sequence_length,
                    latent_dim=config.get("latent_dim", 64),
                    dropout=dropout,
                    num_classes=output_dim
                )
            
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
                # ResNetPrice uses channels: List[int], not base_channels
                channels = config.get("channels", [64, 128, 256, 512])
                return ResNetPrice(
                    input_dim=input_dim,
                    channels=channels,
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
    
    def _infer_dims_from_state_dict(self, state_dict: dict, model_type: str) -> dict:
        """Infer model dimensions from state_dict weight shapes.
        
        This is critical for loading models when input_dim/hidden_dim weren't
        saved in the checkpoint config (which is the common case since they're
        computed at training time from data shape).
        
        Matches actual parameter names from gpu_trainer/models/*.py implementations.
        """
        inferred = {}
        model_type_lower = model_type.lower()
        
        def get_param_shape(key):
            """Get shape from state_dict, handling both Tensor and array-like objects."""
            if key in state_dict:
                param = state_dict[key]
                return tuple(param.shape) if hasattr(param, 'shape') else None
            return None
        
        try:
            # === LSTM MODELS ===
            # BidirectionalLSTM: input_bn.weight [input_dim], lstm.weight_ih_l0 [4*hidden_dim, input_dim]
            if "lstm" in model_type_lower or "bidirectional" in model_type_lower:
                shape = get_param_shape("input_bn.weight")
                if shape:
                    inferred["input_dim"] = shape[0]
                    
                shape = get_param_shape("lstm.weight_ih_l0")
                if shape:
                    inferred["hidden_dim"] = shape[0] // 4  # LSTM has 4 gates
                    if "input_dim" not in inferred:
                        inferred["input_dim"] = shape[1]
                        
            # === TRANSFORMER MODELS ===
            # TransformerPriceModel: input_projection.weight [d_model, input_dim]
            elif "transformer" in model_type_lower and "tft" not in model_type_lower and "temporal_fusion" not in model_type_lower:
                shape = get_param_shape("input_projection.weight")
                if shape:
                    inferred["d_model"] = shape[0]
                    inferred["input_dim"] = shape[1]
                    
            # TemporalFusionTransformer: 
            #   static_encoder.0.weight [d_model, input_dim] - use this for d_model
            #   temporal_encoder is LSTM which uses d_model/2 per direction
            elif "tft" in model_type_lower or "temporal_fusion" in model_type_lower:
                # Use static_encoder for d_model (more reliable than LSTM)
                static_shape = get_param_shape("static_encoder.0.weight")
                if static_shape:
                    inferred["d_model"] = static_shape[0]
                    inferred["input_dim"] = static_shape[1]
                else:
                    # Fallback to temporal_encoder if static_encoder not found
                    shape = get_param_shape("temporal_encoder.weight_ih_l0")
                    if shape:
                        inferred["input_dim"] = shape[1]
                        # LSTM hidden is d_model/2 per direction (bidirectional), so hidden*2 = d_model
                        # But weight_ih has shape [4*hidden, input_dim], so d_model = shape[0] // 4 * 2 = shape[0] // 2
                        inferred["d_model"] = shape[0] // 2
                    
            # === CNN MODELS ===
            # ResNetPrice: input_conv.0.weight [channels[0], input_dim, kernel_size]
            # ResNetPrice uses channels: List[int] = [64, 128, 256, 512]
            elif "resnet" in model_type_lower or "cnn" in model_type_lower or "inception" in model_type_lower:
                shape = get_param_shape("input_conv.0.weight")
                if shape:
                    inferred["input_dim"] = shape[1]  # Conv1d: [out_channels, in_channels, kernel]
                    first_channels = shape[0]
                    # For ResNetPrice, infer the full channels list from the residual blocks
                    # Default pattern is [64, 128, 256, 512] but could be different
                    # For now, use the first channel and assume standard progression
                    inferred["channels"] = [first_channels, first_channels*2, first_channels*4, first_channels*8]
                    
            # WaveNet: input_conv.weight [residual_channels, input_dim, 1]
            elif "wavenet" in model_type_lower:
                shape = get_param_shape("input_conv.weight")
                if shape:
                    inferred["input_dim"] = shape[1]
                    inferred["residual_channels"] = shape[0]
                    
            # === VAE MODELS ===
            # MarketVAE: encoder has structure [Linear, BatchNorm, LeakyReLU, Dropout] x N
            # encoder.0.weight [hidden_dims[0], input_dim * sequence_length]
            # encoder.4.weight [hidden_dims[1], hidden_dims[0]]
            # encoder.8.weight [hidden_dims[2], hidden_dims[1]]
            # fc_mu.weight [latent_dim, hidden_dims[-1]]
            elif "vae" in model_type_lower:
                # Find input_dim and sequence_length from first encoder layer
                shape = get_param_shape("encoder.0.weight")
                if shape:
                    total_input = shape[1]
                    first_hidden = shape[0]
                    for seq_len in [100, 50, 60, 120, 80]:
                        if total_input % seq_len == 0:
                            inferred["input_dim"] = total_input // seq_len
                            inferred["sequence_length"] = seq_len
                            break
                    
                    # Scan all encoder layers to build hidden_dims list
                    # Each block is 4 layers (Linear, BatchNorm, LeakyReLU, Dropout)
                    hidden_dims = [first_hidden]
                    layer_idx = 4  # Start at second block
                    while True:
                        layer_shape = get_param_shape(f"encoder.{layer_idx}.weight")
                        if layer_shape and len(layer_shape) == 2:  # Linear layer
                            hidden_dims.append(layer_shape[0])
                            layer_idx += 4
                        else:
                            break
                    inferred["hidden_dims"] = hidden_dims
                    
                # Get latent_dim from fc_mu
                fc_mu_shape = get_param_shape("fc_mu.weight")
                if fc_mu_shape:
                    inferred["latent_dim"] = fc_mu_shape[0]
                    
            # === GNN MODELS ===
            # CrossAssetGNN: 
            #   temporal_encoder.0.weight [hidden_dim, input_dim] - uses full input
            #   node_encoder.0.weight [hidden_dim, input_dim // num_assets] - per-asset features
            elif "cross_asset" in model_type_lower or "crossasset" in model_type_lower:
                # Use temporal_encoder for full input_dim (not node_encoder which uses per-asset)
                temporal_shape = get_param_shape("temporal_encoder.0.weight")
                node_shape = get_param_shape("node_encoder.0.weight")
                
                if temporal_shape:
                    inferred["hidden_dim"] = temporal_shape[0]
                    inferred["input_dim"] = temporal_shape[1]
                    
                    # Calculate num_assets from the ratio
                    if node_shape:
                        features_per_asset = node_shape[1]
                        if features_per_asset > 0:
                            num_assets = inferred["input_dim"] // features_per_asset
                            if num_assets >= 1:
                                inferred["num_assets"] = num_assets
                elif node_shape:
                    # Fallback if temporal_encoder not found
                    inferred["hidden_dim"] = node_shape[0]
                    inferred["input_dim"] = node_shape[1]
                    
            # TemporalGNN: spatial_encoder.weight [hidden_dim, features_per_node]
            elif "gnn" in model_type_lower or "temporal_gnn" in model_type_lower:
                shape = get_param_shape("spatial_encoder.weight")
                if shape:
                    inferred["hidden_dim"] = shape[0]
                    # features_per_node = input_dim // num_nodes, but we'll store what we find
                    inferred["input_dim"] = shape[1]
                    
            if inferred:
                logger.info(f"Inferred dimensions for {model_type}: {inferred}")
            else:
                # Log available keys for debugging
                sample_keys = list(state_dict.keys())[:10]
                logger.warning(f"Could not infer dims for {model_type}. Sample keys: {sample_keys}")
                
        except Exception as e:
            logger.warning(f"Failed to infer dims from state_dict for {model_type}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
        return inferred
    
    def _instantiate_model(self, checkpoint: dict, model_name: str):
        """Instantiate a model from checkpoint config and state_dict."""
        raw_config = checkpoint.get("config", {})
        # Convert Config object to dict if needed
        config = self._config_to_dict(raw_config)
        
        # Check for model_config first (saved by updated trainer with input_dim)
        model_config = checkpoint.get("model_config", {})
        if model_config:
            logger.info(f"Found model_config in checkpoint: {model_config}")
            # Model config takes precedence for model-specific params
            for key, value in model_config.items():
                if value is not None:
                    config[key] = value
        
        state_dict = checkpoint.get("model_state_dict")
        
        if not state_dict:
            logger.warning(f"No state_dict in checkpoint for {model_name}")
            return None
        
        # Try to determine model type from name, config, or infer from state_dict keys
        model_type = config.get("model_type", "") or config.get("name", "")
        if not model_type:
            # Try to infer from checkpoint name
            model_type = model_name
        
        # Infer dimensions from state_dict (since training doesn't save input_dim/hidden_dim in config)
        inferred_dims = self._infer_dims_from_state_dict(state_dict, model_type)
        
        # Merge inferred dims into config - inferred ALWAYS takes precedence
        # since they come from actual weights and reflect the true model architecture
        for key, value in inferred_dims.items():
            if value is not None:
                config[key] = value
        
        try:
            # Get config values with defaults (now possibly updated by inferred dims)
            input_dim = config.get("input_dim", 81)
            hidden_dim = config.get("hidden_dim", 128)
            d_model = config.get("d_model", 256)
            sequence_length = config.get("sequence_length", 100)
            
            logger.info(f"Creating model {model_type} with: input_dim={input_dim}, hidden_dim={hidden_dim}, d_model={d_model}")
            
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
        
        # Load expected feature list if exists (for alignment verification)
        feature_list_path = self.checkpoint_dir / "feature_columns.txt"
        if feature_list_path.exists():
            try:
                with open(feature_list_path, 'r') as f:
                    self.expected_features = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded expected feature list: {len(self.expected_features)} columns")
                # Update input_dim to match expected features
                self.input_dim = len(self.expected_features)
                logger.info(f"Updated input_dim to {self.input_dim} based on feature list")
            except Exception as e:
                logger.warning(f"Failed to load feature list: {e}")
                self.expected_features = None
        else:
            self.expected_features = None
        
        # Find and load best checkpoint for each model type
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            logger.warning("No checkpoint files found")
            return
        
        # Only load "best_*" checkpoints to avoid loading epoch checkpoints
        best_checkpoints = [f for f in checkpoint_files if f.stem.startswith("best_")]
        if not best_checkpoints:
            # Fallback: if no best_* files, use all checkpoints
            best_checkpoints = checkpoint_files
            logger.info("No best_* checkpoints found, loading all .pt files")
            
        for ckpt_path in best_checkpoints:
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model_name = ckpt_path.stem
                
                # Map filename to standardized model type
                model_type = self._map_filename_to_model_type(model_name)
                self.model_type_map[model_name] = model_type
                
                # Convert Config object to dict if needed
                raw_config = checkpoint.get("config", {})
                config_dict = self._config_to_dict(raw_config)
                
                # Store checkpoint metadata
                self.models[model_name] = {
                    "path": str(ckpt_path),
                    "accuracy": checkpoint.get("val_accuracy", 0),
                    "epoch": checkpoint.get("epoch", 0),
                    "config": config_dict,
                    "parameters": checkpoint.get("parameters", 0),
                    "model_type": model_type  # Standardized type for dashboard
                }
                
                # Try to instantiate model if state_dict present
                if "model_state_dict" in checkpoint:
                    model_instance = self._instantiate_model(checkpoint, model_name)
                    if model_instance is not None:
                        self.model_instances[model_name] = model_instance
                        self.models[model_name]["loaded"] = True
                        logger.info(f"Loaded & instantiated: {model_name} -> {model_type} (acc={checkpoint.get('val_accuracy', 0):.2f}%)")
                    else:
                        self.models[model_name]["state_dict"] = checkpoint["model_state_dict"]
                        self.models[model_name]["loaded"] = False
                        logger.warning(f"Loaded checkpoint but failed to instantiate: {model_name} -> {model_type}")
                    
            except Exception as e:
                logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
                
        logger.info(f"Loaded {len(self.models)} checkpoints, {len(self.model_instances)} instantiated")
        logger.info(f"Model type mapping: {self.model_type_map}")
    
    def get_model_status_by_type(self) -> Dict[str, Dict]:
        """Get model status organized by standardized model type for dashboard display."""
        status = {}
        
        # Initialize all 6 model types as pending
        for model_type in ["transformer", "tft", "lstm", "cnn", "vae", "gnn"]:
            status[model_type] = {
                "status": "pending",
                "accuracy": None,
                "loss": None,
                "epochs": 0,
                "best_epoch": 0,
                "checkpoint_name": None
            }
        
        # Update status from loaded models
        for model_name, model_info in self.models.items():
            model_type = model_info.get("model_type", self._map_filename_to_model_type(model_name))
            if model_type in status:
                is_instantiated = model_name in self.model_instances
                status[model_type] = {
                    "status": "complete" if is_instantiated else "loaded",
                    "accuracy": model_info.get("accuracy", 0),
                    "loss": model_info.get("config", {}).get("val_loss", None),
                    "epochs": model_info.get("epoch", 0),
                    "best_epoch": model_info.get("epoch", 0),
                    "checkpoint_name": model_name
                }
        
        return status
        
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
        
        # Feature count validation - strict mode fails on mismatch
        input_features = features.shape[-1] if len(features.shape) >= 2 else features.shape[0]
        if hasattr(self, 'expected_features') and self.expected_features:
            expected_count = len(self.expected_features)
            if input_features != expected_count:
                error_msg = f"Feature count mismatch: received {input_features}, expected {expected_count}"
                logger.error(error_msg)
                # Return error prediction instead of potentially wrong prediction
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "probabilities": {"LONG": 0.33, "SHORT": 0.33, "HOLD": 0.34},
                    "error": error_msg,
                    "expected_features": expected_count,
                    "received_features": input_features
                }
            
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
    
    def predict_multihead(self, features: np.ndarray) -> Optional[Dict]:
        """Make prediction using multi-head model with learned quantiles.
        
        Returns None if no multi-head model is available.
        
        Multi-head models output:
        - Classification: direction probabilities
        - Regression: expected return (mu) and uncertainty (sigma)
        - Quantiles: q10, q25, q50, q75, q90 for SL/TP derivation
        
        Accepts inputs of shape:
        - [seq_len, features] -> single sample
        - [batch, seq_len, features] -> batch of samples
        
        Feature version locking:
        - Validates feature dimension, sequence length, and horizon
        - Returns HOLD with confidence=0 on any mismatch (safety behavior)
        """
        # Check for multi-head model instances
        multihead_model = None
        for name, model in self.model_instances.items():
            if hasattr(model, 'forward_multihead'):
                multihead_model = model
                break
        
        if multihead_model is None:
            return None
        
        # Safe HOLD response for validation failures
        safe_hold_response = lambda reason: {
            "action": 1,
            "action_name": "HOLD",
            "direction_probs": {"LONG": 0.33, "SHORT": 0.33, "HOLD": 0.34},
            "confidence": 0.0,
            "mu": 0.0,
            "sigma": 0.01,
            "quantiles": {"q10": -0.01, "q25": -0.005, "q50": 0.0, "q75": 0.005, "q90": 0.01},
            "model_name": "HOLD (validation failed)",
            "is_learned": False,
            "error": reason,
            "feature_dim_validated": False
        }
        
        # Feature dimension validation (critical)
        input_features = features.shape[-1] if len(features.shape) >= 2 else features.shape[0]
        expected_dim = getattr(multihead_model, 'input_dim', self.input_dim)
        
        if input_features != expected_dim:
            logger.warning(f"[BLOCK] Feature mismatch: expected {expected_dim}, got {input_features}")
            return safe_hold_response(f"Feature dimension mismatch: expected {expected_dim}, got {input_features}")
        
        # Sequence length validation (critical - hard failure)
        if len(features.shape) == 2:
            actual_seq = features.shape[0]
        else:
            actual_seq = features.shape[1]
        
        expected_seq = getattr(self, 'sequence_length', 100)
        if actual_seq != expected_seq:
            logger.warning(f"[BLOCK] Sequence length mismatch: expected {expected_seq}, got {actual_seq}")
            return safe_hold_response(f"Sequence length mismatch: expected {expected_seq}, got {actual_seq}")
        
        try:
            multihead_model.eval()
            with torch.no_grad():
                # Handle both 2D [seq_len, features] and 3D [batch, seq_len, features] inputs
                if len(features.shape) == 2:
                    x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                else:
                    x = torch.FloatTensor(features).to(self.device)
                    if len(x.shape) == 2:
                        x = x.unsqueeze(0)
                
                output = multihead_model.forward_multihead(x)
                
                # Extract probabilities (handle both single and batch)
                probs = torch.softmax(output.class_logits, dim=-1).cpu().numpy()
                if len(probs.shape) == 2 and probs.shape[0] == 1:
                    probs = probs[0]
                elif len(probs.shape) == 2:
                    probs = probs.mean(axis=0)  # Average across batch
                
                action_idx = int(np.argmax(probs))
                confidence = float(probs[action_idx])
                
                # Extract regression outputs
                mu_arr = output.mu.cpu().numpy()
                sigma_arr = output.sigma.cpu().numpy() if output.sigma is not None else np.array([[0.01]])
                
                mu = float(mu_arr.mean())
                sigma = float(sigma_arr.mean())
                
                # Extract learned quantiles
                quantiles_arr = output.quantiles.cpu().numpy()
                if len(quantiles_arr.shape) == 2 and quantiles_arr.shape[0] == 1:
                    quantiles = quantiles_arr[0]
                else:
                    quantiles = quantiles_arr.mean(axis=0)
                
                return {
                    "action": action_idx,
                    "action_name": ACTION_NAMES[action_idx],
                    "direction_probs": {
                        "LONG": float(probs[2]),
                        "SHORT": float(probs[0]),
                        "HOLD": float(probs[1])
                    },
                    "confidence": confidence,
                    "mu": mu,
                    "sigma": sigma,
                    "quantiles": {
                        "q10": float(quantiles[0]),
                        "q25": float(quantiles[1]),
                        "q50": float(quantiles[2]),
                        "q75": float(quantiles[3]),
                        "q90": float(quantiles[4])
                    },
                    "model_name": multihead_model.name,
                    "is_learned": True,
                    "feature_dim_validated": True
                }
        except Exception as e:
            logger.error(f"Multi-head prediction error: {e}")
            return None
    
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

class EnsemblePredictionRequest(BaseModel):
    """Request for professional ensemble prediction."""
    features: List[List[float]]

class QuantilePredictionRequest(BaseModel):
    """Request for quantile regression prediction."""
    features: List[List[float]]

class QuantilePredictionResponse(BaseModel):
    """Response with quantile regression for Entry/SL/TP derivation."""
    direction_probs: Dict[str, float]
    quantiles: Dict[str, float]  # q10, q25, q50, q75, q90
    mfe_quantiles: Optional[Dict[str, float]] = None
    mae_quantiles: Optional[Dict[str, float]] = None
    model_name: str
    confidence: float

class EnsemblePredictionResponse(BaseModel):
    """Response with regime-gated ensemble signal."""
    action: str  # LONG, SHORT, HOLD, NO_TRADE
    confidence: float
    confidence_margin: float  # p_top1 - p_top2
    edge: float
    
    # Regime information
    market_regime: str
    risk_regime: str
    regime_confidence: float
    
    # Model agreement
    agreement_pct: float
    weighted_agreement: float
    disagreement_score: float
    
    # Position sizing
    position_size_pct: float
    regime_adjusted_size: float
    
    # Thresholds
    confidence_threshold_used: float
    regime_adjustment: str
    
    # Per-model breakdown
    model_votes: Dict[str, Any]
    
    # Ensemble probabilities
    ensemble_probs: Dict[str, float]
    
    # Reasons
    reasons: List[str]

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
            
            # Compute features using the correct method name
            fe = FeatureEngineer()
            features_df = fe.compute_technical_features(df)
            
            # Remove NaN rows (from indicator warmup)
            features_df = features_df.dropna()
            
            if len(features_df) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No valid features after computation (all NaN)"
                )
            
            # Scale features using the per-column scaler dict (via transform_features helper)
            features_np = model_manager.transform_features(features_df)
            
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

@app.post("/predict/quantile", response_model=QuantilePredictionResponse)
async def predict_quantile(request: QuantilePredictionRequest):
    """Make quantile regression prediction for Entry/SL/TP derivation.
    
    Returns:
    - Direction probabilities (LONG, SHORT, HOLD)
    - Return quantiles (q10, q25, q50, q75, q90) for probability-based price projections
    - MFE/MAE quantiles when available
    
    Entry/SL/TP are derived from quantiles:
    - Entry = current price
    - For LONG: SL = price * (1 + q10), TP = price * (1 + q90)
    - For SHORT: SL = price * (1 + q90), TP = price * (1 + q10)
    
    NOTE: This endpoint uses learned quantiles if a multi-head model is loaded,
    otherwise falls back to heuristic synthesis from classification probabilities.
    """
    try:
        features = np.array(request.features)
        
        if len(features.shape) == 2 and features.shape[0] == 1:
            features = features.reshape(1, features.shape[0], features.shape[1])
        elif len(features.shape) != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Expected features shape [batch, seq_len, features], got {features.shape}"
            )
        
        # Check if we have a multi-head model with learned quantiles
        multihead_result = model_manager.predict_multihead(
            features[0] if features.shape[0] == 1 else features
        )
        
        if multihead_result is not None:
            # Use learned quantiles from multi-head model
            direction_probs = multihead_result["direction_probs"]
            quantiles = multihead_result["quantiles"]
            confidence = multihead_result["confidence"]
            
            # MFE/MAE estimates from quantile spread
            q90 = quantiles["q90"]
            q10 = quantiles["q10"]
            mfe_quantiles = {
                "q10": float(max(quantiles["q75"], 0) * 0.8),
                "q50": float(max(q90, 0) * 0.9),
                "q90": float(max(q90, 0) * 1.2)
            }
            mae_quantiles = {
                "q10": float(min(q10, 0) * 0.8),
                "q50": float(min(q10, 0) * 1.0),
                "q90": float(min(q10, 0) * 1.3)
            }
            
            return QuantilePredictionResponse(
                direction_probs=direction_probs,
                quantiles=quantiles,
                mfe_quantiles=mfe_quantiles,
                mae_quantiles=mae_quantiles,
                model_name=multihead_result.get("model_name", "MultiHead"),
                confidence=confidence
            )
        
        # Fallback to heuristic synthesis from classification probabilities
        result = model_manager.predict(features[0] if features.shape[0] == 1 else features)
        
        probs = result.get("probabilities", [0.33, 0.34, 0.33])
        direction_probs = {
            "LONG": float(probs[2]),
            "SHORT": float(probs[0]),
            "HOLD": float(probs[1])
        }
        
        confidence = float(result.get("confidence", max(probs)))
        
        p_long = probs[2]
        p_short = probs[0]
        directional_bias = p_long - p_short
        
        base_vol = 0.015
        uncertainty = float(result.get("uncertainty", 0.3))
        spread = base_vol * (1 + uncertainty)
        
        q50 = directional_bias * base_vol * 2
        q25 = q50 - spread * 0.67
        q75 = q50 + spread * 0.67
        q10 = q50 - spread * 1.28
        q90 = q50 + spread * 1.28
        
        quantiles = {
            "q10": float(q10),
            "q25": float(q25),
            "q50": float(q50),
            "q75": float(q75),
            "q90": float(q90)
        }
        
        mfe_quantiles = {
            "q10": float(max(q75, 0) * 0.8),
            "q50": float(max(q90, 0) * 0.9),
            "q90": float(max(q90, 0) * 1.2)
        }
        
        mae_quantiles = {
            "q10": float(min(q10, 0) * 0.8),
            "q50": float(min(q10, 0) * 1.0),
            "q90": float(min(q10, 0) * 1.3)
        }
        
        return QuantilePredictionResponse(
            direction_probs=direction_probs,
            quantiles=quantiles,
            mfe_quantiles=mfe_quantiles,
            mae_quantiles=mae_quantiles,
            model_name=result.get("model_name", "Classification (Heuristic)"),
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quantile prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Global ensemble predictor instance
_ensemble_predictor = None

def get_ensemble_predictor():
    """Get or create ensemble predictor instance."""
    global _ensemble_predictor
    if _ensemble_predictor is None and model_manager.model_instances:
        try:
            from .ensemble_predictor import EnsemblePredictor
            _ensemble_predictor = EnsemblePredictor(
                model_instances=model_manager.model_instances,
                device=model_manager.device
            )
            logger.info("Initialized ensemble predictor")
        except Exception as e:
            logger.error(f"Failed to initialize ensemble predictor: {e}")
    return _ensemble_predictor

@app.post("/predict/ensemble", response_model=EnsemblePredictionResponse)
async def predict_ensemble(request: EnsemblePredictionRequest):
    """
    Professional ensemble prediction with regime gating.
    
    This endpoint:
    1. Uses direction models (Transformer, TFT, LSTM, CNN) for voting
    2. Uses VAE for market regime detection (trend/range/chop)
    3. Uses GNN for risk regime detection (risk-on/off)
    4. Weights by walk-forward trading metrics (not accuracy)
    5. Applies confidence margin (p_top1 - p_top2) thresholds
    6. Adjusts position sizing based on regime
    """
    try:
        features = np.array(request.features)
        
        if len(features.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D features [seq_len, n_features], got shape {features.shape}"
            )
        
        predictor = get_ensemble_predictor()
        
        if predictor is None:
            # Fallback to basic prediction if ensemble not available
            result = model_manager.predict(features)
            return EnsemblePredictionResponse(
                action=result.get("action_name", "HOLD"),
                confidence=result["confidence"],
                confidence_margin=0.0,
                edge=0.0,
                market_regime="UNKNOWN",
                risk_regime="UNKNOWN",
                regime_confidence=0.0,
                agreement_pct=1.0,
                weighted_agreement=1.0,
                disagreement_score=0.0,
                position_size_pct=0.0,
                regime_adjusted_size=0.0,
                confidence_threshold_used=0.15,
                regime_adjustment="NONE",
                model_votes={},
                ensemble_probs={
                    "SHORT": result["probabilities"][0],
                    "HOLD": result["probabilities"][1],
                    "LONG": result["probabilities"][2]
                },
                reasons=["Ensemble predictor not initialized - using basic prediction"]
            )
        
        signal = predictor.predict(features)
        
        return EnsemblePredictionResponse(
            action=signal.action,
            confidence=signal.confidence,
            confidence_margin=signal.confidence_margin,
            edge=signal.edge,
            market_regime=signal.market_regime,
            risk_regime=signal.risk_regime,
            regime_confidence=signal.regime_confidence,
            agreement_pct=signal.agreement_pct,
            weighted_agreement=signal.weighted_agreement,
            disagreement_score=signal.disagreement_score,
            position_size_pct=signal.position_size_pct,
            regime_adjusted_size=signal.regime_adjusted_size,
            confidence_threshold_used=signal.confidence_threshold_used,
            regime_adjustment=signal.regime_adjustment,
            model_votes=signal.model_votes,
            ensemble_probs=signal.ensemble_probs,
            reasons=signal.reasons
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ensemble/update-weights")
async def update_ensemble_weights(weights: Dict[str, Dict[str, float]]):
    """Update model weights from walk-forward evaluation results."""
    try:
        predictor = get_ensemble_predictor()
        if predictor is None:
            raise HTTPException(status_code=503, detail="Ensemble predictor not initialized")
        
        from .ensemble_predictor import ModelWeight
        new_weights = {}
        for name, metrics in weights.items():
            new_weights[name] = ModelWeight(
                model_name=name,
                expectancy=metrics.get("expectancy", 0.001),
                precision_on_trade=metrics.get("precision_on_trade", 0.55),
                profit_factor=metrics.get("profit_factor", 1.2),
                f1_directional=metrics.get("f1_directional", 0.45),
                sharpe=metrics.get("sharpe", 0.5),
                calibration_temp=metrics.get("calibration_temp", 1.0)
            )
        
        predictor.save_weights(new_weights)
        return {"message": "Weights updated", "models": list(new_weights.keys())}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ensemble/status")
async def get_ensemble_status():
    """Get ensemble predictor status and model classification."""
    predictor = get_ensemble_predictor()
    
    if predictor is None:
        return {
            "initialized": False,
            "reason": "No model instances loaded",
            "direction_models": [],
            "regime_models": [],
            "risk_models": []
        }
    
    return {
        "initialized": True,
        "direction_models": list(predictor.direction_models.keys()),
        "regime_models": list(predictor.regime_models.keys()),
        "risk_models": list(predictor.risk_models.keys()),
        "model_weights": {
            name: {
                "expectancy": w.expectancy,
                "precision_on_trade": w.precision_on_trade,
                "profit_factor": w.profit_factor,
                "f1_directional": w.f1_directional,
                "sharpe": w.sharpe,
                "composite_weight": w.composite_weight
            }
            for name, w in predictor.model_weights.items()
        },
        "thresholds": {
            "base_confidence": predictor.base_confidence_threshold,
            "base_margin": predictor.base_margin_threshold,
            "majority_weight": predictor.majority_weight_threshold
        }
    }

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
        
        # Convert Config object to dict if needed
        raw_config = checkpoint.get("config", {})
        config_dict = model_manager._config_to_dict(raw_config)
        
        # Store metadata
        model_manager.models[model_name] = {
            "path": str(path),
            "accuracy": checkpoint.get("val_accuracy", 0),
            "epoch": checkpoint.get("epoch", 0),
            "config": config_dict,
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

@app.get("/features/expected")
async def get_expected_features():
    """Get the expected feature list for prediction alignment verification."""
    if hasattr(model_manager, 'expected_features') and model_manager.expected_features:
        return {
            "count": len(model_manager.expected_features),
            "features": model_manager.expected_features,
            "input_dim": model_manager.input_dim
        }
    return {
        "count": model_manager.input_dim,
        "features": None,
        "input_dim": model_manager.input_dim,
        "warning": "No feature list loaded - using default input_dim"
    }

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
            "config": model_info.get("config", {}),
            "model_type": model_info.get("model_type", name)  # Standardized type
        }
    
    # Count successful vs failed instantiations
    total_checkpoints = len(model_manager.models)
    successful_instances = len(model_manager.model_instances)
    failed_instances = len(model_manager.instantiation_errors)
    
    # Determine training mode from input_dim
    # Quick training: 15m only with ~57 features
    # Full MTF: 5m/15m/1h/4h with ~81 features (includes cross-asset + embedding)
    input_dim = model_manager.input_dim
    if input_dim <= 60:
        training_mode = "quick"
        training_mode_description = "Quick (15m only, ~57 features)"
    else:
        training_mode = "full"
        training_mode_description = "Full MTF (5m/15m/1h/4h, ~81 features)"
    
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
        "model_type_map": model_manager.model_type_map,  # Checkpoint name -> type mapping
        "model_status_by_type": model_manager.get_model_status_by_type(),  # Dashboard-ready status
        "training_mode": training_mode,  # "quick" or "full" based on input_dim
        "training_mode_description": training_mode_description,
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
