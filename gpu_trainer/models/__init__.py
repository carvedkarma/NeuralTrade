"""
Neural Network Models for BTC Futures Trading

This module contains all the deep learning architectures:
- Transformer: Attention-based sequence learning
- LSTM: Recurrent networks for time series
- CNN: Convolutional networks for pattern recognition
- VAE: Variational autoencoders for representation learning
- GNN: Graph neural networks for cross-asset modeling
- RL: Reinforcement learning agents
- Sentiment: NLP models for news/social media
- Ensemble: Meta-learning and model combination
"""

from .base import BaseModel, PositionalEncoding, AttentionBlock, ResidualBlock
from .transformer import TransformerPriceModel, TemporalFusionTransformer
from .lstm import BidirectionalLSTM, StackedLSTM, ConvLSTM
from .cnn import ResNetPrice, InceptionNet, WaveNet
from .vae import MarketVAE, BetaVAE, ConditionalVAE
from .gnn import CrossAssetGNN, TemporalGNN
from .rl_agent import PPOAgent, TradingEnvironment, ActorCritic
from .sentiment import SentimentEncoder, MultiModalSentiment, SentimentPricePredictor
from .ensemble import MetaLearner, DeepEnsemble, MasterEnsemble, OnlineLearningEnsemble
from .multihead import (
    MultiHeadOutput, MultiHeadTransformer, MultiScaleTransformer, MultiHeadTFT, MultiHeadLSTM, 
    MultiHeadCNN, MultiHeadGNN, MultiHeadVAE, get_multihead_model,
    MultiScaleTemporalEmbedding, MultiScaleAttentionBlock
)

__all__ = [
    # Base
    "BaseModel",
    "PositionalEncoding", 
    "AttentionBlock",
    "ResidualBlock",
    
    # Transformer
    "TransformerPriceModel",
    "TemporalFusionTransformer",
    
    # LSTM
    "BidirectionalLSTM",
    "StackedLSTM",
    "ConvLSTM",
    
    # CNN
    "ResNetPrice",
    "InceptionNet",
    "WaveNet",
    
    # VAE
    "MarketVAE",
    "BetaVAE",
    "ConditionalVAE",
    
    # GNN
    "CrossAssetGNN",
    "TemporalGNN",
    
    # RL
    "PPOAgent",
    "TradingEnvironment",
    "ActorCritic",
    
    # Sentiment
    "SentimentEncoder",
    "MultiModalSentiment",
    "SentimentPricePredictor",
    
    # Ensemble
    "MetaLearner",
    "DeepEnsemble",
    "MasterEnsemble",
    "OnlineLearningEnsemble",
    
    # Multi-Head Models
    "MultiHeadOutput",
    "MultiHeadTransformer",
    "MultiScaleTransformer",
    "MultiScaleTemporalEmbedding",
    "MultiScaleAttentionBlock",
    "MultiHeadTFT",
    "MultiHeadLSTM",
    "MultiHeadCNN",
    "MultiHeadGNN",
    "MultiHeadVAE",
    "get_multihead_model",
]
