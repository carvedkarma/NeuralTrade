/**
 * GPU Trainer Bridge
 * 
 * Connects the Replit Node.js app to the local GPU trainer FastAPI server.
 * When running locally with GPU, predictions come from the neural networks.
 * When GPU trainer is unavailable, falls back to the existing ML predictor.
 */

import { FeatureVector } from "./feature-engine";
import { updateGpuTrainerProgress, getUnifiedProgressReport } from "./unified-learning-controller";

interface GPUPredictionResponse {
  action: "LONG" | "SHORT" | "HOLD";
  probabilities: {
    LONG: number;
    SHORT: number;
    HOLD: number;
  };
  confidence: number;
  uncertainty: number;
  model_weights?: Record<string, number>;
  reasoning: string[];
}

interface QuantilePredictionResponse {
  direction_probs: {
    LONG: number;
    SHORT: number;
    HOLD: number;
  };
  quantiles: {
    q10: number;
    q25: number;
    q50: number;
    q75: number;
    q90: number;
  };
  mfe_quantiles?: {
    q10: number;
    q50: number;
    q90: number;
  };
  mae_quantiles?: {
    q10: number;
    q50: number;
    q90: number;
  };
  model_name: string;
  confidence: number;
}

// Multihead prediction response - returns ALL 6 heads
interface MultiHeadPredictionResponse {
  // Direction
  action: "LONG" | "SHORT" | "HOLD";
  direction_probs: {
    SHORT: number;
    HOLD: number;
    LONG: number;
  };
  confidence: number;
  
  // Regression (μ, σ)
  expected_return: number;
  uncertainty: number;
  edge: number;
  
  // Quantiles (learned, not heuristic)
  quantiles: {
    q10: number;
    q25: number;
    q50: number;
    q75: number;
    q90: number;
  };
  
  // Trading levels (learned from MFE/MAE)
  entry_offset_pct: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  
  // Derived price levels
  current_price: number;
  entry_price: number;
  stop_loss_price: number;
  take_profit_price: number;
  
  // Future candle predictions
  predicted_candles: {
    step: number;
    close_delta: number;
    high_delta: number;
    low_delta: number;
  }[] | null;
  
  // Trade plan
  suggested_order_type: "MAKER" | "TAKER";
  urgency: "LOW" | "MEDIUM" | "HIGH";
  position_size_pct: number;
  risk_reward_ratio: number;
  
  // Metadata
  model_name: string;
  is_multihead: boolean;
  reasons: string[];
}

interface EnsemblePredictionResponse {
  action: "LONG" | "SHORT" | "HOLD" | "NO_TRADE";
  confidence: number;
  confidence_margin: number;
  edge: number;
  
  // Regime information
  market_regime: string;
  risk_regime: string;
  regime_confidence: number;
  
  // Model agreement
  agreement_pct: number;
  weighted_agreement: number;
  disagreement_score: number;
  
  // Position sizing
  position_size_pct: number;
  regime_adjusted_size: number;
  
  // Thresholds
  confidence_threshold_used: number;
  regime_adjustment: string;
  
  // Per-model breakdown
  model_votes: Record<string, {
    action: string;
    confidence: number;
    confidence_margin: number;
    weight: number;
    probs: {
      SHORT: number;
      HOLD: number;
      LONG: number;
    };
  }>;
  
  // Ensemble probabilities
  ensemble_probs: {
    SHORT: number;
    HOLD: number;
    LONG: number;
  };
  
  // Reasons
  reasons: string[];
  
  // === Multi-head outputs (aggregated from models with forward_multihead) ===
  quantiles?: {
    q10: number;
    q25: number;
    q50: number;
    q75: number;
    q90: number;
  };
  mu?: number;  // Expected return
  sigma?: number;  // Uncertainty
  entry_offset?: number;  // Entry price offset
  sl_distance?: number;  // Stop loss distance (%)
  tp_distance?: number;  // Take profit distance (%)
  
  // === Flow Forecast outputs (from VolStateHead and AccelerationHead) ===
  vol_state?: "contraction" | "neutral" | "expansion";
  vol_state_probs?: { contraction: number; neutral: number; expansion: number };
  acceleration?: number;
  forecast_mode?: "QUANTILE_PATHS" | "NO_FORECAST";
  quantile_paths?: {
    q10: number[];
    q50: number[];
    q90: number[];
  };
}

interface EnsembleStatus {
  initialized: boolean;
  direction_models: string[];
  regime_models: string[];
  risk_models: string[];
  model_weights?: Record<string, {
    expectancy: number;
    precision_on_trade: number;
    profit_factor: number;
    f1_directional: number;
    sharpe: number;
    composite_weight: number;
  }>;
  thresholds?: {
    base_confidence: number;
    base_margin: number;
    majority_weight: number;
  };
}

interface GPUHealthResponse {
  status: string;
  gpu_available: boolean;
  gpu_name: string | null;
  gpu_memory_used: number | null;
  gpu_memory_total: number | null;
  models_loaded: string[];
  uptime_seconds: number;
}

interface GPUTrainingStatus {
  is_training: boolean;
  current_epoch: number;
  total_epochs: number;
  current_model: string | null;
  progress: number;
  metrics: Record<string, number>;
}

interface ModelStatusEntry {
  status: "pending" | "training" | "complete" | "stopped";
  accuracy: number | null;
  loss: number | null;
  epochs: number;
  best_epoch: number;
}

interface PushedGPUStatus {
  connected: boolean;
  lastPush: number | null;
  gpuAvailable: boolean;
  gpuName: string | null;
  gpuMemoryUsed: number | null;
  gpuMemoryTotal: number | null;
  isTraining: boolean;
  trainingProgress: number;
  currentModel: string | null;
  currentEpoch: number;
  totalEpochs: number;
  trainLoss: number | null;
  valLoss: number | null;
  modelsLoaded: string[];
  modelsCompleted: string[];
  modelStatus?: Record<string, ModelStatusEntry>;
  trainingMode?: "quick" | "full" | null;
  trainingModeDescription?: string | null;
  inputDim?: number | null;
}

class GPUTrainerBridge {
  private baseUrl: string;
  private isAvailable: boolean = false;
  private lastHealthCheck: number = 0;
  private healthCheckInterval: number = 30000; // 30 seconds
  private predictionMode: "stf" | "mtf" = "stf"; // Default to STF for 15m-trained models
  
  // Pushed status from remote GPU trainer
  private pushedStatus: PushedGPUStatus = {
    connected: false,
    lastPush: null,
    gpuAvailable: false,
    gpuName: null,
    gpuMemoryUsed: null,
    gpuMemoryTotal: null,
    isTraining: false,
    trainingProgress: 0,
    currentModel: null,
    currentEpoch: 0,
    totalEpochs: 0,
    trainLoss: null,
    valLoss: null,
    modelsLoaded: [],
    modelsCompleted: [],
    trainingMode: null,
    trainingModeDescription: null,
    inputDim: null
  };
  
  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }
  
  /**
   * Get the current GPU trainer URL
   */
  getUrl(): string {
    return this.baseUrl;
  }
  
  /**
   * Update the GPU trainer URL dynamically
   */
  setUrl(newUrl: string): void {
    this.baseUrl = newUrl;
    this.isAvailable = false;
    this.lastHealthCheck = 0;
    console.log(`[GPU Bridge] URL updated to: ${newUrl}`);
  }
  
  /**
   * Get the current prediction mode (STF or MTF)
   */
  getPredictionMode(): "stf" | "mtf" {
    return this.predictionMode;
  }
  
  /**
   * Set the prediction mode for inference
   * STF: Single-TimeFrame (41 features from compute_technical_features)
   * MTF: Multi-TimeFrame (66 features from MTF fusion)
   */
  setPredictionMode(mode: "stf" | "mtf"): void {
    this.predictionMode = mode;
    console.log(`[GPU Bridge] Prediction mode set to: ${mode.toUpperCase()}`);
  }
  
  /**
   * Update pushed status from remote GPU trainer
   */
  updatePushedStatus(status: PushedGPUStatus): void {
    this.pushedStatus = status;
    
    if (status.connected && status.trainingProgress > 0) {
      const unifiedReport = getUnifiedProgressReport();
      const estimatedIdx = Math.floor((status.trainingProgress / 100) * unifiedReport.trainableCandles) + 50;
      updateGpuTrainerProgress(estimatedIdx, status.trainingProgress >= 100);
    }
  }
  
  /**
   * Get pushed status (for dashboard)
   */
  getPushedStatus(): PushedGPUStatus {
    return this.pushedStatus;
  }
  
  /**
   * Fetch models status from GPU trainer (includes training mode)
   */
  async fetchModelsStatus(): Promise<any | null> {
    try {
      const response = await fetch(`${this.baseUrl}/models/status`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(3000) // 3 second timeout
      });
      
      if (response.ok) {
        return await response.json();
      }
      return null;
    } catch (error) {
      return null;
    }
  }
  
  /**
   * Check if the GPU trainer is available
   */
  async checkHealth(): Promise<GPUHealthResponse | null> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      if (response.ok) {
        const health = await response.json() as GPUHealthResponse;
        this.isAvailable = true;
        this.lastHealthCheck = Date.now();
        return health;
      }
      
      this.isAvailable = false;
      return null;
    } catch (error) {
      this.isAvailable = false;
      console.log("GPU trainer not available, using fallback ML predictor");
      return null;
    }
  }
  
  /**
   * Check if GPU trainer is available (with caching)
   */
  async isGPUAvailable(): Promise<boolean> {
    const now = Date.now();
    if (now - this.lastHealthCheck > this.healthCheckInterval) {
      await this.checkHealth();
    }
    return this.isAvailable;
  }
  
  /**
   * Get prediction from GPU neural networks
   */
  async predict(features: number[][]): Promise<GPUPredictionResponse | null> {
    if (!await this.isGPUAvailable()) {
      return null;
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features: features,
          sequence_length: features.length
        }),
        signal: AbortSignal.timeout(10000) // 10 second timeout
      });
      
      if (response.ok) {
        return await response.json() as GPUPredictionResponse;
      }
      
      return null;
    } catch (error) {
      console.error("GPU prediction failed:", error);
      return null;
    }
  }
  
  /**
   * Convert FeatureVector to array format for GPU prediction
   * Maps all FeatureVector properties including OHLCV for neural network input
   */
  featureVectorToArray(feature: FeatureVector): number[] {
    // Convert kalmanRegime to numeric: bull=1, bear=-1, chop=0
    const kalmanRegimeNum = feature.kalmanRegime === "bull" ? 1 : 
                            feature.kalmanRegime === "bear" ? -1 : 0;
    
    // Convert volatilityRegime to numeric: high=1, medium=0.5, low=0
    const volatilityRegimeNum = feature.volatilityRegime === "high" ? 1 : 
                                feature.volatilityRegime === "medium" ? 0.5 : 0;
    
    // Core features (57 values) + embedding (24 values) = 81 total features
    const coreFeatures = [
      // OHLCV data (10 features) - raw price action for neural networks
      feature.price,
      feature.open,
      feature.high,
      feature.low,
      feature.close,
      feature.volume,
      feature.normalizedPrice,
      feature.normalizedVolume,
      feature.candleBody,
      feature.candleRange,
      
      // Price returns at multiple lookbacks (4 features)
      feature.returns1,
      feature.returns2,
      feature.returns4,
      feature.returns8,
      
      // EMA features (5 features)
      feature.ema20,
      feature.ema50,
      feature.ema20Slope,
      feature.ema50Slope,
      feature.emaDistance,
      
      // Breakout distances (2 features)
      feature.breakoutDistanceHigh,
      feature.breakoutDistanceLow,
      
      // Volatility features (5 features)
      feature.efficiencyRatio,
      feature.atr14,
      feature.volatility,
      feature.bollingerWidth,
      volatilityRegimeNum,
      
      // Momentum indicators (7 features)
      feature.rsi14,
      feature.macd,
      feature.macdSignal,
      feature.macdHist,
      feature.stochK,
      feature.stochD,
      feature.momentum,
      
      // Trend indicators (4 features)
      feature.adx,
      feature.plusDi,
      feature.minusDi,
      feature.trendStrength,
      
      // Volume features (3 features)
      feature.obv,
      feature.obvSlope,
      feature.volumeRatio,
      
      // Kalman filter features (4 features)
      feature.kalmanFast,
      feature.kalmanSlow,
      feature.kalmanSpread,
      kalmanRegimeNum,
      
      // Price position and velocity (3 features)
      feature.pricePosition,
      feature.priceVelocity,
      feature.priceAcceleration,
      
      // Cross-asset features (10 features)
      feature.ethBtcCorrelation,
      feature.solBtcCorrelation,
      feature.bnbBtcCorrelation,
      feature.ethRelativeStrength,
      feature.solRelativeStrength,
      feature.bnbRelativeStrength,
      feature.ethMomentumDivergence,
      feature.solMomentumDivergence,
      feature.bnbMomentumDivergence,
      feature.cryptoSectorMomentum,
    ];
    
    // Append the 24-dimensional embedding for pattern matching
    // Total: 57 core features + 24 embedding = 81 features
    return [...coreFeatures, ...feature.embedding];
  }
  
  /**
   * Get training status
   */
  async getTrainingStatus(): Promise<GPUTrainingStatus | null> {
    if (!await this.isGPUAvailable()) {
      return null;
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/training/status`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        return await response.json() as GPUTrainingStatus;
      }
      
      return null;
    } catch (error) {
      return null;
    }
  }
  
  /**
   * Start training a model
   */
  async startTraining(modelType: string, epochs: number = 100): Promise<boolean> {
    if (!await this.isGPUAvailable()) {
      return false;
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/training/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_type: modelType,
          epochs: epochs,
          batch_size: 64,
          learning_rate: 0.0001
        }),
        signal: AbortSignal.timeout(10000)
      });
      
      return response.ok;
    } catch (error) {
      return false;
    }
  }
  
  /**
   * Get professional ensemble prediction with regime gating
   * 
   * This uses:
   * - Direction models (Transformer, TFT, LSTM, CNN) for voting
   * - VAE for market regime detection (trend/range/chop)
   * - GNN for risk regime detection (risk-on/off)
   * - Walk-forward metric weighting
   * 
   * @deprecated Use predictEnsembleFromCandles for proper MTF feature alignment
   */
  async predictEnsemble(features: number[][]): Promise<EnsemblePredictionResponse | null> {
    if (!await this.isGPUAvailable()) {
      return null;
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/predict/ensemble`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features: features
        }),
        signal: AbortSignal.timeout(15000) // 15 second timeout (ensemble is slower)
      });
      
      if (response.ok) {
        return await response.json() as EnsemblePredictionResponse;
      }
      
      console.error("Ensemble prediction failed:", await response.text());
      return null;
    } catch (error) {
      console.error("Ensemble prediction error:", error);
      return null;
    }
  }
  
  /**
   * Get professional ensemble prediction from raw multi-timeframe candle data.
   * 
   * This endpoint computes MTF features (same as training) server-side,
   * ensuring feature alignment between training and inference.
   * 
   * Features computed: ~66 MTF features (5m/15m/1h/4h)
   */
  async predictEnsembleFromCandles(
    candles15m: { timestamp: number; open: number; high: number; low: number; close: number; volume: number }[],
    candles5m?: { timestamp: number; open: number; high: number; low: number; close: number; volume: number }[],
    candles1h?: { timestamp: number; open: number; high: number; low: number; close: number; volume: number }[],
    candles4h?: { timestamp: number; open: number; high: number; low: number; close: number; volume: number }[],
    symbol: string = "BTCUSDT"
  ): Promise<EnsemblePredictionResponse | null> {
    if (!await this.isGPUAvailable()) {
      return null;
    }
    
    try {
      const requestBody: any = {
        candles_15m: candles15m,
        symbol: symbol
      };
      
      if (candles5m && candles5m.length >= 50) {
        requestBody.candles_5m = candles5m;
      }
      if (candles1h && candles1h.length >= 50) {
        requestBody.candles_1h = candles1h;
      }
      if (candles4h && candles4h.length >= 20) {
        requestBody.candles_4h = candles4h;
      }
      
      // Use the configured prediction mode
      const mode = this.predictionMode;
      const url = `${this.baseUrl}/predict/ensemble/candles?mode=${mode}`;
      console.log(`[GPU Bridge] Prediction request using mode: ${mode.toUpperCase()}`);
      
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal: AbortSignal.timeout(20000) // 20 second timeout (feature computation + ensemble)
      });
      
      if (response.ok) {
        return await response.json() as EnsemblePredictionResponse;
      }
      
      console.error(`${mode.toUpperCase()} Ensemble prediction failed:`, await response.text());
      return null;
    } catch (error) {
      console.error(`${this.predictionMode.toUpperCase()} Ensemble prediction error:`, error);
      return null;
    }
  }
  
  /**
   * Quantile regression prediction for Entry/SL/TP derivation
   */
  async predictQuantile(features: number[]): Promise<QuantilePredictionResponse | null> {
    if (!await this.isGPUAvailable()) {
      return null;
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/predict/quantile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features: [features]  // Wrap in array for batch format
        }),
        signal: AbortSignal.timeout(10000)
      });
      
      if (response.ok) {
        return await response.json() as QuantilePredictionResponse;
      }
      
      console.error("Quantile prediction failed:", await response.text());
      return null;
    } catch (error) {
      console.error("Quantile prediction error:", error);
      return null;
    }
  }
  
  /**
   * Get ensemble predictor status
   */
  async getEnsembleStatus(): Promise<EnsembleStatus | null> {
    if (!await this.isGPUAvailable()) {
      return null;
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/ensemble/status`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        return await response.json() as EnsembleStatus;
      }
      
      return null;
    } catch (error) {
      console.error("Failed to get ensemble status:", error);
      return null;
    }
  }
  
  /**
   * Get multihead prediction from raw candle data.
   * 
   * This is the CANONICAL endpoint for multi-head model inference.
   * Returns ALL 6 heads: direction, μ/σ, quantiles, entry/SL/TP, candles.
   * 
   * Uses forward_multihead() internally for proper multi-head output.
   */
  async predictMultiheadFromCandles(
    candles: { timestamp: number; open: number; high: number; low: number; close: number; volume: number }[]
  ): Promise<MultiHeadPredictionResponse | null> {
    if (!await this.isGPUAvailable()) {
      return null;
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/predict/multihead/candles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          candles: candles
        }),
        signal: AbortSignal.timeout(15000) // 15 second timeout
      });
      
      if (response.ok) {
        return await response.json() as MultiHeadPredictionResponse;
      }
      
      console.error("Multihead prediction failed:", await response.text());
      return null;
    } catch (error) {
      console.error("Multihead prediction error:", error);
      return null;
    }
  }
  
  /**
   * Get GPU metrics for dashboard
   */
  async getGPUMetrics(): Promise<Record<string, any> | null> {
    const health = await this.checkHealth();
    if (!health) {
      return null;
    }
    
    const training = await this.getTrainingStatus();
    const ensembleStatus = await this.getEnsembleStatus();
    
    return {
      gpuAvailable: health.gpu_available,
      gpuName: health.gpu_name,
      gpuMemoryUsed: health.gpu_memory_used,
      gpuMemoryTotal: health.gpu_memory_total,
      gpuMemoryPercent: health.gpu_memory_used && health.gpu_memory_total 
        ? (health.gpu_memory_used / health.gpu_memory_total) * 100 
        : 0,
      modelsLoaded: health.models_loaded,
      uptime: health.uptime_seconds,
      isTraining: training?.is_training || false,
      trainingProgress: training?.progress || 0,
      currentModel: training?.current_model,
      trainingMetrics: training?.metrics || {},
      ensemble: ensembleStatus
    };
  }
}

// Singleton instance
export const gpuBridge = new GPUTrainerBridge(
  process.env.GPU_TRAINER_URL || "http://localhost:8000"
);

export type { GPUPredictionResponse, GPUHealthResponse, GPUTrainingStatus, EnsemblePredictionResponse, EnsembleStatus, MultiHeadPredictionResponse };
