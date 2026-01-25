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
}

class GPUTrainerBridge {
  private baseUrl: string;
  private isAvailable: boolean = false;
  private lastHealthCheck: number = 0;
  private healthCheckInterval: number = 30000; // 30 seconds
  
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
    modelsCompleted: []
  };
  
  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
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
   */
  featureVectorToArray(feature: FeatureVector): number[] {
    return [
      feature.price,
      feature.open,
      feature.high,
      feature.low,
      feature.volume,
      feature.rsi14,
      feature.rsi7,
      feature.macd,
      feature.macdSignal,
      feature.macdHist,
      feature.bbUpper,
      feature.bbMiddle,
      feature.bbLower,
      feature.bbWidth,
      feature.bbPosition,
      feature.atr14,
      feature.atr7,
      feature.adx,
      feature.plusDi,
      feature.minusDi,
      feature.stochK,
      feature.stochD,
      feature.obv,
      feature.obvSma,
      feature.volumeRatio,
      feature.volatility20,
      feature.volatility50,
      feature.priceChange1,
      feature.priceChange5,
      feature.priceChange20,
      feature.ema9,
      feature.ema21,
      feature.sma50,
      feature.sma200,
      feature.kalmanFast,
      feature.kalmanSlow,
      feature.kalmanRegime === "up" ? 1 : feature.kalmanRegime === "down" ? -1 : 0,
      feature.efficiencyRatio,
      feature.trendStrength,
      feature.momentum
    ];
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
   * Get GPU metrics for dashboard
   */
  async getGPUMetrics(): Promise<Record<string, any> | null> {
    const health = await this.checkHealth();
    if (!health) {
      return null;
    }
    
    const training = await this.getTrainingStatus();
    
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
      trainingMetrics: training?.metrics || {}
    };
  }
}

// Singleton instance
export const gpuBridge = new GPUTrainerBridge(
  process.env.GPU_TRAINER_URL || "http://localhost:8000"
);

export type { GPUPredictionResponse, GPUHealthResponse, GPUTrainingStatus };
