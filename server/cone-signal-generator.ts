/**
 * Cone-Based Signal Generator
 * 
 * Uses probability cone quantiles (q10-q90) to generate trading signals.
 * Implements institutional-grade trade gating with auto-calibrated edge thresholds.
 * 
 * Trade Gate Requirements:
 * - p_hold < 0.60 (not too choppy)
 * - abs(mu) >= 0.0020 (at least 0.2% expected move)
 * - edge = abs(mu)/(q90-q10) >= threshold (auto-calibrated)
 * - unc <= 0.030 (uncertainty under 3%)
 * - RR >= 1.2 (risk-reward ratio)
 * 
 * Levels:
 * - Entry = last candle close
 * - SL from q10/q90 tail (adverse quantile)
 * - TP from q75/q25 (favorable inner quantile)
 * 
 * Cooldown: 6 bars after any trade signal
 */

import { SignalType, ConeSignalResponse, FlowForecast as SchemaFlowForecast } from "@shared/schema";

export interface QuantilePrediction {
  q10: number;  // All as decimal returns (e.g., -0.02 = -2%)
  q25: number;
  q50: number;
  q75: number;
  q90: number;
}

export interface DirectionProbabilities {
  probUp: number;
  probDown: number;
  probHold: number;
}

export interface FlowForecast {
  volState: "contraction" | "neutral" | "expansion";
  volStateProbs: { contraction: number; neutral: number; expansion: number };
  acceleration: number;
  forecastMode: "QUANTILE_PATHS" | "NO_FORECAST";
  quantilePaths?: {
    q10: number[];
    q50: number[];
    q90: number[];
  };
}

export interface PredictedCandle {
  step: number;
  close_delta: number;  // Delta from current close
  high_delta?: number;
  low_delta?: number;
}

export interface ConeInput {
  currentPrice: number;
  quantiles: QuantilePrediction;
  probs: DirectionProbabilities;
  mu: number;       // Expected return (decimal)
  sigma?: number;   // Uncertainty (decimal)
  timestamp: number;
  flowForecast?: FlowForecast;  // Optional flow forecast from GPU trainer
  predictedCandles?: PredictedCandle[];  // Model's candle head output (first 5 bars)
}

interface EdgeHistoryEntry {
  timestamp: number;
  edge: number;
}

interface TradeHistoryEntry {
  timestamp: number;
  direction: SignalType;
}

class ConeSignalGenerator {
  // Configuration
  private readonly HORIZON_BARS = 16;
  private readonly COOLDOWN_BARS = 6;
  private readonly MIN_MU = 0.0020;        // 0.2% minimum expected move
  private readonly MAX_UNCERTAINTY = 0.030; // 3% max uncertainty
  private readonly MAX_PROB_HOLD = 0.60;    // 60% max hold probability
  private readonly MIN_RR = 1.2;            // Minimum risk-reward ratio
  private readonly TARGET_TRADES_PER_DAY = 3.5; // Target 3-4 trades/day
  
  // State
  private edgeThreshold = 0.14;  // Default, will be auto-calibrated
  private edgeHistory: EdgeHistoryEntry[] = [];
  private tradeHistory: TradeHistoryEntry[] = [];
  private lastSignalTimestamp: number = 0;
  private barDurationMs = 15 * 60 * 1000;  // 15 minutes
  
  constructor() {
    console.log("[ConeSignal] Initialized with default edge threshold:", this.edgeThreshold);
  }
  
  /**
   * Generate a cone-based trading signal
   */
  generateSignal(input: ConeInput): ConeSignalResponse {
    const { currentPrice, quantiles, probs, mu, sigma, timestamp, flowForecast } = input;
    
    // Calculate derived values
    const coneWidth = quantiles.q90 - quantiles.q10;
    const edge = coneWidth > 0 ? Math.abs(mu) / coneWidth : 0;
    
    // Record edge for auto-calibration
    this.recordEdge(timestamp, edge);
    
    // Check cooldown
    const cooldownBarsRemaining = this.getCooldownBarsRemaining(timestamp);
    
    // Determine direction from mu and probabilities
    const direction = this.determineDirection(mu, probs);
    
    // Calculate trade levels
    const levels = this.calculateLevels(currentPrice, quantiles, direction);
    
    // Check all gate conditions
    const holdReasons: string[] = [];
    
    // Flow Forecast: Volatility Gate
    // NO_FORECAST when vol_state==contraction OR spread < 3×cost
    if (flowForecast?.forecastMode === "NO_FORECAST") {
      holdReasons.push(`No tradeable flow: ${flowForecast.volState} regime (compression)`);
    }
    
    if (cooldownBarsRemaining > 0) {
      holdReasons.push(`Cooldown: ${cooldownBarsRemaining} bars remaining`);
    }
    
    if (probs.probHold >= this.MAX_PROB_HOLD) {
      holdReasons.push(`High chop probability: ${(probs.probHold * 100).toFixed(1)}% >= ${(this.MAX_PROB_HOLD * 100)}%`);
    }
    
    if (Math.abs(mu) < this.MIN_MU) {
      holdReasons.push(`Expected move too small: ${(Math.abs(mu) * 100).toFixed(3)}% < ${(this.MIN_MU * 100).toFixed(1)}%`);
    }
    
    if (edge < this.edgeThreshold) {
      holdReasons.push(`Edge too low: ${edge.toFixed(3)} < ${this.edgeThreshold.toFixed(3)}`);
    }
    
    if (sigma !== undefined && sigma > this.MAX_UNCERTAINTY) {
      holdReasons.push(`Uncertainty too high: ${(sigma * 100).toFixed(2)}% > ${(this.MAX_UNCERTAINTY * 100)}%`);
    }
    
    if (levels.riskReward !== null && levels.riskReward < this.MIN_RR) {
      holdReasons.push(`RR too low: ${levels.riskReward.toFixed(2)} < ${this.MIN_RR}`);
    }
    
    // Final signal decision
    const finalDirection: SignalType = holdReasons.length > 0 ? "HOLD" : direction;
    
    // Record trade if not HOLD
    if (finalDirection !== "HOLD") {
      this.recordTrade(timestamp, finalDirection);
    }
    
    return {
      direction: finalDirection,
      entryPrice: levels.entry,
      stopLoss: levels.stopLoss,
      takeProfit: levels.takeProfit,
      mu,
      sigma,
      edge,
      riskReward: levels.riskReward,
      quantiles,
      probUp: probs.probUp,
      probDown: probs.probDown,
      probHold: probs.probHold,
      holdReasons,
      edgeThreshold: this.edgeThreshold,
      cooldownBarsRemaining,
      timestamp,
      flowForecast: flowForecast,
    };
  }
  
  /**
   * Determine direction from mu and probabilities
   */
  private determineDirection(mu: number, probs: DirectionProbabilities): SignalType {
    // Primary signal from mu
    if (mu > 0 && probs.probUp > probs.probDown) {
      return "LONG";
    } else if (mu < 0 && probs.probDown > probs.probUp) {
      return "SHORT";
    }
    
    // If mu and probs disagree, use the stronger signal
    if (Math.abs(mu) > this.MIN_MU * 2) {
      // Strong mu signal
      return mu > 0 ? "LONG" : "SHORT";
    } else if (Math.abs(probs.probUp - probs.probDown) > 0.2) {
      // Strong probability signal
      return probs.probUp > probs.probDown ? "LONG" : "SHORT";
    }
    
    return "HOLD";
  }
  
  /**
   * Calculate entry/SL/TP levels from quantiles
   */
  private calculateLevels(
    currentPrice: number,
    quantiles: QuantilePrediction,
    direction: SignalType
  ): {
    entry: number;
    stopLoss: number | null;
    takeProfit: number | null;
    riskReward: number | null;
  } {
    const entry = currentPrice;
    
    if (direction === "HOLD") {
      return { entry, stopLoss: null, takeProfit: null, riskReward: null };
    }
    
    let stopLoss: number;
    let takeProfit: number;
    
    if (direction === "LONG") {
      // SL from q10 tail, TP from q75
      stopLoss = currentPrice * (1 + quantiles.q10);
      takeProfit = currentPrice * (1 + quantiles.q75);
    } else {
      // SL from q90 tail, TP from q25
      stopLoss = currentPrice * (1 + quantiles.q90);
      takeProfit = currentPrice * (1 + quantiles.q25);
    }
    
    // Calculate risk-reward
    const risk = Math.abs(entry - stopLoss);
    const reward = Math.abs(takeProfit - entry);
    const riskReward = risk > 0 ? reward / risk : null;
    
    return { entry, stopLoss, takeProfit, riskReward };
  }
  
  /**
   * Record edge value for auto-calibration
   */
  private recordEdge(timestamp: number, edge: number) {
    this.edgeHistory.push({ timestamp, edge });
    
    // Keep last 7 days of edge history (672 bars at 15min)
    const sevenDaysAgo = timestamp - 7 * 24 * 60 * 60 * 1000;
    this.edgeHistory = this.edgeHistory.filter(e => e.timestamp > sevenDaysAgo);
    
    // Auto-calibrate every 96 bars (1 day)
    if (this.edgeHistory.length >= 96 && this.edgeHistory.length % 96 === 0) {
      this.autoCalibrate();
    }
  }
  
  /**
   * Record a trade for cooldown tracking
   */
  private recordTrade(timestamp: number, direction: SignalType) {
    this.tradeHistory.push({ timestamp, direction });
    this.lastSignalTimestamp = timestamp;
    
    // Keep last 7 days
    const sevenDaysAgo = timestamp - 7 * 24 * 60 * 60 * 1000;
    this.tradeHistory = this.tradeHistory.filter(t => t.timestamp > sevenDaysAgo);
  }
  
  /**
   * Get remaining cooldown bars
   */
  private getCooldownBarsRemaining(currentTimestamp: number): number {
    if (this.lastSignalTimestamp === 0) return 0;
    
    const barsSinceLastTrade = Math.floor(
      (currentTimestamp - this.lastSignalTimestamp) / this.barDurationMs
    );
    
    return Math.max(0, this.COOLDOWN_BARS - barsSinceLastTrade);
  }
  
  /**
   * Auto-calibrate edge threshold to target 3-4 trades per day
   */
  private autoCalibrate() {
    if (this.edgeHistory.length < 96) return; // Need at least 1 day
    
    const edges = this.edgeHistory.map(e => e.edge).sort((a, b) => b - a);
    
    // Calculate bars per day
    const barsPerDay = 96; // 15-min bars
    
    // Target trades per day = 3.5 (middle of 3-4)
    // So we want the edge threshold that would pass TARGET_TRADES_PER_DAY signals
    const targetTradesPerPeriod = this.TARGET_TRADES_PER_DAY * (this.edgeHistory.length / barsPerDay);
    
    // Find the percentile that gives us this many trades
    const percentileIndex = Math.floor(targetTradesPerPeriod);
    
    if (percentileIndex < edges.length && percentileIndex >= 0) {
      const newThreshold = edges[percentileIndex];
      
      // Clamp to reasonable range [0.05, 0.30]
      this.edgeThreshold = Math.max(0.05, Math.min(0.30, newThreshold));
      
      console.log(
        `[ConeSignal] Auto-calibrated edge threshold: ${this.edgeThreshold.toFixed(4)} ` +
        `(${((percentileIndex / edges.length) * 100).toFixed(1)}th percentile, ` +
        `targeting ${this.TARGET_TRADES_PER_DAY} trades/day)`
      );
    }
  }
  
  /**
   * Get current edge threshold
   */
  getEdgeThreshold(): number {
    return this.edgeThreshold;
  }
  
  /**
   * Get edge percentile for display
   */
  getEdgePercentile(): number {
    if (this.edgeHistory.length === 0) return 0;
    
    const edges = this.edgeHistory.map(e => e.edge).sort((a, b) => a - b);
    const belowThreshold = edges.filter(e => e < this.edgeThreshold).length;
    
    return (belowThreshold / edges.length) * 100;
  }
  
  /**
   * Get recent trade count (last 24 hours)
   */
  getRecentTradeCount(): number {
    const dayAgo = Date.now() - 24 * 60 * 60 * 1000;
    return this.tradeHistory.filter(t => t.timestamp > dayAgo).length;
  }
  
  /**
   * Force update the edge threshold (for testing)
   */
  setEdgeThreshold(threshold: number) {
    this.edgeThreshold = Math.max(0.05, Math.min(0.30, threshold));
    console.log(`[ConeSignal] Edge threshold set to: ${this.edgeThreshold.toFixed(4)}`);
  }
  
  /**
   * Reset cooldown (for testing or after position close)
   */
  resetCooldown() {
    this.lastSignalTimestamp = 0;
  }
  
  /**
   * Get statistics for display
   */
  getStats() {
    return {
      edgeThreshold: this.edgeThreshold,
      edgePercentile: this.getEdgePercentile(),
      recentTradeCount: this.getRecentTradeCount(),
      edgeHistorySize: this.edgeHistory.length,
      cooldownBarsRemaining: this.getCooldownBarsRemaining(Date.now()),
      config: {
        horizonBars: this.HORIZON_BARS,
        cooldownBars: this.COOLDOWN_BARS,
        minMu: this.MIN_MU,
        maxUncertainty: this.MAX_UNCERTAINTY,
        maxProbHold: this.MAX_PROB_HOLD,
        minRR: this.MIN_RR,
        targetTradesPerDay: this.TARGET_TRADES_PER_DAY,
      },
    };
  }
  
  /**
   * Generate multi-step band forecast (upgraded from simple cone)
   * 
   * Uses two data sources blended together:
   * 1. Near-term (steps 1-5): Use predicted candles from model's candle head
   * 2. Far-term (steps 6-16): Use quantile cone with alpha shaping based on vol_state
   * 
   * Alpha shaping:
   * - expansion: 1.5 (fast growth)
   * - neutral: 1.0 (linear)
   * - contraction: 0.7 (concave, slower growth)
   */
  generateMultiStepBand(
    currentPrice: number,
    quantiles: QuantilePrediction,
    predictedCandles?: PredictedCandle[],
    volState: "contraction" | "neutral" | "expansion" = "neutral"
  ): { q10: number[]; q50: number[]; q90: number[] } {
    const horizon = this.HORIZON_BARS;
    const nearTermSteps = 5;  // First 5 bars from candle head
    
    // Alpha shaping based on vol_state
    const alphaMap = {
      expansion: 1.5,    // Fast growth curve
      neutral: 1.0,      // Linear
      contraction: 0.7,  // Concave, slower growth
    };
    const alpha = alphaMap[volState] || 1.0;
    
    const q10Path: number[] = [];
    const q50Path: number[] = [];
    const q90Path: number[] = [];
    
    for (let k = 1; k <= horizon; k++) {
      const t = k / horizon;  // 0 to 1
      
      if (k <= nearTermSteps && predictedCandles && predictedCandles.length >= k) {
        // Near-term: Use predicted candle head output
        const candle = predictedCandles[k - 1];
        const closeDelta = candle.close_delta;
        
        // Use high/low deltas if available, otherwise estimate from quantile width
        const highDelta = candle.high_delta ?? closeDelta + Math.abs(quantiles.q90 - quantiles.q50) * t;
        const lowDelta = candle.low_delta ?? closeDelta - Math.abs(quantiles.q50 - quantiles.q10) * t;
        
        q10Path.push(currentPrice * (1 + lowDelta));
        q50Path.push(currentPrice * (1 + closeDelta));
        q90Path.push(currentPrice * (1 + highDelta));
      } else {
        // Far-term: Use quantile cone with alpha shaping
        // path[k] = close * exp((k/h)^α * quantile)
        const shapedT = Math.pow(t, alpha);
        
        // Apply alpha-shaped projection
        q10Path.push(currentPrice * Math.exp(shapedT * quantiles.q10));
        q50Path.push(currentPrice * Math.exp(shapedT * quantiles.q50));
        q90Path.push(currentPrice * Math.exp(shapedT * quantiles.q90));
      }
    }
    
    return { q10: q10Path, q50: q50Path, q90: q90Path };
  }
}

// Singleton instance
export const coneSignalGenerator = new ConeSignalGenerator();

/**
 * Outcome tracker for cone signals
 * Monitors pending signals and updates their outcomes when price hits SL/TP
 */
export interface OutcomeUpdate {
  id: number;
  outcome: "HIT_TP" | "HIT_SL" | "EXPIRED";
  exitPrice: number;
  exitTimestamp: number;
  pnlPercent: number;
  candlesHeld: number;
  mfe: number;
  mae: number;
}

export async function checkConeSignalOutcomes(
  pendingSignals: Array<{
    id: number;
    timestamp: number;
    direction: string;
    entryPrice: number;
    stopLoss: number | null;
    takeProfit: number | null;
  }>,
  currentCandle: { timestamp: number; high: number; low: number; close: number },
  horizonBars: number = 16,
  barDurationMs: number = 15 * 60 * 1000
): Promise<OutcomeUpdate[]> {
  const updates: OutcomeUpdate[] = [];
  const now = currentCandle.timestamp;
  
  for (const signal of pendingSignals) {
    // Check expiry (horizon bars passed)
    const barsSinceSignal = Math.floor((now - signal.timestamp) / barDurationMs);
    const isExpired = barsSinceSignal >= horizonBars;
    
    // Check if TP or SL was hit
    let outcome: "HIT_TP" | "HIT_SL" | "EXPIRED" | null = null;
    let exitPrice = currentCandle.close;
    
    if (signal.direction === "LONG") {
      if (signal.takeProfit && currentCandle.high >= signal.takeProfit) {
        outcome = "HIT_TP";
        exitPrice = signal.takeProfit;
      } else if (signal.stopLoss && currentCandle.low <= signal.stopLoss) {
        outcome = "HIT_SL";
        exitPrice = signal.stopLoss;
      }
    } else if (signal.direction === "SHORT") {
      if (signal.takeProfit && currentCandle.low <= signal.takeProfit) {
        outcome = "HIT_TP";
        exitPrice = signal.takeProfit;
      } else if (signal.stopLoss && currentCandle.high >= signal.stopLoss) {
        outcome = "HIT_SL";
        exitPrice = signal.stopLoss;
      }
    }
    
    // If expired and no SL/TP hit, mark as expired
    if (!outcome && isExpired) {
      outcome = "EXPIRED";
      exitPrice = currentCandle.close;
    }
    
    if (outcome) {
      // Calculate PnL
      let pnlPercent = 0;
      if (signal.direction === "LONG") {
        pnlPercent = ((exitPrice - signal.entryPrice) / signal.entryPrice) * 100;
      } else if (signal.direction === "SHORT") {
        pnlPercent = ((signal.entryPrice - exitPrice) / signal.entryPrice) * 100;
      }
      
      updates.push({
        id: signal.id,
        outcome,
        exitPrice,
        exitTimestamp: now,
        pnlPercent,
        candlesHeld: barsSinceSignal,
        mfe: 0, // Would need to track max favorable excursion over time
        mae: 0, // Would need to track max adverse excursion over time
      });
    }
  }
  
  return updates;
}
