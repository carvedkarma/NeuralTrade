/**
 * Institution-Grade Trade Decision Engine
 * 
 * Implements research-backed trading logic with:
 * - Horizon-specific edge thresholds and confidence gates
 * - Multi-horizon decision logic (15 & 60 primary, 240 trend filter)
 * - Enhanced NO-TRADE conditions
 * - Bounded edge-weighted Kelly sizing
 * - Time stops at horizon expiry
 */

import { HORIZON_CONFIG, NO_TRADE_CONDITIONS, TRADING_COSTS } from "./gpu-data-export";

export interface PredictionWithUncertainty {
  horizon: number;
  mu: number;          // Expected edge (μ)
  sigma: number;       // Uncertainty (σ)
  direction: number;   // 1 = long, -1 = short, 0 = neutral
  confidence: number;  // μ/σ ratio
  meetsThreshold: boolean;
}

export interface TradeDecision {
  action: "LONG" | "SHORT" | "HOLD";
  confidence: number;
  primaryHorizon: number;
  edge: number;
  size: number;           // Position size as fraction of equity
  stopLoss: number;       // ATR-based stop loss distance
  takeProfit: number;     // Initial take profit distance
  maxHoldBars: number;    // Time stop
  reasons: string[];
  vetoes: string[];
}

export interface HorizonPredictions {
  h15: PredictionWithUncertainty;
  h60: PredictionWithUncertainty;
  h240: PredictionWithUncertainty;
}

export interface MarketContext {
  atr14: number;
  volatility20: number;
  volatilityPercentile: number;  // Where current vol sits in historical distribution
  recentLosses: number;          // Count of recent consecutive losses
  fundingRate: number;
  fundingRateChange: number;     // Change over last N bars
  spreadProxy: number;           // Current spread estimate
}

/**
 * Compute confidence ratio (μ/σ) for a prediction
 */
export function computeConfidenceRatio(mu: number, sigma: number): number {
  if (sigma <= 0 || !Number.isFinite(sigma)) return 0;
  if (!Number.isFinite(mu)) return 0;
  return mu / sigma;
}

/**
 * Get the minimum confidence threshold for a horizon
 */
export function getMinConfidenceForHorizon(horizon: number): number {
  if (horizon <= 15) return HORIZON_CONFIG.h15.minConfidence;
  if (horizon <= 60) return HORIZON_CONFIG.h60.minConfidence;
  return HORIZON_CONFIG.h240.minConfidence;
}

/**
 * Get the minimum edge threshold for a horizon
 */
export function getMinEdgeForHorizon(horizon: number): number {
  if (horizon <= 15) return HORIZON_CONFIG.h15.minEdge;
  if (horizon <= 60) return HORIZON_CONFIG.h60.minEdge;
  return HORIZON_CONFIG.h240.minEdge;
}

/**
 * Check if a prediction meets horizon-specific thresholds
 */
export function predictionMeetsThresholds(
  mu: number,
  sigma: number,
  horizon: number
): { meets: boolean; confidence: number; minConfidence: number; minEdge: number } {
  const confidence = computeConfidenceRatio(Math.abs(mu), sigma);
  const minConfidence = getMinConfidenceForHorizon(horizon);
  const minEdge = getMinEdgeForHorizon(horizon);
  
  const meetsEdge = Math.abs(mu) > minEdge;
  const meetsConfidence = confidence >= minConfidence;
  
  return {
    meets: meetsEdge && meetsConfidence,
    confidence,
    minConfidence,
    minEdge
  };
}

/**
 * Check NO-TRADE conditions and return list of vetoes
 */
export function checkNoTradeConditions(
  predictions: HorizonPredictions,
  context: MarketContext
): string[] {
  const vetoes: string[] = [];
  const config = NO_TRADE_CONDITIONS;
  
  // 1. Dead zone check: μ_15 too small
  if (Math.abs(predictions.h15.mu) < config.deadZoneThreshold) {
    vetoes.push(`DEAD_ZONE: |μ_15|=${Math.abs(predictions.h15.mu).toFixed(4)} < ${config.deadZoneThreshold}`);
  }
  
  // 2. Uncertainty spike: volatility in panic territory
  if (context.volatilityPercentile > config.uncertaintyPercentile) {
    vetoes.push(`UNCERTAINTY_SPIKE: vol_percentile=${(context.volatilityPercentile * 100).toFixed(0)}% > ${(config.uncertaintyPercentile * 100).toFixed(0)}%`);
  }
  
  // 3. Horizon disagreement: μ_15 and μ_60 have opposite signs
  if (config.horizonDisagreementVeto) {
    const sign15 = Math.sign(predictions.h15.mu);
    const sign60 = Math.sign(predictions.h60.mu);
    if (sign15 !== 0 && sign60 !== 0 && sign15 !== sign60) {
      vetoes.push(`HORIZON_DISAGREEMENT: μ_15 sign=${sign15}, μ_60 sign=${sign60}`);
    }
  }
  
  // 4. Loss streak: too many recent losses
  if (context.recentLosses >= config.maxLossStreak) {
    vetoes.push(`LOSS_STREAK: ${context.recentLosses} >= ${config.maxLossStreak} consecutive losses`);
  }
  
  // 5. Funding rate flip: aggressive change
  if (Math.abs(context.fundingRateChange) > config.fundingFlipThreshold) {
    vetoes.push(`FUNDING_FLIP: |Δfunding|=${Math.abs(context.fundingRateChange).toFixed(4)} > ${config.fundingFlipThreshold}`);
  }
  
  return vetoes;
}

/**
 * Compute bounded edge-weighted Kelly position size
 * 
 * Formula: size = (E / V) * risk_cap, clamped to [min_size, max_size]
 * 
 * @param edge Expected edge (μ)
 * @param uncertainty Standard deviation (σ)
 * @param riskCap Maximum risk per trade (default 0.25% = 0.0025)
 * @param minSize Minimum position size (default 0.05% = 0.0005)
 * @param maxSize Maximum position size (default 0.30% = 0.003)
 */
export function computePositionSize(
  edge: number,
  uncertainty: number,
  riskCap: number = 0.0025,
  minSize: number = 0.0005,
  maxSize: number = 0.003,
  sizeMultiplier: number = 1.0  // Can be reduced after loss streak
): number {
  if (uncertainty <= 0 || !Number.isFinite(uncertainty)) return minSize;
  if (edge <= 0 || !Number.isFinite(edge)) return minSize;
  
  const rawSize = (edge / uncertainty) * riskCap * sizeMultiplier;
  return Math.max(minSize, Math.min(maxSize, rawSize));
}

/**
 * Compute stop loss and take profit based on ATR
 * 
 * @param atr14 14-period ATR
 * @param direction Trade direction (1 = long, -1 = short)
 * @param slMultiplier Stop loss ATR multiplier (default 1.2)
 * @param tpMultiplier Take profit as multiple of SL (default 1.5R)
 */
export function computeStopsAndTargets(
  atr14: number,
  direction: number,
  slMultiplier: number = 1.2,
  tpMultiplier: number = 1.5
): { stopLoss: number; takeProfit: number } {
  const slDistance = atr14 * slMultiplier;
  const tpDistance = slDistance * tpMultiplier;
  
  return {
    stopLoss: slDistance,
    takeProfit: tpDistance
  };
}

/**
 * Main trade decision engine
 * 
 * Implements institution-grade decision logic:
 * 1. Check if predictions meet horizon-specific thresholds
 * 2. Apply NO-TRADE conditions
 * 3. Use multi-horizon confluence
 * 4. Compute position size and stops
 */
export function makeTradeDecision(
  predictions: HorizonPredictions,
  context: MarketContext
): TradeDecision {
  const reasons: string[] = [];
  const vetoes = checkNoTradeConditions(predictions, context);
  
  // Default HOLD decision
  const holdDecision: TradeDecision = {
    action: "HOLD",
    confidence: 0,
    primaryHorizon: 0,
    edge: 0,
    size: 0,
    stopLoss: 0,
    takeProfit: 0,
    maxHoldBars: 0,
    reasons: ["No trade signal"],
    vetoes
  };
  
  // If too many vetoes, don't trade
  if (vetoes.length > 0) {
    holdDecision.reasons = vetoes;
    return holdDecision;
  }
  
  // Check if 15-bar prediction meets thresholds
  const h15Check = predictionMeetsThresholds(predictions.h15.mu, predictions.h15.sigma, 15);
  const h60Check = predictionMeetsThresholds(predictions.h60.mu, predictions.h60.sigma, 60);
  const h240Check = predictionMeetsThresholds(predictions.h240.mu, predictions.h240.sigma, 240);
  
  // Primary trading on 15-bar if it meets thresholds
  let primaryHorizon = 0;
  let primaryMu = 0;
  let primarySigma = 0;
  
  if (h15Check.meets) {
    // Additional check: short-term should dominate
    // |μ_15| > |μ_60| * 0.6
    const shortTermDominates = Math.abs(predictions.h15.mu) > Math.abs(predictions.h60.mu) * 0.6;
    
    if (shortTermDominates) {
      primaryHorizon = 15;
      primaryMu = predictions.h15.mu;
      primarySigma = predictions.h15.sigma;
      reasons.push(`H15 meets thresholds: edge=${primaryMu.toFixed(4)}, conf=${h15Check.confidence.toFixed(2)}`);
      reasons.push(`Short-term dominates: |μ_15|=${Math.abs(predictions.h15.mu).toFixed(4)} > |μ_60|*0.6=${(Math.abs(predictions.h60.mu) * 0.6).toFixed(4)}`);
    }
  }
  
  // Fall back to 60-bar if 15 doesn't qualify
  if (primaryHorizon === 0 && h60Check.meets) {
    primaryHorizon = 60;
    primaryMu = predictions.h60.mu;
    primarySigma = predictions.h60.sigma;
    reasons.push(`H60 meets thresholds: edge=${primaryMu.toFixed(4)}, conf=${h60Check.confidence.toFixed(2)}`);
  }
  
  // 240-bar is confirmation only
  if (primaryHorizon > 0) {
    // Check if 240-bar trend supports the trade
    const trendSupports = (primaryMu > 0 && predictions.h240.mu > -0.001) || 
                          (primaryMu < 0 && predictions.h240.mu < 0.001);
    
    if (!trendSupports) {
      reasons.push(`VETO: 240-bar trend opposes trade: μ_240=${predictions.h240.mu.toFixed(4)}`);
      holdDecision.reasons = reasons;
      return holdDecision;
    }
    reasons.push(`H240 trend supports: μ_240=${predictions.h240.mu.toFixed(4)}`);
  }
  
  // No valid primary horizon found
  if (primaryHorizon === 0) {
    holdDecision.reasons = ["No horizon meets thresholds"];
    return holdDecision;
  }
  
  // Determine direction
  const direction = primaryMu > 0 ? 1 : -1;
  const action = direction > 0 ? "LONG" : "SHORT";
  
  // Compute position size with loss streak reduction
  let sizeMultiplier = 1.0;
  if (context.recentLosses >= 2) {
    sizeMultiplier = NO_TRADE_CONDITIONS.lossStreakSizeReduction;
    reasons.push(`Size reduced by ${((1 - sizeMultiplier) * 100).toFixed(0)}% due to ${context.recentLosses} recent losses`);
  }
  
  // Reduce size if trend is weak
  if (Math.abs(predictions.h240.mu) < 0.002) {
    sizeMultiplier *= 0.7;
    reasons.push(`Size reduced 30% due to weak trend: |μ_240|=${Math.abs(predictions.h240.mu).toFixed(4)} < 0.002`);
  }
  
  const size = computePositionSize(
    Math.abs(primaryMu),
    primarySigma,
    0.0025,  // risk_cap = 0.25%
    0.0005,  // min_size = 0.05%
    0.003,   // max_size = 0.30%
    sizeMultiplier
  );
  
  // Compute stops and targets
  const { stopLoss, takeProfit } = computeStopsAndTargets(context.atr14, direction);
  
  // Get max hold bars for time stop
  const maxHoldBars = primaryHorizon === 15 ? HORIZON_CONFIG.h15.maxHoldBars :
                      primaryHorizon === 60 ? HORIZON_CONFIG.h60.maxHoldBars :
                      HORIZON_CONFIG.h240.maxHoldBars;
  
  return {
    action,
    confidence: computeConfidenceRatio(Math.abs(primaryMu), primarySigma),
    primaryHorizon,
    edge: primaryMu,
    size,
    stopLoss,
    takeProfit,
    maxHoldBars,
    reasons,
    vetoes
  };
}

/**
 * Create predictions from raw model outputs
 */
export function createPredictions(
  mu15: number, sigma15: number,
  mu60: number, sigma60: number,
  mu240: number, sigma240: number
): HorizonPredictions {
  const makePred = (mu: number, sigma: number, horizon: number): PredictionWithUncertainty => {
    const confidence = computeConfidenceRatio(Math.abs(mu), sigma);
    const { meets } = predictionMeetsThresholds(mu, sigma, horizon);
    return {
      horizon,
      mu,
      sigma,
      direction: mu > getMinEdgeForHorizon(horizon) ? 1 : mu < -getMinEdgeForHorizon(horizon) ? -1 : 0,
      confidence,
      meetsThreshold: meets
    };
  };
  
  return {
    h15: makePred(mu15, sigma15, 15),
    h60: makePred(mu60, sigma60, 60),
    h240: makePred(mu240, sigma240, 240)
  };
}

/**
 * Check if a position should be closed due to time stop
 */
export function checkTimeStop(
  entryBar: number,
  currentBar: number,
  horizon: number
): { shouldClose: boolean; reason: string } {
  const maxBars = horizon === 15 ? HORIZON_CONFIG.h15.maxHoldBars :
                  horizon === 60 ? HORIZON_CONFIG.h60.maxHoldBars :
                  HORIZON_CONFIG.h240.maxHoldBars;
  
  const barsHeld = currentBar - entryBar;
  
  if (barsHeld >= maxBars) {
    return {
      shouldClose: true,
      reason: `TIME_STOP: held ${barsHeld} bars >= max ${maxBars} bars`
    };
  }
  
  return { shouldClose: false, reason: "" };
}

// ===============================================
// SIGNAL THRESHOLD TUNING FOR CLASSIFICATION MODELS
// ===============================================

export interface ClassificationProbabilities {
  pShort: number;   // P(SHORT) - class 0
  pNeutral: number; // P(NEUTRAL) - class 1  
  pLong: number;    // P(LONG) - class 2
}

export interface SignalThresholdConfig {
  // Score threshold for emitting signals (default 0.15 for ~2-3 signals/day)
  scoreThreshold: number;
  // Minimum confidence (max prob) to consider signal valid
  minConfidence: number;
  // Margin between top 2 probabilities
  minMargin: number;
}

export const DEFAULT_SIGNAL_THRESHOLD: SignalThresholdConfig = {
  scoreThreshold: 0.15,   // |pLong - pShort| must exceed this
  minConfidence: 0.45,    // max(pLong, pShort, pNeutral) must exceed this
  minMargin: 0.10         // Difference between 1st and 2nd highest prob
};

/**
 * Compute directional score from classification probabilities
 * 
 * Score = pLong - pShort
 * - Positive = bullish bias
 * - Negative = bearish bias
 * - Near zero = no clear direction
 */
export function computeDirectionalScore(probs: ClassificationProbabilities): number {
  return probs.pLong - probs.pShort;
}

/**
 * Compute confidence from probabilities (max probability)
 */
export function computeProbConfidence(probs: ClassificationProbabilities): number {
  return Math.max(probs.pShort, probs.pNeutral, probs.pLong);
}

/**
 * Compute margin between top 2 probabilities
 * Higher margin = more decisive prediction
 */
export function computeProbMargin(probs: ClassificationProbabilities): number {
  const sorted = [probs.pShort, probs.pNeutral, probs.pLong].sort((a, b) => b - a);
  return sorted[0] - sorted[1];
}

/**
 * Apply signal threshold to classification probabilities
 * 
 * This is the key function for tuning signal frequency:
 * - Higher scoreThreshold = fewer, higher-conviction signals
 * - Lower scoreThreshold = more signals, lower average quality
 * 
 * Recommended tuning: Adjust scoreThreshold to hit ~2-3 signals/day
 */
export function applySignalThreshold(
  probs: ClassificationProbabilities,
  config: SignalThresholdConfig = DEFAULT_SIGNAL_THRESHOLD
): { 
  action: "LONG" | "SHORT" | "HOLD";
  score: number;
  confidence: number;
  margin: number;
  meetsThreshold: boolean;
  reasons: string[];
} {
  const score = computeDirectionalScore(probs);
  const confidence = computeProbConfidence(probs);
  const margin = computeProbMargin(probs);
  
  const reasons: string[] = [];
  
  // Check if score exceeds threshold
  const scoreExceedsThreshold = Math.abs(score) >= config.scoreThreshold;
  const confidenceOk = confidence >= config.minConfidence;
  const marginOk = margin >= config.minMargin;
  
  if (!scoreExceedsThreshold) {
    reasons.push(`Score ${score.toFixed(3)} below threshold ${config.scoreThreshold}`);
  }
  if (!confidenceOk) {
    reasons.push(`Confidence ${confidence.toFixed(3)} below min ${config.minConfidence}`);
  }
  if (!marginOk) {
    reasons.push(`Margin ${margin.toFixed(3)} below min ${config.minMargin}`);
  }
  
  const meetsThreshold = scoreExceedsThreshold && confidenceOk && marginOk;
  
  if (!meetsThreshold) {
    return {
      action: "HOLD",
      score,
      confidence,
      margin,
      meetsThreshold: false,
      reasons
    };
  }
  
  // Determine direction from score sign
  const action = score > 0 ? "LONG" : "SHORT";
  reasons.push(`Score ${score.toFixed(3)} meets threshold ${config.scoreThreshold}`);
  reasons.push(`Confidence ${confidence.toFixed(3)} >= ${config.minConfidence}`);
  reasons.push(`Margin ${margin.toFixed(3)} >= ${config.minMargin}`);
  
  return {
    action,
    score,
    confidence,
    margin,
    meetsThreshold: true,
    reasons
  };
}

/**
 * Compute optimal threshold for target signal frequency
 * 
 * Given historical probability data and target signals per day,
 * returns the threshold that would achieve that frequency.
 * 
 * @param historicalScores Array of |pLong - pShort| values from backtest
 * @param targetSignalsPerDay Desired number of signals per day
 * @param barsPerDay Number of bars per day (e.g., 96 for 15m candles)
 */
export function computeOptimalThreshold(
  historicalScores: number[],
  targetSignalsPerDay: number,
  barsPerDay: number
): number {
  if (historicalScores.length === 0) return 0.15;
  
  // Sort scores in descending order
  const sorted = [...historicalScores].sort((a, b) => b - a);
  
  // Calculate total days of data
  const totalDays = historicalScores.length / barsPerDay;
  
  // Number of signals needed for target frequency
  const totalSignalsNeeded = targetSignalsPerDay * totalDays;
  
  // Find the threshold that would give us this many signals
  const index = Math.min(Math.floor(totalSignalsNeeded), sorted.length - 1);
  
  return sorted[index] || 0.15;
}
