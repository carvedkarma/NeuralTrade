import { db } from "./db";
import { patterns, patternClusters as patternClustersTable, predictionEpisodes } from "./db/schema";
import { desc, sql, eq, isNull, and, lt } from "drizzle-orm";
import type { FeatureVector } from "./feature-engine";
import type { InsertPredictionEpisode, PredictionEpisode } from "@shared/schema";
import * as fs from 'fs';
import * as path from 'path';

const MAX_PATTERNS_TOTAL = 30;
const MIN_SAMPLES_PER_PATTERN = 100;  // Increased from 50 for statistical reliability (research-backed)
const MIN_BACKTEST_TRADES = 500;
const MIN_CANDLES_15M = 2000; // ~20 days of 15m data
const EMBARGO_CANDLES = 16;  // Temporal embargo (4 hours of 15m candles) - legacy, use PRECISION_MODE_CONFIG for new code
const EMBARGO_SIMILARITY_THRESHOLD = 0.95;  // Feature-vector embargo: exclude patterns with >95% similarity to boundary patterns

// ============================================================================
// PRECISION MODE CONFIGURATION (P0 Fixes - Research-Backed)
// ============================================================================
export interface PrecisionModeConfig {
  enabled: boolean;
  similarityThreshold: number;      // Min cosine similarity (0.86 for sniper mode)
  minMatchesFloor: number;          // Min matches after filtering (35 for reliability)
  embargoHorizon: number;           // Forward-looking candles (e.g., 8 for 8-candle prediction)
  embargoSequenceLength: number;    // Sequence length used in features (e.g., 16)
  candleIntervalMs: number;         // Candle interval in ms (15min = 900000)
  regimeFilterEnabled: boolean;     // Hard-filter by regime before cosine similarity
  volatilityFilterEnabled: boolean; // Hard-filter by ATR percentile bucket
  timeTauDays: number;              // Time decay tau (60 days default)
  bayesianAlpha: number;            // Beta prior alpha (smoothing parameter)
  bayesianBeta: number;             // Beta prior beta (smoothing parameter)
  minPosteriorPWin: number;         // Min posterior P(win) to trade (0.58-0.62)
  maxUncertainty: number;           // Max credible interval width to trade
}

// Default precision mode config (sniper mode)
export const PRECISION_MODE_CONFIG: PrecisionModeConfig = {
  enabled: true,
  similarityThreshold: 0.86,        // P0-2: Strict similarity for precision
  minMatchesFloor: 35,              // P0-2: Min matches for statistical reliability
  embargoHorizon: 8,                // 8 candles forward (2 hours at 15m)
  embargoSequenceLength: 16,        // 16-candle sequence for features
  candleIntervalMs: 15 * 60 * 1000, // 15 minutes
  regimeFilterEnabled: true,        // P0-3: Regime gating
  volatilityFilterEnabled: true,    // P0-3: Volatility bucket gating
  timeTauDays: 60,                  // P1-5: Time decay tau
  bayesianAlpha: 1,                 // Beta(1,1) = uniform prior
  bayesianBeta: 1,
  minPosteriorPWin: 0.58,           // P1-2: Min win probability to trade
  maxUncertainty: 0.25,             // P1-2: Max uncertainty to trade
};

// Legacy constant for backward compatibility (use PRECISION_MODE_CONFIG instead)
const MIN_SIMILARITY_THRESHOLD = PRECISION_MODE_CONFIG.similarityThreshold;

// ============================================================================
// VOLATILITY BUCKET SYSTEM (P0-3)
// ============================================================================
export type VolatilityBucket = "low" | "medium" | "high" | "extreme";

// ATR percentile thresholds (calibrated on BTC historical data)
const ATR_PERCENTILE_THRESHOLDS = {
  low: 0.25,      // Bottom 25% of ATR values
  medium: 0.50,   // 25-50%
  high: 0.75,     // 50-75%
  extreme: 1.0,   // Top 25%
};

// Track rolling ATR statistics for bucket assignment
let atrHistory: number[] = [];
const ATR_HISTORY_SIZE = 1000;  // Keep last 1000 ATR values for percentile calculation

export function updateAtrHistory(atr: number): void {
  atrHistory.push(atr);
  if (atrHistory.length > ATR_HISTORY_SIZE) {
    atrHistory = atrHistory.slice(-ATR_HISTORY_SIZE);
  }
}

export function getVolatilityBucket(atr: number): VolatilityBucket {
  if (atrHistory.length < 50) {
    // Not enough history, use simple heuristics
    const normalizedAtr = atr / 100000; // Rough BTC normalization
    if (normalizedAtr < 0.005) return "low";
    if (normalizedAtr < 0.01) return "medium";
    if (normalizedAtr < 0.02) return "high";
    return "extreme";
  }
  
  // Compute percentile of current ATR
  const sorted = [...atrHistory].sort((a, b) => a - b);
  let percentile = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (atr <= sorted[i]) {
      percentile = i / sorted.length;
      break;
    }
    percentile = 1.0;
  }
  
  if (percentile <= ATR_PERCENTILE_THRESHOLDS.low) return "low";
  if (percentile <= ATR_PERCENTILE_THRESHOLDS.medium) return "medium";
  if (percentile <= ATR_PERCENTILE_THRESHOLDS.high) return "high";
  return "extreme";
}

// ============================================================================
// HARD-NEGATIVE MINING (P1-3) - Track "false friend" patterns
// ============================================================================
interface FalseFriendPenalty {
  patternId: number;
  penalty: number;         // Cumulative penalty (0-1, higher = worse)
  failureCount: number;    // Number of times this pattern caused losses
  lastFailureTs: number;   // Last failure timestamp
  regime: string;          // Regime where failure occurred
}

const falseFriendPenalties: Map<number, FalseFriendPenalty> = new Map();
const PENALTY_DECAY_RATE = 0.95;  // Decay penalty by 5% per day
const PENALTY_INCREMENT = 0.1;    // Add 10% penalty per failure
const PENALTIES_FILE = 'false_friend_penalties.json';

// P1-3 FIX: Persist false friend penalties to file
function saveFalseFriendPenalties(): void {
  try {
    const data = Array.from(falseFriendPenalties.values());
    fs.writeFileSync(PENALTIES_FILE, JSON.stringify(data, null, 2));
  } catch (err) {
    console.error('[Pattern Memory] Failed to save false friend penalties:', err);
  }
}

export function loadFalseFriendPenalties(): void {
  try {
    if (fs.existsSync(PENALTIES_FILE)) {
      const data = JSON.parse(fs.readFileSync(PENALTIES_FILE, 'utf8')) as FalseFriendPenalty[];
      falseFriendPenalties.clear();
      for (const entry of data) {
        falseFriendPenalties.set(entry.patternId, entry);
      }
      console.log(`[Pattern Memory] Loaded ${data.length} false friend penalties from file`);
    }
  } catch (err) {
    console.error('[Pattern Memory] Failed to load false friend penalties:', err);
  }
}

export function recordPatternFailure(patternId: number, regime: string): void {
  const existing = falseFriendPenalties.get(patternId);
  if (existing) {
    existing.penalty = Math.min(1.0, existing.penalty + PENALTY_INCREMENT);
    existing.failureCount++;
    existing.lastFailureTs = Date.now();
    existing.regime = regime;
  } else {
    falseFriendPenalties.set(patternId, {
      patternId,
      penalty: PENALTY_INCREMENT,
      failureCount: 1,
      lastFailureTs: Date.now(),
      regime,
    });
  }
  // Persist after each failure (debounced in production, immediate for now)
  saveFalseFriendPenalties();
}

export function getFalseFriendPenalty(patternId: number): number {
  const entry = falseFriendPenalties.get(patternId);
  if (!entry) return 0;
  
  // Apply time decay
  const daysSinceFailure = (Date.now() - entry.lastFailureTs) / (24 * 60 * 60 * 1000);
  const decayedPenalty = entry.penalty * Math.pow(PENALTY_DECAY_RATE, daysSinceFailure);
  
  return decayedPenalty;
}

export function decayAllPenalties(): void {
  const now = Date.now();
  const entries = Array.from(falseFriendPenalties.entries());
  for (const [id, entry] of entries) {
    const daysSinceFailure = (now - entry.lastFailureTs) / (24 * 60 * 60 * 1000);
    entry.penalty *= Math.pow(PENALTY_DECAY_RATE, daysSinceFailure);
    entry.lastFailureTs = now;
    
    // Remove if penalty is negligible
    if (entry.penalty < 0.01) {
      falseFriendPenalties.delete(id);
    }
  }
}

// ============================================================================
// BAYESIAN CLUSTER RELIABILITY (P1-2)
// ============================================================================
export interface BayesianClusterStats {
  clusterId: string;
  wins: number;
  losses: number;
  posteriorMean: number;         // E[P(win)] = (wins + α) / (wins + losses + α + β)
  credibleIntervalLow: number;   // 5th percentile
  credibleIntervalHigh: number;  // 95th percentile
  uncertainty: number;           // Width of credible interval
  evLong: number;                // Expected value for LONG
  evShort: number;               // Expected value for SHORT
}

// ============================================================================
// EXACT BETA DISTRIBUTION FUNCTIONS (P1-2 Fix)
// Implements proper Beta quantiles using incomplete beta function approximation
// Much more accurate than normal approximation, especially for small samples
// ============================================================================

// Log Gamma function using Lanczos approximation
function logGamma(z: number): number {
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  const g = 7;
  const coefficients = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
  ];
  let x = coefficients[0];
  for (let i = 1; i < g + 2; i++) {
    x += coefficients[i] / (z + i);
  }
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

// Beta function B(a,b)
function betaFunction(a: number, b: number): number {
  return Math.exp(logGamma(a) + logGamma(b) - logGamma(a + b));
}

// Regularized incomplete beta function I_x(a,b) using continued fraction
// This is the CDF of the Beta distribution
function regularizedBeta(x: number, a: number, b: number): number {
  if (x < 0 || x > 1) return x < 0 ? 0 : 1;
  if (x === 0) return 0;
  if (x === 1) return 1;
  
  // Use symmetry relation for numerical stability
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - regularizedBeta(1 - x, b, a);
  }
  
  // Continued fraction representation (Lentz's algorithm)
  const maxIterations = 200;
  const epsilon = 1e-14;
  
  const prefactor = Math.exp(
    a * Math.log(x) + b * Math.log(1 - x) - Math.log(a) - logGamma(a) - logGamma(b) + logGamma(a + b)
  );
  
  let cf = 1;
  let delta = 1;
  let h = 1;
  
  for (let m = 0; m <= maxIterations; m++) {
    // Compute numerator coefficients
    let numerator: number;
    if (m === 0) {
      numerator = 1;
    } else if (m % 2 === 0) {
      const k = m / 2;
      numerator = (k * (b - k) * x) / ((a + 2 * k - 1) * (a + 2 * k));
    } else {
      const k = (m - 1) / 2 + 1;
      numerator = -((a + k - 1) * (a + b + k - 1) * x) / ((a + 2 * k - 2) * (a + 2 * k - 1));
    }
    
    h = 1 + numerator / h;
    if (Math.abs(h) < 1e-30) h = 1e-30;
    delta = 1 / h;
    cf *= delta;
    
    if (Math.abs(delta - 1) < epsilon) {
      break;
    }
  }
  
  return prefactor * cf;
}

// Beta distribution quantile using bisection search on regularized incomplete beta
// This is much more accurate than normal approximation, especially for small alpha/beta
function betaQuantile(alpha: number, beta: number, p: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  if (alpha <= 0 || beta <= 0) return 0.5;
  
  // Bisection search for quantile
  let lo = 0;
  let hi = 1;
  const tolerance = 1e-10;
  const maxIterations = 100;
  
  // Initial guess using normal approximation for starting point
  const mean = alpha / (alpha + beta);
  const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
  let x = Math.max(0.001, Math.min(0.999, mean));
  
  for (let i = 0; i < maxIterations; i++) {
    const cdf = regularizedBeta(x, alpha, beta);
    
    if (Math.abs(cdf - p) < tolerance) {
      return x;
    }
    
    if (cdf < p) {
      lo = x;
    } else {
      hi = x;
    }
    
    x = (lo + hi) / 2;
  }
  
  return x;
}

export function computeBayesianStats(
  wins: number, 
  losses: number,
  avgWinReturn: number = 0,
  avgLossReturn: number = 0,
  config: PrecisionModeConfig = PRECISION_MODE_CONFIG
): BayesianClusterStats {
  const alpha = wins + config.bayesianAlpha;
  const beta_param = losses + config.bayesianBeta;
  
  const posteriorMean = alpha / (alpha + beta_param);
  const posteriorLossMean = 1 - posteriorMean;  // P(loss) = 1 - P(win)
  const credibleIntervalLow = betaQuantile(alpha, beta_param, 0.05);
  const credibleIntervalHigh = betaQuantile(alpha, beta_param, 0.95);
  const uncertainty = credibleIntervalHigh - credibleIntervalLow;
  
  // PROPER EV CALCULATION (P1-2 Fix):
  // EV = P(win) * avgWinReturn + P(loss) * avgLossReturn
  // Note: avgLossReturn should be negative, so this properly subtracts losses
  // If avgLossReturn is stored as positive magnitude, we subtract it explicitly
  const avgLossAbs = Math.abs(avgLossReturn);
  const evLong = posteriorMean * avgWinReturn - posteriorLossMean * avgLossAbs;
  const evShort = posteriorMean * avgWinReturn - posteriorLossMean * avgLossAbs;
  
  return {
    clusterId: "",
    wins,
    losses,
    posteriorMean,
    credibleIntervalLow,
    credibleIntervalHigh,
    uncertainty,
    evLong,
    evShort,
  };
}

let currentCandleCount = 0;
let currentBacktestTrades = 0;

export function updateDataCounts(candles: number, trades: number) {
  currentCandleCount = candles;
  currentBacktestTrades = trades;
}

export function canCreateNewPatterns(): boolean {
  const meetsTradeReq = currentBacktestTrades >= MIN_BACKTEST_TRADES;
  const meetsCandleReq = currentCandleCount >= MIN_CANDLES_15M;
  return meetsTradeReq && meetsCandleReq;
}

export function canCreateNewPatternsWithCounts(candles: number, trades: number): boolean {
  return trades >= MIN_BACKTEST_TRADES && candles >= MIN_CANDLES_15M;
}

export function getPatternRequirements() {
  return { minTrades: MIN_BACKTEST_TRADES, minCandles: MIN_CANDLES_15M };
}

export interface PatternCluster {
  id: string;
  regime: "trend_up" | "trend_down" | "chop" | "shock";
  centroid: number[];
  support: number;
  wins: number;
  winRate: number;
  avgReturn: number;
  maturity: number;
  samples: PatternSample[];
  // Direction-specific stats for accurate win rate calculation
  longWins: number;
  longTotal: number;
  longAvgPnL: number;
  shortWins: number;
  shortTotal: number;
  shortAvgPnL: number;
}

export interface PatternSample {
  timestamp: number;
  embedding: number[];
  forwardReturn8: number;
  won: boolean;
  direction: "LONG" | "SHORT" | "HOLD";  // Direction of the trade taken
  actualPnL: number;  // Actual P&L after direction adjustment
}

export let patternClusters: Map<string, PatternCluster> = new Map();

// Reset all pattern clusters to empty state
export function resetPatternClusters(): void {
  console.log("[Pattern Memory] Resetting all pattern clusters...");
  patternClusters = new Map();
  console.log("[Pattern Memory] All pattern clusters have been reset");
}

export interface PatternMatch {
  timestamp: number;
  similarity: number;
  forwardReturn8: number;
  forwardReturn16: number;
  maxDrawdown: number;
  maxRunup: number;
  timeToMfe: number;
  won: boolean;
  regime: string;
  label: string;
  atrAtEntry: number;
  dynamicThreshold: number;
  clusterId?: string;
  clusterMaturity?: number;
  direction?: "LONG" | "SHORT" | "HOLD";  // Trade direction
  actualPnL?: number;  // Direction-adjusted P&L
  longWinRate?: number;  // Cluster's LONG win rate
  shortWinRate?: number;  // Cluster's SHORT win rate
}

export function mapKalmanToRegime(kalmanRegime: string): "trend_up" | "trend_down" | "chop" | "shock" {
  switch (kalmanRegime) {
    case "bull": return "trend_up";
    case "bear": return "trend_down";
    case "chop": return "chop";
    case "shock": return "shock";
    default: return "chop";
  }
}

export function computeDynamicThreshold(atr: number, price: number): number {
  const volatility = atr / price;
  return Math.max(0.0015, 0.9 * volatility);
}

export function determineActualOutcome(
  forwardReturn8: number,
  threshold: number
): "up" | "down" | "chop" {
  if (forwardReturn8 > threshold) return "up";
  if (forwardReturn8 < -threshold) return "down";
  return "chop";
}

export function determinePrediction(
  regime: "trend_up" | "trend_down" | "chop" | "shock"
): "up" | "down" | "chop" {
  if (regime === "trend_up") return "up";
  if (regime === "trend_down") return "down";
  return "chop";
}

export function determineWin(
  prediction: "up" | "down" | "chop",
  actualOutcome: "up" | "down" | "chop"
): boolean {
  // DEPRECATED: This only checks regime match, not actual profitability
  // Use determineWinByPnL instead for accurate win determination
  return prediction === actualOutcome;
}

// CORRECT: Win determination based on actual P&L (the only metric that matters)
export function determineWinByPnL(
  direction: "LONG" | "SHORT" | "HOLD",
  forwardReturn: number,
  costs: number = 0.0010  // Standardized: 0.04% maker fees round-trip + 0.04% slippage + 0.02% funding
): { won: boolean; actualPnL: number } {
  if (direction === "HOLD") {
    return { won: false, actualPnL: 0 };
  }
  
  // For LONG: profit if price went up (positive return)
  // For SHORT: profit if price went down (negate the return)
  const directionMultiplier = direction === "LONG" ? 1 : -1;
  const grossPnL = forwardReturn * directionMultiplier;
  const actualPnL = grossPnL - costs;
  
  // A win is when actual P&L after costs is positive
  const won = actualPnL > 0;
  
  return { won, actualPnL };
}

// CRITICAL FIX: Determine win based on actual forward return, not regime-derived direction
// For training labels, we use the OPTIMAL direction based on what actually happened
export function determineWinFromReturn(
  forwardReturn: number,
  dynamicThreshold: number,
  costs: number = 0.0010
): { won: boolean; optimalDirection: "LONG" | "SHORT" | "HOLD"; actualPnL: number } {
  const netReturnLong = forwardReturn - costs;
  const netReturnShort = -forwardReturn - costs;
  
  // Check if either direction would have been profitable above threshold
  if (netReturnLong > dynamicThreshold) {
    return { won: true, optimalDirection: "LONG", actualPnL: netReturnLong };
  }
  if (netReturnShort > dynamicThreshold) {
    return { won: true, optimalDirection: "SHORT", actualPnL: netReturnShort };
  }
  
  // No profitable trade - this is a HOLD pattern
  // But we still track if there was significant movement (volatility expansion)
  const absReturn = Math.abs(forwardReturn);
  if (absReturn > dynamicThreshold * 1.5) {
    // Significant move but didn't exceed cost threshold - edge case
    return { won: false, optimalDirection: "HOLD", actualPnL: 0 };
  }
  
  return { won: false, optimalDirection: "HOLD", actualPnL: 0 };
}

// Determine direction based on regime and features
export function determineDirection(
  regime: "trend_up" | "trend_down" | "chop" | "shock"
): "LONG" | "SHORT" | "HOLD" {
  switch (regime) {
    case "trend_up": return "LONG";
    case "trend_down": return "SHORT";
    case "chop": return "HOLD";
    case "shock": return "HOLD";
    default: return "HOLD";
  }
}

export interface PatternStats {
  matchCount: number;
  avgReturn8: number;
  avgReturn16: number;
  winRate: number;
  avgDrawdown: number;
  avgRunup: number;
  avgTimeToMfe: number;
  mae70thPercentile: number;
  mfe70thPercentile: number;
  bestCase: number;
  worstCase: number;
  consistency: number;
  regimeBreakdown: Record<string, number>;
  matureMatchCount: number;
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

export interface StorePatternParams {
  feature: FeatureVector;
  forwardReturn8: number;
  forwardReturn16: number;
  maxDrawdown: number;
  maxRunup: number;
  timeToMfe: number;
  atrAtEntry: number;
  dynamicThreshold: number;
  direction?: "LONG" | "SHORT" | "HOLD";  // Explicit direction override
  isTestSet?: boolean;  // For train/test split (walk-forward validation)
}

function computeMaturity(support: number): number {
  return Math.min(1, Math.max(0, support / 200));
}

function isPatternMature(cluster: PatternCluster): boolean {
  return cluster.support >= MIN_SAMPLES_PER_PATTERN && cluster.maturity >= 0.25;
}

function generateClusterId(regime: string, index: number): string {
  return `${regime}_cluster_${index}`;
}

async function findNearestCluster(
  embedding: number[],
  regime: "trend_up" | "trend_down" | "chop" | "shock"
): Promise<{ cluster: PatternCluster | null; similarity: number }> {
  let bestCluster: PatternCluster | null = null;
  let bestSimilarity = 0;
  
  const clusters = Array.from(patternClusters.values());
  for (const cluster of clusters) {
    if (cluster.regime !== regime) continue;
    
    const similarity = cosineSimilarity(embedding, cluster.centroid);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestCluster = cluster;
    }
  }
  
  return { cluster: bestCluster, similarity: bestSimilarity };
}

function updateClusterCentroid(cluster: PatternCluster, newEmbedding: number[]): number[] {
  const n = cluster.support;
  return cluster.centroid.map((c, i) => (c * n + newEmbedding[i]) / (n + 1));
}

export async function storePattern(params: StorePatternParams): Promise<void> {
  const { feature, forwardReturn8, forwardReturn16, maxDrawdown, maxRunup, timeToMfe, atrAtEntry, dynamicThreshold } = params;
  
  const regime = mapKalmanToRegime(feature.kalmanRegime);
  const actualOutcome = determineActualOutcome(forwardReturn8, dynamicThreshold);
  
  // CRITICAL FIX: Determine win based on ACTUAL forward return, not regime-derived direction
  // This fixes the 71% false negative rate where chop regime patterns were marked as losses
  // regardless of actual price movement
  const { won, optimalDirection, actualPnL } = determineWinFromReturn(forwardReturn8, dynamicThreshold);
  
  // Use explicit direction if provided, otherwise use optimal direction from actual return
  const direction = params.direction || optimalDirection;
  
  const embedding = normalizeEmbedding(feature.embedding);
  if (!isValidEmbedding(embedding)) {
    return;
  }
  
  const { cluster: nearestCluster, similarity } = await findNearestCluster(embedding, regime);
  
  const totalClusters = patternClusters.size;
  const canCreate = canCreateNewPatterns();
  
  // Helper to update direction-specific stats on a cluster
  const updateClusterDirectionStats = (cluster: PatternCluster, dir: "LONG" | "SHORT" | "HOLD", pnl: number, isWin: boolean) => {
    if (dir === "LONG") {
      cluster.longTotal = (cluster.longTotal || 0) + 1;
      if (isWin) cluster.longWins = (cluster.longWins || 0) + 1;
      const prevAvg = cluster.longAvgPnL || 0;
      const prevCount = cluster.longTotal - 1;
      cluster.longAvgPnL = prevCount > 0 ? (prevAvg * prevCount + pnl) / cluster.longTotal : pnl;
    } else if (dir === "SHORT") {
      cluster.shortTotal = (cluster.shortTotal || 0) + 1;
      if (isWin) cluster.shortWins = (cluster.shortWins || 0) + 1;
      const prevAvg = cluster.shortAvgPnL || 0;
      const prevCount = cluster.shortTotal - 1;
      cluster.shortAvgPnL = prevCount > 0 ? (prevAvg * prevCount + pnl) / cluster.shortTotal : pnl;
    }
    // Update overall win rate from direction-specific stats
    const totalTrades = (cluster.longTotal || 0) + (cluster.shortTotal || 0);
    const totalWins = (cluster.longWins || 0) + (cluster.shortWins || 0);
    cluster.winRate = totalTrades > 0 ? totalWins / totalTrades : 0;
    cluster.wins = totalWins;
  };
  
  if (nearestCluster && similarity >= MIN_SIMILARITY_THRESHOLD) {
    nearestCluster.centroid = updateClusterCentroid(nearestCluster, embedding);
    nearestCluster.support++;
    nearestCluster.avgReturn = (nearestCluster.avgReturn * (nearestCluster.support - 1) + forwardReturn8) / nearestCluster.support;
    nearestCluster.maturity = computeMaturity(nearestCluster.support);
    
    // CRITICAL: Update direction-specific stats
    updateClusterDirectionStats(nearestCluster, direction, actualPnL, won);
    
    nearestCluster.samples.push({
      timestamp: feature.timestamp,
      embedding,
      forwardReturn8,
      won,
      direction,
      actualPnL,
    });
    
    if (nearestCluster.samples.length > 200) {
      nearestCluster.samples = nearestCluster.samples.slice(-200);
    }
  } else if (canCreate && totalClusters < MAX_PATTERNS_TOTAL) {
    const regimeClusters = Array.from(patternClusters.values()).filter(c => c.regime === regime);
    const newId = generateClusterId(regime, regimeClusters.length);
    
    const newCluster: PatternCluster = {
      id: newId,
      regime,
      centroid: embedding,
      support: 1,
      wins: won ? 1 : 0,
      winRate: won ? 1 : 0,
      avgReturn: forwardReturn8,
      maturity: computeMaturity(1),
      samples: [{
        timestamp: feature.timestamp,
        embedding,
        forwardReturn8,
        won,
        direction,
        actualPnL,
      }],
      // Initialize direction-specific stats
      longWins: direction === "LONG" && won ? 1 : 0,
      longTotal: direction === "LONG" ? 1 : 0,
      longAvgPnL: direction === "LONG" ? actualPnL : 0,
      shortWins: direction === "SHORT" && won ? 1 : 0,
      shortTotal: direction === "SHORT" ? 1 : 0,
      shortAvgPnL: direction === "SHORT" ? actualPnL : 0,
    };
    
    patternClusters.set(newId, newCluster);
    console.log(`Created new pattern cluster: ${newId} (total: ${patternClusters.size}/${MAX_PATTERNS_TOTAL})`);
  } else if (nearestCluster) {
    nearestCluster.centroid = updateClusterCentroid(nearestCluster, embedding);
    nearestCluster.support++;
    nearestCluster.avgReturn = (nearestCluster.avgReturn * (nearestCluster.support - 1) + forwardReturn8) / nearestCluster.support;
    nearestCluster.maturity = computeMaturity(nearestCluster.support);
    updateClusterDirectionStats(nearestCluster, direction, actualPnL, won);
  }
  
  // TRAIN/TEST SPLIT: Assign patterns to train or test window
  // Default 80/20 split based on data timestamp (not random) for walk-forward validation
  // This prevents look-ahead bias by ensuring test patterns are always from later periods
  const trainingWindow = params.isTestSet ? "test" : "train";
  
  await db.insert(patterns).values({
    timestamp: feature.timestamp,
    embedding: feature.embedding,
    featureHash: JSON.stringify(feature.embedding).slice(0, 64),
    forwardReturn8: forwardReturn8,
    forwardReturn16: forwardReturn16,
    forwardMaxDrawdown: maxDrawdown,
    forwardMaxRunup: maxRunup,
    timeToMfe: timeToMfe,
    forwardWin: won,
    regime: regime,
    label: actualOutcome,
    atrAtEntry: atrAtEntry,
    dynamicThreshold: dynamicThreshold,
    direction: direction,
    actualPnl: actualPnL,
    createdAt: Date.now(),
    trainingWindow: trainingWindow,
  });
}

// Recalculate forward_win for all existing patterns using correct return-based logic
// This fixes the mislabeled patterns from the old regime-based logic
// Uses direct SQL for performance (updates 500k+ patterns in seconds instead of hours)
export async function recalculatePatternLabels(): Promise<{ updated: number; errors: number }> {
  console.log("[Pattern Memory] Starting pattern label recalculation using optimized SQL...");
  
  const startTime = Date.now();
  const COSTS = 0.0010; // Standardized transaction costs
  
  try {
    // Single SQL update that recalculates all labels based on forward_return_8 and dynamic_threshold
    // Logic: 
    //   - LONG win: forward_return_8 - costs > threshold
    //   - SHORT win: -forward_return_8 - costs > threshold
    //   - If neither direction profitable, direction = HOLD, forward_win = false
    const result = await db.execute(sql`
      UPDATE patterns SET
        direction = CASE
          WHEN forward_return_8 - ${COSTS} > COALESCE(dynamic_threshold, 0.003) THEN 'LONG'
          WHEN -forward_return_8 - ${COSTS} > COALESCE(dynamic_threshold, 0.003) THEN 'SHORT'
          ELSE 'HOLD'
        END,
        forward_win = CASE
          WHEN forward_return_8 - ${COSTS} > COALESCE(dynamic_threshold, 0.003) THEN true
          WHEN -forward_return_8 - ${COSTS} > COALESCE(dynamic_threshold, 0.003) THEN true
          ELSE false
        END,
        actual_pnl = CASE
          WHEN forward_return_8 - ${COSTS} > COALESCE(dynamic_threshold, 0.003) THEN forward_return_8 - ${COSTS}
          WHEN -forward_return_8 - ${COSTS} > COALESCE(dynamic_threshold, 0.003) THEN -forward_return_8 - ${COSTS}
          ELSE 0
        END
    `);
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    
    // Get count of updated rows
    const countResult = await db.execute(sql`SELECT COUNT(*) as count FROM patterns`);
    const countRow = countResult[0] as { count: string | number } | undefined;
    const totalPatterns = countRow ? Number(countRow.count) : 0;
    
    console.log(`[Pattern Memory] Recalculation complete: ${totalPatterns} patterns updated in ${duration}s`);
    
    // Log new distribution
    const distResult = await db.execute(sql`
      SELECT 
        direction, 
        COUNT(*) as count, 
        SUM(CASE WHEN forward_win THEN 1 ELSE 0 END) as wins,
        AVG(forward_return_8) as avg_return
      FROM patterns 
      GROUP BY direction
    `);
    console.log("[Pattern Memory] New distribution by direction:", distResult);
    
    return { updated: totalPatterns, errors: 0 };
  } catch (err) {
    console.error("[Pattern Memory] Recalculation error:", err);
    return { updated: 0, errors: 1 };
  }
}

export interface SimilarityDistribution {
  min: number;
  max: number;
  mean: number;
  median: number;
  count: number;
}

let lastSimilarityDist: SimilarityDistribution = { min: 0, max: 0, mean: 0, median: 0, count: 0 };

export function getLastSimilarityDistribution(): SimilarityDistribution {
  return lastSimilarityDist;
}

function normalizeEmbedding(embedding: number[]): number[] {
  const magnitude = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
  if (magnitude === 0 || !isFinite(magnitude)) {
    return embedding.map(() => 0);
  }
  return embedding.map(v => v / magnitude);
}

function isValidEmbedding(embedding: number[]): boolean {
  if (!Array.isArray(embedding) || embedding.length === 0) return false;
  const allZero = embedding.every(v => v === 0);
  const hasInvalid = embedding.some(v => !isFinite(v));
  return !allZero && !hasInvalid;
}

// ============================================================================
// PRECISION MODE PATTERN MATCHING (P0 Fixes Implementation)
// ============================================================================

export interface FindSimilarPatternsOptions {
  topK?: number;
  minSimilarity?: number;
  currentTimestamp: number;           // P0-1: MANDATORY - the candle timestamp being predicted
  currentRegime?: string;             // P0-3: Filter by regime (optional but recommended)
  currentAtr?: number;                // P0-3: Current ATR for volatility bucket filtering
  config?: PrecisionModeConfig;       // Custom precision config (defaults to PRECISION_MODE_CONFIG)
}

export interface PrecisionPatternMatch extends PatternMatch {
  timeDecayWeight: number;            // P1-5: Recency weight
  falseFriendPenalty: number;         // P1-3: Penalty from past failures
  effectiveSimilarity: number;        // Final similarity after adjustments
  volatilityBucket: VolatilityBucket; // P0-3: Volatility bucket at pattern time
  patternId: number;                  // For tracking in episodes
}

export async function findSimilarPatterns(
  currentEmbedding: number[],
  topK: number = 50,
  minSimilarity: number = MIN_SIMILARITY_THRESHOLD,
  embargoTimestamp?: number,
  currentTimestamp?: number
): Promise<PatternMatch[]> {
  // P0-1: CRITICAL - currentTimestamp is MANDATORY to prevent backtest leakage
  // In precision mode, we throw if not provided - this is intentional to catch bugs
  if (currentTimestamp === undefined) {
    if (PRECISION_MODE_CONFIG.enabled) {
      throw new Error(
        "[Pattern Memory] CRITICAL ERROR: currentTimestamp is MANDATORY in precision mode. " +
        "Passing undefined leaks future data in backtests. Pass the candle timestamp, not Date.now()."
      );
    }
    // Legacy fallback only when precision mode is disabled
    console.warn("[Pattern Memory] WARNING: currentTimestamp not provided - using Date.now(). Disable this warning by passing timestamp.");
    currentTimestamp = Date.now();
  }
  
  // Use new precision function internally
  const precisionMatches = await findSimilarPatternsPrecision(currentEmbedding, {
    topK,
    minSimilarity,
    currentTimestamp,
    config: { ...PRECISION_MODE_CONFIG, regimeFilterEnabled: false, volatilityFilterEnabled: false },
  });
  
  // Convert back to legacy format
  return precisionMatches;
}

/**
 * PRECISION MODE: Find similar patterns with all P0/P1 fixes
 * - P0-1: Mandatory timestamp with proper embargo calculation
 * - P0-3: Regime + volatility gating before cosine similarity
 * - P1-3: Hard-negative mining (false friend penalties)
 * - P1-5: Time-decay weighting
 */
export async function findSimilarPatternsPrecision(
  currentEmbedding: number[],
  options: FindSimilarPatternsOptions
): Promise<PrecisionPatternMatch[]> {
  const config = options.config || PRECISION_MODE_CONFIG;
  const currentTs = options.currentTimestamp;
  
  // P0-1: Compute proper embargo based on horizon + sequence length
  const embargoCandles = config.embargoHorizon + 2 * config.embargoSequenceLength;
  const embargoMs = currentTs - (embargoCandles * config.candleIntervalMs);
  
  // Determine current volatility bucket for filtering
  let currentVolBucket: VolatilityBucket | null = null;
  if (config.volatilityFilterEnabled && options.currentAtr !== undefined) {
    currentVolBucket = getVolatilityBucket(options.currentAtr);
  }
  
  const normalizedCurrent = normalizeEmbedding(currentEmbedding);
  if (!isValidEmbedding(normalizedCurrent)) {
    console.warn("[Pattern Memory Precision] Invalid embedding - all zeros or NaN");
    return [];
  }
  
  // Fetch patterns from DB
  const allPatterns = await db.select().from(patterns).orderBy(desc(patterns.timestamp)).limit(15000);
  
  const matches: PrecisionPatternMatch[] = [];
  const allSimilarities: number[] = [];
  let filteredByRegime = 0;
  let filteredByVolatility = 0;
  let filteredByEmbargo = 0;
  
  for (const pattern of allPatterns) {
    // P0-1: Strict embargo based on currentTimestamp
    if (pattern.timestamp > embargoMs) {
      filteredByEmbargo++;
      continue;
    }
    
    // P0-3: Regime gating - only match within same regime
    if (config.regimeFilterEnabled && options.currentRegime) {
      const patternRegime = pattern.regime || "chop";
      if (patternRegime !== options.currentRegime) {
        filteredByRegime++;
        continue;
      }
    }
    
    // P0-3: Volatility bucket gating
    if (config.volatilityFilterEnabled && currentVolBucket !== null) {
      const patternAtr = pattern.atrAtEntry || 0;
      const patternVolBucket = getVolatilityBucket(patternAtr);
      // Allow adjacent buckets (low-medium, medium-high) but not extremes
      const bucketDistance = Math.abs(
        ["low", "medium", "high", "extreme"].indexOf(currentVolBucket) -
        ["low", "medium", "high", "extreme"].indexOf(patternVolBucket)
      );
      if (bucketDistance > 1) {
        filteredByVolatility++;
        continue;
      }
    }
    
    const embedding = pattern.embedding as number[];
    if (!isValidEmbedding(embedding)) continue;
    
    const normalizedPattern = normalizeEmbedding(embedding);
    const rawSimilarity = cosineSimilarity(normalizedCurrent, normalizedPattern);
    
    if (!isFinite(rawSimilarity)) continue;
    allSimilarities.push(rawSimilarity);
    
    // Skip if below threshold or exact match (data leak)
    const threshold = options.minSimilarity ?? config.similarityThreshold;
    if (rawSimilarity < threshold || rawSimilarity >= 0.995) continue;
    
    // P1-5: Time decay weight
    const ageMs = currentTs - pattern.timestamp;
    const ageDays = ageMs / (24 * 60 * 60 * 1000);
    const timeDecayWeight = Math.exp(-ageDays / config.timeTauDays);
    
    // P1-3: False friend penalty
    const patternId = pattern.id;
    const ffPenalty = getFalseFriendPenalty(patternId);
    
    // Compute effective similarity (adjusted by time decay and penalty)
    const effectiveSimilarity = rawSimilarity * timeDecayWeight * (1 - ffPenalty);
    
    // Get cluster info
    const clusterInfo = findClusterForPattern(pattern.regime || "chop", embedding);
    
    // Compute volatility bucket
    const volBucket = getVolatilityBucket(pattern.atrAtEntry || 0);
    
    matches.push({
      timestamp: pattern.timestamp,
      similarity: rawSimilarity,
      forwardReturn8: pattern.forwardReturn8 || 0,
      forwardReturn16: pattern.forwardReturn16 || 0,
      maxDrawdown: pattern.forwardMaxDrawdown || 0,
      maxRunup: pattern.forwardMaxRunup || 0,
      timeToMfe: pattern.timeToMfe || 0,
      won: pattern.forwardWin || false,
      regime: pattern.regime || "unknown",
      label: pattern.label || "unknown",
      atrAtEntry: pattern.atrAtEntry || 0,
      dynamicThreshold: pattern.dynamicThreshold || 0.004,
      clusterId: clusterInfo?.clusterId,
      clusterMaturity: clusterInfo?.maturity,
      direction: (pattern.direction as "LONG" | "SHORT" | "HOLD") || "HOLD",
      actualPnL: pattern.actualPnl || 0,
      // Precision mode fields
      timeDecayWeight,
      falseFriendPenalty: ffPenalty,
      effectiveSimilarity,
      volatilityBucket: volBucket,
      patternId,
    });
  }
  
  // Log similarity distribution
  if (allSimilarities.length > 0) {
    allSimilarities.sort((a, b) => a - b);
    const min = allSimilarities[0];
    const max = allSimilarities[allSimilarities.length - 1];
    const mean = allSimilarities.reduce((a, b) => a + b, 0) / allSimilarities.length;
    const medianIdx = Math.floor(allSimilarities.length / 2);
    const median = allSimilarities[medianIdx];
    
    lastSimilarityDist = { min, max, mean, median, count: allSimilarities.length };
  }
  
  // Sort by effective similarity (includes time decay and penalties)
  matches.sort((a, b) => b.effectiveSimilarity - a.effectiveSimilarity);
  
  const topMatches = matches.slice(0, options.topK || 50);
  
  // P0-2: Check minimum matches floor
  if (config.enabled && topMatches.length < config.minMatchesFloor) {
    console.log(`[Pattern Memory Precision] Insufficient matches: ${topMatches.length}/${config.minMatchesFloor} - returning empty (filters: embargo=${filteredByEmbargo}, regime=${filteredByRegime}, vol=${filteredByVolatility})`);
    return [];
  }
  
  if (topMatches.length > 0) {
    const avgEffSim = topMatches.reduce((s, m) => s + m.effectiveSimilarity, 0) / topMatches.length;
    console.log(`[Pattern Memory Precision] Found ${topMatches.length} matches (avgEffSim=${(avgEffSim * 100).toFixed(1)}%, filters: embargo=${filteredByEmbargo}, regime=${filteredByRegime}, vol=${filteredByVolatility})`);
  }
  
  return topMatches;
}

function findClusterForPattern(regime: string, embedding: number[]): { clusterId: string; maturity: number } | null {
  const normalizedEmb = normalizeEmbedding(embedding);
  let bestCluster: PatternCluster | null = null;
  let bestSimilarity = 0;
  
  const clusters = Array.from(patternClusters.values());
  for (const cluster of clusters) {
    if (cluster.regime !== regime) continue;
    
    const similarity = cosineSimilarity(normalizedEmb, cluster.centroid);
    if (similarity > bestSimilarity && similarity >= MIN_SIMILARITY_THRESHOLD) {
      bestSimilarity = similarity;
      bestCluster = cluster;
    }
  }
  
  if (bestCluster) {
    return { clusterId: bestCluster.id, maturity: bestCluster.maturity };
  }
  return null;
}

// ============================================================================
// ROBUST STATISTICS (P0-4) - Similarity-Weighted Median, Trimmed Mean, Quantiles
// ============================================================================

/**
 * Compute weighted median (more robust than mean for outliers)
 */
function weightedMedian(values: number[], weights: number[]): number {
  if (values.length === 0) return 0;
  if (values.length === 1) return values[0];
  
  // Pair values with weights and sort by value
  const paired = values.map((v, i) => ({ value: v, weight: weights[i] }));
  paired.sort((a, b) => a.value - b.value);
  
  const totalWeight = paired.reduce((sum, p) => sum + p.weight, 0);
  if (totalWeight === 0) return paired[Math.floor(paired.length / 2)].value;
  
  let cumWeight = 0;
  for (const p of paired) {
    cumWeight += p.weight;
    if (cumWeight >= totalWeight / 2) {
      return p.value;
    }
  }
  
  return paired[paired.length - 1].value;
}

/**
 * Compute weighted trimmed mean (exclude top/bottom 10% by weight)
 */
function weightedTrimmedMean(values: number[], weights: number[], trimPct: number = 0.10): number {
  if (values.length === 0) return 0;
  if (values.length <= 2) return values.reduce((a, b) => a + b, 0) / values.length;
  
  // Pair values with weights and sort by value
  const paired = values.map((v, i) => ({ value: v, weight: weights[i] }));
  paired.sort((a, b) => a.value - b.value);
  
  const totalWeight = paired.reduce((sum, p) => sum + p.weight, 0);
  const trimWeight = totalWeight * trimPct;
  
  let lowerCum = 0;
  let upperCum = 0;
  let trimmedSum = 0;
  let trimmedWeightSum = 0;
  
  for (let i = 0; i < paired.length; i++) {
    const p = paired[i];
    lowerCum += p.weight;
    
    // Calculate how much of this point is in the lower trimmed region
    const lowerExclude = Math.max(0, trimWeight - (lowerCum - p.weight));
    
    // Calculate upper trimmed region
    const upperStart = totalWeight - trimWeight;
    const upperExclude = Math.max(0, lowerCum - upperStart);
    
    // Include the portion that's not trimmed
    const includeWeight = Math.max(0, p.weight - lowerExclude - upperExclude);
    if (includeWeight > 0) {
      trimmedSum += p.value * includeWeight;
      trimmedWeightSum += includeWeight;
    }
  }
  
  return trimmedWeightSum > 0 ? trimmedSum / trimmedWeightSum : 0;
}

/**
 * Compute weighted quantile
 */
function weightedQuantile(values: number[], weights: number[], q: number): number {
  if (values.length === 0) return 0;
  if (values.length === 1) return values[0];
  
  const paired = values.map((v, i) => ({ value: v, weight: weights[i] }));
  paired.sort((a, b) => a.value - b.value);
  
  const totalWeight = paired.reduce((sum, p) => sum + p.weight, 0);
  const targetWeight = totalWeight * q;
  
  let cumWeight = 0;
  for (const p of paired) {
    cumWeight += p.weight;
    if (cumWeight >= targetWeight) {
      return p.value;
    }
  }
  
  return paired[paired.length - 1].value;
}

// ============================================================================
// P1-4: EV + UNCERTAINTY OUTPUT FORMAT
// ============================================================================

export interface PatternPrediction {
  // Direction decision
  direction: "LONG" | "SHORT" | "HOLD";
  shouldTrade: boolean;
  
  // Expected values (P1-4)
  evLong: number;           // Expected value if going LONG
  evShort: number;          // Expected value if going SHORT
  pWinLong: number;         // P(win | LONG)
  pWinShort: number;        // P(win | SHORT)
  
  // Uncertainty metrics (P1-2)
  uncertainty: number;       // Credible interval width
  credibleIntervalLow: number;
  credibleIntervalHigh: number;
  
  // Trade levels derived from MAE/MFE quantiles
  suggestedSL: number;       // Based on MAE percentile
  suggestedTP1: number;      // Based on MFE 50th percentile
  suggestedTP2: number;      // Based on MFE 75th percentile
  
  // Robust stats (P0-4)
  weightedMedianReturn: number;
  trimmedMeanReturn: number;
  returnQ10: number;         // 10th percentile return
  returnQ25: number;
  returnQ50: number;         // Median
  returnQ75: number;
  returnQ90: number;         // 90th percentile return
  maeQ70: number;            // 70th percentile MAE (for SL)
  mfeQ70: number;            // 70th percentile MFE (for TP)
  
  // Match quality
  matchCount: number;
  avgEffectiveSimilarity: number;
  avgTimeDecay: number;
  
  // Reasoning
  reasoning: string;
}

/**
 * Compute precision-mode pattern prediction with EV + uncertainty
 * This is the institutional-grade output format
 */
export function computePrecisionPrediction(
  matches: PrecisionPatternMatch[],
  config: PrecisionModeConfig = PRECISION_MODE_CONFIG
): PatternPrediction {
  const emptyPrediction: PatternPrediction = {
    direction: "HOLD",
    shouldTrade: false,
    evLong: 0,
    evShort: 0,
    pWinLong: 0,
    pWinShort: 0,
    uncertainty: 1.0,
    credibleIntervalLow: 0,
    credibleIntervalHigh: 1,
    suggestedSL: 0,
    suggestedTP1: 0,
    suggestedTP2: 0,
    weightedMedianReturn: 0,
    trimmedMeanReturn: 0,
    returnQ10: 0,
    returnQ25: 0,
    returnQ50: 0,
    returnQ75: 0,
    returnQ90: 0,
    maeQ70: 0,
    mfeQ70: 0,
    matchCount: 0,
    avgEffectiveSimilarity: 0,
    avgTimeDecay: 0,
    reasoning: "Insufficient pattern matches for prediction",
  };
  
  if (matches.length < config.minMatchesFloor) {
    return emptyPrediction;
  }
  
  // Weights based on effective similarity (includes time decay and penalties)
  const weights = matches.map(m => m.effectiveSimilarity);
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  const normalizedWeights = weights.map(w => w / totalWeight);
  
  // Separate by direction
  const longMatches = matches.filter(m => m.direction === "LONG");
  const shortMatches = matches.filter(m => m.direction === "SHORT");
  
  // Compute direction-specific stats
  const longWins = longMatches.filter(m => m.won).length;
  const longLosses = longMatches.length - longWins;
  const shortWins = shortMatches.filter(m => m.won).length;
  const shortLosses = shortMatches.length - shortWins;
  
  // P1-2 FIX: Separate average win and loss returns for proper EV calculation
  const longWinMatches = longMatches.filter(m => m.won);
  const longLossMatches = longMatches.filter(m => !m.won);
  const shortWinMatches = shortMatches.filter(m => m.won);
  const shortLossMatches = shortMatches.filter(m => !m.won);
  
  const longAvgWin = longWinMatches.length > 0 
    ? longWinMatches.reduce((s, m) => s + Math.abs(m.actualPnL || 0), 0) / longWinMatches.length 
    : 0;
  const longAvgLoss = longLossMatches.length > 0 
    ? longLossMatches.reduce((s, m) => s + Math.abs(m.actualPnL || 0), 0) / longLossMatches.length 
    : 0;
  const shortAvgWin = shortWinMatches.length > 0 
    ? shortWinMatches.reduce((s, m) => s + Math.abs(m.actualPnL || 0), 0) / shortWinMatches.length 
    : 0;
  const shortAvgLoss = shortLossMatches.length > 0 
    ? shortLossMatches.reduce((s, m) => s + Math.abs(m.actualPnL || 0), 0) / shortLossMatches.length 
    : 0;
  
  // Bayesian posteriors for win probability with proper EV calculation
  const longBayesian = computeBayesianStats(
    longWins, 
    longLosses,
    longAvgWin,
    longAvgLoss,
    config
  );
  
  const shortBayesian = computeBayesianStats(
    shortWins,
    shortLosses,
    shortAvgWin,
    shortAvgLoss,
    config
  );
  
  // P0-4: Robust statistics for returns using weighted median/trimmed mean
  const returns8 = matches.map(m => m.forwardReturn8);
  const weightedMedianReturn = weightedMedian(returns8, weights);
  const trimmedMeanReturn = weightedTrimmedMean(returns8, weights, 0.10);
  
  // Return quantiles
  const returnQ10 = weightedQuantile(returns8, weights, 0.10);
  const returnQ25 = weightedQuantile(returns8, weights, 0.25);
  const returnQ50 = weightedQuantile(returns8, weights, 0.50);
  const returnQ75 = weightedQuantile(returns8, weights, 0.75);
  const returnQ90 = weightedQuantile(returns8, weights, 0.90);
  
  // MAE/MFE quantiles for SL/TP
  const maes = matches.map(m => Math.abs(m.maxDrawdown));
  const mfes = matches.map(m => m.maxRunup);
  const maeQ70 = weightedQuantile(maes, weights, 0.70);
  const mfeQ70 = weightedQuantile(mfes, weights, 0.70);
  const mfeQ50 = weightedQuantile(mfes, weights, 0.50);
  
  // P1-2 FIX: Use EV from Bayesian stats (already computed with proper win/loss separation)
  // EV = P(win) * avgWin - P(loss) * avgLoss (computed inside computeBayesianStats)
  const evLong = longBayesian.evLong;
  const evShort = shortBayesian.evShort;
  
  // Determine best direction based on EV
  let direction: "LONG" | "SHORT" | "HOLD" = "HOLD";
  let shouldTrade = false;
  let selectedBayesian = longBayesian;
  
  if (evLong > 0 && evLong > evShort && longBayesian.posteriorMean >= config.minPosteriorPWin) {
    direction = "LONG";
    selectedBayesian = longBayesian;
    shouldTrade = longBayesian.uncertainty <= config.maxUncertainty;
  } else if (evShort > 0 && evShort > evLong && shortBayesian.posteriorMean >= config.minPosteriorPWin) {
    direction = "SHORT";
    selectedBayesian = shortBayesian;
    shouldTrade = shortBayesian.uncertainty <= config.maxUncertainty;
  }
  
  // If neither direction has positive EV or meets threshold, HOLD
  if (direction !== "HOLD" && !shouldTrade) {
    direction = "HOLD";
  }
  
  // Average match quality metrics
  const avgEffectiveSimilarity = matches.reduce((s, m) => s + m.effectiveSimilarity, 0) / matches.length;
  const avgTimeDecay = matches.reduce((s, m) => s + m.timeDecayWeight, 0) / matches.length;
  
  // Build reasoning
  const reasoning = `${matches.length} matches (${longMatches.length}L/${shortMatches.length}S). ` +
    `EV: L=${(evLong * 100).toFixed(2)}% S=${(evShort * 100).toFixed(2)}%. ` +
    `P(win): L=${(longBayesian.posteriorMean * 100).toFixed(0)}% S=${(shortBayesian.posteriorMean * 100).toFixed(0)}%. ` +
    `Uncertainty: ${(selectedBayesian.uncertainty * 100).toFixed(0)}%. ` +
    `MedianRet: ${(weightedMedianReturn * 100).toFixed(2)}%`;
  
  return {
    direction,
    shouldTrade,
    evLong,
    evShort,
    pWinLong: longBayesian.posteriorMean,
    pWinShort: shortBayesian.posteriorMean,
    uncertainty: selectedBayesian.uncertainty,
    credibleIntervalLow: selectedBayesian.credibleIntervalLow,
    credibleIntervalHigh: selectedBayesian.credibleIntervalHigh,
    suggestedSL: maeQ70,
    suggestedTP1: mfeQ50,
    suggestedTP2: mfeQ70,
    weightedMedianReturn,
    trimmedMeanReturn,
    returnQ10,
    returnQ25,
    returnQ50,
    returnQ75,
    returnQ90,
    maeQ70,
    mfeQ70,
    matchCount: matches.length,
    avgEffectiveSimilarity,
    avgTimeDecay,
    reasoning,
  };
}

// Legacy function - kept for backward compatibility
export function computePatternStats(matches: PatternMatch[]): PatternStats {
  const matureMatches = matches.filter(m => (m.clusterMaturity || 0) >= 0.25);
  
  if (matureMatches.length === 0) {
    return {
      matchCount: matches.length,
      avgReturn8: 0,
      avgReturn16: 0,
      winRate: 0,
      avgDrawdown: 0,
      avgRunup: 0,
      avgTimeToMfe: 0,
      mae70thPercentile: 0,
      mfe70thPercentile: 0,
      bestCase: 0,
      worstCase: 0,
      consistency: 0,
      regimeBreakdown: {},
      matureMatchCount: 0,
    };
  }
  
  // If we have PrecisionPatternMatch, use weighted stats (P0-4)
  const isPrecisionMatch = (m: PatternMatch): m is PrecisionPatternMatch => 
    'effectiveSimilarity' in m;
  
  let weights: number[];
  if (matureMatches.length > 0 && isPrecisionMatch(matureMatches[0])) {
    weights = (matureMatches as PrecisionPatternMatch[]).map(m => m.effectiveSimilarity);
  } else {
    // Equal weights for legacy matches
    weights = matureMatches.map(() => 1);
  }
  
  const returns8 = matureMatches.map(m => m.forwardReturn8);
  const returns16 = matureMatches.map(m => m.forwardReturn16);
  const drawdowns = matureMatches.map(m => m.maxDrawdown);
  const runups = matureMatches.map(m => m.maxRunup);
  const timesToMfe = matureMatches.map(m => m.timeToMfe);
  const wins = matureMatches.filter(m => m.won).length;
  
  const regimeBreakdown: Record<string, number> = {};
  for (const m of matureMatches) {
    regimeBreakdown[m.regime] = (regimeBreakdown[m.regime] || 0) + 1;
  }
  
  // P0-4: Use weighted trimmed mean instead of simple average
  const avgReturn8 = weightedTrimmedMean(returns8, weights, 0.10);
  const avgReturn16 = weightedTrimmedMean(returns16, weights, 0.10);
  
  // Use weighted quantiles for MAE/MFE
  const mae70thPercentile = weightedQuantile(drawdowns.map(Math.abs), weights, 0.70);
  const mfe70thPercentile = weightedQuantile(runups, weights, 0.70);
  
  const variance = returns8.reduce((sum, r) => sum + Math.pow(r - avgReturn8, 2), 0) / matureMatches.length;
  const stdDev = Math.sqrt(variance);
  
  return {
    matchCount: matches.length,
    avgReturn8,
    avgReturn16,
    winRate: wins / matureMatches.length,
    avgDrawdown: weightedTrimmedMean(drawdowns, weights, 0.10),
    avgRunup: weightedTrimmedMean(runups, weights, 0.10),
    avgTimeToMfe: weightedTrimmedMean(timesToMfe, weights, 0.10),
    mae70thPercentile,
    mfe70thPercentile,
    bestCase: Math.max(...returns8),
    worstCase: Math.min(...returns8),
    consistency: avgReturn8 === 0 ? 0 : 1 - (stdDev / Math.abs(avgReturn8)),
    regimeBreakdown,
    matureMatchCount: matureMatches.length,
  };
}

export function getPatternConfidence(stats: PatternStats): {
  direction: "LONG" | "SHORT" | "HOLD";
  confidence: number;
  reasoning: string;
} {
  // SELECTIVE MODE: Require minimum mature patterns for reliable signals
  // Research shows 10+ mature patterns needed for statistical reliability
  if (stats.matureMatchCount < 10) {
    return {
      direction: "HOLD",
      confidence: 0.7,
      reasoning: `Insufficient pattern data: ${stats.matureMatchCount}/10 mature patterns. HOLD until more data available.`,
    };
  }
  
  const expectedReturn = stats.avgReturn8 * 100;
  const winRate = stats.winRate;
  
  // SELECTIVE MODE: Require positive edge for action
  // Only trade when historical patterns show > 45% win rate AND meaningful expected return
  if (winRate < 0.45 || Math.abs(expectedReturn) < 0.1) {
    return {
      direction: "HOLD",
      confidence: 0.6,
      reasoning: `No edge detected: Win rate ${(winRate * 100).toFixed(1)}%, Expected return ${expectedReturn.toFixed(2)}%. HOLD.`,
    };
  }
  
  // Determine direction based on expected return with proper thresholds
  let direction: "LONG" | "SHORT" | "HOLD";
  if (expectedReturn > 0.15 && winRate > 0.50) {
    direction = "LONG";
  } else if (expectedReturn < -0.15 && winRate > 0.50) {
    direction = "SHORT";
  } else {
    direction = "HOLD";
  }
  
  const confidence = Math.max(0.3, Math.min(0.9, (winRate * 0.6 + stats.consistency * 0.4)));
  
  return {
    direction,
    confidence,
    reasoning: `${stats.matchCount} similar patterns (${stats.matureMatchCount} mature). Win rate: ${(winRate * 100).toFixed(1)}%, Avg return: ${expectedReturn.toFixed(2)}%`,
  };
}

export interface RegimeStats {
  count: number;
  winRate: number;
  avgReturn: number;
}

export interface ClusterSummary {
  id: string;
  regime: string;
  support: number;
  maturity: number;
  winRate: number;
  avgReturn: number;
  isMature: boolean;
}

export interface StoredPatternStats {
  totalPatterns: number;
  activePatterns: number;
  immaturePatterns: number;
  maxPatterns: number;
  winRate: number;
  regimeBreakdown: Record<string, number>;
  regimeStats: Record<string, RegimeStats>;
  avgThreshold: number;
  recentWinRate: number;
  similarityHealthy: boolean;
  canCreatePatterns: boolean;
  patternClusters: ClusterSummary[];
  rawSampleCount: number;
  trainSetCount?: number;  // Patterns used for training (look-ahead bias free)
  testSetCount?: number;   // Patterns held out for validation (out-of-sample)
  testSetWinRate?: number; // Win rate on test set only (true performance metric)
}

export function getActivePatternClusters(): ClusterSummary[] {
  const summaries: ClusterSummary[] = [];
  
  const clusters = Array.from(patternClusters.values());
  for (const cluster of clusters) {
    summaries.push({
      id: cluster.id,
      regime: cluster.regime,
      support: cluster.support,
      maturity: cluster.maturity,
      winRate: cluster.winRate,
      avgReturn: cluster.avgReturn,
      isMature: isPatternMature(cluster),
    });
  }
  
  return summaries.sort((a, b) => b.support - a.support);
}

/**
 * Feature-vector-based embargo filter
 * Removes patterns from train/test sets that are too similar to boundary patterns
 * This prevents data leakage from temporally adjacent but feature-similar patterns
 */
function applyFeatureVectorEmbargo(
  trainPatterns: any[], 
  testPatterns: any[], 
  boundaryWindow: number = EMBARGO_CANDLES
): { filteredTrain: any[], filteredTest: any[], embargoedCount: number } {
  if (trainPatterns.length === 0 || testPatterns.length === 0) {
    return { filteredTrain: trainPatterns, filteredTest: testPatterns, embargoedCount: 0 };
  }
  
  // Find the boundary timestamp (last train pattern or first test pattern)
  const sortedTrain = [...trainPatterns].sort((a, b) => b.timestamp - a.timestamp);
  const sortedTest = [...testPatterns].sort((a, b) => a.timestamp - b.timestamp);
  
  const boundaryTime = sortedTrain[0]?.timestamp || sortedTest[0]?.timestamp;
  const msPerCandle = 15 * 60 * 1000; // 15 minutes
  const temporalEmbargoMs = boundaryWindow * msPerCandle;
  
  // Identify boundary patterns (within temporal embargo window of boundary)
  const boundaryTrainPatterns = sortedTrain.filter(p => 
    Math.abs(p.timestamp - boundaryTime) <= temporalEmbargoMs
  );
  const boundaryTestPatterns = sortedTest.filter(p => 
    Math.abs(p.timestamp - boundaryTime) <= temporalEmbargoMs
  );
  
  // Collect boundary embeddings
  const boundaryEmbeddings: number[][] = [];
  for (const p of [...boundaryTrainPatterns, ...boundaryTestPatterns]) {
    if (p.embedding && Array.isArray(p.embedding)) {
      boundaryEmbeddings.push(normalizeEmbedding(p.embedding));
    }
  }
  
  if (boundaryEmbeddings.length === 0) {
    return { filteredTrain: trainPatterns, filteredTest: testPatterns, embargoedCount: 0 };
  }
  
  // Check each non-boundary pattern for similarity to boundary patterns
  let embargoedCount = 0;
  
  const isEmbargoedBySimilarity = (pattern: any): boolean => {
    if (!pattern.embedding || !Array.isArray(pattern.embedding)) return false;
    
    // Skip if already in boundary window (already excluded by temporal embargo)
    if (Math.abs(pattern.timestamp - boundaryTime) <= temporalEmbargoMs) return false;
    
    const patternEmb = normalizeEmbedding(pattern.embedding);
    
    // Check similarity against all boundary embeddings
    for (const boundaryEmb of boundaryEmbeddings) {
      const similarity = cosineSimilarity(patternEmb, boundaryEmb);
      if (similarity > EMBARGO_SIMILARITY_THRESHOLD) {
        embargoedCount++;
        return true;  // Too similar to boundary pattern - exclude
      }
    }
    return false;
  };
  
  const filteredTrain = trainPatterns.filter(p => !isEmbargoedBySimilarity(p));
  const filteredTest = testPatterns.filter(p => !isEmbargoedBySimilarity(p));
  
  if (embargoedCount > 0) {
    console.log(`[Pattern Memory] Feature-vector embargo excluded ${embargoedCount} patterns (similarity > ${EMBARGO_SIMILARITY_THRESHOLD})`);
  }
  
  return { filteredTrain, filteredTest, embargoedCount };
}

// Get pattern statistics with optional test-set-only mode for unbiased metrics
// testSetOnly=true returns out-of-sample statistics (research-backed for true accuracy)
export async function getStoredPatternStats(testSetOnly: boolean = false): Promise<StoredPatternStats> {
  const emptyStats: StoredPatternStats = {
    totalPatterns: 0,
    activePatterns: 0,
    immaturePatterns: 0,
    maxPatterns: MAX_PATTERNS_TOTAL,
    winRate: 0,
    regimeBreakdown: { trend_up: 0, trend_down: 0, chop: 0, shock: 0 },
    regimeStats: {
      trend_up: { count: 0, winRate: 0, avgReturn: 0 },
      trend_down: { count: 0, winRate: 0, avgReturn: 0 },
      chop: { count: 0, winRate: 0, avgReturn: 0 },
      shock: { count: 0, winRate: 0, avgReturn: 0 },
    },
    avgThreshold: 0.004,
    recentWinRate: 0,
    similarityHealthy: false,
    canCreatePatterns: canCreateNewPatterns(),
    patternClusters: [],
    rawSampleCount: 0,
    trainSetCount: 0,
    testSetCount: 0,
    testSetWinRate: 0,
  };
  
  try {
    // Fetch patterns with optional filter for test set only
    let query = db.select().from(patterns).orderBy(desc(patterns.timestamp)).limit(10000);
    const allPatterns = await query;
    
    if (allPatterns.length === 0) {
      return emptyStats;
    }
    
    // Separate train and test sets
    const rawTrainPatterns = allPatterns.filter(p => p.trainingWindow !== "test");
    const rawTestPatterns = allPatterns.filter(p => p.trainingWindow === "test");
    
    // Apply feature-vector-based embargo to prevent data leakage from similar patterns
    // This filters out patterns that are too similar to the train/test boundary
    const { filteredTrain: trainPatterns, filteredTest: testPatterns, embargoedCount } = 
      applyFeatureVectorEmbargo(rawTrainPatterns, rawTestPatterns);
    
    // Use test set for metrics if requested (unbiased out-of-sample performance)
    const patternsForStats = testSetOnly && testPatterns.length > 0 ? testPatterns : allPatterns;
    
    const totalWins = patternsForStats.filter(p => p.forwardWin).length;
    const winRate = totalWins / patternsForStats.length;
    
    const regimeBreakdown: Record<string, number> = { trend_up: 0, trend_down: 0, chop: 0, shock: 0 };
    const regimeWins: Record<string, number> = { trend_up: 0, trend_down: 0, chop: 0, shock: 0 };
    const regimeReturns: Record<string, number[]> = { trend_up: [], trend_down: [], chop: [], shock: [] };
    let thresholdSum = 0;
    let thresholdCount = 0;
    
    for (const p of patternsForStats) {
      const regime = p.regime || "chop";
      const validRegime = regime in regimeBreakdown ? regime : "chop";
      
      regimeBreakdown[validRegime]++;
      if (p.forwardWin) regimeWins[validRegime]++;
      if (p.forwardReturn8 !== null) regimeReturns[validRegime].push(p.forwardReturn8);
      
      if (p.dynamicThreshold) {
        thresholdSum += p.dynamicThreshold;
        thresholdCount++;
      }
    }
    
    const regimeStats: Record<string, RegimeStats> = {};
    for (const regime of ["trend_up", "trend_down", "chop", "shock"]) {
      const count = regimeBreakdown[regime];
      const wins = regimeWins[regime];
      const returns = regimeReturns[regime];
      regimeStats[regime] = {
        count,
        winRate: count > 0 ? wins / count : 0,
        avgReturn: returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0,
      };
    }
    
    const recentPatterns = patternsForStats.slice(0, Math.min(100, patternsForStats.length));
    const recentWins = recentPatterns.filter(p => p.forwardWin).length;
    const recentWinRate = recentPatterns.length > 0 ? recentWins / recentPatterns.length : 0;
    
    // Compute test-set-specific metrics even when using all patterns
    const testWins = testPatterns.filter(p => p.forwardWin).length;
    const testSetWinRate = testPatterns.length > 0 ? testWins / testPatterns.length : 0;
    
    const simDist = getLastSimilarityDistribution();
    const similarityHealthy = simDist.mean > 0 && simDist.mean < 0.90;
    
    const clusterSummaries = getActivePatternClusters();
    const matureClusters = clusterSummaries.filter(c => c.isMature);
    const immatureClusters = clusterSummaries.filter(c => !c.isMature);
    
    // === VALIDATION LOGGING (Research-backed verification) ===
    // Verify win rate calculations match actual returns using same criteria as forwardWin
    // forwardWin is set when forwardReturn8 > dynamicThreshold (includes costs)
    if (patternsForStats.length >= 20) {
      // Compute win rate using same logic as forwardWin: return > threshold
      const patternsWithReturns = patternsForStats.filter(p => 
        p.forwardReturn8 !== null && p.dynamicThreshold !== undefined
      );
      const winsFromThreshold = patternsWithReturns.filter(p => 
        p.forwardReturn8! > (p.dynamicThreshold || 0)
      );
      const computedWinRate = patternsWithReturns.length > 0 
        ? winsFromThreshold.length / patternsWithReturns.length 
        : 0;
      
      // Compare with stored win rate (should match since same criteria)
      const storedWinRate = winRate;
      const discrepancy = Math.abs(computedWinRate - storedWinRate);
      
      if (discrepancy > 0.02) {
        console.warn(`[Pattern Validation] Win rate mismatch: stored=${(storedWinRate * 100).toFixed(1)}%, computed=${(computedWinRate * 100).toFixed(1)}% (diff: ${(discrepancy * 100).toFixed(1)}%) - check forwardWin logic`);
      } else if (embargoedCount > 0) {
        console.log(`[Pattern Validation] Win rates verified: ${(storedWinRate * 100).toFixed(1)}% (${patternsForStats.length} patterns, ${embargoedCount} embargoed for boundary protection)`);
      }
    }
    
    return {
      totalPatterns: patternClusters.size,
      activePatterns: matureClusters.length,
      immaturePatterns: immatureClusters.length,
      maxPatterns: MAX_PATTERNS_TOTAL,
      winRate,
      regimeBreakdown,
      regimeStats,
      avgThreshold: thresholdCount > 0 ? thresholdSum / thresholdCount : 0.004,
      recentWinRate,
      similarityHealthy,
      canCreatePatterns: canCreateNewPatterns(),
      patternClusters: clusterSummaries,
      rawSampleCount: patternsForStats.length,
      trainSetCount: trainPatterns.length,
      testSetCount: testPatterns.length,
      testSetWinRate,
    };
  } catch (error) {
    console.error("Error getting stored pattern stats:", error);
    return emptyStats;
  }
}

export async function initializePatternClusters(): Promise<void> {
  try {
    const allPatterns = await db.select().from(patterns).orderBy(desc(patterns.timestamp)).limit(10000);
    
    if (allPatterns.length === 0) {
      console.log("No patterns to cluster");
      return;
    }
    
    patternClusters.clear();
    
    const regimePatterns: Record<string, typeof allPatterns> = {
      trend_up: [],
      trend_down: [],
      chop: [],
      shock: [],
    };
    
    for (const p of allPatterns) {
      const regime = p.regime || "chop";
      if (regime in regimePatterns) {
        regimePatterns[regime].push(p);
      }
    }
    
    const clustersPerRegime = Math.floor(MAX_PATTERNS_TOTAL / 4);
    
    for (const regime of ["trend_up", "trend_down", "chop", "shock"] as const) {
      const regimeData = regimePatterns[regime];
      if (regimeData.length === 0) continue;
      
      const sampleSize = Math.min(regimeData.length, 1000);
      const sampledPatterns = regimeData.slice(0, sampleSize);
      
      const numClusters = Math.min(clustersPerRegime, Math.ceil(sampledPatterns.length / MIN_SAMPLES_PER_PATTERN));
      
      if (numClusters === 0) continue;
      
      const clustersForRegime = kMeansClustering(sampledPatterns, numClusters, regime);
      
      for (const cluster of clustersForRegime) {
        patternClusters.set(cluster.id, cluster);
      }
    }
    
    console.log(`Initialized ${patternClusters.size} pattern clusters (max: ${MAX_PATTERNS_TOTAL})`);
    const clustersForLog = Array.from(patternClusters.entries());
    for (const [id, cluster] of clustersForLog) {
      console.log(`  ${id}: support=${cluster.support}, maturity=${cluster.maturity.toFixed(2)}, winRate=${(cluster.winRate * 100).toFixed(1)}%`);
    }
  } catch (error) {
    console.error("Error initializing pattern clusters:", error);
  }
}

function kMeansClustering(
  patternsData: any[],
  k: number,
  regime: "trend_up" | "trend_down" | "chop" | "shock"
): PatternCluster[] {
  if (patternsData.length === 0 || k === 0) return [];
  
  const validPatterns = patternsData.filter(p => {
    const emb = p.embedding as number[];
    return isValidEmbedding(emb);
  });
  
  if (validPatterns.length < k) {
    k = validPatterns.length;
  }
  
  if (k === 0) return [];
  
  const step = Math.floor(validPatterns.length / k);
  let centroids: number[][] = [];
  for (let i = 0; i < k; i++) {
    const p = validPatterns[i * step];
    centroids.push(normalizeEmbedding(p.embedding as number[]));
  }
  
  const assignments: number[] = new Array(validPatterns.length).fill(0);
  
  for (let iter = 0; iter < 10; iter++) {
    for (let i = 0; i < validPatterns.length; i++) {
      const emb = normalizeEmbedding(validPatterns[i].embedding as number[]);
      let bestCluster = 0;
      let bestSim = -1;
      
      for (let c = 0; c < k; c++) {
        const sim = cosineSimilarity(emb, centroids[c]);
        if (sim > bestSim) {
          bestSim = sim;
          bestCluster = c;
        }
      }
      
      assignments[i] = bestCluster;
    }
    
    const newCentroids: number[][] = [];
    for (let c = 0; c < k; c++) {
      const clusterPatterns = validPatterns.filter((_, i) => assignments[i] === c);
      if (clusterPatterns.length === 0) {
        newCentroids.push(centroids[c]);
        continue;
      }
      
      const embLength = (clusterPatterns[0].embedding as number[]).length;
      const avgEmb = new Array(embLength).fill(0);
      
      for (const p of clusterPatterns) {
        const emb = normalizeEmbedding(p.embedding as number[]);
        for (let i = 0; i < embLength; i++) {
          avgEmb[i] += emb[i];
        }
      }
      
      for (let i = 0; i < embLength; i++) {
        avgEmb[i] /= clusterPatterns.length;
      }
      
      newCentroids.push(normalizeEmbedding(avgEmb));
    }
    
    centroids = newCentroids;
  }
  
  const clusters: PatternCluster[] = [];
  
  for (let c = 0; c < k; c++) {
    const clusterPatterns = validPatterns.filter((_, i) => assignments[i] === c);
    if (clusterPatterns.length < 5) continue;
    
    const wins = clusterPatterns.filter(p => p.forwardWin).length;
    const returns = clusterPatterns.map(p => p.forwardReturn8 || 0);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    
    const cluster: PatternCluster = {
      id: generateClusterId(regime, c),
      regime,
      centroid: centroids[c],
      support: clusterPatterns.length,
      wins,
      winRate: wins / clusterPatterns.length,
      avgReturn,
      maturity: computeMaturity(clusterPatterns.length),
      samples: clusterPatterns.slice(0, 200).map(p => ({
        timestamp: p.timestamp,
        embedding: normalizeEmbedding(p.embedding as number[]),
        forwardReturn8: p.forwardReturn8 || 0,
        won: p.forwardWin || false,
        direction: "HOLD" as "LONG" | "SHORT" | "HOLD",  // Historical patterns default to HOLD
        actualPnL: p.forwardReturn8 || 0,  // Use forward return as P&L
      })),
      // Direction-aware properties (initialize from historical data)
      longWins: 0,
      longTotal: 0,
      longAvgPnL: 0,
      shortWins: 0,
      shortTotal: 0,
      shortAvgPnL: 0,
    };
    
    clusters.push(cluster);
  }
  
  return clusters;
}

export function getMatureClusterCount(): number {
  let count = 0;
  const clusters = Array.from(patternClusters.values());
  for (const cluster of clusters) {
    if (isPatternMature(cluster)) count++;
  }
  return count;
}

export function getPatternClusterStats(): {
  total: number;
  mature: number;
  immature: number;
  byRegime: Record<string, { total: number; mature: number }>;
} {
  const byRegime: Record<string, { total: number; mature: number }> = {
    trend_up: { total: 0, mature: 0 },
    trend_down: { total: 0, mature: 0 },
    chop: { total: 0, mature: 0 },
    shock: { total: 0, mature: 0 },
  };
  
  let total = 0;
  let mature = 0;
  
  const clusters = Array.from(patternClusters.values());
  for (const cluster of clusters) {
    total++;
    byRegime[cluster.regime].total++;
    
    if (isPatternMature(cluster)) {
      mature++;
      byRegime[cluster.regime].mature++;
    }
  }
  
  return {
    total,
    mature,
    immature: total - mature,
    byRegime,
  };
}

export async function loadPatternClustersFromDb(): Promise<void> {
  try {
    const rows = await db.select().from(patternClustersTable);
    
    if (rows.length === 0) {
      console.log("[Pattern Persistence] No saved clusters found in database");
      return;
    }
    
    patternClusters.clear();
    
    for (const row of rows) {
      const centroid = Array.isArray(row.centroid) ? row.centroid as number[] : [];
      
      const cluster: PatternCluster = {
        id: `${row.regime}_${row.clusterId}`,
        regime: row.regime as "trend_up" | "trend_down" | "chop" | "shock",
        centroid,
        support: row.sampleCount || 0,
        wins: Math.round((row.winRate || 0) * (row.sampleCount || 0)),
        winRate: row.winRate || 0,
        avgReturn: row.avgReturn || 0,
        maturity: (row.sampleCount || 0) >= MIN_SAMPLES_PER_PATTERN ? 1 : (row.sampleCount || 0) / MIN_SAMPLES_PER_PATTERN,
        samples: [],
        // Direction-aware properties (initialize from historical data)
        longWins: 0,
        longTotal: 0,
        longAvgPnL: 0,
        shortWins: 0,
        shortTotal: 0,
        shortAvgPnL: 0,
      };
      
      patternClusters.set(cluster.id, cluster);
    }
    
    console.log(`[Pattern Persistence] Loaded ${rows.length} clusters from database`);
    
  } catch (error) {
    console.error("[Pattern Persistence] Error loading clusters:", error);
  }
}

export async function savePatternClustersToDb(): Promise<void> {
  try {
    const now = Date.now();
    const clusters = Array.from(patternClusters.values());
    
    if (clusters.length === 0) {
      return;
    }
    
    for (const cluster of clusters) {
      const parts = cluster.id.split("_");
      const clusterId = parseInt(parts[parts.length - 1]) || 0;
      
      const clusterData = {
        clusterId,
        regime: cluster.regime,
        centroid: cluster.centroid,
        sampleCount: cluster.support,
        winRate: cluster.winRate,
        avgReturn: cluster.avgReturn,
        avgMfe: 0,
        avgMae: 0,
        avgTimeToMfe: 0,
        isMature: isPatternMature(cluster),
        createdTs: now,
        updatedTs: now,
      };
      
      const [existing] = await db.select()
        .from(patternClustersTable)
        .where(eq(patternClustersTable.clusterId, clusterId))
        .limit(1);
      
      if (existing) {
        await db.update(patternClustersTable)
          .set({ 
            sampleCount: clusterData.sampleCount,
            winRate: clusterData.winRate,
            avgReturn: clusterData.avgReturn,
            isMature: clusterData.isMature,
            updatedTs: now,
          })
          .where(eq(patternClustersTable.id, existing.id));
      } else {
        await db.insert(patternClustersTable).values(clusterData);
      }
    }
    
    console.log(`[Pattern Persistence] Saved ${clusters.length} clusters to database`);
    
  } catch (error) {
    console.error("[Pattern Persistence] Error saving clusters:", error);
  }
}

// ============================================================================
// PREDICTION EPISODE LOGGING (P1-1) - Self-Learning Feedback Loop
// ============================================================================

/**
 * Log a prediction episode for later feedback/learning
 * Called whenever a prediction is made
 */
export async function logPredictionEpisode(
  prediction: PatternPrediction,
  matches: PrecisionPatternMatch[],
  embedding: number[],
  currentPrice: number,
  regime: string,
  volatilityBucket: VolatilityBucket
): Promise<number | null> {
  try {
    const episode: InsertPredictionEpisode = {
      timestamp: Date.now(),
      symbol: "BTCUSDT",
      timeframe: "15m",
      embedding,
      matchedPatternIds: matches.map(m => m.patternId),
      matchedSimilarities: matches.map(m => m.similarity),
      action: prediction.direction,
      entryPrice: currentPrice,
      suggestedSL: prediction.suggestedSL,
      suggestedTP1: prediction.suggestedTP1,
      suggestedTP2: prediction.suggestedTP2,
      evLong: prediction.evLong,
      evShort: prediction.evShort,
      pWinLong: prediction.pWinLong,
      pWinShort: prediction.pWinShort,
      uncertainty: prediction.uncertainty,
      confidence: prediction.shouldTrade ? 0.7 : 0.3,
      regime,
      volatilityBucket,
      horizon: PRECISION_MODE_CONFIG.embargoHorizon,
      outcome: "PENDING",
      createdAt: Date.now(),
    };
    
    const [inserted] = await db.insert(predictionEpisodes).values(episode).returning();
    console.log(`[Episode Logger] Logged prediction episode #${inserted.id}: ${prediction.direction} @ ${currentPrice}`);
    return inserted.id;
  } catch (error) {
    console.error("[Episode Logger] Error logging episode:", error);
    return null;
  }
}

/**
 * Daily feedback loop: Label past episodes with actual outcomes
 * Called by scheduler (daily) or can be triggered manually
 */
export async function labelPendingEpisodes(): Promise<{ labeled: number; errors: number }> {
  console.log("[Daily Feedback] Starting episode labeling...");
  
  const horizonMs = PRECISION_MODE_CONFIG.embargoHorizon * PRECISION_MODE_CONFIG.candleIntervalMs;
  const cutoffTime = Date.now() - horizonMs - (60 * 60 * 1000); // Extra 1hr buffer
  
  try {
    // Find pending episodes that are old enough to have outcomes
    const pendingEpisodes = await db.select()
      .from(predictionEpisodes)
      .where(
        and(
          eq(predictionEpisodes.outcome, "PENDING"),
          lt(predictionEpisodes.timestamp, cutoffTime)
        )
      )
      .limit(100);
    
    if (pendingEpisodes.length === 0) {
      console.log("[Daily Feedback] No pending episodes to label");
      return { labeled: 0, errors: 0 };
    }
    
    let labeled = 0;
    let errors = 0;
    
    for (const episode of pendingEpisodes) {
      try {
        // Get forward candles to compute outcome
        // NOTE: This requires candles table access - simplified for now
        const outcome = await computeEpisodeOutcome(episode);
        
        if (outcome) {
          await db.update(predictionEpisodes)
            .set({
              outcome: outcome.outcome,
              outcomeTimestamp: Date.now(),
              actualReturn: outcome.actualReturn,
              actualMAE: outcome.mae,
              actualMFE: outcome.mfe,
              timeToOutcome: outcome.timeToOutcome,
              hitTP: outcome.hitTP,
              hitSL: outcome.hitSL,
              falsePositive: outcome.falsePositive,
            })
            .where(eq(predictionEpisodes.id, episode.id));
          
          // P1-3: If false positive, penalize matched patterns
          if (outcome.falsePositive && episode.matchedPatternIds) {
            const patternIds = episode.matchedPatternIds as number[];
            for (const pid of patternIds.slice(0, 10)) { // Top 10 matches
              recordPatternFailure(pid, episode.regime || "unknown");
            }
            console.log(`[Hard-Negative Mining] Penalized ${Math.min(10, patternIds.length)} patterns for false positive`);
          }
          
          labeled++;
        }
      } catch (err) {
        console.error(`[Daily Feedback] Error labeling episode ${episode.id}:`, err);
        errors++;
      }
    }
    
    console.log(`[Daily Feedback] Labeled ${labeled} episodes (${errors} errors)`);
    return { labeled, errors };
  } catch (error) {
    console.error("[Daily Feedback] Error in labeling loop:", error);
    return { labeled: 0, errors: 1 };
  }
}

interface EpisodeOutcome {
  outcome: "WIN" | "LOSS" | "SCRATCH" | "EXPIRED";
  actualReturn: number;
  mae: number;
  mfe: number;
  timeToOutcome: number;
  hitTP: boolean;
  hitSL: boolean;
  falsePositive: boolean;
}

/**
 * Compute outcome for a single episode by looking at forward candles
 * Uses the episode's entry price and SL/TP to determine outcome
 */
async function computeEpisodeOutcome(episode: PredictionEpisode): Promise<EpisodeOutcome | null> {
  // For now, use a simplified outcome computation
  // In production, this would query forward candles and compute actual MAE/MFE
  
  if (!episode.entryPrice || !episode.action || episode.action === "HOLD") {
    return {
      outcome: "EXPIRED",
      actualReturn: 0,
      mae: 0,
      mfe: 0,
      timeToOutcome: 0,
      hitTP: false,
      hitSL: false,
      falsePositive: false,
    };
  }
  
  // Query forward candles from timestamp
  const forwardCandles = await db.execute(sql`
    SELECT high, low, close, timestamp 
    FROM candles 
    WHERE timestamp > ${episode.timestamp} 
    AND timeframe = '15m'
    ORDER BY timestamp ASC 
    LIMIT ${PRECISION_MODE_CONFIG.embargoHorizon}
  `);
  
  if (!forwardCandles || (forwardCandles as unknown as any[]).length === 0) {
    return null; // Not enough forward data yet
  }
  
  const candles = forwardCandles as unknown as { high: number; low: number; close: number; timestamp: number }[];
  const entryPrice = episode.entryPrice;
  const isLong = episode.action === "LONG";
  
  let maxFavorable = 0;
  let maxAdverse = 0;
  let hitTP = false;
  let hitSL = false;
  let timeToOutcome = candles.length;
  
  const slPct = episode.suggestedSL || 0.02; // Default 2%
  const tpPct = episode.suggestedTP1 || 0.03; // Default 3%
  
  for (let i = 0; i < candles.length; i++) {
    const candle = candles[i];
    
    // Compute excursions
    if (isLong) {
      const favorable = (candle.high - entryPrice) / entryPrice;
      const adverse = (entryPrice - candle.low) / entryPrice;
      maxFavorable = Math.max(maxFavorable, favorable);
      maxAdverse = Math.max(maxAdverse, adverse);
      
      if (adverse >= slPct && !hitSL) {
        hitSL = true;
        timeToOutcome = i + 1;
        if (!hitTP) break;
      }
      if (favorable >= tpPct && !hitTP) {
        hitTP = true;
        timeToOutcome = i + 1;
        if (!hitSL) break;
      }
    } else {
      // SHORT
      const favorable = (entryPrice - candle.low) / entryPrice;
      const adverse = (candle.high - entryPrice) / entryPrice;
      maxFavorable = Math.max(maxFavorable, favorable);
      maxAdverse = Math.max(maxAdverse, adverse);
      
      if (adverse >= slPct && !hitSL) {
        hitSL = true;
        timeToOutcome = i + 1;
        if (!hitTP) break;
      }
      if (favorable >= tpPct && !hitTP) {
        hitTP = true;
        timeToOutcome = i + 1;
        if (!hitSL) break;
      }
    }
  }
  
  // Compute final return (last close vs entry)
  const lastClose = candles[candles.length - 1].close;
  const actualReturn = isLong 
    ? (lastClose - entryPrice) / entryPrice 
    : (entryPrice - lastClose) / entryPrice;
  
  // Determine outcome
  let outcome: "WIN" | "LOSS" | "SCRATCH" | "EXPIRED";
  if (hitTP && !hitSL) {
    outcome = "WIN";
  } else if (hitSL && !hitTP) {
    outcome = "LOSS";
  } else if (hitTP && hitSL) {
    // Both hit - use actual return to determine
    outcome = actualReturn > 0 ? "WIN" : "LOSS";
  } else if (Math.abs(actualReturn) < 0.001) {
    outcome = "SCRATCH";
  } else {
    outcome = actualReturn > 0 ? "WIN" : "LOSS";
  }
  
  // A false positive is when we predicted a trade but it lost
  const falsePositive = outcome === "LOSS" && episode.action !== "HOLD";
  
  return {
    outcome,
    actualReturn,
    mae: maxAdverse,
    mfe: maxFavorable,
    timeToOutcome,
    hitTP,
    hitSL,
    falsePositive,
  };
}

/**
 * Get episode statistics for dashboard display
 */
export async function getEpisodeStats(): Promise<{
  total: number;
  pending: number;
  wins: number;
  losses: number;
  scratches: number;
  winRate: number;
  avgReturn: number;
  avgMAE: number;
  avgMFE: number;
}> {
  try {
    const stats = await db.execute(sql`
      SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN outcome = 'PENDING' THEN 1 ELSE 0 END) as pending,
        SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
        SUM(CASE WHEN outcome = 'SCRATCH' THEN 1 ELSE 0 END) as scratches,
        AVG(actual_return) as avg_return,
        AVG(actual_mae) as avg_mae,
        AVG(actual_mfe) as avg_mfe
      FROM prediction_episodes
      WHERE outcome IS NOT NULL AND outcome != 'PENDING'
    `);
    
    const row = (stats as any[])[0] || {};
    const completed = (Number(row.wins) || 0) + (Number(row.losses) || 0);
    
    return {
      total: Number(row.total) || 0,
      pending: Number(row.pending) || 0,
      wins: Number(row.wins) || 0,
      losses: Number(row.losses) || 0,
      scratches: Number(row.scratches) || 0,
      winRate: completed > 0 ? (Number(row.wins) || 0) / completed : 0,
      avgReturn: Number(row.avg_return) || 0,
      avgMAE: Number(row.avg_mae) || 0,
      avgMFE: Number(row.avg_mfe) || 0,
    };
  } catch (error) {
    console.error("[Episode Stats] Error fetching stats:", error);
    return {
      total: 0, pending: 0, wins: 0, losses: 0, scratches: 0,
      winRate: 0, avgReturn: 0, avgMAE: 0, avgMFE: 0,
    };
  }
}

/**
 * Memory consolidation: Prune low-quality patterns (weekly job)
 */
export async function consolidateMemory(): Promise<{ pruned: number; merged: number }> {
  console.log("[Memory Consolidation] Starting weekly consolidation...");
  
  // Decay all false friend penalties
  decayAllPenalties();
  
  // For now, just log the action - full pruning logic would go here
  // This would:
  // 1. Remove patterns with consistently negative EV
  // 2. Remove old patterns that no longer match current regime
  // 3. Merge near-duplicate patterns into centroids
  
  console.log("[Memory Consolidation] Penalties decayed, consolidation complete");
  return { pruned: 0, merged: 0 };
}

// ============================================================================
// SELF-LEARNING SCHEDULER (P1-3 FIX)
// ============================================================================
let episodeLabelingInterval: NodeJS.Timeout | null = null;

/**
 * Initialize Pattern Memory self-learning system
 * - Loads persisted false friend penalties
 * - Starts episode labeling scheduler (runs every 2 hours)
 */
export function initializeSelfLearning(): void {
  console.log("[Pattern Memory] Initializing self-learning system...");
  
  // Load persisted false friend penalties
  loadFalseFriendPenalties();
  
  // Start episode labeling scheduler (every 2 hours)
  if (!episodeLabelingInterval) {
    const LABELING_INTERVAL_MS = 2 * 60 * 60 * 1000; // 2 hours
    
    episodeLabelingInterval = setInterval(async () => {
      try {
        const result = await labelPendingEpisodes();
        if (result.labeled > 0 || result.errors > 0) {
          console.log(`[Self-Learning Scheduler] Labeled ${result.labeled} episodes (${result.errors} errors)`);
        }
      } catch (err) {
        console.error("[Self-Learning Scheduler] Error during episode labeling:", err);
      }
    }, LABELING_INTERVAL_MS);
    
    console.log(`[Pattern Memory] Episode labeling scheduled every ${LABELING_INTERVAL_MS / (60 * 1000)} minutes`);
    
    // Also run labeling immediately on startup (with a short delay)
    setTimeout(async () => {
      try {
        await labelPendingEpisodes();
      } catch (err) {
        console.error("[Self-Learning] Initial labeling failed:", err);
      }
    }, 30000); // 30 second delay after startup
  }
  
  console.log("[Pattern Memory] Self-learning system initialized");
}

/**
 * Stop the self-learning scheduler (for cleanup)
 */
export function stopSelfLearning(): void {
  if (episodeLabelingInterval) {
    clearInterval(episodeLabelingInterval);
    episodeLabelingInterval = null;
    console.log("[Pattern Memory] Self-learning scheduler stopped");
  }
}

/**
 * Get current false friend penalties summary for diagnostics
 */
export function getFalseFriendStats(): { 
  totalPenalties: number; 
  avgPenalty: number; 
  maxPenalty: number;
  topOffenders: Array<{ patternId: number; penalty: number; failures: number }>;
} {
  const entries = Array.from(falseFriendPenalties.values());
  if (entries.length === 0) {
    return { totalPenalties: 0, avgPenalty: 0, maxPenalty: 0, topOffenders: [] };
  }
  
  const penalties = entries.map(e => getFalseFriendPenalty(e.patternId));
  const avgPenalty = penalties.reduce((a, b) => a + b, 0) / penalties.length;
  const maxPenalty = Math.max(...penalties);
  
  const topOffenders = entries
    .map(e => ({
      patternId: e.patternId,
      penalty: getFalseFriendPenalty(e.patternId),
      failures: e.failureCount,
    }))
    .sort((a, b) => b.penalty - a.penalty)
    .slice(0, 10);
  
  return { totalPenalties: entries.length, avgPenalty, maxPenalty, topOffenders };
}
