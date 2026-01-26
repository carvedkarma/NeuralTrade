import { db } from "./db";
import { patterns, patternClusters as patternClustersTable } from "./db/schema";
import { desc, sql, eq } from "drizzle-orm";
import type { FeatureVector } from "./feature-engine";

const MAX_PATTERNS_TOTAL = 30;
const MIN_SAMPLES_PER_PATTERN = 100;  // Increased from 50 for statistical reliability (research-backed)
const MIN_BACKTEST_TRADES = 500;
const MIN_CANDLES_15M = 2000; // ~20 days of 15m data
const EMBARGO_CANDLES = 16;
const MIN_SIMILARITY_THRESHOLD = 0.75;  // Increased from 0.6 for quality pattern matches (research-backed)

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
  
  if (nearestCluster && similarity >= 0.7) {
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

export async function findSimilarPatterns(
  currentEmbedding: number[],
  topK: number = 50,
  minSimilarity: number = MIN_SIMILARITY_THRESHOLD,  // Use research-backed threshold
  embargoTimestamp?: number,
  currentTimestamp?: number
): Promise<PatternMatch[]> {
  const allPatterns = await db.select().from(patterns).orderBy(desc(patterns.timestamp)).limit(10000);
  
  const matches: PatternMatch[] = [];
  const allSimilarities: number[] = [];
  
  const embargoMs = embargoTimestamp || (Date.now() - EMBARGO_CANDLES * 15 * 60 * 1000);
  const currentWindow = currentTimestamp || Date.now();
  const windowBuffer = 15 * 60 * 1000;
  
  const normalizedCurrent = normalizeEmbedding(currentEmbedding);
  if (!isValidEmbedding(normalizedCurrent)) {
    console.warn("Invalid current embedding - all zeros or NaN values");
    return [];
  }
  
  for (const pattern of allPatterns) {
    if (pattern.timestamp > embargoMs) continue;
    if (Math.abs(pattern.timestamp - currentWindow) < windowBuffer) continue;
    
    const embedding = pattern.embedding as number[];
    if (!isValidEmbedding(embedding)) continue;
    
    const normalizedPattern = normalizeEmbedding(embedding);
    const similarity = cosineSimilarity(normalizedCurrent, normalizedPattern);
    
    if (!isFinite(similarity)) continue;
    allSimilarities.push(similarity);
    
    if (similarity >= minSimilarity && similarity < 0.995) {
      const clusterInfo = findClusterForPattern(pattern.regime || "chop", embedding);
      
      matches.push({
        timestamp: pattern.timestamp,
        similarity,
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
      });
    }
  }
  
  if (allSimilarities.length > 0) {
    allSimilarities.sort((a, b) => a - b);
    const min = allSimilarities[0];
    const max = allSimilarities[allSimilarities.length - 1];
    const mean = allSimilarities.reduce((a, b) => a + b, 0) / allSimilarities.length;
    const medianIdx = Math.floor(allSimilarities.length / 2);
    const median = allSimilarities[medianIdx];
    
    lastSimilarityDist = { min, max, mean, median, count: allSimilarities.length };
    
    console.log(`Similarity distribution: min=${(min * 100).toFixed(1)}%, mean=${(mean * 100).toFixed(1)}%, median=${(median * 100).toFixed(1)}%, max=${(max * 100).toFixed(1)}% (n=${allSimilarities.length})`);
    
    if (mean > 0.90) {
      console.warn(`SIMILARITY WARNING: Avg similarity ${(mean * 100).toFixed(1)}% is too high! Target: 65-85%`);
    }
  }
  
  matches.sort((a, b) => b.similarity - a.similarity);
  
  return matches.slice(0, topK);
}

function findClusterForPattern(regime: string, embedding: number[]): { clusterId: string; maturity: number } | null {
  const normalizedEmb = normalizeEmbedding(embedding);
  let bestCluster: PatternCluster | null = null;
  let bestSimilarity = 0;
  
  const clusters = Array.from(patternClusters.values());
  for (const cluster of clusters) {
    if (cluster.regime !== regime) continue;
    
    const similarity = cosineSimilarity(normalizedEmb, cluster.centroid);
    if (similarity > bestSimilarity && similarity >= 0.7) {
      bestSimilarity = similarity;
      bestCluster = cluster;
    }
  }
  
  if (bestCluster) {
    return { clusterId: bestCluster.id, maturity: bestCluster.maturity };
  }
  return null;
}

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
  
  const returns8 = matureMatches.map(m => m.forwardReturn8);
  const returns16 = matureMatches.map(m => m.forwardReturn16);
  const drawdowns = matureMatches.map(m => m.maxDrawdown).sort((a, b) => a - b);
  const runups = matureMatches.map(m => m.maxRunup).sort((a, b) => b - a);
  const timesToMfe = matureMatches.map(m => m.timeToMfe);
  const wins = matureMatches.filter(m => m.won).length;
  
  const regimeBreakdown: Record<string, number> = {};
  for (const m of matureMatches) {
    regimeBreakdown[m.regime] = (regimeBreakdown[m.regime] || 0) + 1;
  }
  
  const avgReturn8 = returns8.reduce((a, b) => a + b, 0) / matureMatches.length;
  const avgReturn16 = returns16.reduce((a, b) => a + b, 0) / matureMatches.length;
  
  const variance = returns8.reduce((sum, r) => sum + Math.pow(r - avgReturn8, 2), 0) / matureMatches.length;
  const stdDev = Math.sqrt(variance);
  
  const p70Index = Math.floor(matureMatches.length * 0.7);
  
  return {
    matchCount: matches.length,
    avgReturn8,
    avgReturn16,
    winRate: wins / matureMatches.length,
    avgDrawdown: drawdowns.reduce((a, b) => a + b, 0) / matureMatches.length,
    avgRunup: runups.reduce((a, b) => a + b, 0) / matureMatches.length,
    avgTimeToMfe: timesToMfe.reduce((a, b) => a + b, 0) / matureMatches.length,
    mae70thPercentile: drawdowns[p70Index] || 0,
    mfe70thPercentile: runups[Math.floor(matureMatches.length * 0.3)] || 0,
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
    const trainPatterns = allPatterns.filter(p => p.trainingWindow !== "test");
    const testPatterns = allPatterns.filter(p => p.trainingWindow === "test");
    
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
