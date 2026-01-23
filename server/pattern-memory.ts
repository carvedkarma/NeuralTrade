import { db } from "./db";
import { patterns } from "./db/schema";
import { desc, sql } from "drizzle-orm";
import type { FeatureVector } from "./feature-engine";

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
}

export function mapKalmanToRegime(kalmanRegime: string): string {
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
  return prediction === actualOutcome;
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

function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) return Infinity;
  
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.pow(a[i] - b[i], 2);
  }
  return Math.sqrt(sum);
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
}

export async function storePattern(params: StorePatternParams): Promise<void> {
  const { feature, forwardReturn8, forwardReturn16, maxDrawdown, maxRunup, timeToMfe, atrAtEntry, dynamicThreshold } = params;
  
  const regime = mapKalmanToRegime(feature.kalmanRegime);
  const prediction = determinePrediction(regime);
  const actualOutcome = determineActualOutcome(forwardReturn8, dynamicThreshold);
  const won = determineWin(prediction, actualOutcome);
  
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
  });
}

export async function findSimilarPatterns(
  currentEmbedding: number[],
  topK: number = 50,
  minSimilarity: number = 0.7,
  embargoTimestamp?: number
): Promise<PatternMatch[]> {
  const allPatterns = await db.select().from(patterns).orderBy(desc(patterns.timestamp)).limit(10000);
  
  const matches: PatternMatch[] = [];
  const embargoMs = embargoTimestamp || (Date.now() - 8 * 15 * 60 * 1000);
  
  for (const pattern of allPatterns) {
    if (pattern.timestamp > embargoMs) continue;
    
    const embedding = pattern.embedding as number[];
    if (!Array.isArray(embedding)) continue;
    
    const similarity = cosineSimilarity(currentEmbedding, embedding);
    
    if (similarity >= minSimilarity) {
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
      });
    }
  }
  
  matches.sort((a, b) => b.similarity - a.similarity);
  
  return matches.slice(0, topK);
}

export function computePatternStats(matches: PatternMatch[]): PatternStats {
  if (matches.length === 0) {
    return {
      matchCount: 0,
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
    };
  }
  
  const returns8 = matches.map(m => m.forwardReturn8);
  const returns16 = matches.map(m => m.forwardReturn16);
  const drawdowns = matches.map(m => m.maxDrawdown).sort((a, b) => a - b);
  const runups = matches.map(m => m.maxRunup).sort((a, b) => b - a);
  const timesToMfe = matches.map(m => m.timeToMfe);
  const wins = matches.filter(m => m.won).length;
  
  const regimeBreakdown: Record<string, number> = {};
  for (const m of matches) {
    regimeBreakdown[m.regime] = (regimeBreakdown[m.regime] || 0) + 1;
  }
  
  const avgReturn8 = returns8.reduce((a, b) => a + b, 0) / matches.length;
  const avgReturn16 = returns16.reduce((a, b) => a + b, 0) / matches.length;
  
  const variance = returns8.reduce((sum, r) => sum + Math.pow(r - avgReturn8, 2), 0) / matches.length;
  const stdDev = Math.sqrt(variance);
  
  const p70Index = Math.floor(matches.length * 0.7);
  
  return {
    matchCount: matches.length,
    avgReturn8,
    avgReturn16,
    winRate: wins / matches.length,
    avgDrawdown: drawdowns.reduce((a, b) => a + b, 0) / matches.length,
    avgRunup: runups.reduce((a, b) => a + b, 0) / matches.length,
    avgTimeToMfe: timesToMfe.reduce((a, b) => a + b, 0) / matches.length,
    mae70thPercentile: drawdowns[p70Index] || 0,
    mfe70thPercentile: runups[Math.floor(matches.length * 0.3)] || 0,
    bestCase: Math.max(...returns8),
    worstCase: Math.min(...returns8),
    consistency: avgReturn8 === 0 ? 0 : 1 - (stdDev / Math.abs(avgReturn8)),
    regimeBreakdown,
  };
}

export function getPatternConfidence(stats: PatternStats): {
  direction: "LONG" | "SHORT" | "HOLD";
  confidence: number;
  reasoning: string;
} {
  if (stats.matchCount < 10) {
    return {
      direction: "HOLD",
      confidence: 0,
      reasoning: `Insufficient pattern matches (${stats.matchCount}). Need at least 10 similar historical setups.`,
    };
  }
  
  const expectedReturn = stats.avgReturn8 * 100;
  const winRate = stats.winRate;
  
  if (winRate < 0.45 || Math.abs(expectedReturn) < 0.1) {
    return {
      direction: "HOLD",
      confidence: 0.3,
      reasoning: `Weak pattern signal. Win rate: ${(winRate * 100).toFixed(1)}%, Avg return: ${expectedReturn.toFixed(2)}%`,
    };
  }
  
  const direction: "LONG" | "SHORT" = expectedReturn > 0 ? "LONG" : "SHORT";
  const confidence = Math.min(0.9, (winRate * 0.6 + stats.consistency * 0.4));
  
  return {
    direction,
    confidence,
    reasoning: `${stats.matchCount} similar patterns found. Win rate: ${(winRate * 100).toFixed(1)}%, Avg return: ${expectedReturn.toFixed(2)}%, Best: ${(stats.bestCase * 100).toFixed(2)}%, Worst: ${(stats.worstCase * 100).toFixed(2)}%`,
  };
}

export interface StoredPatternStats {
  totalPatterns: number;
  winRate: number;
  regimeBreakdown: Record<string, number>;
  avgThreshold: number;
  recentWinRate: number;
}

export async function getStoredPatternStats(): Promise<StoredPatternStats> {
  try {
    const allPatterns = await db.select().from(patterns).orderBy(desc(patterns.timestamp)).limit(5000);
    
    if (allPatterns.length === 0) {
      return {
        totalPatterns: 0,
        winRate: 0,
        regimeBreakdown: { trend_up: 0, trend_down: 0, chop: 0, shock: 0 },
        avgThreshold: 0.004,
        recentWinRate: 0,
      };
    }
    
    const totalWins = allPatterns.filter(p => p.forwardWin).length;
    const winRate = totalWins / allPatterns.length;
    
    const regimeBreakdown: Record<string, number> = { trend_up: 0, trend_down: 0, chop: 0, shock: 0 };
    let thresholdSum = 0;
    let thresholdCount = 0;
    
    for (const p of allPatterns) {
      const regime = p.regime || "chop";
      if (regime in regimeBreakdown) {
        regimeBreakdown[regime]++;
      } else {
        regimeBreakdown["chop"]++;
      }
      if (p.dynamicThreshold) {
        thresholdSum += p.dynamicThreshold;
        thresholdCount++;
      }
    }
    
    const recentPatterns = allPatterns.slice(0, Math.min(100, allPatterns.length));
    const recentWins = recentPatterns.filter(p => p.forwardWin).length;
    const recentWinRate = recentPatterns.length > 0 ? recentWins / recentPatterns.length : 0;
    
    return {
      totalPatterns: allPatterns.length,
      winRate,
      regimeBreakdown,
      avgThreshold: thresholdCount > 0 ? thresholdSum / thresholdCount : 0.004,
      recentWinRate,
    };
  } catch (error) {
    console.error("Error getting stored pattern stats:", error);
    return {
      totalPatterns: 0,
      winRate: 0,
      regimeBreakdown: { trend_up: 0, trend_down: 0, chop: 0, shock: 0 },
      avgThreshold: 0.004,
      recentWinRate: 0,
    };
  }
}
