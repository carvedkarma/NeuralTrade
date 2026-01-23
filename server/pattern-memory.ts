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
  won: boolean;
  regime: string;
  label: string;
}

export interface PatternStats {
  matchCount: number;
  avgReturn8: number;
  avgReturn16: number;
  winRate: number;
  avgDrawdown: number;
  bestCase: number;
  worstCase: number;
  consistency: number;
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

export async function storePattern(
  feature: FeatureVector,
  forwardReturn8: number,
  forwardReturn16: number,
  maxDrawdown: number,
  label: "up" | "down" | "chop"
): Promise<void> {
  const won = label === "up" ? forwardReturn8 > 0 : label === "down" ? forwardReturn8 < 0 : false;
  
  await db.insert(patterns).values({
    timestamp: feature.timestamp,
    embedding: feature.embedding,
    featureHash: JSON.stringify(feature.embedding).slice(0, 64),
    forwardReturn8: forwardReturn8,
    forwardReturn16: forwardReturn16,
    forwardMaxDrawdown: maxDrawdown,
    forwardWin: won,
    regime: feature.kalmanRegime,
    label,
  });
}

export async function findSimilarPatterns(
  currentEmbedding: number[],
  topK: number = 50,
  minSimilarity: number = 0.7
): Promise<PatternMatch[]> {
  const allPatterns = await db.select().from(patterns).orderBy(desc(patterns.timestamp)).limit(10000);
  
  const matches: PatternMatch[] = [];
  
  for (const pattern of allPatterns) {
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
        won: pattern.forwardWin || false,
        regime: pattern.regime || "unknown",
        label: pattern.label || "unknown",
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
      bestCase: 0,
      worstCase: 0,
      consistency: 0,
    };
  }
  
  const returns8 = matches.map(m => m.forwardReturn8);
  const returns16 = matches.map(m => m.forwardReturn16);
  const drawdowns = matches.map(m => m.maxDrawdown);
  const wins = matches.filter(m => m.won).length;
  
  const avgReturn8 = returns8.reduce((a, b) => a + b, 0) / matches.length;
  const avgReturn16 = returns16.reduce((a, b) => a + b, 0) / matches.length;
  
  const variance = returns8.reduce((sum, r) => sum + Math.pow(r - avgReturn8, 2), 0) / matches.length;
  const stdDev = Math.sqrt(variance);
  
  return {
    matchCount: matches.length,
    avgReturn8,
    avgReturn16,
    winRate: wins / matches.length,
    avgDrawdown: drawdowns.reduce((a, b) => a + b, 0) / matches.length,
    bestCase: Math.max(...returns8),
    worstCase: Math.min(...returns8),
    consistency: avgReturn8 === 0 ? 0 : 1 - (stdDev / Math.abs(avgReturn8)),
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
