import { db } from "./db";
import { sql } from "drizzle-orm";
import { candles } from "@shared/schema";
import { eq, asc } from "drizzle-orm";

// GPU Trainer multi-timeframe stats (cached)
let gpuTrainerStats = {
  totalCandles: 0,
  byTimeframe: {} as Record<string, number>,
  lastUpdated: 0,
};

export async function refreshGpuTrainerStats(): Promise<void> {
  try {
    const result = await db.execute(sql`
      SELECT timeframe, COUNT(*) as count 
      FROM candles 
      GROUP BY timeframe
    `);
    
    const rows = result as any[];
    gpuTrainerStats.byTimeframe = {};
    gpuTrainerStats.totalCandles = 0;
    
    for (const row of rows) {
      const count = parseInt(row.count);
      gpuTrainerStats.byTimeframe[row.timeframe] = count;
      gpuTrainerStats.totalCandles += count;
    }
    gpuTrainerStats.lastUpdated = Date.now();
    
    console.log(`[Unified Controller] GPU Trainer stats: ${gpuTrainerStats.totalCandles.toLocaleString()} total candles`);
    console.log(`[Unified Controller] By timeframe:`, gpuTrainerStats.byTimeframe);
  } catch (error) {
    console.error("[Unified Controller] Failed to refresh GPU trainer stats:", error);
  }
}

export function getGpuTrainerStats() {
  return { ...gpuTrainerStats };
}

export interface UnifiedLearningState {
  sharedProgressIndex: number;
  totalCandlesAvailable: number;
  forwardLookWindow: number;
  maxTrainableIndex: number;
  isComplete: boolean;
  lastUpdateTs: number;
  
  strategyLearnerIdx: number;
  patternMemoryIdx: number;
  gpuTrainerIdx: number;
  
  strategyLearnerComplete: boolean;
  patternMemoryComplete: boolean;
  gpuTrainerExported: boolean;
  
  strategyLearnerStartedAt: number | null;
  patternMemoryStartedAt: number | null;
  gpuTrainerStartedAt: number | null;
}

let candleTimestamps: number[] = [];

export async function loadCandleTimestamps(): Promise<void> {
  try {
    const rows = await db.select({ timestamp: candles.timestamp })
      .from(candles)
      .where(eq(candles.symbol, "BTCUSDT"))
      .orderBy(asc(candles.timestamp))
      .limit(2000000);
    
    candleTimestamps = rows.map(r => r.timestamp);
    console.log(`[Unified Controller] Loaded ${candleTimestamps.length} candle timestamps for date tracking`);
  } catch (error) {
    console.error("[Unified Controller] Failed to load candle timestamps:", error);
  }
}

export function getIndexTimestamp(index: number): number | null {
  if (index >= 0 && index < candleTimestamps.length) {
    return candleTimestamps[index];
  }
  return null;
}

export function formatTrainingDate(timestamp: number | null): string {
  if (!timestamp) return "Unknown";
  return new Date(timestamp).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric"
  });
}

const FORWARD_LOOK_WINDOW = 16;
const START_INDEX = 50;

let unifiedState: UnifiedLearningState = {
  sharedProgressIndex: START_INDEX,
  totalCandlesAvailable: 0,
  forwardLookWindow: FORWARD_LOOK_WINDOW,
  maxTrainableIndex: 0,
  isComplete: false,
  lastUpdateTs: Date.now(),
  
  strategyLearnerIdx: START_INDEX,
  patternMemoryIdx: START_INDEX,
  gpuTrainerIdx: START_INDEX,
  
  strategyLearnerComplete: false,
  patternMemoryComplete: false,
  gpuTrainerExported: false,
  
  strategyLearnerStartedAt: null,
  patternMemoryStartedAt: null,
  gpuTrainerStartedAt: null,
};

export function getUnifiedLearningState(): UnifiedLearningState {
  return { ...unifiedState };
}

export function initializeUnifiedLearning(totalCandles: number): void {
  unifiedState.totalCandlesAvailable = totalCandles;
  unifiedState.maxTrainableIndex = totalCandles - FORWARD_LOOK_WINDOW;
  
  console.log(`[Unified Controller] Initialized with ${totalCandles} candles, max trainable index: ${unifiedState.maxTrainableIndex}`);
}

export function getSharedProgressIndex(): number {
  return Math.min(
    unifiedState.strategyLearnerIdx,
    unifiedState.patternMemoryIdx,
    unifiedState.gpuTrainerIdx
  );
}

export function updateStrategyLearnerProgress(index: number): void {
  if (unifiedState.strategyLearnerStartedAt === null && index > START_INDEX) {
    unifiedState.strategyLearnerStartedAt = Date.now();
  }
  unifiedState.strategyLearnerIdx = index;
  unifiedState.strategyLearnerComplete = index >= unifiedState.maxTrainableIndex;
  unifiedState.sharedProgressIndex = getSharedProgressIndex();
  unifiedState.lastUpdateTs = Date.now();
  checkAllComplete();
}

export function updatePatternMemoryProgress(index: number): void {
  if (unifiedState.patternMemoryStartedAt === null && index > START_INDEX) {
    unifiedState.patternMemoryStartedAt = Date.now();
  }
  unifiedState.patternMemoryIdx = index;
  unifiedState.patternMemoryComplete = index >= unifiedState.maxTrainableIndex;
  unifiedState.sharedProgressIndex = getSharedProgressIndex();
  unifiedState.lastUpdateTs = Date.now();
  checkAllComplete();
}

export function updateGpuTrainerProgress(index: number, exported: boolean = false): void {
  if (unifiedState.gpuTrainerStartedAt === null && index > START_INDEX) {
    unifiedState.gpuTrainerStartedAt = Date.now();
  }
  unifiedState.gpuTrainerIdx = index;
  unifiedState.gpuTrainerExported = exported;
  unifiedState.sharedProgressIndex = getSharedProgressIndex();
  unifiedState.lastUpdateTs = Date.now();
  checkAllComplete();
}

function checkAllComplete(): void {
  unifiedState.isComplete = 
    unifiedState.strategyLearnerComplete &&
    unifiedState.patternMemoryComplete &&
    unifiedState.gpuTrainerExported;
    
  if (unifiedState.isComplete) {
    console.log(`[Unified Controller] All 3 learning systems completed training on ${unifiedState.totalCandlesAvailable} candles!`);
  }
}

export function resetUnifiedLearning(): void {
  unifiedState = {
    sharedProgressIndex: START_INDEX,
    totalCandlesAvailable: unifiedState.totalCandlesAvailable,
    forwardLookWindow: FORWARD_LOOK_WINDOW,
    maxTrainableIndex: unifiedState.maxTrainableIndex,
    isComplete: false,
    lastUpdateTs: Date.now(),
    
    strategyLearnerIdx: START_INDEX,
    patternMemoryIdx: START_INDEX,
    gpuTrainerIdx: START_INDEX,
    
    strategyLearnerComplete: false,
    patternMemoryComplete: false,
    gpuTrainerExported: false,
    
    strategyLearnerStartedAt: null,
    patternMemoryStartedAt: null,
    gpuTrainerStartedAt: null,
  };
  
  console.log(`[Unified Controller] Reset all learning progress to index ${START_INDEX}`);
}

function calculateETA(currentIdx: number, startedAt: number | null): { etaSeconds: number | null; etaFormatted: string } {
  if (!startedAt || currentIdx <= START_INDEX) {
    return { etaSeconds: null, etaFormatted: "Waiting..." };
  }
  
  const elapsed = Date.now() - startedAt;
  const processed = currentIdx - START_INDEX;
  const remaining = unifiedState.maxTrainableIndex - currentIdx;
  
  if (processed <= 0 || remaining <= 0) {
    return { etaSeconds: null, etaFormatted: remaining <= 0 ? "Complete" : "Calculating..." };
  }
  
  const msPerCandle = elapsed / processed;
  const etaMs = msPerCandle * remaining;
  const etaSeconds = Math.round(etaMs / 1000);
  
  if (etaSeconds < 60) {
    return { etaSeconds, etaFormatted: `${etaSeconds}s` };
  } else if (etaSeconds < 3600) {
    const mins = Math.floor(etaSeconds / 60);
    const secs = etaSeconds % 60;
    return { etaSeconds, etaFormatted: `${mins}m ${secs}s` };
  } else {
    const hours = Math.floor(etaSeconds / 3600);
    const mins = Math.floor((etaSeconds % 3600) / 60);
    return { etaSeconds, etaFormatted: `${hours}h ${mins}m` };
  }
}

export interface SystemTrainingStatus {
  name: string;
  index: number;
  progress: number;
  complete: boolean;
  currentDate: string | null;
  currentTimestamp: number | null;
  eta: { etaSeconds: number | null; etaFormatted: string };
  isActive: boolean;
}

export interface DecisionInfo {
  signal: "LONG" | "SHORT" | "HOLD" | "NO_SIGNAL";
  confidence: number;
  source: string;
  ready: boolean;
}

export interface EnhancedProgressReport {
  overallProgress: number;
  systems: SystemTrainingStatus[];
  totalCandles: number;
  trainableCandles: number;
  allAligned: boolean;
  completedCount: number;
  stagedDecisionReady: boolean;
  stagedDecisionWeight: number;
  
  // GPU Trainer specific stats
  gpuTrainerStats: {
    totalCandles: number;
    byTimeframe: Record<string, number>;
  };
  
  // Two separate decision displays
  combinedLearningDecision: DecisionInfo;
  gpuDecision: DecisionInfo;
}

export function getUnifiedProgressReport(): EnhancedProgressReport {
  const maxIdx = unifiedState.maxTrainableIndex - START_INDEX;
  const calcProgress = (idx: number) => maxIdx > 0 ? Math.min(100, ((idx - START_INDEX) / maxIdx) * 100) : 0;
  
  const slTimestamp = getIndexTimestamp(unifiedState.strategyLearnerIdx);
  const pmTimestamp = getIndexTimestamp(unifiedState.patternMemoryIdx);
  const gpuTimestamp = getIndexTimestamp(unifiedState.gpuTrainerIdx);
  
  const systems: SystemTrainingStatus[] = [
    {
      name: "Strategy Learner",
      index: unifiedState.strategyLearnerIdx,
      progress: calcProgress(unifiedState.strategyLearnerIdx),
      complete: unifiedState.strategyLearnerComplete,
      currentDate: formatTrainingDate(slTimestamp),
      currentTimestamp: slTimestamp,
      eta: calculateETA(unifiedState.strategyLearnerIdx, unifiedState.strategyLearnerStartedAt),
      isActive: unifiedState.strategyLearnerStartedAt !== null && !unifiedState.strategyLearnerComplete,
    },
    {
      name: "Pattern Memory",
      index: unifiedState.patternMemoryIdx,
      progress: calcProgress(unifiedState.patternMemoryIdx),
      complete: unifiedState.patternMemoryComplete,
      currentDate: formatTrainingDate(pmTimestamp),
      currentTimestamp: pmTimestamp,
      eta: calculateETA(unifiedState.patternMemoryIdx, unifiedState.patternMemoryStartedAt),
      isActive: unifiedState.patternMemoryStartedAt !== null && !unifiedState.patternMemoryComplete,
    },
    {
      name: "GPU Trainer",
      index: unifiedState.gpuTrainerIdx,
      progress: calcProgress(unifiedState.gpuTrainerIdx),
      complete: unifiedState.gpuTrainerExported,
      currentDate: formatTrainingDate(gpuTimestamp),
      currentTimestamp: gpuTimestamp,
      eta: calculateETA(unifiedState.gpuTrainerIdx, unifiedState.gpuTrainerStartedAt),
      isActive: unifiedState.gpuTrainerStartedAt !== null && !unifiedState.gpuTrainerExported,
    },
  ];
  
  const allAligned = Math.abs(unifiedState.strategyLearnerIdx - unifiedState.patternMemoryIdx) <= 500;
  const avgProgress = systems.reduce((sum, s) => sum + s.progress, 0) / systems.length;
  
  const completedCount = systems.filter(s => s.complete).length;
  const stagedDecisionReady = completedCount >= 1;
  const stagedDecisionWeight = completedCount / 3;
  
  // Combined Learning Decision (Strategy Learner + Pattern Memory consensus)
  const slComplete = unifiedState.strategyLearnerComplete;
  const pmComplete = unifiedState.patternMemoryComplete;
  const combinedLearningReady = slComplete && pmComplete;
  
  const combinedLearningDecision: DecisionInfo = {
    signal: combinedLearningReady ? "HOLD" : "NO_SIGNAL", // Will be updated by actual prediction system
    confidence: combinedLearningReady ? 0.5 : 0,
    source: "Strategy Learner + Pattern Memory",
    ready: combinedLearningReady,
  };
  
  // GPU Decision (separate GPU Trainer prediction)
  const gpuReady = unifiedState.gpuTrainerExported;
  const gpuDecision: DecisionInfo = {
    signal: gpuReady ? "HOLD" : "NO_SIGNAL", // Will be updated by GPU trainer predictions
    confidence: gpuReady ? 0.5 : 0,
    source: "GPU Neural Network",
    ready: gpuReady,
  };
  
  return {
    overallProgress: avgProgress,
    systems,
    totalCandles: unifiedState.totalCandlesAvailable,
    trainableCandles: unifiedState.maxTrainableIndex - START_INDEX,
    allAligned,
    completedCount,
    stagedDecisionReady,
    stagedDecisionWeight,
    gpuTrainerStats: {
      totalCandles: gpuTrainerStats.totalCandles,
      byTimeframe: gpuTrainerStats.byTimeframe,
    },
    combinedLearningDecision,
    gpuDecision,
  };
}

export function shouldContinueTraining(): boolean {
  return !unifiedState.isComplete && unifiedState.maxTrainableIndex > START_INDEX;
}

export function getTrainingRange(): { startIdx: number; endIdx: number } {
  return {
    startIdx: START_INDEX,
    endIdx: unifiedState.maxTrainableIndex,
  };
}
