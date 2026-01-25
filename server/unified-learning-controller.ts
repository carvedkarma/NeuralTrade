import { db } from "./db";
import { sql } from "drizzle-orm";

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
  unifiedState.strategyLearnerIdx = index;
  unifiedState.strategyLearnerComplete = index >= unifiedState.maxTrainableIndex;
  unifiedState.sharedProgressIndex = getSharedProgressIndex();
  unifiedState.lastUpdateTs = Date.now();
  checkAllComplete();
}

export function updatePatternMemoryProgress(index: number): void {
  unifiedState.patternMemoryIdx = index;
  unifiedState.patternMemoryComplete = index >= unifiedState.maxTrainableIndex;
  unifiedState.sharedProgressIndex = getSharedProgressIndex();
  unifiedState.lastUpdateTs = Date.now();
  checkAllComplete();
}

export function updateGpuTrainerProgress(index: number, exported: boolean = false): void {
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
  };
  
  console.log(`[Unified Controller] Reset all learning progress to index ${START_INDEX}`);
}

export function getUnifiedProgressReport(): {
  overallProgress: number;
  systems: {
    name: string;
    index: number;
    progress: number;
    complete: boolean;
  }[];
  totalCandles: number;
  trainableCandles: number;
  allAligned: boolean;
} {
  const maxIdx = unifiedState.maxTrainableIndex - START_INDEX;
  const calcProgress = (idx: number) => maxIdx > 0 ? Math.min(100, ((idx - START_INDEX) / maxIdx) * 100) : 0;
  
  const systems = [
    {
      name: "Strategy Learner",
      index: unifiedState.strategyLearnerIdx,
      progress: calcProgress(unifiedState.strategyLearnerIdx),
      complete: unifiedState.strategyLearnerComplete,
    },
    {
      name: "Pattern Memory",
      index: unifiedState.patternMemoryIdx,
      progress: calcProgress(unifiedState.patternMemoryIdx),
      complete: unifiedState.patternMemoryComplete,
    },
    {
      name: "GPU Trainer",
      index: unifiedState.gpuTrainerIdx,
      progress: calcProgress(unifiedState.gpuTrainerIdx),
      complete: unifiedState.gpuTrainerExported,
    },
  ];
  
  const allAligned = Math.abs(unifiedState.strategyLearnerIdx - unifiedState.patternMemoryIdx) <= 500;
  const avgProgress = systems.reduce((sum, s) => sum + s.progress, 0) / systems.length;
  
  return {
    overallProgress: avgProgress,
    systems,
    totalCandles: unifiedState.totalCandlesAvailable,
    trainableCandles: unifiedState.maxTrainableIndex - START_INDEX,
    allAligned,
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
