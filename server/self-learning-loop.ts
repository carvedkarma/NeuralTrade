import { db } from "./db";
import { 
  candles, labeledSamples, replayBuffer, trainingRuns, learningJobStatus,
  type InsertLabeledSample, type InsertReplayBuffer, type InsertTrainingRun,
  type LabeledSample, type TrainingRun
} from "@shared/schema";
import { eq, desc, lt, sql, and, gte, lte } from "drizzle-orm";
import { getKlines, getMarkPrice } from "./binance";
import { getBTCCandlesBinanceVision, getBTCPriceBinanceVision } from "./binance-vision";

const FIFTEEN_MIN_MS = 15 * 60 * 1000;
const HORIZON_BARS = 16;
const TRAIN_THRESHOLD = 500;
const REPLAY_BUFFER_MAX_SIZE = 100000;
const NEW_SAMPLES_PCT = 0.3;
const GPU_TRAINER_URL = process.env.GPU_TRAINER_URL || "http://localhost:8000";

interface SelfLearningConfig {
  enabled: boolean;
  intervalMs: number;
  horizonBars: number;
  trainThreshold: number;
  replayBufferMaxSize: number;
  gpuTrainerUrl: string;
}

const config: SelfLearningConfig = {
  enabled: false,
  intervalMs: FIFTEEN_MIN_MS,
  horizonBars: HORIZON_BARS,
  trainThreshold: TRAIN_THRESHOLD,
  replayBufferMaxSize: REPLAY_BUFFER_MAX_SIZE,
  gpuTrainerUrl: GPU_TRAINER_URL,
};

let jobInterval: NodeJS.Timeout | null = null;

export function getSelfLearningConfig(): SelfLearningConfig {
  return { ...config };
}

export function setSelfLearningEnabled(enabled: boolean): void {
  config.enabled = enabled;
  if (enabled && !jobInterval) {
    startSelfLearningLoop();
  } else if (!enabled && jobInterval) {
    stopSelfLearningLoop();
  }
}

export function startSelfLearningLoop(): void {
  if (jobInterval) return;
  console.log("[Self-Learning] Starting 15-minute job loop");
  jobInterval = setInterval(runSelfLearningJob, config.intervalMs);
  runSelfLearningJob();
}

export function stopSelfLearningLoop(): void {
  if (jobInterval) {
    clearInterval(jobInterval);
    jobInterval = null;
    console.log("[Self-Learning] Stopped job loop");
  }
}

async function updateJobStatus(
  jobType: string, 
  updates: Partial<{ 
    lastRunTs: number;
    lastSuccessTs: number;
    lastErrorTs: number;
    lastError: string | null;
    runsTotal: number;
    runsSuccess: number;
    runsFailed: number;
    gapBarsMissing: number;
    lastCandleTs: number;
    lastCandleClose: number;
    tickerPrice: number;
    priceGapPct: number;
    pendingSamplesCount: number;
    labeledSamplesCount: number;
    replayBufferSize: number;
    samplesUntilTrain: number;
    deployedModelId: string;
    deployedTs: number;
  }>
): Promise<void> {
  const now = Date.now();
  const existing = await db.select().from(learningJobStatus).where(eq(learningJobStatus.jobType, jobType)).limit(1);
  
  if (existing.length === 0) {
    await db.insert(learningJobStatus).values({
      jobType,
      ...updates,
      updatedTs: now,
    });
  } else {
    await db.update(learningJobStatus)
      .set({ ...updates, updatedTs: now })
      .where(eq(learningJobStatus.jobType, jobType));
  }
}

async function runSelfLearningJob(): Promise<void> {
  if (!config.enabled) return;
  
  const now = Date.now();
  console.log("[Self-Learning] Running 15-minute job at", new Date(now).toISOString());
  
  try {
    await updateJobStatus("main", { lastRunTs: now });
    
    const gapFillResult = await gapFillCandles();
    const labelingResult = await labelMatureSamples();
    const triggerResult = await checkTrainingTrigger();
    
    await updateJobStatus("main", {
      lastSuccessTs: now,
      gapBarsMissing: gapFillResult.gapBars,
      lastCandleTs: gapFillResult.lastCandleTs,
      lastCandleClose: gapFillResult.lastCandleClose,
      tickerPrice: gapFillResult.tickerPrice,
      priceGapPct: gapFillResult.priceGapPct,
      pendingSamplesCount: labelingResult.pendingCount,
      labeledSamplesCount: labelingResult.labeledCount,
      replayBufferSize: await getReplayBufferSize(),
      samplesUntilTrain: config.trainThreshold - labelingResult.labeledCount,
    });
    
    console.log("[Self-Learning] Job completed successfully");
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error("[Self-Learning] Job failed:", errorMsg);
    await updateJobStatus("main", {
      lastErrorTs: now,
      lastError: errorMsg,
    });
  }
}

interface GapFillResult {
  gapBars: number;
  lastCandleTs: number;
  lastCandleClose: number;
  tickerPrice: number;
  priceGapPct: number;
  candlesFilled: number;
}

async function gapFillCandles(): Promise<GapFillResult> {
  // Use Binance Vision (works from restricted regions) instead of Binance Futures API
  let tickerPrice = 0;
  try {
    const visionPrice = await getBTCPriceBinanceVision();
    tickerPrice = visionPrice || 0;
  } catch (e) {
    // Fallback to original method
    const markPriceResult = await getMarkPrice();
    tickerPrice = markPriceResult ? parseFloat(markPriceResult.markPrice) : 0;
  }
  
  const lastCandle = await db.select()
    .from(candles)
    .where(and(
      eq(candles.symbol, "BTCUSDT"),
      eq(candles.timeframe, "15m")
    ))
    .orderBy(desc(candles.timestamp))
    .limit(1);
  
  const lastCandleTs = lastCandle[0]?.timestamp || 0;
  const lastCandleClose = lastCandle[0]?.close || 0;
  
  const now = Date.now();
  const expectedLastCandleTs = Math.floor(now / FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS - FIFTEEN_MIN_MS;
  const gapMs = expectedLastCandleTs - lastCandleTs;
  const gapBars = Math.floor(gapMs / FIFTEEN_MIN_MS);
  
  const priceGapPct = tickerPrice > 0 && lastCandleClose > 0
    ? Math.abs(lastCandleClose - tickerPrice) / tickerPrice * 100
    : 0;
  
  let candlesFilled = 0;
  
  if (gapBars > 0 && gapBars < 1000) {
    console.log(`[Gap Fill] Detected ${gapBars} missing 15m bars, fetching from Binance Vision...`);
    
    // Use Binance Vision API (works from restricted regions)
    let newCandles = await getBTCCandlesBinanceVision("15m", Math.min(gapBars + 10, 500));
    
    // Fallback to original if Vision fails
    if (newCandles.length === 0) {
      console.log(`[Gap Fill] Binance Vision failed, trying original API...`);
      newCandles = await getKlines("BTCUSDT", "15m", Math.min(gapBars + 5, 500));
    }
    
    for (const candle of newCandles) {
      if (candle.timestamp > lastCandleTs) {
        try {
          await db.insert(candles).values({
            symbol: "BTCUSDT",
            timeframe: "15m",
            timestamp: candle.timestamp,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: candle.volume,
          }).onConflictDoNothing();
          candlesFilled++;
        } catch (e) {
        }
      }
    }
    
    if (candlesFilled > 0) {
      console.log(`[Gap Fill] Inserted ${candlesFilled} new candles via Binance Vision`);
    }
  }
  
  if (priceGapPct > 0.2) {
    console.warn(`[Gap Fill] STALE WARNING: Last candle close ${lastCandleClose.toFixed(2)} vs ticker ${tickerPrice.toFixed(2)} (${priceGapPct.toFixed(2)}% gap)`);
  }
  
  const updatedLastCandle = await db.select()
    .from(candles)
    .where(and(
      eq(candles.symbol, "BTCUSDT"),
      eq(candles.timeframe, "15m")
    ))
    .orderBy(desc(candles.timestamp))
    .limit(1);
  
  return {
    gapBars: Math.max(0, gapBars - candlesFilled),
    lastCandleTs: updatedLastCandle[0]?.timestamp || lastCandleTs,
    lastCandleClose: updatedLastCandle[0]?.close || lastCandleClose,
    tickerPrice,
    priceGapPct,
    candlesFilled,
  };
}

interface LabelingResult {
  pendingCount: number;
  labeledCount: number;
  samplesLabeled: number;
}

async function labelMatureSamples(): Promise<LabelingResult> {
  const now = Date.now();
  const horizonMs = config.horizonBars * FIFTEEN_MIN_MS;
  const maturityThreshold = now - horizonMs;
  
  const pendingSamples = await db.select()
    .from(labeledSamples)
    .where(and(
      eq(labeledSamples.status, "pending"),
      lt(labeledSamples.timestamp, maturityThreshold)
    ))
    .limit(100);
  
  let samplesLabeled = 0;
  
  for (const sample of pendingSamples) {
    try {
      const futureCandles = await db.select()
        .from(candles)
        .where(and(
          eq(candles.symbol, sample.symbol || "BTCUSDT"),
          eq(candles.timeframe, sample.timeframe || "15m"),
          gte(candles.timestamp, sample.timestamp),
          lte(candles.timestamp, sample.timestamp + horizonMs)
        ))
        .orderBy(candles.timestamp)
        .limit(config.horizonBars + 1);
      
      if (futureCandles.length >= config.horizonBars) {
        const entryPrice = sample.currentPrice;
        const exitCandle = futureCandles[futureCandles.length - 1];
        const exitPrice = exitCandle.close;
        
        const actualReturn = (exitPrice - entryPrice) / entryPrice;
        
        let mfe = 0;
        let mae = 0;
        for (const fc of futureCandles) {
          const highRet = (fc.high - entryPrice) / entryPrice;
          const lowRet = (fc.low - entryPrice) / entryPrice;
          mfe = Math.max(mfe, highRet);
          mae = Math.min(mae, lowRet);
        }
        
        const direction = actualReturn > 0.001 ? "UP" : actualReturn < -0.001 ? "DOWN" : "FLAT";
        
        await db.update(labeledSamples)
          .set({
            status: "labeled",
            actualReturn,
            mfe,
            mae: Math.abs(mae),
            direction,
            labeledTs: now,
          })
          .where(eq(labeledSamples.id, sample.id));
        
        await addToReplayBuffer({
          sampleId: sample.sampleId,
          timestamp: sample.timestamp,
          features: sample.features as Record<string, unknown>,
          actualReturn,
          mfe,
          mae: Math.abs(mae),
          direction,
          outcome: actualReturn > 0 ? "WIN" : actualReturn < 0 ? "LOSS" : "SCRATCH",
          priority: 1.0 + Math.abs(actualReturn) * 10,
          regime: sample.regime,
          createdTs: now,
        });
        
        samplesLabeled++;
      }
    } catch (e) {
      console.error("[Labeling] Error labeling sample:", e);
    }
  }
  
  const [pendingResult] = await db.select({ count: sql<number>`count(*)` })
    .from(labeledSamples)
    .where(eq(labeledSamples.status, "pending"));
  
  const [labeledResult] = await db.select({ count: sql<number>`count(*)` })
    .from(labeledSamples)
    .where(eq(labeledSamples.status, "labeled"));
  
  if (samplesLabeled > 0) {
    console.log(`[Labeling] Labeled ${samplesLabeled} samples with outcomes`);
  }
  
  return {
    pendingCount: pendingResult?.count || 0,
    labeledCount: labeledResult?.count || 0,
    samplesLabeled,
  };
}

async function addToReplayBuffer(sample: InsertReplayBuffer): Promise<void> {
  const currentSize = await getReplayBufferSize();
  
  if (currentSize >= config.replayBufferMaxSize) {
    await db.delete(replayBuffer)
      .where(eq(replayBuffer.id, sql`(SELECT id FROM replay_buffer ORDER BY priority ASC, created_ts ASC LIMIT 1)`));
  }
  
  await db.insert(replayBuffer).values(sample).onConflictDoNothing();
}

async function getReplayBufferSize(): Promise<number> {
  const [result] = await db.select({ count: sql<number>`count(*)` }).from(replayBuffer);
  return result?.count || 0;
}

interface TrainingTriggerResult {
  triggered: boolean;
  newSamplesCount: number;
  runId?: string;
}

async function checkTrainingTrigger(): Promise<TrainingTriggerResult> {
  const [labeledResult] = await db.select({ count: sql<number>`count(*)` })
    .from(labeledSamples)
    .where(eq(labeledSamples.status, "labeled"));
  
  const newSamplesCount = labeledResult?.count || 0;
  
  if (newSamplesCount >= config.trainThreshold) {
    console.log(`[Training Trigger] ${newSamplesCount} labeled samples >= threshold ${config.trainThreshold}, triggering candidate training`);
    
    const runId = await triggerCandidateTraining(newSamplesCount);
    
    return { triggered: true, newSamplesCount, runId };
  }
  
  return { triggered: false, newSamplesCount };
}

async function triggerCandidateTraining(newSamplesCount: number): Promise<string> {
  const runId = `run_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const now = Date.now();
  
  const currentDeployed = await db.select()
    .from(trainingRuns)
    .where(eq(trainingRuns.isDeployed, true))
    .orderBy(desc(trainingRuns.deployedTs))
    .limit(1);
  
  const warmStartFrom = currentDeployed[0]?.runId || null;
  
  const replayCount = Math.floor(newSamplesCount * (1 - NEW_SAMPLES_PCT) / NEW_SAMPLES_PCT);
  
  await db.insert(trainingRuns).values({
    runId,
    status: "pending",
    modelType: "transformer",
    warmStartFromRunId: warmStartFrom,
    learningRate: 0.0001,
    epochs: 10,
    newSamplesPct: NEW_SAMPLES_PCT,
    replaySamplesPct: 1 - NEW_SAMPLES_PCT,
    newSamplesCount,
    replaySamplesCount: replayCount,
    totalTrainingSamples: newSamplesCount + replayCount,
    createdTs: now,
    updatedTs: now,
  });
  
  try {
    const response = await fetch(`${config.gpuTrainerUrl}/train/candidate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: runId,
        warm_start_from: warmStartFrom,
        new_samples_count: newSamplesCount,
        replay_samples_count: replayCount,
        learning_rate: 0.0001,
        epochs: 10,
      }),
    });
    
    if (response.ok) {
      await db.update(trainingRuns)
        .set({ status: "training", startedTs: now, updatedTs: now })
        .where(eq(trainingRuns.runId, runId));
      console.log(`[Training] Started candidate training run ${runId}`);
    } else {
      const error = await response.text();
      await db.update(trainingRuns)
        .set({ status: "failed", deploymentReason: `GPU trainer error: ${error}`, updatedTs: now })
        .where(eq(trainingRuns.runId, runId));
      console.error(`[Training] Failed to start training: ${error}`);
    }
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    await db.update(trainingRuns)
      .set({ status: "failed", deploymentReason: `Connection error: ${errorMsg}`, updatedTs: now })
      .where(eq(trainingRuns.runId, runId));
    console.error(`[Training] Failed to connect to GPU trainer: ${errorMsg}`);
  }
  
  return runId;
}

export async function getLearningStatus(): Promise<{
  enabled: boolean;
  lastRunTs: number | null;
  lastSuccessTs: number | null;
  lastError: string | null;
  gapBarsMissing: number;
  lastCandleTs: number | null;
  lastCandleClose: number | null;
  tickerPrice: number | null;
  priceGapPct: number | null;
  isStale: boolean;
  pendingSamplesCount: number;
  labeledSamplesCount: number;
  replayBufferSize: number;
  replayBufferMaxSize: number;
  samplesUntilTrain: number;
  trainThreshold: number;
  deployedModelId: string | null;
  deployedTs: number | null;
  recentRuns: TrainingRun[];
}> {
  const status = await db.select()
    .from(learningJobStatus)
    .where(eq(learningJobStatus.jobType, "main"))
    .limit(1);
  
  const recentRuns = await db.select()
    .from(trainingRuns)
    .orderBy(desc(trainingRuns.createdTs))
    .limit(10);
  
  const s = status[0];
  const priceGapPct = s?.priceGapPct || 0;
  
  return {
    enabled: config.enabled,
    lastRunTs: s?.lastRunTs || null,
    lastSuccessTs: s?.lastSuccessTs || null,
    lastError: s?.lastError || null,
    gapBarsMissing: s?.gapBarsMissing || 0,
    lastCandleTs: s?.lastCandleTs || null,
    lastCandleClose: s?.lastCandleClose || null,
    tickerPrice: s?.tickerPrice || null,
    priceGapPct,
    isStale: priceGapPct > 0.2,
    pendingSamplesCount: s?.pendingSamplesCount || 0,
    labeledSamplesCount: s?.labeledSamplesCount || 0,
    replayBufferSize: s?.replayBufferSize || 0,
    replayBufferMaxSize: config.replayBufferMaxSize,
    samplesUntilTrain: s?.samplesUntilTrain || config.trainThreshold,
    trainThreshold: config.trainThreshold,
    deployedModelId: s?.deployedModelId || null,
    deployedTs: s?.deployedTs || null,
    recentRuns,
  };
}

export async function addPendingSample(
  timestamp: number,
  features: any,
  currentPrice: number,
  regime?: string
): Promise<void> {
  const sampleId = `sample_${timestamp}_${Math.random().toString(36).substr(2, 9)}`;
  
  await db.insert(labeledSamples).values({
    sampleId,
    timestamp,
    symbol: "BTCUSDT",
    timeframe: "15m",
    features,
    currentPrice,
    horizonBars: config.horizonBars,
    status: "pending",
    regime,
    createdTs: Date.now(),
  }).onConflictDoNothing();
}

export async function handleTrainingCallback(
  runId: string,
  status: "passed" | "failed",
  metrics?: {
    expectancy?: number;
    sharpe?: number;
    maxDrawdown?: number;
    winRate?: number;
    profitFactor?: number;
    trades?: number;
    stability?: number;
    checkpointPath?: string;
  }
): Promise<{ deployed: boolean; reason: string }> {
  const now = Date.now();
  
  if (status === "failed") {
    await db.update(trainingRuns)
      .set({ status: "failed", completedTs: now, updatedTs: now })
      .where(eq(trainingRuns.runId, runId));
    return { deployed: false, reason: "Training failed" };
  }
  
  const run = await db.select()
    .from(trainingRuns)
    .where(eq(trainingRuns.runId, runId))
    .limit(1);
  
  if (!run[0]) {
    return { deployed: false, reason: "Run not found" };
  }
  
  await db.update(trainingRuns)
    .set({
      status: "evaluating",
      candidateExpectancy: metrics?.expectancy,
      candidateSharpe: metrics?.sharpe,
      candidateMaxDrawdown: metrics?.maxDrawdown,
      candidateWinRate: metrics?.winRate,
      candidateProfitFactor: metrics?.profitFactor,
      candidateTrades: metrics?.trades,
      candidateStability: metrics?.stability,
      checkpointPath: metrics?.checkpointPath,
      completedTs: now,
      updatedTs: now,
    })
    .where(eq(trainingRuns.runId, runId));
  
  const currentDeployed = await db.select()
    .from(trainingRuns)
    .where(eq(trainingRuns.isDeployed, true))
    .orderBy(desc(trainingRuns.deployedTs))
    .limit(1);
  
  const current = currentDeployed[0];
  const candidate = { ...run[0], ...metrics };
  
  const beatsCurrent = evaluateCandidate(candidate, current);
  
  if (beatsCurrent.passes) {
    await db.update(trainingRuns)
      .set({ isDeployed: false })
      .where(eq(trainingRuns.isDeployed, true));
    
    await db.update(trainingRuns)
      .set({
        status: "deployed",
        beatsCurrent: true,
        isDeployed: true,
        deployedTs: now,
        deploymentReason: beatsCurrent.reason,
        rollbackPath: current?.checkpointPath,
        updatedTs: now,
      })
      .where(eq(trainingRuns.runId, runId));
    
    await db.update(labeledSamples)
      .set({ status: "used", usedInRunId: runId })
      .where(eq(labeledSamples.status, "labeled"));
    
    await updateJobStatus("main", { deployedModelId: runId, deployedTs: now });
    
    console.log(`[Training] Deployed new model ${runId}: ${beatsCurrent.reason}`);
    return { deployed: true, reason: beatsCurrent.reason };
  } else {
    await db.update(trainingRuns)
      .set({
        status: "passed",
        beatsCurrent: false,
        deploymentReason: beatsCurrent.reason,
        updatedTs: now,
      })
      .where(eq(trainingRuns.runId, runId));
    
    console.log(`[Training] Candidate ${runId} did not beat current: ${beatsCurrent.reason}`);
    return { deployed: false, reason: beatsCurrent.reason };
  }
}

function evaluateCandidate(
  candidate: any,
  current: TrainingRun | undefined
): { passes: boolean; reason: string } {
  if (!current) {
    return { passes: true, reason: "No current model deployed, accepting candidate" };
  }
  
  const candidateExp = candidate.expectancy || candidate.candidateExpectancy || 0;
  const currentExp = current.currentExpectancy || current.candidateExpectancy || 0;
  
  const candidateSharpe = candidate.sharpe || candidate.candidateSharpe || 0;
  const currentSharpe = current.currentSharpe || current.candidateSharpe || 0;
  
  const candidateDD = candidate.maxDrawdown || candidate.candidateMaxDrawdown || 1;
  const currentDD = current.currentMaxDrawdown || current.candidateMaxDrawdown || 1;
  
  if (candidateExp > currentExp * 1.05 && candidateSharpe >= currentSharpe * 0.95) {
    return { passes: true, reason: `Expectancy improved ${(currentExp*100).toFixed(2)}% -> ${(candidateExp*100).toFixed(2)}%` };
  }
  
  if (candidateSharpe > currentSharpe * 1.1 && candidateDD <= currentDD * 1.1) {
    return { passes: true, reason: `Sharpe improved ${currentSharpe.toFixed(2)} -> ${candidateSharpe.toFixed(2)}` };
  }
  
  if (candidateDD < currentDD * 0.8 && candidateExp >= currentExp * 0.95) {
    return { passes: true, reason: `Drawdown reduced ${(currentDD*100).toFixed(1)}% -> ${(candidateDD*100).toFixed(1)}%` };
  }
  
  return { 
    passes: false, 
    reason: `Did not beat thresholds: Exp ${(candidateExp*100).toFixed(2)}% vs ${(currentExp*100).toFixed(2)}%, Sharpe ${candidateSharpe.toFixed(2)} vs ${currentSharpe.toFixed(2)}`
  };
}

export async function rollbackToPrevious(): Promise<{ success: boolean; reason: string }> {
  const current = await db.select()
    .from(trainingRuns)
    .where(eq(trainingRuns.isDeployed, true))
    .limit(1);
  
  if (!current[0]?.rollbackPath) {
    return { success: false, reason: "No rollback checkpoint available" };
  }
  
  const previous = await db.select()
    .from(trainingRuns)
    .where(eq(trainingRuns.checkpointPath, current[0].rollbackPath))
    .limit(1);
  
  if (!previous[0]) {
    return { success: false, reason: "Previous model not found in database" };
  }
  
  const now = Date.now();
  
  await db.update(trainingRuns)
    .set({ isDeployed: false, status: "rollback", updatedTs: now })
    .where(eq(trainingRuns.runId, current[0].runId));
  
  await db.update(trainingRuns)
    .set({ isDeployed: true, deployedTs: now, updatedTs: now })
    .where(eq(trainingRuns.runId, previous[0].runId));
  
  await updateJobStatus("main", { deployedModelId: previous[0].runId, deployedTs: now });
  
  console.log(`[Training] Rolled back from ${current[0].runId} to ${previous[0].runId}`);
  return { success: true, reason: `Rolled back to ${previous[0].runId}` };
}
