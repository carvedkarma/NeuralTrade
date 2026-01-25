import type { Candle } from "@shared/schema";
import { patternClusters, findSimilarPatterns, type PatternCluster, type PatternMatch } from "./pattern-memory";
import type { FeatureVector } from "./feature-engine";

interface ActionOutcome {
  action: "LONG" | "SHORT" | "HOLD";
  pnl: number;
  mae: number;
  mfe: number;
  winRate: number;
  expectancy: number;
  sampleCount: number;
}

interface PolicyPrediction {
  pLongProfitable: number;
  pShortProfitable: number;
  pHoldOptimal: number;
  recommendedAction: "LONG" | "SHORT" | "HOLD";
  confidence: number;
}

interface ExpectedValue {
  longEV: number;
  shortEV: number;
  holdEV: number;
  bestAction: "LONG" | "SHORT" | "HOLD";
  bestEV: number;
  variance: number;
  riskAdjustedScore: number;
}

interface ExpansionForecast {
  probability: number;
  expectedBars: number;
  currentATR: number;
  predictedATR: number;
  signal: "expanding" | "contracting" | "stable";
}

interface ActionPattern {
  patternId: string;
  regime: string;
  longWinRate: number;
  shortWinRate: number;
  longAvgReward: number;
  shortAvgReward: number;
  bestAction: "LONG" | "SHORT" | "HOLD";
  actionAdvantage: number;
  sampleCount: number;
}

interface ActionLabeledSample {
  timestamp: number;
  stateFeatures: number[];
  action: "LONG" | "SHORT" | "HOLD";
  pnl: number;
  mae: number;
  mfe: number;
  holdBars: number;
  costAdjustedExpectancy: number;
  regime: string;
}

interface TrainingProgress {
  totalSamples: number;
  epochsCompleted: number;
  lastTrainingTime: number | null;
  modelAccuracy: number;
  isTraining: boolean;
}

export interface StrategyLearnerData {
  actionOutcomes: ActionOutcome[];
  policyPrediction: PolicyPrediction;
  expectedValue: ExpectedValue;
  expansionForecast: ExpansionForecast;
  actionPatterns: ActionPattern[];
  trainingProgress: TrainingProgress;
  comparisonWithCurrent: {
    currentSignal: string;
    currentConfidence: number;
    learnerSignal: string;
    learnerEV: number;
    agreement: boolean;
  };
}

export interface CombinedIntelligence {
  mlDirection: "LONG" | "SHORT" | "HOLD";
  mlConfidence: number;
  strategyAction: "LONG" | "SHORT" | "HOLD";
  strategyEV: number;
  patternWinRate: number;
  patternSupport: number;
  combinedScore: number;
  systemsAgree: boolean;
  finalSignal: "LONG" | "SHORT" | "HOLD";
  finalConfidence: number;
  reasoning: string[];
  vetoes: string[];
}

export class StrategyLearner {
  private actionSamples: ActionLabeledSample[] = [];
  private policyWeights: Map<string, number[]> = new Map();
  private expansionModel: { weights: number[]; bias: number } = { weights: [], bias: 0 };
  private trainingEpochs = 0;
  private lastTrainingTime: number | null = null;
  private isTraining = false;
  private modelAccuracy = 0;
  private trainingProgressIdx = 0;  // Tracks how far through historical candles we've trained
  
  private longOutcomes: { pnl: number; mae: number; mfe: number }[] = [];
  private shortOutcomes: { pnl: number; mae: number; mfe: number }[] = [];
  private holdOutcomes: { pnl: number }[] = [];
  
  private actionPatterns: Map<string, {
    longWins: number;
    longTotal: number;
    shortWins: number;
    shortTotal: number;
    longRewards: number[];
    shortRewards: number[];
  }> = new Map();

  constructor() {
    console.log("[Strategy Learner] Initialized");
  }

  async simulateActionsForCandle(
    candles: Candle[],
    idx: number,
    lookForward: number = 8,
    fees: number = 0.0008
  ): Promise<ActionLabeledSample[]> {
    if (idx < 20 || idx + lookForward >= candles.length) return [];

    const currentCandle = candles[idx];
    const entryPrice = currentCandle.close;
    
    const features = this.computeStateFeatures(candles, idx);
    const regime = this.detectRegime(candles, idx);
    
    const samples: ActionLabeledSample[] = [];

    for (const action of ["LONG", "SHORT", "HOLD"] as const) {
      if (action === "HOLD") {
        samples.push({
          timestamp: currentCandle.timestamp,
          stateFeatures: features,
          action,
          pnl: 0,
          mae: 0,
          mfe: 0,
          holdBars: 0,
          costAdjustedExpectancy: 0,
          regime,
        });
        continue;
      }

      let mae = 0;
      let mfe = 0;
      let exitPrice = entryPrice;

      for (let i = 1; i <= lookForward; i++) {
        const futureCandle = candles[idx + i];
        const movePercent = ((futureCandle.close - entryPrice) / entryPrice) * 100;
        const adjustedMove = action === "LONG" ? movePercent : -movePercent;

        if (adjustedMove > mfe) mfe = adjustedMove;
        if (adjustedMove < mae) mae = adjustedMove;
        
        exitPrice = futureCandle.close;
      }

      const rawPnl = action === "LONG" 
        ? ((exitPrice - entryPrice) / entryPrice) * 100
        : ((entryPrice - exitPrice) / entryPrice) * 100;
      
      const costPercent = fees * 2 * 100;
      const netPnl = rawPnl - costPercent;
      
      const expectancy = netPnl / Math.max(0.1, Math.abs(mae));

      samples.push({
        timestamp: currentCandle.timestamp,
        stateFeatures: features,
        action,
        pnl: netPnl,
        mae: Math.abs(mae),
        mfe,
        holdBars: lookForward,
        costAdjustedExpectancy: expectancy,
        regime,
      });
    }

    return samples;
  }

  private computeStateFeatures(candles: Candle[], idx: number): number[] {
    const lookback = 14;
    const slice = candles.slice(Math.max(0, idx - lookback), idx + 1);
    
    if (slice.length < 2) return Array(10).fill(0);

    const closes = slice.map(c => c.close);
    const volumes = slice.map(c => c.volume);
    const highs = slice.map(c => c.high);
    const lows = slice.map(c => c.low);

    const returns = closes.slice(1).map((c, i) => (c - closes[i]) / closes[i]);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const volatility = Math.sqrt(returns.reduce((a, r) => a + (r - avgReturn) ** 2, 0) / returns.length);

    const rsi = this.computeRSI(closes);
    
    const atrValues = highs.slice(1).map((h, i) => 
      Math.max(h - lows[i + 1], Math.abs(h - closes[i]), Math.abs(lows[i + 1] - closes[i]))
    );
    const atr = atrValues.reduce((a, b) => a + b, 0) / atrValues.length;

    const sma = closes.reduce((a, b) => a + b, 0) / closes.length;
    const priceVsSma = (closes[closes.length - 1] - sma) / sma;

    const volumeAvg = volumes.reduce((a, b) => a + b, 0) / volumes.length;
    const volumeRatio = volumes[volumes.length - 1] / volumeAvg;

    const momentum = (closes[closes.length - 1] - closes[0]) / closes[0];

    const ema8 = this.computeEMA(closes, 8);
    const ema21 = this.computeEMA(closes, Math.min(21, closes.length));
    const emaCross = (ema8 - ema21) / ema21;

    return [
      rsi / 100,
      volatility * 100,
      atr / closes[closes.length - 1],
      priceVsSma,
      volumeRatio,
      momentum,
      emaCross,
      avgReturn,
      closes[closes.length - 1] > closes[closes.length - 2] ? 1 : 0,
      volumes[volumes.length - 1] > volumeAvg ? 1 : 0,
    ];
  }

  private computeRSI(closes: number[], period: number = 14): number {
    if (closes.length < period + 1) return 50;
    
    let gains = 0, losses = 0;
    for (let i = closes.length - period; i < closes.length; i++) {
      const change = closes[i] - closes[i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  private computeEMA(values: number[], period: number): number {
    if (values.length === 0) return 0;
    const k = 2 / (period + 1);
    let ema = values[0];
    for (let i = 1; i < values.length; i++) {
      ema = values[i] * k + ema * (1 - k);
    }
    return ema;
  }

  private detectRegime(candles: Candle[], idx: number): string {
    const lookback = 20;
    const slice = candles.slice(Math.max(0, idx - lookback), idx + 1);
    if (slice.length < 10) return "chop";

    const closes = slice.map(c => c.close);
    const returns = closes.slice(1).map((c, i) => (c - closes[i]) / closes[i]);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const volatility = Math.sqrt(returns.reduce((a, r) => a + (r - avgReturn) ** 2, 0) / returns.length);

    const highestHigh = Math.max(...slice.map(c => c.high));
    const lowestLow = Math.min(...slice.map(c => c.low));
    const range = (highestHigh - lowestLow) / lowestLow;

    if (avgReturn > 0.002 && closes[closes.length - 1] > closes[0]) return "trend_up";
    if (avgReturn < -0.002 && closes[closes.length - 1] < closes[0]) return "trend_down";
    if (range < 0.02 || volatility < 0.005) return "chop";
    return "chop";
  }

  async trainOnHistoricalData(candles: Candle[], batchSize: number = 500): Promise<void> {
    if (candles.length < 50) return;
    
    this.isTraining = true;
    // Use trainingProgressIdx to continue from where we left off (persisted to DB)
    const startIdx = Math.max(20, this.trainingProgressIdx);
    const endIdx = Math.min(candles.length - 16, startIdx + batchSize);
    
    // If we've already trained past this point, skip
    if (startIdx >= endIdx) {
      this.isTraining = false;
      return;
    }

    for (let i = startIdx; i < endIdx; i++) {
      const samples = await this.simulateActionsForCandle(candles, i);
      
      for (const sample of samples) {
        this.actionSamples.push(sample);
        
        if (sample.action === "LONG") {
          this.longOutcomes.push({ pnl: sample.pnl, mae: sample.mae, mfe: sample.mfe });
        } else if (sample.action === "SHORT") {
          this.shortOutcomes.push({ pnl: sample.pnl, mae: sample.mae, mfe: sample.mfe });
        } else {
          this.holdOutcomes.push({ pnl: 0 });
        }

        const patternKey = `${sample.regime}_${this.quantizeFeatures(sample.stateFeatures)}`;
        if (!this.actionPatterns.has(patternKey)) {
          this.actionPatterns.set(patternKey, {
            longWins: 0, longTotal: 0, shortWins: 0, shortTotal: 0,
            longRewards: [], shortRewards: [],
          });
        }
        
        const pattern = this.actionPatterns.get(patternKey)!;
        if (sample.action === "LONG") {
          pattern.longTotal++;
          if (sample.pnl > 0) pattern.longWins++;
          pattern.longRewards.push(sample.pnl);
        } else if (sample.action === "SHORT") {
          pattern.shortTotal++;
          if (sample.pnl > 0) pattern.shortWins++;
          pattern.shortRewards.push(sample.pnl);
        }
      }
    }
    
    // Update trainingProgressIdx to where we just finished
    this.trainingProgressIdx = endIdx;

    this.updatePolicyModel();
    this.updateExpansionModel(candles);
    
    this.trainingEpochs++;
    this.lastTrainingTime = Date.now();
    this.isTraining = false;

    if (this.trainingEpochs % 5 === 0) {
      console.log(`[Strategy Learner] Epoch ${this.trainingEpochs}: idx=${this.trainingProgressIdx}, ${this.actionPatterns.size} patterns`);
      await this.saveStateToDb();
    }
  }

  private quantizeFeatures(features: number[]): string {
    return features.slice(0, 5).map(f => Math.round(f * 10)).join("_");
  }

  private updatePolicyModel(): void {
    const longWinRate = this.longOutcomes.filter(o => o.pnl > 0).length / Math.max(1, this.longOutcomes.length);
    const shortWinRate = this.shortOutcomes.filter(o => o.pnl > 0).length / Math.max(1, this.shortOutcomes.length);
    
    const longAvgPnl = this.longOutcomes.reduce((a, o) => a + o.pnl, 0) / Math.max(1, this.longOutcomes.length);
    const shortAvgPnl = this.shortOutcomes.reduce((a, o) => a + o.pnl, 0) / Math.max(1, this.shortOutcomes.length);

    this.policyWeights.set("long", [longWinRate, longAvgPnl]);
    this.policyWeights.set("short", [shortWinRate, shortAvgPnl]);
    
    const total = this.longOutcomes.length + this.shortOutcomes.length;
    const correct = this.longOutcomes.filter(o => o.pnl > 0).length + 
                    this.shortOutcomes.filter(o => o.pnl > 0).length;
    this.modelAccuracy = total > 0 ? correct / total : 0;
  }

  private updateExpansionModel(candles: Candle[]): void {
    if (candles.length < 30) return;

    const expansionSamples: { features: number[]; expanded: boolean }[] = [];
    
    for (let i = 20; i < candles.length - 5; i++) {
      const features = this.computeStateFeatures(candles, i);
      
      const currentATR = this.computeATR(candles, i);
      const futureATR = this.computeATR(candles, i + 2);
      const expanded = futureATR > currentATR * 1.15;
      
      expansionSamples.push({ features, expanded });
    }

    if (expansionSamples.length > 100) {
      const expandedCount = expansionSamples.filter(s => s.expanded).length;
      const baseProb = expandedCount / expansionSamples.length;
      this.expansionModel.bias = baseProb;
    }
  }

  private computeATR(candles: Candle[], idx: number, period: number = 14): number {
    const slice = candles.slice(Math.max(0, idx - period), idx + 1);
    if (slice.length < 2) return 0;

    let atrSum = 0;
    for (let i = 1; i < slice.length; i++) {
      const tr = Math.max(
        slice[i].high - slice[i].low,
        Math.abs(slice[i].high - slice[i - 1].close),
        Math.abs(slice[i].low - slice[i - 1].close)
      );
      atrSum += tr;
    }
    return atrSum / (slice.length - 1);
  }

  getPolicyPrediction(candles: Candle[]): PolicyPrediction {
    if (candles.length < 20) {
      return {
        pLongProfitable: 0.33,
        pShortProfitable: 0.33,
        pHoldOptimal: 0.34,
        recommendedAction: "HOLD",
        confidence: 0.34,
      };
    }

    const idx = candles.length - 1;
    const features = this.computeStateFeatures(candles, idx);
    const regime = this.detectRegime(candles, idx);
    const patternKey = `${regime}_${this.quantizeFeatures(features)}`;

    let pLong = 0.33, pShort = 0.33, pHold = 0.34;

    const pattern = this.actionPatterns.get(patternKey);
    if (pattern && pattern.longTotal > 5 && pattern.shortTotal > 5) {
      const longWinRate = pattern.longWins / pattern.longTotal;
      const shortWinRate = pattern.shortWins / pattern.shortTotal;
      const longAvg = pattern.longRewards.reduce((a, b) => a + b, 0) / pattern.longRewards.length;
      const shortAvg = pattern.shortRewards.reduce((a, b) => a + b, 0) / pattern.shortRewards.length;

      pLong = longWinRate * 0.5 + (longAvg > 0 ? 0.3 : 0.1);
      pShort = shortWinRate * 0.5 + (shortAvg > 0 ? 0.3 : 0.1);
      pHold = 1 - pLong - pShort;
      if (pHold < 0) pHold = 0.1;

      const total = pLong + pShort + pHold;
      pLong /= total;
      pShort /= total;
      pHold /= total;
    } else {
      const longStats = this.policyWeights.get("long") || [0.33, 0];
      const shortStats = this.policyWeights.get("short") || [0.33, 0];
      
      pLong = longStats[0];
      pShort = shortStats[0];
      pHold = Math.max(0.1, 1 - pLong - pShort);
      
      const total = pLong + pShort + pHold;
      pLong /= total;
      pShort /= total;
      pHold /= total;
    }

    let recommendedAction: "LONG" | "SHORT" | "HOLD" = "HOLD";
    let confidence = pHold;

    if (pLong > pShort && pLong > pHold) {
      recommendedAction = "LONG";
      confidence = pLong;
    } else if (pShort > pLong && pShort > pHold) {
      recommendedAction = "SHORT";
      confidence = pShort;
    }

    return { pLongProfitable: pLong, pShortProfitable: pShort, pHoldOptimal: pHold, recommendedAction, confidence };
  }

  getExpectedValue(candles: Candle[]): ExpectedValue {
    const policy = this.getPolicyPrediction(candles);

    const longAvgPnl = this.longOutcomes.length > 0 
      ? this.longOutcomes.reduce((a, o) => a + o.pnl, 0) / this.longOutcomes.length 
      : 0;
    const shortAvgPnl = this.shortOutcomes.length > 0 
      ? this.shortOutcomes.reduce((a, o) => a + o.pnl, 0) / this.shortOutcomes.length 
      : 0;

    const longEV = policy.pLongProfitable * longAvgPnl;
    const shortEV = policy.pShortProfitable * shortAvgPnl;
    const holdEV = 0;

    let bestAction: "LONG" | "SHORT" | "HOLD" = "HOLD";
    let bestEV = holdEV;

    if (longEV > bestEV) { bestEV = longEV; bestAction = "LONG"; }
    if (shortEV > bestEV) { bestEV = shortEV; bestAction = "SHORT"; }

    const allPnl = [...this.longOutcomes.map(o => o.pnl), ...this.shortOutcomes.map(o => o.pnl)];
    const avgPnl = allPnl.length > 0 ? allPnl.reduce((a, b) => a + b, 0) / allPnl.length : 0;
    const variance = allPnl.length > 0 
      ? allPnl.reduce((a, p) => a + (p - avgPnl) ** 2, 0) / allPnl.length 
      : 0;

    const riskAdjustedScore = variance > 0 ? bestEV / Math.sqrt(variance) : bestEV;

    return { longEV, shortEV, holdEV, bestAction, bestEV, variance, riskAdjustedScore };
  }

  getExpansionForecast(candles: Candle[]): ExpansionForecast {
    if (candles.length < 20) {
      return { probability: 0.5, expectedBars: 2, currentATR: 0, predictedATR: 0, signal: "stable" };
    }

    const idx = candles.length - 1;
    const currentATR = this.computeATR(candles, idx);
    
    const baseProb = this.expansionModel.bias || 0.3;
    
    const recentATRs = [];
    for (let i = Math.max(0, idx - 5); i <= idx; i++) {
      recentATRs.push(this.computeATR(candles, i));
    }
    const atrTrend = recentATRs.length > 1 
      ? (recentATRs[recentATRs.length - 1] - recentATRs[0]) / recentATRs[0] 
      : 0;

    let probability = baseProb;
    if (atrTrend > 0.1) probability += 0.2;
    if (atrTrend < -0.1) probability -= 0.2;
    probability = Math.max(0.1, Math.min(0.9, probability));

    const predictedATR = currentATR * (1 + (probability - 0.5) * 0.3);

    let signal: "expanding" | "contracting" | "stable" = "stable";
    if (probability > 0.6) signal = "expanding";
    if (probability < 0.4) signal = "contracting";

    return { probability, expectedBars: 2, currentATR, predictedATR, signal };
  }

  getActionOutcomes(): ActionOutcome[] {
    const calcStats = (outcomes: { pnl: number; mae: number; mfe: number }[]) => {
      if (outcomes.length === 0) return { pnl: 0, mae: 0, mfe: 0, winRate: 0, expectancy: 0 };
      const avgPnl = outcomes.reduce((a, o) => a + o.pnl, 0) / outcomes.length;
      const avgMae = outcomes.reduce((a, o) => a + o.mae, 0) / outcomes.length;
      const avgMfe = outcomes.reduce((a, o) => a + o.mfe, 0) / outcomes.length;
      const winRate = outcomes.filter(o => o.pnl > 0).length / outcomes.length;
      const expectancy = avgPnl / Math.max(0.1, avgMae);
      return { pnl: avgPnl, mae: avgMae, mfe: avgMfe, winRate, expectancy };
    };

    const longStats = calcStats(this.longOutcomes);
    const shortStats = calcStats(this.shortOutcomes);

    return [
      { action: "LONG", ...longStats, sampleCount: this.longOutcomes.length },
      { action: "SHORT", ...shortStats, sampleCount: this.shortOutcomes.length },
      { action: "HOLD", pnl: 0, mae: 0, mfe: 0, winRate: 1, expectancy: 0, sampleCount: this.holdOutcomes.length },
    ];
  }

  getActionPatterns(): ActionPattern[] {
    const patterns: ActionPattern[] = [];

    const entries = Array.from(this.actionPatterns.entries());
    for (const [patternId, data] of entries) {
      if (data.longTotal < 5 || data.shortTotal < 5) continue;

      const longWinRate = data.longWins / data.longTotal;
      const shortWinRate = data.shortWins / data.shortTotal;
      const longAvgReward = data.longRewards.length > 0 
        ? data.longRewards.reduce((a: number, b: number) => a + b, 0) / data.longRewards.length 
        : 0;
      const shortAvgReward = data.shortRewards.length > 0 
        ? data.shortRewards.reduce((a: number, b: number) => a + b, 0) / data.shortRewards.length 
        : 0;

      let bestAction: "LONG" | "SHORT" | "HOLD" = "HOLD";
      let actionAdvantage = 0;

      if (longAvgReward > shortAvgReward && longAvgReward > 0) {
        bestAction = "LONG";
        actionAdvantage = longAvgReward - shortAvgReward;
      } else if (shortAvgReward > longAvgReward && shortAvgReward > 0) {
        bestAction = "SHORT";
        actionAdvantage = shortAvgReward - longAvgReward;
      }

      const [regime] = patternId.split("_");
      
      patterns.push({
        patternId,
        regime,
        longWinRate,
        shortWinRate,
        longAvgReward,
        shortAvgReward,
        bestAction,
        actionAdvantage: actionAdvantage / 100,
        sampleCount: data.longTotal + data.shortTotal,
      });
    }

    return patterns.sort((a, b) => b.sampleCount - a.sampleCount).slice(0, 20);
  }

  getTrainingProgress(): TrainingProgress {
    return {
      totalSamples: this.trainingProgressIdx,  // Use persisted index, not in-memory array
      epochsCompleted: this.trainingEpochs,
      lastTrainingTime: this.lastTrainingTime,
      modelAccuracy: this.modelAccuracy,
      isTraining: this.isTraining,
    };
  }

  getData(candles: Candle[], currentSignal: string, currentConfidence: number): StrategyLearnerData {
    const policy = this.getPolicyPrediction(candles);
    const ev = this.getExpectedValue(candles);

    return {
      actionOutcomes: this.getActionOutcomes(),
      policyPrediction: policy,
      expectedValue: ev,
      expansionForecast: this.getExpansionForecast(candles),
      actionPatterns: this.getActionPatterns(),
      trainingProgress: this.getTrainingProgress(),
      comparisonWithCurrent: {
        currentSignal,
        currentConfidence,
        learnerSignal: ev.bestAction,
        learnerEV: ev.bestEV,
        agreement: currentSignal === ev.bestAction,
      },
    };
  }

  async getCombinedIntelligence(
    candles: Candle[],
    feature: FeatureVector,
    mlDirection: "LONG" | "SHORT" | "HOLD",
    mlConfidence: number
  ): Promise<CombinedIntelligence> {
    const ev = this.getExpectedValue(candles);
    const policy = this.getPolicyPrediction(candles);
    const reasoning: string[] = [];
    const vetoes: string[] = [];
    
    let patternWinRate = 0;
    let patternSupport = 0;
    
    try {
      const patternMatches = await findSimilarPatterns(feature.embedding, 20, 0.65);
      if (patternMatches.length > 0) {
        patternSupport = patternMatches.length;
        const wins = patternMatches.filter(p => p.won).length;
        patternWinRate = wins / patternMatches.length;
        
        if (patternWinRate > 0.55) {
          reasoning.push(`Pattern memory: ${(patternWinRate * 100).toFixed(0)}% win rate from ${patternSupport} matches`);
        }
      }
      
      for (const [id, cluster] of Array.from(patternClusters.entries())) {
        if (cluster.maturity > 0.5 && cluster.support > 50) {
          patternSupport += cluster.support;
          const clusterWeight = cluster.support / Math.max(1, patternSupport);
          patternWinRate = patternWinRate * (1 - clusterWeight) + cluster.winRate * clusterWeight;
        }
      }
    } catch (e) {
      console.warn("[Combined Intelligence] Pattern lookup failed:", e);
    }
    
    const strategyAction = ev.bestAction;
    const strategyEV = ev.bestEV;
    
    // CRITICAL FIX: Systems only agree when they give the SAME direction
    // (ML=LONG, Strategy=HOLD) is NOT agreement - it's a caution signal
    const systemsAgree = mlDirection === strategyAction;
    
    const directionMatch = mlDirection === strategyAction;
    
    let combinedScore = 0;
    combinedScore += mlConfidence * 0.35;
    combinedScore += (strategyEV > 0 ? Math.min(strategyEV * 10, 0.3) : strategyEV * 5) * 0.25;
    combinedScore += patternWinRate * 0.25;
    combinedScore += (directionMatch ? 0.15 : 0);
    combinedScore = Math.max(0, Math.min(1, combinedScore));
    
    if (mlDirection !== "HOLD") {
      reasoning.push(`ML predicts ${mlDirection} with ${(mlConfidence * 100).toFixed(0)}% confidence`);
    }
    if (strategyEV > 0) {
      reasoning.push(`Strategy Learner: ${strategyAction} has positive EV (${(strategyEV * 100).toFixed(2)}%)`);
    }
    if (policy.pLongProfitable > 0.5 && mlDirection === "LONG") {
      reasoning.push(`Policy model: ${(policy.pLongProfitable * 100).toFixed(0)}% LONG profitability`);
    }
    if (policy.pShortProfitable > 0.5 && mlDirection === "SHORT") {
      reasoning.push(`Policy model: ${(policy.pShortProfitable * 100).toFixed(0)}% SHORT profitability`);
    }
    
    // CRITICAL FIX: Check EV for the SPECIFIC direction, not overall bestEV
    const directionEV = mlDirection === "LONG" ? ev.longEV : 
                        mlDirection === "SHORT" ? ev.shortEV : 0;
    if (mlDirection !== "HOLD" && directionEV <= 0) {
      vetoes.push(`Strategy Learner: ${mlDirection} has negative EV (${(directionEV * 100).toFixed(2)}%)`);
    }
    if (mlDirection !== "HOLD" && !directionMatch && strategyAction !== "HOLD") {
      vetoes.push(`Systems disagree: ML says ${mlDirection}, Strategy says ${strategyAction}`);
    }
    if (patternWinRate < 0.45 && patternSupport > 10) {
      vetoes.push(`Pattern history: Only ${(patternWinRate * 100).toFixed(0)}% win rate`);
    }
    
    let finalSignal: "LONG" | "SHORT" | "HOLD" = "HOLD";
    let finalConfidence = combinedScore;
    
    // CRITICAL FIX: Use already-computed directionEV for consistency
    const strategyApproves = directionEV > 0;
    
    if (mlDirection !== "HOLD" && strategyApproves && patternWinRate >= 0.45 && vetoes.length === 0) {
      finalSignal = mlDirection;
      finalConfidence = combinedScore * 1.1;
    } else if (mlDirection !== "HOLD" && !strategyApproves) {
      finalSignal = "HOLD";
      finalConfidence = Math.max(0.3, combinedScore * 0.6);
      if (vetoes.length === 0) {
        vetoes.push(`Strategy Learner does not approve ${mlDirection} trade`);
      }
    }
    
    finalConfidence = Math.max(0, Math.min(1, finalConfidence));
    
    return {
      mlDirection,
      mlConfidence,
      strategyAction,
      strategyEV,
      patternWinRate,
      patternSupport,
      combinedScore,
      systemsAgree,
      finalSignal,
      finalConfidence,
      reasoning,
      vetoes,
    };
  }

  async saveStateToDb(): Promise<void> {
    try {
      const { db } = await import("./db");
      const { strategyLearnerState } = await import("./db/schema");
      const { eq } = await import("drizzle-orm");
      
      const longPnls = this.longOutcomes.map(o => o.pnl);
      const shortPnls = this.shortOutcomes.map(o => o.pnl);
      const longStats = this.computeActionStats(longPnls);
      const shortStats = this.computeActionStats(shortPnls);
      
      // Count wins and losses for proper restoration
      const longWins = longPnls.filter(p => p > 0).length;
      const longLosses = longPnls.length - longWins;
      const shortWins = shortPnls.filter(p => p > 0).length;
      const shortLosses = shortPnls.length - shortWins;
      
      // Calculate average win/loss PnL for distribution recreation
      const allPnls = [...longPnls, ...shortPnls];
      const winPnls = allPnls.filter(p => p > 0);
      const lossPnls = allPnls.filter(p => p <= 0);
      const avgWinPnl = winPnls.length > 0 ? winPnls.reduce((a, b) => a + b, 0) / winPnls.length : 0;
      const avgLossPnl = lossPnls.length > 0 ? lossPnls.reduce((a, b) => a + b, 0) / lossPnls.length : 0;
      
      const existingState = await db.select().from(strategyLearnerState).limit(1);
      const now = Date.now();
      
      const stateData = {
        epochsCompleted: this.trainingEpochs,
        totalSamples: this.trainingProgressIdx,  // Use training index, not samples array length
        modelAccuracy: this.modelAccuracy,
        longWinRate: longStats.winRate,
        shortWinRate: shortStats.winRate,
        holdWinRate: 1,
        longExpectancy: longStats.expectancy,
        shortExpectancy: shortStats.expectancy,
        holdExpectancy: 0,
        longPnl: longStats.avgPnl,
        shortPnl: shortStats.avgPnl,
        lastTrainingTs: this.lastTrainingTime,
        trainingProgressIdx: this.trainingProgressIdx,  // Critical: track where we stopped
        // New fields for proper restoration
        longWins,
        longLosses,
        shortWins,
        shortLosses,
        avgWinPnl,
        avgLossPnl,
        updatedTs: now,
      };

      if (existingState.length > 0) {
        await db.update(strategyLearnerState)
          .set(stateData)
          .where(eq(strategyLearnerState.id, existingState[0].id));
      } else {
        await db.insert(strategyLearnerState).values(stateData);
      }
      
      console.log(`[Strategy Learner] State saved: ${this.trainingEpochs} epochs, idx=${this.trainingProgressIdx}, wins L:${longWins} S:${shortWins}`);
    } catch (err) {
      console.error("[Strategy Learner] Failed to save state:", err);
    }
  }

  private computeActionStats(pnls: number[]): { winRate: number; expectancy: number; avgPnl: number } {
    if (pnls.length === 0) return { winRate: 0, expectancy: 0, avgPnl: 0 };
    const wins = pnls.filter(p => p > 0).length;
    const winRate = wins / pnls.length;
    const avgPnl = pnls.reduce((a, b) => a + b, 0) / pnls.length;
    const avgWin = pnls.filter(p => p > 0).reduce((a, b) => a + b, 0) / Math.max(1, wins);
    const avgLoss = pnls.filter(p => p <= 0).reduce((a, b) => a + b, 0) / Math.max(1, pnls.length - wins);
    const expectancy = (winRate * avgWin) - ((1 - winRate) * Math.abs(avgLoss));
    return { winRate, expectancy, avgPnl };
  }

  async loadStateFromDb(): Promise<boolean> {
    try {
      const { db } = await import("./db");
      const { strategyLearnerState } = await import("./db/schema");
      
      const rows = await db.select().from(strategyLearnerState).limit(1);
      if (rows.length === 0) return false;
      
      const state = rows[0];
      this.trainingEpochs = state.epochsCompleted || 0;
      this.modelAccuracy = state.modelAccuracy || 0;
      this.lastTrainingTime = state.lastTrainingTs || null;
      
      // CRITICAL: Restore training progress index so we don't retrain from beginning
      this.trainingProgressIdx = state.trainingProgressIdx || 0;
      
      // Restore win/loss outcomes with proper distribution (not all same PnL)
      const longWins = state.longWins || 0;
      const longLosses = state.longLosses || 0;
      const shortWins = state.shortWins || 0;
      const shortLosses = state.shortLosses || 0;
      const avgWinPnl = state.avgWinPnl || 0.5;  // Default positive PnL for wins
      const avgLossPnl = state.avgLossPnl || -0.5;  // Default negative PnL for losses
      
      // Recreate long outcomes with proper win/loss distribution
      for (let i = 0; i < longWins; i++) {
        // Add some variance to avoid all identical PnL values
        const variance = (Math.random() - 0.5) * 0.2 * avgWinPnl;
        this.longOutcomes.push({ 
          pnl: avgWinPnl + variance, 
          mae: 0.3 + Math.random() * 0.2, 
          mfe: 0.5 + Math.random() * 0.3 
        });
      }
      for (let i = 0; i < longLosses; i++) {
        const variance = (Math.random() - 0.5) * 0.2 * Math.abs(avgLossPnl);
        this.longOutcomes.push({ 
          pnl: avgLossPnl - variance, 
          mae: 0.5 + Math.random() * 0.3, 
          mfe: 0.2 + Math.random() * 0.2 
        });
      }
      
      // Recreate short outcomes with proper win/loss distribution
      for (let i = 0; i < shortWins; i++) {
        const variance = (Math.random() - 0.5) * 0.2 * avgWinPnl;
        this.shortOutcomes.push({ 
          pnl: avgWinPnl + variance, 
          mae: 0.3 + Math.random() * 0.2, 
          mfe: 0.5 + Math.random() * 0.3 
        });
      }
      for (let i = 0; i < shortLosses; i++) {
        const variance = (Math.random() - 0.5) * 0.2 * Math.abs(avgLossPnl);
        this.shortOutcomes.push({ 
          pnl: avgLossPnl - variance, 
          mae: 0.5 + Math.random() * 0.3, 
          mfe: 0.2 + Math.random() * 0.2 
        });
      }
      
      // Recreate hold outcomes (neutral)
      const holdCount = Math.floor((longWins + longLosses + shortWins + shortLosses) / 2);
      for (let i = 0; i < holdCount; i++) {
        this.holdOutcomes.push({ pnl: (Math.random() - 0.5) * 0.1 });  // Small random around 0
      }
      
      console.log(`[Strategy Learner] State restored: ${this.trainingEpochs} epochs, idx=${this.trainingProgressIdx}, L:${longWins}W/${longLosses}L S:${shortWins}W/${shortLosses}L`);
      return true;
    } catch (err) {
      console.error("[Strategy Learner] Failed to load state:", err);
      return false;
    }
  }

  // Reset all learning state and clear database
  reset(): void {
    console.log("[Strategy Learner] Resetting all learning data...");
    
    // Clear in-memory state
    this.actionSamples = [];
    this.policyWeights.clear();
    this.expansionModel = { weights: [], bias: 0 };
    this.trainingEpochs = 0;
    this.lastTrainingTime = null;
    this.isTraining = false;
    this.modelAccuracy = 0;
    this.trainingProgressIdx = 0;  // Reset training progress
    
    this.longOutcomes = [];
    this.shortOutcomes = [];
    this.holdOutcomes = [];
    
    this.actionPatterns.clear();
    
    // Clear pattern clusters in memory
    import("./pattern-memory").then(({ resetPatternClusters }) => {
      resetPatternClusters();
      console.log("[Strategy Learner] Pattern clusters cleared");
    }).catch(err => console.error("[Strategy Learner] Failed to clear pattern clusters:", err));
    
    // Clear database state (fire and forget)
    this.clearDbState().catch(err => console.error("[Strategy Learner] Failed to clear DB state:", err));
    
    console.log("[Strategy Learner] All learning data has been reset");
  }

  private async clearDbState(): Promise<void> {
    try {
      const { db } = await import("./db");
      const { strategyLearnerState } = await import("./db/schema");
      await db.delete(strategyLearnerState);
      console.log("[Strategy Learner] Database state cleared");
    } catch (err) {
      console.error("[Strategy Learner] Failed to clear DB state:", err);
    }
  }
}

export const strategyLearner = new StrategyLearner();
