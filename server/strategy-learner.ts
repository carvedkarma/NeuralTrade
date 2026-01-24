import type { Candle } from "@shared/schema";

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

export class StrategyLearner {
  private actionSamples: ActionLabeledSample[] = [];
  private policyWeights: Map<string, number[]> = new Map();
  private expansionModel: { weights: number[]; bias: number } = { weights: [], bias: 0 };
  private trainingEpochs = 0;
  private lastTrainingTime: number | null = null;
  private isTraining = false;
  private modelAccuracy = 0;
  
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
    const startIdx = Math.max(20, this.actionSamples.length);
    const endIdx = Math.min(candles.length - 16, startIdx + batchSize);

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

    this.updatePolicyModel();
    this.updateExpansionModel(candles);
    
    this.trainingEpochs++;
    this.lastTrainingTime = Date.now();
    this.isTraining = false;

    if (this.trainingEpochs % 10 === 0) {
      console.log(`[Strategy Learner] Epoch ${this.trainingEpochs}: ${this.actionSamples.length} samples, ${this.actionPatterns.size} patterns`);
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

    for (const [patternId, data] of this.actionPatterns.entries()) {
      if (data.longTotal < 5 || data.shortTotal < 5) continue;

      const longWinRate = data.longWins / data.longTotal;
      const shortWinRate = data.shortWins / data.shortTotal;
      const longAvgReward = data.longRewards.length > 0 
        ? data.longRewards.reduce((a, b) => a + b, 0) / data.longRewards.length 
        : 0;
      const shortAvgReward = data.shortRewards.length > 0 
        ? data.shortRewards.reduce((a, b) => a + b, 0) / data.shortRewards.length 
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
      totalSamples: this.actionSamples.length,
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
}

export const strategyLearner = new StrategyLearner();
