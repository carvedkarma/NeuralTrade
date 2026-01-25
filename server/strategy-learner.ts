import type { Candle } from "@shared/schema";
import { patternClusters, findSimilarPatterns, type PatternCluster, type PatternMatch } from "./pattern-memory";
import type { FeatureVector } from "./feature-engine";
import { classifyRegime, getRegimeRiskParams, type MarketRegime } from "./feature-engine";
import { updateStrategyLearnerProgress, getUnifiedProgressReport } from "./unified-learning-controller";

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
  atrPercentile: number;  // Used by Meta-Labeling for volatility-adjusted gating
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
    maxBars: number = 16,
    fees: number = 0.0008
  ): Promise<ActionLabeledSample[]> {
    if (idx < 20 || idx + maxBars >= candles.length) return [];

    const currentCandle = candles[idx];
    const entryPrice = currentCandle.close;
    
    const features = this.computeStateFeatures(candles, idx);
    const regime = this.detectRegime(candles, idx);
    
    // Compute ATR for dynamic SL/TP (matching paper engine)
    const atr = this.computeATR(candles, idx);
    const atrPercent = (atr / entryPrice) * 100;
    
    // Use shared regime risk params for consistency across all components
    // This ensures training labels match real-time signal generation and paper trading
    const regimeParams = getRegimeRiskParams(regime);
    const stopMultiplier = regimeParams.stopMultiplier;
    const rrRatio = regimeParams.rrRatio;
    
    // Calculate stop and TP distances
    const stopDistance = Math.max(atrPercent * stopMultiplier, 0.3);  // Min 0.3%
    const tpDistance = stopDistance * rrRatio;  // TP based on R:R ratio
    
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
      let holdBars = 0;
      let exitReason = "TIME";  // Default: exit on max bars

      // TRIPLE BARRIER LABELING: Simulate bar-by-bar with proper intrabar timing
      // Research shows this improves accuracy by 20-50% over simple forward-return labeling
      for (let i = 1; i <= maxBars; i++) {
        const futureCandle = candles[idx + i];
        holdBars = i;
        
        // Calculate price levels for barriers
        const tpLevel = action === "LONG" 
          ? entryPrice * (1 + tpDistance / 100)
          : entryPrice * (1 - tpDistance / 100);
        const slLevel = action === "LONG"
          ? entryPrice * (1 - stopDistance / 100)
          : entryPrice * (1 + stopDistance / 100);
        
        // Check if barriers were hit
        const tpHit = action === "LONG" 
          ? futureCandle.high >= tpLevel
          : futureCandle.low <= tpLevel;
        const slHit = action === "LONG"
          ? futureCandle.low <= slLevel
          : futureCandle.high >= slLevel;
        
        // Track MAE and MFE
        const highMove = ((futureCandle.high - entryPrice) / entryPrice) * 100;
        const lowMove = ((futureCandle.low - entryPrice) / entryPrice) * 100;
        const adjustedHighMove = action === "LONG" ? highMove : -lowMove;
        const adjustedLowMove = action === "LONG" ? lowMove : -highMove;
        if (adjustedHighMove > mfe) mfe = adjustedHighMove;
        if (adjustedLowMove < mae) mae = adjustedLowMove;
        
        // CRITICAL: Intrabar timing logic using distance-to-open
        // This determines which barrier was hit FIRST within the bar
        if (tpHit && slHit) {
          // Both barriers hit in same bar - use distance-to-open heuristic
          // Closer extreme to open is assumed to have happened first
          const distToHigh = Math.abs(futureCandle.high - futureCandle.open);
          const distToLow = Math.abs(futureCandle.low - futureCandle.open);
          
          if (action === "LONG") {
            // For LONG: high=TP, low=SL - which extreme is closer to open?
            if (distToHigh < distToLow) {
              // High was closer to open → TP hit first
              exitReason = "TP";
              exitPrice = tpLevel;
            } else {
              // Low was closer to open → SL hit first
              exitReason = "SL";
              exitPrice = slLevel;
            }
          } else {
            // For SHORT: low=TP, high=SL
            if (distToLow < distToHigh) {
              // Low was closer to open → TP hit first
              exitReason = "TP";
              exitPrice = tpLevel;
            } else {
              // High was closer to open → SL hit first
              exitReason = "SL";
              exitPrice = slLevel;
            }
          }
          break;
        } else if (slHit) {
          // Only SL hit
          exitReason = "SL";
          exitPrice = slLevel;
          break;
        } else if (tpHit) {
          // Only TP hit
          exitReason = "TP";
          exitPrice = tpLevel;
          break;
        }
        
        // If neither hit, use close for running P&L
        exitPrice = futureCandle.close;
      }

      const rawPnl = action === "LONG" 
        ? ((exitPrice - entryPrice) / entryPrice) * 100
        : ((entryPrice - exitPrice) / entryPrice) * 100;
      
      const costPercent = fees * 2 * 100;  // Entry + exit fees
      const netPnl = rawPnl - costPercent;
      
      const expectancy = netPnl / Math.max(0.1, Math.abs(mae));

      samples.push({
        timestamp: currentCandle.timestamp,
        stateFeatures: features,
        action,
        pnl: netPnl,
        mae: Math.abs(mae),
        mfe,
        holdBars,
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

  private detectRegime(candles: Candle[], idx: number): MarketRegime {
    // Use shared ATR-percentile regime classifier for consistency across system
    // This ensures training labels match real-time signal generation
    const regimeAnalysis = classifyRegime(candles, idx);
    return regimeAnalysis.regime;
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
    updateStrategyLearnerProgress(endIdx);

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
    // IMPROVED QUANTIZATION: Use 10 features with 100x rounding for finer discrimination
    // Research shows coarse quantization loses important pattern distinctions
    return features.slice(0, 10).map(f => Math.round(f * 100)).join("_");
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
  
  // Get backtest training stats for ML ensemble display
  getBacktestStats(): { 
    longCount: number; 
    shortCount: number; 
    longWins: number; 
    shortWins: number;
    longWinRate: number;
    shortWinRate: number;
    totalTrades: number;
    overallWinRate: number;
  } {
    const longCount = this.longOutcomes.length;
    const shortCount = this.shortOutcomes.length;
    const longWins = this.longOutcomes.filter(o => o.pnl > 0).length;
    const shortWins = this.shortOutcomes.filter(o => o.pnl > 0).length;
    const totalTrades = longCount + shortCount;
    const totalWins = longWins + shortWins;
    
    return {
      longCount,
      shortCount,
      longWins,
      shortWins,
      longWinRate: longCount > 0 ? longWins / longCount : 0,
      shortWinRate: shortCount > 0 ? shortWins / shortCount : 0,
      totalTrades,
      overallWinRate: totalTrades > 0 ? totalWins / totalTrades : 0,
    };
  }

  /**
   * HALF-KELLY POSITION SIZING
   * Research shows Half-Kelly captures ~75% of optimal growth with ~50% less drawdown
   * Formula: Kelly% = (W × R - L) / R, then use 50% of that
   * Cap at 20% maximum position size regardless of Kelly calculation
   */
  getKellyPositionSize(direction: "LONG" | "SHORT"): {
    kellyFraction: number;
    halfKelly: number;
    recommendedSize: number;
    reasoning: string;
  } {
    const outcomes = direction === "LONG" ? this.longOutcomes : this.shortOutcomes;
    const MIN_SAMPLES = 100;  // Need sufficient data for reliable Kelly estimate
    
    if (outcomes.length < MIN_SAMPLES) {
      return {
        kellyFraction: 0,
        halfKelly: 0,
        recommendedSize: 0.02,  // Default 2% when insufficient data
        reasoning: `Insufficient samples (${outcomes.length}/${MIN_SAMPLES}). Using conservative 2% position.`,
      };
    }
    
    // Calculate win rate and average win/loss sizes
    const wins = outcomes.filter(o => o.pnl > 0);
    const losses = outcomes.filter(o => o.pnl <= 0);
    
    const winRate = wins.length / outcomes.length;
    const lossRate = 1 - winRate;
    
    const avgWin = wins.length > 0 
      ? wins.reduce((sum, o) => sum + o.pnl, 0) / wins.length 
      : 0;
    const avgLoss = losses.length > 0 
      ? Math.abs(losses.reduce((sum, o) => sum + o.pnl, 0) / losses.length)
      : 1;  // Prevent division by zero
    
    // Calculate reward-to-risk ratio (R)
    const R = avgLoss > 0 ? avgWin / avgLoss : 0;
    
    // Full Kelly formula: K = (W × R - L) / R
    // where W = win rate, L = loss rate, R = avg win / avg loss
    const kellyFraction = R > 0 ? (winRate * R - lossRate) / R : 0;
    
    // Half-Kelly for reduced volatility
    const halfKelly = Math.max(0, kellyFraction * 0.5);
    
    // Cap at 20% maximum position size
    const MAX_POSITION = 0.20;
    const MIN_POSITION = 0.01;  // Minimum 1% if we're trading
    
    const recommendedSize = Math.min(MAX_POSITION, Math.max(MIN_POSITION, halfKelly));
    
    return {
      kellyFraction,
      halfKelly,
      recommendedSize,
      reasoning: `Win rate: ${(winRate * 100).toFixed(1)}%, Avg Win: ${avgWin.toFixed(2)}%, ` +
                 `Avg Loss: ${avgLoss.toFixed(2)}%, R: ${R.toFixed(2)}. ` +
                 `Full Kelly: ${(kellyFraction * 100).toFixed(1)}%, Half-Kelly: ${(halfKelly * 100).toFixed(1)}%`,
    };
  }

  /**
   * META-LABELING FILTER
   * Secondary model that evaluates "Should I trust this signal?"
   * Based on: regime, volatility, model agreement, recent win rate
   * Returns confidence score (0-1) for position sizing
   */
  getMetaLabelConfidence(
    direction: "LONG" | "SHORT" | "HOLD",
    regime: string,
    mlConfidence: number,
    patternWinRate: number,
    systemsAgree: boolean,
    atrPercentile: number
  ): {
    shouldTrade: boolean;
    metaConfidence: number;
    reasoning: string[];
    vetoes: string[];
  } {
    const reasoning: string[] = [];
    const vetoes: string[] = [];
    let metaScore = 0.5;  // Start neutral
    
    if (direction === "HOLD") {
      return {
        shouldTrade: false,
        metaConfidence: 0.9,  // High confidence in HOLD decision
        reasoning: ["Primary signal is HOLD - no trade evaluation needed"],
        vetoes: [],
      };
    }
    
    // Factor 1: Regime suitability (weight: 25%)
    const trendingRegimes = ["trend_up", "trend_down"];
    const favorableRegime = trendingRegimes.includes(regime);
    const shockRegime = regime === "shock";
    const chopRegime = regime === "chop";
    
    if (favorableRegime) {
      metaScore += 0.15;
      reasoning.push(`Favorable regime: ${regime}`);
    } else if (shockRegime) {
      metaScore -= 0.15;
      vetoes.push(`Shock regime - high volatility risk`);
    } else if (chopRegime) {
      metaScore -= 0.10;
      vetoes.push(`Chop regime - no clear directional edge`);
    }
    
    // Factor 2: ML confidence (weight: 25%)
    if (mlConfidence >= 0.7) {
      metaScore += 0.20;
      reasoning.push(`High ML confidence: ${(mlConfidence * 100).toFixed(0)}%`);
    } else if (mlConfidence >= 0.5) {
      metaScore += 0.10;
    } else {
      metaScore -= 0.10;
      vetoes.push(`Low ML confidence: ${(mlConfidence * 100).toFixed(0)}%`);
    }
    
    // Factor 3: Pattern win rate (weight: 25%)
    if (patternWinRate >= 0.55) {
      metaScore += 0.15;
      reasoning.push(`Strong pattern history: ${(patternWinRate * 100).toFixed(0)}% win rate`);
    } else if (patternWinRate < 0.45) {
      metaScore -= 0.15;
      vetoes.push(`Weak pattern history: ${(patternWinRate * 100).toFixed(0)}% win rate`);
    }
    
    // Factor 4: System agreement (weight: 15%)
    if (systemsAgree) {
      metaScore += 0.10;
      reasoning.push("All systems agree on direction");
    } else {
      metaScore -= 0.10;
      vetoes.push("Systems disagree on direction");
    }
    
    // Factor 5: Volatility regime (weight: 10%)
    if (atrPercentile < 25) {
      // Very low volatility - potential breakout, neutral
      metaScore += 0.05;
    } else if (atrPercentile > 75) {
      // Very high volatility - reduce size
      metaScore -= 0.10;
      vetoes.push(`High volatility: ATR at ${atrPercentile.toFixed(0)}th percentile`);
    }
    
    // Clamp meta confidence
    const metaConfidence = Math.max(0.1, Math.min(0.95, metaScore));
    
    // Threshold for trade execution
    const TRADE_THRESHOLD = 0.55;
    const shouldTrade = metaConfidence >= TRADE_THRESHOLD && vetoes.length <= 2;
    
    if (!shouldTrade) {
      vetoes.push(`Meta-label score ${(metaConfidence * 100).toFixed(0)}% below ${(TRADE_THRESHOLD * 100).toFixed(0)}% threshold`);
    }
    
    return {
      shouldTrade,
      metaConfidence,
      reasoning,
      vetoes,
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
      const patternMatches = await findSimilarPatterns(feature.embedding, 20);  // Uses MIN_SIMILARITY_THRESHOLD (0.75)
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
    
    // Get ATR percentile for Meta-Labeling volatility filter
    const regimeAnalysis = classifyRegime(candles);
    const atrPercentile = regimeAnalysis.atrPercentile;
    
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
      atrPercentile,
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

  // Manual trigger for strategy learner training
  async startManual(): Promise<{ success: boolean; message: string; epochsRun: number }> {
    try {
      const { loadCandlesFromDb, getMultiAssetDataSummary } = await import("./historical-data");
      
      // Check if historical data exists
      const dataSummary = await getMultiAssetDataSummary();
      const btcData = dataSummary.assets.find((a: { symbol: string }) => a.symbol === "BTCUSDT");
      
      if (!btcData || btcData.totalCandles < 1000) {
        return { 
          success: false, 
          message: `Need at least 1,000 candles to train. Currently have ${btcData?.totalCandles || 0}. Download historical data first.`,
          epochsRun: 0
        };
      }
      
      // Load candles from database
      const candles = await loadCandlesFromDb("BTCUSDT", "15m");
      
      if (candles.length < 1000) {
        return { 
          success: false, 
          message: `Not enough candles loaded. Have ${candles.length}, need 1,000+`,
          epochsRun: 0
        };
      }
      
      // Reset training progress to allow fresh training
      this.trainingProgressIdx = 0;
      this.trainingEpochs = 0;
      
      console.log(`[Strategy Learner] Manual training triggered with ${candles.length.toLocaleString()} candles`);
      
      // Run multiple training epochs
      const epochsToRun = 5;
      for (let i = 0; i < epochsToRun; i++) {
        await this.trainOnHistoricalData(candles, 2000);
      }
      
      return { 
        success: true, 
        message: `Training completed: ${this.trainingEpochs} epochs on ${candles.length.toLocaleString()} candles`,
        epochsRun: this.trainingEpochs
      };
    } catch (error) {
      console.error("[Strategy Learner] Manual start error:", error);
      return { success: false, message: "Failed to start training", epochsRun: 0 };
    }
  }

  // Check if training has started
  hasStartedTraining(): boolean {
    return this.trainingEpochs > 0 || this.actionSamples.length > 0;
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
