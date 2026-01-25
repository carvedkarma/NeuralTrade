import type { Candle, FuturesData, Signal } from "@shared/schema";
import type { FeatureVector } from "./feature-engine";
import { classifyRegime, getRegimeRiskParams, type MarketRegime } from "./feature-engine";
import { getEnsemblePrediction, type EnsemblePrediction } from "./ml-predictor";
import { findSimilarPatterns, computePatternStats, type PatternMatch } from "./pattern-memory";
import { getSentimentData, interpretFearGreed } from "./sentiment-api";
import { strategyLearner, type CombinedIntelligence } from "./strategy-learner";

function checkNewsFilter(newsScore: number, fearGreedValue: number): { shouldVeto: boolean; reason: string | null } {
  if (fearGreedValue >= 85 || fearGreedValue <= 10) {
    return { 
      shouldVeto: true, 
      reason: `Extreme sentiment (F&G: ${fearGreedValue}) - macro blackout period` 
    };
  }
  if (Math.abs(newsScore) >= 0.8) {
    return { 
      shouldVeto: true, 
      reason: `Extreme news bias (${(newsScore * 100).toFixed(0)}%) - wait for stabilization` 
    };
  }
  return { shouldVeto: false, reason: null };
}

export interface ExpansionGate {
  impulseCandle: boolean;
  atrExpansion: boolean;
  rangeBreak: boolean;
  confirmed: boolean;
  details: string;
}

export type EdgeBucket = "none" | "weak" | "moderate" | "strong";

export interface ShotPlan {
  signal: "LONG" | "SHORT" | "HOLD";
  confidence: number;
  regime: "trend_up" | "trend_down" | "chop" | "shock" | "quiet" | "ranging";
  strategy: string;
  entryZone: { low: number; high: number } | null;
  stopLoss: number | null;
  takeProfit1: number | null;
  takeProfit2: number | null;
  trailingStop: number | null;
  riskReward: number;
  expectedHoldTime: string;
  estimatedCosts: number;
  edge: number;
  edgeBucket: EdgeBucket;
  edgeMultiple: number;
  probUp: number;
  probDown: number;
  probChop: number;
  expectedMove: number;
  reasons: string[];
  vetoReasons: string[];
  patternMatches: PatternMatch[];
  mlPredictions: EnsemblePrediction;
  expansionGate: ExpansionGate;
  combinedIntelligence?: CombinedIntelligence;
  // Quality score (0-100) combining EV, expansion, regime clarity, maturity
  qualityScore: number;
  qualityBreakdown: {
    evScore: number;           // 40% weight: EV relative to costs (0-1)
    expansionScore: number;    // 30% weight: expansion probability (0-1)
    regimeClarity: number;     // 20% weight: |trend_up - trend_down| (0-1)
    maturityScore: number;     // 10% weight: pattern sample maturity (0-1)
  };
}

const FEES = 0.0004;
const SLIPPAGE = 0.0002;
const FUNDING_RISK = 0.0001;
const EDGE_MULTIPLE_MIN = 1.5;
const MIN_QUALITY_SCORE = 70;  // Minimum quality score required for trades

// Calculate quality score (0-100) based on:
// - 40% EV score (edge relative to costs)
// - 30% expansion score (predicted volatility expansion)
// - 20% regime clarity (|trend_up - trend_down|)
// - 10% maturity score (pattern sample count)
function calculateQualityScore(
  edge: number,
  costs: number,
  expansionGate: ExpansionGate,
  probUp: number,
  probDown: number,
  patternMatches: PatternMatch[]
): { qualityScore: number; breakdown: { evScore: number; expansionScore: number; regimeClarity: number; maturityScore: number } } {
  // EV score: edge / costs, capped at 0-1 (edge needs to be 3x costs for max score)
  const evScore = costs > 0 ? Math.min(1, Math.max(0, (edge / costs) / 3)) : 0;
  
  // Expansion score: how many primary conditions are met (0-1)
  // NOTE: "confirmed" is derived from other conditions, so we don't count it separately
  const primaryExpansionConditions = [
    expansionGate.impulseCandle,
    expansionGate.atrExpansion,
    expansionGate.rangeBreak
  ].filter(Boolean).length;
  const expansionScore = primaryExpansionConditions / 3;
  
  // Regime clarity: |probUp - probDown| (0-1)
  const regimeClarity = Math.abs(probUp - probDown);
  
  // Maturity score: log(samples) / log(target_samples), target = 100
  const sampleCount = patternMatches.length;
  const targetSamples = 100;
  const maturityScore = sampleCount > 0 
    ? Math.min(1, Math.log(sampleCount + 1) / Math.log(targetSamples + 1))
    : 0;
  
  // Weighted combination
  const qualityScore = (
    evScore * 0.40 +
    expansionScore * 0.30 +
    regimeClarity * 0.20 +
    maturityScore * 0.10
  ) * 100;  // Convert to 0-100 scale
  
  return {
    qualityScore,
    breakdown: {
      evScore,
      expansionScore,
      regimeClarity,
      maturityScore,
    }
  };
}

function estimateCosts(holdCandles: number): number {
  const fundingPeriods = Math.ceil(holdCandles / 32);
  return FEES * 2 + SLIPPAGE * 2 + FUNDING_RISK * fundingPeriods;
}

function classifyEdgeBucket(edge: number, costs: number): EdgeBucket {
  if (edge <= 0 || costs <= 0) return "none";
  const multiple = edge / costs;
  if (multiple < 1.0) return "none";
  if (multiple < 1.5) return "weak";
  if (multiple < 2.5) return "moderate";
  return "strong";
}

function checkExpansionGate(
  candles: Candle[],
  atr14: number,
  direction: "LONG" | "SHORT" | "HOLD"
): ExpansionGate {
  if (candles.length < 20 || direction === "HOLD") {
    return {
      impulseCandle: false,
      atrExpansion: false,
      rangeBreak: false,
      confirmed: false,
      details: "Insufficient data or HOLD signal"
    };
  }

  const lastCandle = candles[candles.length - 1];
  
  const impulseThreshold = 0.6 * atr14;
  const candleBody = Math.abs(lastCandle.close - lastCandle.open);
  const impulseCandle = candleBody >= impulseThreshold;
  
  const currentRange = lastCandle.high - lastCandle.low;
  const atrExpansion = currentRange >= atr14 * 1.15;
  
  const rangeLookback = candles.slice(-12, -1);
  let rangeBreak = false;
  
  if (direction === "SHORT") {
    const rangeLow = Math.min(...rangeLookback.map(c => c.low));
    rangeBreak = lastCandle.close < rangeLow;
  } else if (direction === "LONG") {
    const rangeHigh = Math.max(...rangeLookback.map(c => c.high));
    rangeBreak = lastCandle.close > rangeHigh;
  }
  
  const volumeLookback = candles.slice(-20, -1);
  const volumes = volumeLookback.map(c => c.volume).sort((a, b) => a - b);
  const medianVolume = volumes[Math.floor(volumes.length / 2)];
  const volumeConfirms = lastCandle.volume > medianVolume;
  
  const rangeBreakWithVolume = rangeBreak && volumeConfirms;
  
  const confirmed = impulseCandle || atrExpansion || rangeBreakWithVolume;
  
  const details: string[] = [];
  if (impulseCandle) details.push(`Impulse: body ${candleBody.toFixed(0)} >= 0.6*ATR(${impulseThreshold.toFixed(0)})`);
  if (atrExpansion) details.push(`ATR exp: range ${currentRange.toFixed(0)} >= 1.15*ATR(${(atr14 * 1.15).toFixed(0)})`);
  if (rangeBreakWithVolume) details.push(`Range break + vol > median`);
  if (!confirmed) details.push("Waiting for expansion confirmation");

  return {
    impulseCandle,
    atrExpansion,
    rangeBreak: rangeBreakWithVolume,
    confirmed,
    details: details.join("; ")
  };
}

function calculateRiskReward(entry: number, stop: number, tp: number): number {
  const risk = Math.abs(entry - stop);
  const reward = Math.abs(tp - entry);
  return risk === 0 ? 0 : reward / risk;
}

// Note: getRegime is now replaced by classifyRegime from feature-engine
// This wrapper function allows signal-engine to use classifyRegime with candle data

function selectStrategy(regime: string, feature: FeatureVector): string {
  if (regime === "shock") {
    return "Shock Recovery";
  }
  if (regime === "chop") {
    return "Mean Reversion";
  }
  if (regime === "quiet") {
    return "Breakout Anticipation";
  }
  if (regime === "ranging") {
    return "Range Trading";
  }
  if (feature.efficiencyRatio > 0.6) {
    return "Kalman Trend Follow";
  }
  if (Math.abs(feature.kalmanSpread) < feature.atr14 * 0.3) {
    return "Kalman Retest";
  }
  return "Kalman Trend";
}

function getEstimatedHoldTime(regime: string, feature: FeatureVector): string {
  if (regime === "shock") return "1-2 candles (15-30min)";
  if (regime === "chop") return "2-4 candles (30min-1h)";
  if (regime === "quiet") return "6-12 candles (1.5-3h) - await breakout";
  if (regime === "ranging") return "3-6 candles (45min-1.5h)";
  if (feature.volatilityRegime === "high") return "2-4 candles (30min-1h)";
  if (feature.adx > 40) return "4-8 candles (1-2h)";
  return "4-6 candles (1-1.5h)";
}

export async function generateShotPlan(
  candles: Candle[],
  feature: FeatureVector,
  futuresData: FuturesData,
  includeAI: boolean = true
): Promise<ShotPlan> {
  const currentPrice = candles[candles.length - 1].close;
  
  // Use shared ATR-percentile regime classifier for consistency across system
  const regimeAnalysis = classifyRegime(candles);
  const regime = regimeAnalysis.regime;
  const strategy = selectStrategy(regime, feature);
  const holdTimeStr = getEstimatedHoldTime(regime, feature);
  
  const [ensemble, sentiment] = await Promise.all([
    getEnsemblePrediction(candles, feature, futuresData, includeAI),
    getSentimentData()
  ]);
  
  const patternMatches = await findSimilarPatterns(feature.embedding, 20);  // Uses MIN_SIMILARITY_THRESHOLD (0.75)
  const patternStats = computePatternStats(patternMatches);
  
  const reasons: string[] = [];
  const vetoReasons: string[] = [];
  
  const isChopRegime = regime === "chop";
  
  if (isChopRegime) {
    vetoReasons.push("CHOP GATE: Kalman regime is chop - no trade allowed");
  }
  
  const holdCandles = regime === "shock" ? 2 : regime === "chop" ? 3 : 6;
  const costs = estimateCosts(holdCandles);
  
  const fearGreedValue = sentiment.fearGreed?.value ?? 50;
  const newsFilter = checkNewsFilter(sentiment.newsScore, fearGreedValue);
  if (newsFilter.shouldVeto && newsFilter.reason) {
    vetoReasons.push(newsFilter.reason);
  }
  
  if (ensemble.probChop > 0.55 && !isChopRegime) {
    vetoReasons.push(`High chop probability: ${(ensemble.probChop * 100).toFixed(1)}%`);
  }
  if (Math.abs(ensemble.expectedMove) < costs * 3) {
    vetoReasons.push(`Edge too small: Expected ${(ensemble.expectedMove / currentPrice * 100).toFixed(2)}% vs costs ${(costs * 100).toFixed(2)}%`);
  }
  if (feature.volatilityRegime === "high" && regime === "chop") {
    vetoReasons.push("High volatility in choppy regime - dangerous");
  }
  if (ensemble.consensus < 0.5) {
    vetoReasons.push(`Low model consensus: ${(ensemble.consensus * 100).toFixed(0)}%`);
  }
  if (ensemble.confidence < 0.65) {
    vetoReasons.push(`Low confidence: ${(ensemble.confidence * 100).toFixed(0)}% (need 65%+)`);
  }
  if (patternMatches.length < 10) {
    vetoReasons.push(`Insufficient pattern matches: ${patternMatches.length} (need 10+)`);
  }
  
  if (ensemble.direction === "LONG") {
    if (feature.rsi14 < 40) reasons.push("RSI oversold, bounce potential");
    if (feature.kalmanRegime === "bull") reasons.push("Kalman filter bullish regime");
    if (feature.macdHist > 0) reasons.push("MACD histogram positive");
    if (feature.stochK < 30) reasons.push("Stochastic oversold");
    if (patternStats.winRate > 0.55) reasons.push(`Pattern history: ${(patternStats.winRate * 100).toFixed(0)}% win rate`);
  } else if (ensemble.direction === "SHORT") {
    if (feature.rsi14 > 60) reasons.push("RSI overbought, pullback potential");
    if (feature.kalmanRegime === "bear") reasons.push("Kalman filter bearish regime");
    if (feature.macdHist < 0) reasons.push("MACD histogram negative");
    if (feature.stochK > 70) reasons.push("Stochastic overbought");
    if (patternStats.winRate > 0.55) reasons.push(`Pattern history: ${(patternStats.winRate * 100).toFixed(0)}% win rate`);
  }
  
  // SELECTIVE MODE: Require high confidence for trades
  // Research shows 65%+ confidence threshold reduces false positives significantly
  const baseShouldTrade = ensemble.confidence >= 0.65 &&  // Restored proper threshold
                       ensemble.direction !== "HOLD";     // Must have directional signal
  
  let combinedIntelligence: CombinedIntelligence | undefined;
  let shouldTrade = baseShouldTrade;
  
  try {
    combinedIntelligence = await strategyLearner.getCombinedIntelligence(
      candles,
      feature,
      ensemble.direction,
      ensemble.confidence
    );
    
    if (combinedIntelligence.vetoes.length > 0) {
      combinedIntelligence.vetoes.forEach(v => {
        if (!vetoReasons.includes(v)) vetoReasons.push(v);
      });
    }
    
    combinedIntelligence.reasoning.forEach(r => {
      if (!reasons.includes(r)) reasons.push(r);
    });
    
    // SELECTIVE MODE: Respect combined intelligence HOLD signals
    // If the strategy learner says HOLD, we should listen
    if (baseShouldTrade && combinedIntelligence.finalSignal === "HOLD") {
      shouldTrade = false;
      vetoReasons.push("Combined Intelligence recommends HOLD");
    }
    
    if (baseShouldTrade && combinedIntelligence.strategyEV > 0) {
      reasons.push(`Strategy EV: +${(combinedIntelligence.strategyEV * 100).toFixed(2)}%`);
    } else if (baseShouldTrade && combinedIntelligence.strategyEV <= 0) {
      // Negative EV is a strong signal to not trade
      shouldTrade = false;
      vetoReasons.push(`Negative Strategy EV: ${(combinedIntelligence.strategyEV * 100).toFixed(2)}%`);
    }
  } catch (e) {
    console.warn("[Signal Engine] Combined intelligence failed:", e);
  }
  
  let entryZone: { low: number; high: number } | null = null;
  let stopLoss: number | null = null;
  let takeProfit1: number | null = null;
  let takeProfit2: number | null = null;
  let trailingStop: number | null = null;
  
  // AGGRESSIVE MODE: Always generate trade levels when there's a directional signal
  // This allows paper trading to act on signals even with veto reasons
  if (ensemble.direction !== "HOLD") {
    const atr = feature.atr14;
    
    if (ensemble.direction === "LONG") {
      entryZone = {
        low: currentPrice - atr * 0.25,
        high: currentPrice + atr * 0.1,
      };
      stopLoss = currentPrice - atr * 1.5;
      takeProfit1 = currentPrice + atr * 1.5;
      takeProfit2 = currentPrice + atr * 3;
      trailingStop = feature.kalmanFast;
    } else {
      entryZone = {
        low: currentPrice - atr * 0.1,
        high: currentPrice + atr * 0.25,
      };
      stopLoss = currentPrice + atr * 1.5;
      takeProfit1 = currentPrice - atr * 1.5;
      takeProfit2 = currentPrice - atr * 3;
      trailingStop = feature.kalmanFast;
    }
  }
  
  const riskReward = entryZone && stopLoss && takeProfit1 
    ? calculateRiskReward(currentPrice, stopLoss, takeProfit1) 
    : 0;
  
  const edge = Math.abs(ensemble.expectedMove) / currentPrice - costs;
  const edgeBucket = classifyEdgeBucket(edge, costs);
  const edgeMultiple = costs > 0 ? edge / costs : 0;
  
  const expansionGate = checkExpansionGate(candles, feature.atr14, ensemble.direction);
  
  // Calculate quality score for trade setup
  const qualityResult = calculateQualityScore(
    edge,
    costs,
    expansionGate,
    ensemble.probUp,
    ensemble.probDown,
    patternMatches
  );
  
  // SELECTIVE MODE: Enforce quality gate - only take high-quality setups
  const MIN_QUALITY_SCORE = 0.5;  // Minimum quality threshold
  if (shouldTrade && qualityResult.qualityScore < MIN_QUALITY_SCORE) {
    shouldTrade = false;
    vetoReasons.push(`Quality score ${qualityResult.qualityScore.toFixed(2)} < ${MIN_QUALITY_SCORE} minimum`);
  }
  
  // SELECTIVE MODE: Adjust confidence based on quality
  const displayConfidence = shouldTrade ? ensemble.confidence : Math.min(ensemble.confidence, 0.4);
  
  // SELECTIVE MODE: Respect shouldTrade flag - output HOLD when gates block
  const finalSignal = shouldTrade ? ensemble.direction : "HOLD";
  
  return {
    signal: finalSignal,
    confidence: displayConfidence,
    regime,
    strategy,
    entryZone,
    stopLoss,
    takeProfit1,
    takeProfit2,
    trailingStop,
    riskReward,
    expectedHoldTime: holdTimeStr,
    estimatedCosts: costs,
    edge,
    edgeBucket,
    edgeMultiple,
    probUp: ensemble.probUp,
    probDown: ensemble.probDown,
    probChop: ensemble.probChop,
    expectedMove: ensemble.expectedMove,
    reasons,
    vetoReasons,
    patternMatches,
    mlPredictions: ensemble,
    expansionGate,
    combinedIntelligence,
    qualityScore: qualityResult.qualityScore,
    qualityBreakdown: qualityResult.breakdown,
  };
}

export function formatShotPlanForDisplay(plan: ShotPlan): string {
  let output = `\n=== SIGNAL: ${plan.signal} ===\n`;
  output += `Confidence: ${(plan.confidence * 100).toFixed(1)}%\n`;
  output += `Regime: ${plan.regime}\n`;
  output += `Strategy: ${plan.strategy}\n\n`;
  
  if (plan.entryZone) {
    output += `Entry Zone: $${plan.entryZone.low.toFixed(2)} - $${plan.entryZone.high.toFixed(2)}\n`;
    output += `Stop Loss: $${plan.stopLoss?.toFixed(2)}\n`;
    output += `Take Profit 1: $${plan.takeProfit1?.toFixed(2)}\n`;
    output += `Take Profit 2: $${plan.takeProfit2?.toFixed(2)}\n`;
    output += `Risk/Reward: ${plan.riskReward.toFixed(2)}\n\n`;
  }
  
  output += `Expected Hold: ${plan.expectedHoldTime}\n`;
  output += `Estimated Costs: ${(plan.estimatedCosts * 100).toFixed(3)}%\n`;
  output += `Edge: ${(plan.edge * 100).toFixed(3)}%\n\n`;
  
  output += `Probabilities:\n`;
  output += `  Up: ${(plan.probUp * 100).toFixed(1)}%\n`;
  output += `  Down: ${(plan.probDown * 100).toFixed(1)}%\n`;
  output += `  Chop: ${(plan.probChop * 100).toFixed(1)}%\n\n`;
  
  if (plan.reasons.length > 0) {
    output += `Reasons:\n`;
    plan.reasons.forEach((r, i) => output += `  ${i + 1}. ${r}\n`);
  }
  
  if (plan.vetoReasons.length > 0) {
    output += `\nVeto Reasons (why HOLD):\n`;
    plan.vetoReasons.forEach((r, i) => output += `  ${i + 1}. ${r}\n`);
  }
  
  return output;
}
