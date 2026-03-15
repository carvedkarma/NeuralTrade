import { getConfig, getTotalCostsPct, isPaperTradingEnabled, type PaperTradingConfig } from "./config";
import * as storage from "./storage";
import type { PaperPosition } from "@shared/schema";
import type { Candle } from "@shared/schema";
import { candles as candlesTable } from "@shared/schema";
import type { ShotPlan } from "../signal-engine";
import { getRegimeRiskParams, classifyRegime, type MarketRegime } from "../feature-engine";
import { strategyLearner } from "../strategy-learner";
import { 
  checkNoTradeConditions,
  computePositionSize as computeDecisionEngineSize,
  makeTradeDecision,
  type HorizonPredictions,
  type MarketContext,
  type TradeDecision
} from "../trade-decision-engine";
import { HORIZON_CONFIG, NO_TRADE_CONDITIONS } from "../gpu-data-export";
import { edgeTracker } from "../edge-tracker";
import { db } from "../db";
import { eq, and, desc } from "drizzle-orm";

export async function getMarketPrice(symbol: string): Promise<number> {
  return getCurrentMarketPrice(symbol);
}

async function getCurrentMarketPrice(symbol: string): Promise<number> {
  try {
    const { getLatestPrice } = await import("../binance-ws");
    const wsPrice = getLatestPrice(symbol);
    if (wsPrice && wsPrice > 0) return wsPrice;
  } catch {}
  const rows = await db
    .select({ close: candlesTable.close })
    .from(candlesTable)
    .where(and(eq(candlesTable.symbol, symbol), eq(candlesTable.timeframe, "15m")))
    .orderBy(desc(candlesTable.timestamp))
    .limit(1);
  return rows[0]?.close ?? 0;
}

export type ExitReason = "SL" | "TP1" | "TP2" | "TRAIL" | "TIME" | "FLIP" | "MANUAL" | "FAILURE" | "MFE_GIVEBACK" | "NEURAL_FLIP" | "NEURAL_MFE" | "NEURAL_DECAY" | "NEURAL_LOW_CONVICTION" | "NEURAL_MFE_AGGRESSIVE" | "NEURAL_CHOP_EXIT";

interface TradeContext {
  candle: Candle;
  markPrice: number;
  fundingRate: number;
  atr: number;
  kalmanFast: number;
  shotPlan: ShotPlan | null;
  macdHistogram?: number;       // For failure stop detection
  prevMacdHistogram?: number;   // Previous MACD histogram value
}

interface TradeAudit {
  timestamp: number;
  signal: string;
  confidence: number;
  regime: string;
  edge: number;
  costs: number;
  edgeVsCosts: string;
  edgeBucket: string;
  edgeMultiple: number;
  expansionConfirmed: boolean;
  expansionDetails: string;
  positionSize: number;
  sizingMethod: string;  // Track whether Half-Kelly or fixed sizing was used
  exposureAfter: number;
  decision: "ALLOWED" | "BLOCKED";
  reason: string;
}

const auditLog: TradeAudit[] = [];

const EDGE_MULTIPLE_MIN = 1.5;

// Institution-grade loss streak tracking (in-memory for fast access, initialized from DB)
let recentLossStreak = 0;
let lossStreakInitialized = false;

/**
 * Create HorizonPredictions from shot plan data
 * Uses edge as h15 mu, derives h60 and h240 from trend context
 */
function createHorizonPredictions(
  shotPlan: ShotPlan | null,
  edge: number,
  confidence: number
): HorizonPredictions {
  // Default uncertainty estimate based on confidence
  // Higher confidence = lower sigma
  const baseSigma = 0.003; // 30 bps base uncertainty
  const sigmaMultiplier = confidence > 0.6 ? 0.7 : (confidence > 0.4 ? 1.0 : 1.5);
  const sigma15 = baseSigma * sigmaMultiplier;
  
  // H15: Use shot plan edge directly
  const mu15 = edge;
  const direction15 = edge > 0 ? 1 : (edge < 0 ? -1 : 0);
  
  // H60: Assume similar trend direction, slightly dampened
  const mu60 = mu15 * 0.8;
  const sigma60 = sigma15 * 1.2;
  const direction60 = direction15;
  
  // H240: Trend filter - use shot plan regime to infer
  let mu240 = mu15 * 0.5; // Default: weaker version of short-term signal
  if (shotPlan?.regime === "trend_up") {
    mu240 = Math.abs(mu15) * 0.4; // Positive trend
  } else if (shotPlan?.regime === "trend_down") {
    mu240 = -Math.abs(mu15) * 0.4; // Negative trend
  } else if (shotPlan?.regime === "chop") {
    mu240 = 0; // Neutral in chop
  }
  const sigma240 = sigma15 * 1.5;
  const direction240 = mu240 > 0 ? 1 : (mu240 < 0 ? -1 : 0);
  
  return {
    h15: {
      horizon: 15,
      mu: mu15,
      sigma: sigma15,
      direction: direction15,
      confidence: Math.abs(mu15) / sigma15,
      meetsThreshold: Math.abs(mu15) >= HORIZON_CONFIG.h15.minEdge
    },
    h60: {
      horizon: 60,
      mu: mu60,
      sigma: sigma60,
      direction: direction60,
      confidence: Math.abs(mu60) / sigma60,
      meetsThreshold: Math.abs(mu60) >= HORIZON_CONFIG.h60.minEdge
    },
    h240: {
      horizon: 240,
      mu: mu240,
      sigma: sigma240,
      direction: direction240,
      confidence: Math.abs(mu240) / sigma240,
      meetsThreshold: Math.abs(mu240) >= HORIZON_CONFIG.h240.minEdge
    }
  };
}

/**
 * Create MarketContext from TradeContext
 */
function createMarketContext(
  ctx: TradeContext,
  volatilityPercentile: number = 0.5
): MarketContext {
  return {
    atr14: ctx.atr,
    volatility20: ctx.atr / ctx.markPrice, // ATR as % of price
    volatilityPercentile,
    recentLosses: recentLossStreak,
    fundingRate: ctx.fundingRate,
    fundingRateChange: 0, // Would need historical funding data
    spreadProxy: 0.0001 // Default spread estimate
  };
}

/**
 * Get trade decision from the institution-grade decision engine
 */
function getTradeDecision(
  ctx: TradeContext,
  shotPlan: ShotPlan | null,
  edge: number,
  confidence: number
): TradeDecision {
  const predictions = createHorizonPredictions(shotPlan, edge, confidence);
  const marketCtx = createMarketContext(ctx);
  return makeTradeDecision(predictions, marketCtx);
}

export async function initializeLossStreak(): Promise<void> {
  if (!lossStreakInitialized) {
    recentLossStreak = await storage.getRecentLossStreak();
    lossStreakInitialized = true;
    console.log(`[Paper] Initialized loss streak from DB: ${recentLossStreak}`);
  }
}

export function getRecentLossStreakSync(): number {
  return recentLossStreak;
}

export function recordTradeResult(isWin: boolean): void {
  if (isWin) {
    recentLossStreak = 0;
  } else {
    recentLossStreak++;
  }
}

function logAudit(audit: TradeAudit): void {
  auditLog.push(audit);
  if (auditLog.length > 100) auditLog.shift();
  
  const edgeStr = audit.edge > 0 ? `+${(audit.edge * 100).toFixed(3)}%` : `${(audit.edge * 100).toFixed(3)}%`;
  const costsStr = `${(audit.costs * 100).toFixed(3)}%`;
  
  console.log(`[Paper Audit] ${audit.decision}: ${audit.reason}`);
  console.log(`  Signal: ${audit.signal}, Confidence: ${(audit.confidence * 100).toFixed(1)}%, Regime: ${audit.regime}`);
  console.log(`  Edge: ${edgeStr} vs Costs: ${costsStr} (${audit.edgeVsCosts})`);
  console.log(`  Edge Bucket: ${audit.edgeBucket}, Multiple: ${audit.edgeMultiple.toFixed(2)}x costs`);
  console.log(`  Expansion Gate: ${audit.expansionConfirmed ? "CONFIRMED" : "PENDING"} - ${audit.expansionDetails}`);
  console.log(`  Position Size: ${audit.positionSize.toFixed(6)}, Sizing: ${audit.sizingMethod}`);
  console.log(`  Exposure After: ${(audit.exposureAfter * 100).toFixed(1)}%`);
}

export function getAuditLog(): TradeAudit[] {
  return [...auditLog];
}

function applySlippage(price: number, side: "LONG" | "SHORT", slippageBps: number): number {
  const mult = side === "LONG" ? 1 + slippageBps / 10000 : 1 - slippageBps / 10000;
  return price * mult;
}

function calculateFee(notional: number, feePct: number): number {
  return notional * (feePct / 100);
}

function calculateStopDistance(
  entryPrice: number,
  atr: number,
  side: "LONG" | "SHORT",
  config: PaperTradingConfig,
  regime?: string,
  hasExpansion?: boolean
): { stopLoss: number; stopDistance: number; atrPct: number } {
  // ATR as percentage of price
  const atrPct = (atr / entryPrice) * 100;
  
  // Use shared regime risk params for consistency across system
  const regimeParams = getRegimeRiskParams((regime as MarketRegime) || "chop");
  
  // Apply regime-specific stop multiplier
  const atrMultiplier = regimeParams.stopMultiplier;
  
  const atrStop = atr * atrMultiplier;
  const minStopAbs = entryPrice * (config.minStopDistancePct / 100);
  const minCostStop = entryPrice * getTotalCostsPct() * 1.5;
  
  const stopDistance = Math.max(atrStop, minStopAbs, minCostStop);
  
  const stopLoss = side === "LONG" 
    ? entryPrice - stopDistance 
    : entryPrice + stopDistance;
    
  return { stopLoss, stopDistance, atrPct };
}

// Calculate regime-based take profit targets using shared params
function calculateTakeProfits(
  entryPrice: number,
  atr: number,
  side: "LONG" | "SHORT",
  config: PaperTradingConfig,
  regime?: string,
  hasExpansion?: boolean
): { tp1: number; tp2: number | null } {
  // Use shared regime risk params for consistency
  const regimeParams = getRegimeRiskParams((regime as MarketRegime) || "chop");
  const rrRatio = regimeParams.rrRatio;
  const stopMult = regimeParams.stopMultiplier;
  
  const isTrend = regime === "trend_up" || regime === "trend_down";
  
  // Calculate TP based on regime R:R ratio and stop distance
  let tp1Mult: number;
  let tp2Mult: number | null;
  
  if (isTrend && hasExpansion) {
    // Trend + expansion: use full R:R for TP1, 2x for TP2
    tp1Mult = stopMult * rrRatio;        // 1.0 * 2.0 = 2.0x ATR
    tp2Mult = stopMult * rrRatio * 1.5;  // Extended target
  } else if (isTrend) {
    // Trend without expansion: conservative R:R
    tp1Mult = stopMult * rrRatio * 0.8;  // Slightly reduced
    tp2Mult = null;
  } else if (regime === "shock") {
    // Shock: quick exits at reduced R:R
    tp1Mult = stopMult * rrRatio;        // 0.7 * 1.2 = 0.84x ATR
    tp2Mult = null;
  } else if (regime === "quiet") {
    // Quiet: await breakout with wide target
    tp1Mult = stopMult * rrRatio;        // 0.8 * 2.0 = 1.6x ATR
    tp2Mult = stopMult * rrRatio * 1.5;  // Extended for breakout
  } else if (regime === "ranging") {
    // Ranging: mean reversion targets
    tp1Mult = stopMult * rrRatio;        // 0.75 * 1.5 = 1.125x ATR
    tp2Mult = null;
  } else {
    // Chop/default: conservative quick exits
    tp1Mult = stopMult * rrRatio;        // 0.6 * 1.3 = 0.78x ATR
    tp2Mult = null;
  }
  
  const tp1Distance = atr * tp1Mult;
  const tp2Distance = tp2Mult ? atr * tp2Mult : null;
  
  if (side === "LONG") {
    return {
      tp1: entryPrice + tp1Distance,
      tp2: tp2Distance ? entryPrice + tp2Distance : null,  // null when no TP2 intended
    };
  } else {
    return {
      tp1: entryPrice - tp1Distance,
      tp2: tp2Distance ? entryPrice - tp2Distance : null,  // null when no TP2 intended
    };
  }
}

function calculatePositionSize(
  equity: number,
  stopDistance: number,
  riskPct: number,
  maxRiskPct: number,
  side?: "LONG" | "SHORT"
): { qty: number; riskUsdt: number; sizingMethod: string } {
  // HALF-KELLY POSITION SIZING: Dynamic sizing based on historical edge
  // Captures ~75% of optimal growth with ~50% less drawdown than full Kelly
  let actualRiskPct: number;
  let sizingMethod: string;
  
  if (side) {
    const kellyResult = strategyLearner.getKellyPositionSize(side);
    
    if (kellyResult.recommendedSize > 0.01) {
      // Use Half-Kelly sizing (convert from fraction to percentage)
      actualRiskPct = Math.min(kellyResult.recommendedSize * 100, maxRiskPct);
      sizingMethod = `Half-Kelly: ${(kellyResult.recommendedSize * 100).toFixed(1)}% (${kellyResult.reasoning})`;
      console.log(`[Paper] Half-Kelly position sizing: ${sizingMethod}`);
    } else {
      // Fallback to fixed sizing if Kelly recommends very small position
      actualRiskPct = Math.min(riskPct, maxRiskPct);
      sizingMethod = `Fixed risk: ${actualRiskPct.toFixed(1)}% (Kelly insufficient)`;
    }
  } else {
    // No side provided, use fixed sizing
    actualRiskPct = Math.min(riskPct, maxRiskPct);
    sizingMethod = `Fixed risk: ${actualRiskPct.toFixed(1)}%`;
  }
  
  const riskUsdt = equity * (actualRiskPct / 100);
  
  if (stopDistance <= 0) {
    return { qty: 0, riskUsdt: 0, sizingMethod: "Invalid stop distance" };
  }
  
  const qty = riskUsdt / stopDistance;
  return { qty, riskUsdt, sizingMethod };
}

function calculateUnrealizedPnl(position: PaperPosition, currentPrice: number): number {
  const priceDiff = position.side === "LONG" 
    ? currentPrice - position.entryPrice 
    : position.entryPrice - currentPrice;
  return priceDiff * position.qty;
}

function checkStopLoss(position: PaperPosition, candle: Candle): boolean {
  if (!position.stopLoss) return false;
  if (position.side === "LONG") {
    return candle.low <= position.stopLoss;
  } else {
    return candle.high >= position.stopLoss;
  }
}

function checkTp1(position: PaperPosition, candle: Candle): boolean {
  if (!position.tp1) return false;
  if (position.side === "LONG") {
    return candle.high >= position.tp1;
  } else {
    return candle.low <= position.tp1;
  }
}

function checkTp2(position: PaperPosition, candle: Candle): boolean {
  if (!position.tp2) return false;
  if (position.side === "LONG") {
    return candle.high >= position.tp2;
  } else {
    return candle.low <= position.tp2;
  }
}

function checkTrailingStop(position: PaperPosition, candle: Candle): boolean {
  if (!position.trailPrice || position.trailMode === "none") return false;
  if (position.side === "LONG") {
    return candle.low <= position.trailPrice;
  } else {
    return candle.high >= position.trailPrice;
  }
}

// MFE-aware trailing stop with giveback logic
function updateMfeAwareTrailingStop(
  position: PaperPosition, 
  currentPrice: number,
  atr: number,
  config: PaperTradingConfig
): { newTrailPrice: number | null; shouldExit: boolean; exitReason: string; newPeakProfit: number } {
  const atrPct = atr / position.entryPrice;
  const currentProfitPct = position.side === "LONG"
    ? (currentPrice - position.entryPrice) / position.entryPrice
    : (position.entryPrice - currentPrice) / position.entryPrice;
  
  // peakProfit is stored as absolute PnL: priceDiff * qty
  // Convert to percentage for comparison: peakProfit / (entryPrice * qty)
  const notionalValue = position.entryPrice * position.qty;
  const peakProfitPct = position.peakProfit && notionalValue > 0
    ? position.peakProfit / notionalValue 
    : Math.max(currentProfitPct, 0);
  
  // Track new peak
  const newPeakProfitPct = Math.max(peakProfitPct, currentProfitPct);
  const newPeakProfit = newPeakProfitPct * notionalValue;  // Store as absolute PnL
  
  if (position.trailMode === "none") {
    return { newTrailPrice: null, shouldExit: false, exitReason: "", newPeakProfit };
  }

  // FIX: Don't allow MFE to act on brand-new positions (< 2 bars old).
  // This prevents the trail from arming and firing within the very first candle
  // when price noise can briefly cross the activation threshold.
  if ((position.barsOpen ?? 0) < 2) {
    return { newTrailPrice: null, shouldExit: false, exitReason: "", newPeakProfit };
  }
  
  // Only activate trailing after reaching activation threshold (now 1.5x ATR_pct)
  const activationThreshold = config.mfeTrailActivation * atrPct;
  if (newPeakProfitPct < activationThreshold) {
    return { newTrailPrice: null, shouldExit: false, exitReason: "", newPeakProfit };
  }
  
  // Calculate giveback from peak
  const givebackPct = newPeakProfitPct - currentProfitPct;
  
  // Check if giveback exceeds threshold: max(0.35×ATR_pct, 0.5×TP1_distance)
  const tp1Distance = position.tp1 
    ? Math.abs(position.tp1 - position.entryPrice) / position.entryPrice 
    : atrPct;
  const minGiveback = config.mfeMinGiveback * atrPct;
  const maxGiveback = Math.max(minGiveback, config.mfeGivebackPct * tp1Distance);

  // FIX: Require profit to cover round-trip fees before MFE can close the position.
  // Previously checked currentProfitPct > 0 (raw price), which allowed exits at
  // breakeven-raw-price that became net losses after fees were applied.
  const feeCoverPct = getTotalCostsPct();
  if (givebackPct >= maxGiveback && currentProfitPct > feeCoverPct) {
    return { 
      newTrailPrice: null, 
      shouldExit: true, 
      exitReason: `MFE Giveback: ${(givebackPct * 100).toFixed(2)}% from peak ${(newPeakProfitPct * 100).toFixed(2)}%`,
      newPeakProfit
    };
  }
  
  // Standard Kalman-based trailing (as backup)
  const buffer = atr * config.trailBufferAtrMultiplier;
  let newTrailPrice = position.trailPrice;
  
  if (position.side === "LONG") {
    const kalmanTrail = currentPrice - buffer;
    if (!position.trailPrice || kalmanTrail > position.trailPrice) {
      newTrailPrice = kalmanTrail;
    }
  } else {
    const kalmanTrail = currentPrice + buffer;
    if (!position.trailPrice || kalmanTrail < position.trailPrice) {
      newTrailPrice = kalmanTrail;
    }
  }
  
  return { newTrailPrice, shouldExit: false, exitReason: "", newPeakProfit };
}

// Failure stop: exit early when setup is invalidated (Kalman + MACD flip)
function checkFailureStop(
  position: PaperPosition,
  ctx: TradeContext,
  config: PaperTradingConfig
): { shouldExit: boolean; reason: string } {
  if (!config.failureStopEnabled) {
    return { shouldExit: false, reason: "" };
  }
  
  const currentPrice = ctx.candle.close;
  const macdHist = ctx.macdHistogram ?? 0;
  const prevMacdHist = ctx.prevMacdHistogram ?? 0;
  
  // For LONG: exit if close loses Kalman fast AND MACD histogram flips negative
  if (position.side === "LONG") {
    const losesKalman = currentPrice < ctx.kalmanFast;
    const macdFlipsNegative = macdHist < 0 && prevMacdHist >= 0;
    
    if (losesKalman && macdFlipsNegative) {
      return { 
        shouldExit: true, 
        reason: "Failure Stop: Price below Kalman fast + MACD histogram flipped negative"
      };
    }
  }
  
  // For SHORT: exit if close reclaims Kalman fast AND MACD histogram flips positive
  if (position.side === "SHORT") {
    const reclaimsKalman = currentPrice > ctx.kalmanFast;
    const macdFlipsPositive = macdHist > 0 && prevMacdHist <= 0;
    
    if (reclaimsKalman && macdFlipsPositive) {
      return { 
        shouldExit: true, 
        reason: "Failure Stop: Price above Kalman fast + MACD histogram flipped positive"
      };
    }
  }
  
  return { shouldExit: false, reason: "" };
}

// Legacy trailing stop for compatibility
function updateTrailingStop(
  position: PaperPosition, 
  kalmanFast: number, 
  atr: number,
  config: PaperTradingConfig
): number | null {
  if (position.trailMode === "none") return null;
  
  const buffer = atr * config.trailBufferAtrMultiplier;
  if (position.side === "LONG") {
    const newTrail = kalmanFast - buffer;
    if (!position.trailPrice || newTrail > position.trailPrice) {
      return newTrail;
    }
  } else {
    const newTrail = kalmanFast + buffer;
    if (!position.trailPrice || newTrail < position.trailPrice) {
      return newTrail;
    }
  }
  return position.trailPrice;
}

function checkTimeStop(position: PaperPosition, pnlR: number, config: PaperTradingConfig): boolean {
  const barsOpen = position.barsOpen || 0;
  
  // INSTITUTION-GRADE: Use horizon-specific maxHoldBars if primaryHorizon is set
  const primaryHorizon = position.primaryHorizon ?? 15;
  let maxBars = config.timeStopBars;
  
  if (primaryHorizon === 15) {
    maxBars = HORIZON_CONFIG.h15.maxHoldBars;
  } else if (primaryHorizon === 60) {
    maxBars = HORIZON_CONFIG.h60.maxHoldBars;
  } else if (primaryHorizon === 240) {
    maxBars = HORIZON_CONFIG.h240.maxHoldBars;
  }
  
  return barsOpen >= maxBars && pnlR < config.minPnlForTimeStop;
}

function shouldFlip(
  position: PaperPosition,
  shotPlan: ShotPlan | null,
  config: PaperTradingConfig
): boolean {
  if (!shotPlan || shotPlan.signal === "HOLD") return false;
  const isOpposite = 
    (position.side === "LONG" && shotPlan.signal === "SHORT") ||
    (position.side === "SHORT" && shotPlan.signal === "LONG");
  if (!isOpposite) return false;
  const costs = shotPlan.estimatedCosts || getTotalCostsPct();
  return shotPlan.confidence >= config.flipConfidenceThreshold && 
         shotPlan.edge > costs * config.flipEdgeMultiplier;
}

interface GatingResult {
  allowed: boolean;
  reason: string;
  tradeDecision?: TradeDecision;  // Include decision for use in openPosition
}

function checkShotPlanGating(shotPlan: ShotPlan | null, config: PaperTradingConfig, ctx?: TradeContext): GatingResult {
  if (!shotPlan) {
    return { allowed: false, reason: "No shot plan available" };
  }
  
  // Use combined intelligence signal when available (smarter - considers pattern history + strategy learner)
  const effectiveSignal = shotPlan.combinedIntelligence?.finalSignal || shotPlan.signal;
  const effectiveConfidence = shotPlan.combinedIntelligence?.finalConfidence || shotPlan.confidence;
  
  if (effectiveSignal === "HOLD") {
    console.log("[Paper] Combined Intelligence says HOLD - respecting smart signal");
    return { allowed: false, reason: "Combined Intelligence: HOLD signal" };
  }
  
  if (effectiveSignal !== "LONG" && effectiveSignal !== "SHORT") {
    return { allowed: false, reason: `Invalid signal: ${effectiveSignal}` };
  }
  
  if (shotPlan.confidence < config.minConfidence) {
    return { 
      allowed: false, 
      reason: `Confidence ${(shotPlan.confidence * 100).toFixed(1)}% < ${(config.minConfidence * 100).toFixed(1)}% minimum` 
    };
  }
  
  // SELECTIVE MODE: Require positive edge for trades
  const edge = shotPlan.edge ?? 0;
  if (edge <= 0) {
    return { 
      allowed: false, 
      reason: `Negative or zero edge: ${(edge * 100).toFixed(3)}%. No trade without positive expected value.` 
    };
  }
  
  // INSTITUTION-GRADE: Use trade decision engine for comprehensive veto checks
  let tradeDecision: TradeDecision | undefined;
  if (ctx) {
    tradeDecision = getTradeDecision(ctx, shotPlan, edge, effectiveConfidence);
    
    // If decision engine says HOLD, veto the trade with its reasons
    if (tradeDecision.action === "HOLD") {
      const reason = tradeDecision.vetoes.length > 0 
        ? `DECISION_ENGINE: ${tradeDecision.vetoes.join('; ')}`
        : `DECISION_ENGINE: ${tradeDecision.reasons.join('; ')}`;
      console.log(`[Paper] Trade decision engine vetoed: ${reason}`);
      return { allowed: false, reason, tradeDecision };
    }
  } else {
    // Fallback to basic checks if no TradeContext
    // INSTITUTION-GRADE: Edge dead zone check
    if (Math.abs(edge) < NO_TRADE_CONDITIONS.deadZoneThreshold) {
      return {
        allowed: false,
        reason: `DEAD_ZONE: |edge|=${(Math.abs(edge) * 100).toFixed(2)} bps < ${(NO_TRADE_CONDITIONS.deadZoneThreshold * 100).toFixed(0)} bps minimum`
      };
    }
    
    // INSTITUTION-GRADE: Loss streak check
    const currentLossStreak = recentLossStreak;
    if (currentLossStreak >= NO_TRADE_CONDITIONS.maxLossStreak) {
      return {
        allowed: false,
        reason: `LOSS_STREAK: ${currentLossStreak} consecutive losses >= ${NO_TRADE_CONDITIONS.maxLossStreak} max. Cool down required.`
      };
    }
  }
  
  // SELECTIVE MODE: Block low-probability regime trades
  const blockedRegimes = ["chop"];  // Only block pure chop - allow quiet/ranging/shock with caution
  if (blockedRegimes.includes(shotPlan.regime)) {
    return { 
      allowed: false, 
      reason: `${shotPlan.regime} regime detected - no directional edge. HOLD until trend emerges.` 
    };
  }
  
  // SELECTIVE MODE: Require extra confidence for volatile regimes
  if (shotPlan.regime === "shock" && shotPlan.confidence < 0.75) {
    return {
      allowed: false,
      reason: `Shock regime requires 75%+ confidence (got ${(shotPlan.confidence * 100).toFixed(0)}%)`
    };
  }
  
  const isTrendTrade = shotPlan.regime === "trend_up" || shotPlan.regime === "trend_down";
  if (isTrendTrade && shotPlan.expansionGate && !shotPlan.expansionGate.confirmed) {
    return { 
      allowed: false, 
      reason: `Expansion gate not confirmed for trend trade: ${shotPlan.expansionGate.details}` 
    };
  }
  
  if (!shotPlan.entryZone || !shotPlan.stopLoss || !shotPlan.takeProfit1 || !shotPlan.takeProfit2) {
    return { allowed: false, reason: "Missing trade levels (entry/stop/TP)" };
  }
  
  // SELECTIVE MODE: Allow up to 2 vetoes, block trades with 3+ vetoes
  // Research: Meta-labeling + small veto tolerance improves selectivity while maintaining trade flow
  const vetoCount = shotPlan.vetoReasons?.length ?? 0;
  const MAX_ALLOWED_VETOES = 2;
  if (vetoCount > MAX_ALLOWED_VETOES) {
    return { 
      allowed: false, 
      reason: `Too many vetoes: ${vetoCount} > ${MAX_ALLOWED_VETOES} max. Issues: ${shotPlan.vetoReasons!.slice(0, 3).join('; ')}` 
    };
  }
  
  // SELECTIVE MODE: Require at least 2 supporting reasons for a trade
  if (!shotPlan.reasons || shotPlan.reasons.length < 2) {
    return { 
      allowed: false, 
      reason: `Insufficient confluence: ${shotPlan.reasons?.length ?? 0}/2 minimum reasons required` 
    };
  }
  
  // SELECTIVE MODE: Full combined intelligence checks
  if (shotPlan.combinedIntelligence) {
    const ci = shotPlan.combinedIntelligence;
    
    // Block if ML system says HOLD
    if (ci.finalSignal === "HOLD") {
      return { 
        allowed: false, 
        reason: `Combined Intelligence: HOLD signal (${(ci.finalConfidence * 100).toFixed(0)}% confidence)` 
      };
    }
    
    // SELECTIVE MODE: Block negative EV trades
    if (ci.strategyEV <= 0) {
      return { 
        allowed: false, 
        reason: `Negative strategy EV: ${(ci.strategyEV * 100).toFixed(2)}%. No trade without positive expectancy.` 
      };
    }
    
    // SELECTIVE MODE: Require system agreement OR high pattern win rate
    if (!ci.systemsAgree && ci.patternWinRate < 0.5) {
      return { 
        allowed: false, 
        reason: `Systems disagree and pattern win rate ${(ci.patternWinRate * 100).toFixed(0)}% < 50%. HOLD.` 
      };
    }
    
    // META-LABELING FILTER: Secondary model evaluates "should I trust this signal?"
    // This provides an extra layer of filtering to improve precision from ~37% to ~56%
    const metaLabel = strategyLearner.getMetaLabelConfidence(
      effectiveSignal as "LONG" | "SHORT" | "HOLD",
      shotPlan.regime,
      effectiveConfidence,
      ci.patternWinRate,
      ci.systemsAgree,
      ci.atrPercentile ?? 50  // Default to 50th percentile if not available
    );
    
    if (!metaLabel.shouldTrade) {
      return {
        allowed: false,
        reason: `Meta-Label VETO: ${metaLabel.vetoes.slice(0, 2).join('; ')} (${(metaLabel.metaConfidence * 100).toFixed(0)}% confidence)`
      };
    }
    
    console.log(`[Paper] Meta-Label PASS: ${(metaLabel.metaConfidence * 100).toFixed(0)}% confidence - ${metaLabel.reasoning.join(', ')}`);
  }
  
  return { allowed: true, reason: "SELECTIVE MODE: All quality gates passed (including Meta-Label)", tradeDecision };
}

async function checkExposureLimits(
  notional: number,
  equity: number,
  config: PaperTradingConfig
): Promise<GatingResult> {
  const existingPosition = await storage.getOpenPosition();
  
  if (existingPosition) {
    return { allowed: false, reason: "Position already open - one position at a time" };
  }
  
  const newExposure = notional;
  const exposurePct = (newExposure / equity) * 100;
  
  if (exposurePct > config.maxAccountExposurePct) {
    return { 
      allowed: false, 
      reason: `Exposure ${exposurePct.toFixed(1)}% > ${config.maxAccountExposurePct}% max` 
    };
  }
  
  return { allowed: true, reason: "Exposure within limits" };
}

export async function openPosition(
  ctx: TradeContext,
  side: "LONG" | "SHORT",
  shotPlanStopLoss: number,
  shotPlanTp1: number,
  shotPlanTp2: number,
  confidence: number,
  edge: number,
  tradeDecision?: TradeDecision  // Institution-grade decision for horizon selection
): Promise<PaperPosition | null> {
  const config = getConfig();
  const portfolio = await storage.getOrCreatePortfolio();
  
  const entryPrice = applySlippage(ctx.candle.open, side, config.slippageBps);
  const regime = ctx.shotPlan?.regime || "unknown";
  const hasExpansion = ctx.shotPlan?.expansionGate?.confirmed || false;
  
  // Use regime-based ATR stop calculation
  const { stopLoss, stopDistance, atrPct } = calculateStopDistance(
    entryPrice,
    ctx.atr,
    side,
    config,
    regime,
    hasExpansion
  );
  
  // Use regime-based take profit targets
  const { tp1, tp2 } = calculateTakeProfits(
    entryPrice,
    ctx.atr,
    side,
    config,
    regime,
    hasExpansion
  );
  
  const { qty, riskUsdt, sizingMethod } = calculatePositionSize(
    portfolio.currentEquityUsdt,
    stopDistance,
    config.riskPerTradePct,
    config.maxRiskPerTradePct,
    side  // Pass side for Half-Kelly position sizing
  );

  if (qty <= 0) {
    console.log("[Paper] Invalid position size calculated, skipping");
    return null;
  }

  const notional = qty * entryPrice;
  
  const exposureCheck = await checkExposureLimits(notional, portfolio.currentEquityUsdt, config);
  const costs = getTotalCostsPct();
  const edgeMultiple = costs > 0 ? edge / costs : 0;
  
  if (!exposureCheck.allowed) {
    logAudit({
      timestamp: Date.now(),
      signal: side,
      confidence,
      regime: ctx.shotPlan?.regime || "unknown",
      edge,
      costs,
      edgeVsCosts: edge > costs ? "PASS" : "FAIL",
      edgeBucket: ctx.shotPlan?.edgeBucket || "none",
      edgeMultiple,
      expansionConfirmed: ctx.shotPlan?.expansionGate?.confirmed || false,
      expansionDetails: ctx.shotPlan?.expansionGate?.details || "N/A",
      positionSize: qty,
      sizingMethod,
      exposureAfter: notional / portfolio.currentEquityUsdt,
      decision: "BLOCKED",
      reason: exposureCheck.reason,
    });
    return null;
  }

  const entryFee = calculateFee(notional, config.takerFeePct);

  logAudit({
    timestamp: Date.now(),
    signal: side,
    confidence,
    regime: ctx.shotPlan?.regime || "unknown",
    edge,
    costs,
    edgeVsCosts: "PASS",
    edgeBucket: ctx.shotPlan?.edgeBucket || "none",
    edgeMultiple,
    expansionConfirmed: ctx.shotPlan?.expansionGate?.confirmed || false,
    expansionDetails: ctx.shotPlan?.expansionGate?.details || "N/A",
    positionSize: qty,
    sizingMethod,
    exposureAfter: notional / portfolio.currentEquityUsdt,
    decision: "ALLOWED",
    reason: "All checks passed - opening position",
  });

  const position = await storage.createPosition({
    symbol: "BTCUSDT",
    side,
    status: "OPEN",
    entryTs: ctx.candle.timestamp,
    entryPrice,
    qty,
    notionalUsdt: notional,
    leverage: 1,
    stopLoss,
    tp1,
    tp2,
    trailMode: "kalman",
    trailPrice: null,
    timeStopBars: config.timeStopBars,
    barsOpen: 0,
    primaryHorizon: tradeDecision?.primaryHorizon || 15,  // Use decision engine's selected horizon
    initialRiskUsdt: riskUsdt,
    feesPaidUsdt: entryFee,
    fundingPaidUsdt: 0,
    exitTs: null,
    exitPrice: null,
    realizedPnlUsdt: null,
    exitReason: null,
    signalConfidence: confidence,
    signalEdge: edge,
    // MFE tracking and R-multiple fields
    peakProfit: 0,                      // Initialize MFE at 0
    initialStopDistance: stopDistance,  // Store for R-multiple calculation
    regime: regime,                     // Store regime for analysis
  });

  await storage.createTrade({
    positionId: position.id,
    ts: ctx.candle.timestamp,
    action: "OPEN",
    price: entryPrice,
    qty,
    feeUsdt: entryFee,
    slippageUsdt: Math.abs(entryPrice - ctx.candle.open) * qty,
    fundingUsdt: 0,
    pnlUsdt: 0,
    reason: `${side} entry @ ${entryPrice.toFixed(2)} | SL: ${stopLoss.toFixed(2)} | Risk: $${riskUsdt.toFixed(2)}`,
  });

  await storage.updatePortfolio({
    availableBalanceUsdt: portfolio.availableBalanceUsdt - riskUsdt,
  });

  console.log(`[Paper] OPENED ${side}: ${qty.toFixed(6)} BTC @ ${entryPrice.toFixed(2)}`);
  console.log(`  Regime: ${regime}, Expansion: ${hasExpansion ? "YES" : "NO"}`);
  console.log(`  Stop: ${stopLoss.toFixed(2)} (${stopDistance.toFixed(2)} distance, R=1)`);
  console.log(`  Risk: $${riskUsdt.toFixed(2)} (${config.riskPerTradePct}% of equity)`);
  console.log(`  TP1: ${tp1.toFixed(2)}, TP2: ${tp2 ? tp2.toFixed(2) : "N/A"}`);
  
  return position;
}

export async function closePosition(
  position: PaperPosition,
  exitPrice: number,
  reason: ExitReason,
  ctx: TradeContext,
  partialQty?: number
): Promise<{ pnl: number; isPartial: boolean }> {
  const config = getConfig();
  const portfolio = await storage.getOrCreatePortfolio();
  
  const qtyToClose = partialQty || position.qty;
  const isPartial = partialQty !== undefined && partialQty < position.qty;
  
  const slippedExitPrice = applySlippage(
    exitPrice, 
    position.side === "LONG" ? "SHORT" : "LONG",
    config.slippageBps
  );

  const priceDiff = position.side === "LONG"
    ? slippedExitPrice - position.entryPrice
    : position.entryPrice - slippedExitPrice;
  const grossPnl = priceDiff * qtyToClose;
  
  const exitNotional = slippedExitPrice * qtyToClose;
  const exitFee = calculateFee(exitNotional, config.takerFeePct);
  const netPnl = grossPnl - exitFee;

  await storage.createTrade({
    positionId: position.id,
    ts: ctx.candle.timestamp,
    action: isPartial ? "PARTIAL_CLOSE" : "CLOSE",
    price: slippedExitPrice,
    qty: qtyToClose,
    feeUsdt: exitFee,
    slippageUsdt: Math.abs(slippedExitPrice - exitPrice) * qtyToClose,
    fundingUsdt: 0,
    pnlUsdt: netPnl,
    reason: `${reason}: Exit @ ${slippedExitPrice.toFixed(2)} | PnL: ${netPnl >= 0 ? '+' : ''}$${netPnl.toFixed(2)}`,
  });

  if (isPartial) {
    await storage.updatePosition(position.id, {
      qty: position.qty - qtyToClose,
      notionalUsdt: position.notionalUsdt * ((position.qty - qtyToClose) / position.qty),
    });
  } else {
    const totalRealizedPnl = netPnl - (position.fundingPaidUsdt || 0);
    const totalFees = (position.feesPaidUsdt || 0) + exitFee;
    const riskUsdt = position.initialRiskUsdt ?? 1;
    const grossRMultiple = riskUsdt > 0 ? (totalRealizedPnl + totalFees) / riskUsdt : 0;
    const netRMultiple = riskUsdt > 0 ? totalRealizedPnl / riskUsdt : 0;
    const costRMultiple = riskUsdt > 0 ? totalFees / riskUsdt : 0;
    const maxFavorableR = riskUsdt > 0 && position.peakProfit != null ? position.peakProfit / riskUsdt : 0;

    await storage.updatePosition(position.id, {
      status: "CLOSED",
      exitTs: ctx.candle.timestamp,
      exitPrice: slippedExitPrice,
      realizedPnlUsdt: totalRealizedPnl,
      exitReason: reason,
      feesPaidUsdt: totalFees,
    });

    await storage.recordTradeClose({
      positionId: position.id,
      symbol: position.symbol,
      side: position.side,
      entryTs: position.entryTs,
      entryPrice: position.entryPrice,
      exitTs: ctx.candle.timestamp,
      exitPrice: slippedExitPrice,
      grossR: Math.round(grossRMultiple * 10000) / 10000,
      netR: Math.round(netRMultiple * 10000) / 10000,
      costR: Math.round(costRMultiple * 10000) / 10000,
      pnlUsdt: totalRealizedPnl,
      riskUsdt,
      barsHeld: position.barsOpen ?? 0,
      exitReason: reason,
      maxFavorableR: Math.round(maxFavorableR * 10000) / 10000,
      regime: position.regime ?? null,
      signalConfidence: position.signalConfidence ?? null,
      signalEdge: position.signalEdge ?? null,
    });

    const newEquity = portfolio.currentEquityUsdt + totalRealizedPnl;
    const newPeak = Math.max(portfolio.peakEquityUsdt, newEquity);
    const drawdown = newPeak > 0 ? ((newPeak - newEquity) / newPeak) * 100 : 0;
    const maxDrawdown = Math.max(portfolio.maxDrawdownPct ?? 0, drawdown);

    await storage.updatePortfolio({
      currentEquityUsdt: newEquity,
      availableBalanceUsdt: newEquity,
      realizedPnlUsdt: (portfolio.realizedPnlUsdt ?? 0) + totalRealizedPnl,
      unrealizedPnlUsdt: 0,
      peakEquityUsdt: newPeak,
      maxDrawdownPct: maxDrawdown,
      totalTrades: (portfolio.totalTrades ?? 0) + 1,
      winningTrades: totalRealizedPnl > 0 ? (portfolio.winningTrades ?? 0) + 1 : portfolio.winningTrades ?? 0,
      losingTrades: totalRealizedPnl <= 0 ? (portfolio.losingTrades ?? 0) + 1 : portfolio.losingTrades ?? 0,
    });

    await storage.recordEquityPoint(newEquity, drawdown);
    console.log(`[Paper] CLOSED ${position.side}: ${reason} | PnL: ${totalRealizedPnl >= 0 ? '+' : ''}$${totalRealizedPnl.toFixed(2)}`);
    
    // INSTITUTION-GRADE: Track loss streak for NO-TRADE conditions
    recordTradeResult(totalRealizedPnl > 0);
    
    // EDGE TRACKING: Record signal result for edge metrics
    edgeTracker.recordSignal(
      position.side as "LONG" | "SHORT",
      position.entryPrice,
      slippedExitPrice,
      16,  // Default horizon (can be enhanced to store actual horizon per position)
      (position as any).signalConfidence || 0.5
    );
  }

  return { pnl: netPnl, isPartial };
}

export async function processCandle(ctx: TradeContext): Promise<void> {
  const config = getConfig();
  
  // Initialize loss streak from DB on first run
  await initializeLossStreak();
  
  if (!isPaperTradingEnabled()) {
    return;
  }
  
  const position = await storage.getOpenPosition();
  const portfolio = await storage.getOrCreatePortfolio();
  
  if (position && position.source === "v5_signal") {
    return;
  }
  
  if (!position) {
    const gating = checkShotPlanGating(ctx.shotPlan, config, ctx);
    const shotPlanCosts = ctx.shotPlan?.estimatedCosts || getTotalCostsPct();
    const shotPlanEdge = ctx.shotPlan?.edge || 0;
    const shotPlanEdgeMultiple = shotPlanCosts > 0 ? shotPlanEdge / shotPlanCosts : 0;
    
    // Use combined intelligence signal for logging (the actual signal being used)
    const effectiveSignal = ctx.shotPlan?.combinedIntelligence?.finalSignal || ctx.shotPlan?.signal || "NONE";
    const effectiveConfidence = ctx.shotPlan?.combinedIntelligence?.finalConfidence || ctx.shotPlan?.confidence || 0;
    
    logAudit({
      timestamp: Date.now(),
      signal: effectiveSignal,
      confidence: effectiveConfidence,
      regime: ctx.shotPlan?.regime || "unknown",
      edge: shotPlanEdge,
      costs: shotPlanCosts,
      edgeVsCosts: shotPlanEdge > shotPlanCosts ? "PASS" : "FAIL",
      edgeBucket: ctx.shotPlan?.edgeBucket || "none",
      edgeMultiple: shotPlanEdgeMultiple,
      expansionConfirmed: ctx.shotPlan?.expansionGate?.confirmed || false,
      expansionDetails: ctx.shotPlan?.expansionGate?.details || "N/A",
      positionSize: 0,
      sizingMethod: "N/A (pre-gating)",
      exposureAfter: 0,
      decision: gating.allowed ? "ALLOWED" : "BLOCKED",
      reason: gating.reason,
    });
    
    if (!gating.allowed) {
      return;
    }
    
    const shotPlan = ctx.shotPlan!;
    // Use combined intelligence signal when available
    const tradeSignal = (shotPlan.combinedIntelligence?.finalSignal || shotPlan.signal) as "LONG" | "SHORT";
    const tradeConfidence = shotPlan.combinedIntelligence?.finalConfidence || shotPlan.confidence;
    
    await openPosition(
      ctx,
      tradeSignal,
      shotPlan.stopLoss!,
      shotPlan.takeProfit1!,
      shotPlan.takeProfit2!,
      tradeConfidence,
      shotPlan.edge,
      gating.tradeDecision  // Pass trade decision for horizon selection
    );
    return;
  }

  // Track MFE (Maximum Favorable Excursion) - peak profit so far
  const currentPnl = calculateUnrealizedPnl(position, ctx.candle.close);
  const currentPeakProfit = position.peakProfit ?? 0;
  const newPeakProfit = Math.max(currentPeakProfit, currentPnl);
  
  await storage.updatePosition(position.id, {
    barsOpen: (position.barsOpen || 0) + 1,
    peakProfit: newPeakProfit,  // Update MFE tracking
  });

  const slHit = checkStopLoss(position, ctx.candle);
  const tp1Hit = checkTp1(position, ctx.candle);
  const tp2Hit = checkTp2(position, ctx.candle);
  const trailHit = checkTrailingStop(position, ctx.candle);
  
  // Check failure stop (Kalman + MACD flip invalidation)
  const failureCheck = checkFailureStop(position, ctx, config);
  if (failureCheck.shouldExit) {
    console.log(`[Paper] FAILURE STOP: ${failureCheck.reason}`);
    await closePosition(position, ctx.candle.close, "FAILURE", ctx);
    return;
  }
  
  // Check MFE-aware trailing with giveback logic
  const mfeTrail = updateMfeAwareTrailingStop(position, ctx.candle.close, ctx.atr, config);
  
  // Update peak profit tracking
  if (mfeTrail.newPeakProfit && mfeTrail.newPeakProfit > (position.peakProfit || 0)) {
    await storage.updatePositionPeakProfit(position.id, mfeTrail.newPeakProfit);
  }
  
  if (mfeTrail.shouldExit) {
    console.log(`[Paper] MFE GIVEBACK EXIT: ${mfeTrail.exitReason}`);
    await closePosition(position, ctx.candle.close, "MFE_GIVEBACK", ctx);
    return;
  }

  if (slHit && tp1Hit) {
    await closePosition(position, position.stopLoss!, "SL", ctx);
    return;
  }

  if (slHit) {
    await closePosition(position, position.stopLoss!, "SL", ctx);
    return;
  }

  if (tp1Hit && position.qty > 0) {
    const partialQty = position.qty * 0.5;
    await closePosition(position, position.tp1!, "TP1", ctx, partialQty);
    const updatedPosition = await storage.getOpenPosition();
    if (!updatedPosition) return;
    
    if (tp2Hit) {
      await closePosition(updatedPosition, position.tp2!, "TP2", ctx);
      return;
    }
  }

  if (tp2Hit) {
    await closePosition(position, position.tp2!, "TP2", ctx);
    return;
  }

  if (trailHit) {
    await closePosition(position, position.trailPrice!, "TRAIL", ctx);
    return;
  }

  const unrealizedPnl = calculateUnrealizedPnl(position, ctx.markPrice);
  const pnlR = position.initialRiskUsdt ? unrealizedPnl / position.initialRiskUsdt : 0;

  if (checkTimeStop(position, pnlR, config)) {
    await closePosition(position, ctx.markPrice, "TIME", ctx);
    return;
  }

  if (shouldFlip(position, ctx.shotPlan, config)) {
    const gating = checkShotPlanGating(ctx.shotPlan, config, ctx);
    if (gating.allowed) {
      await closePosition(position, ctx.markPrice, "FLIP", ctx);
      const shotPlan = ctx.shotPlan!;
      await openPosition(
        ctx,
        shotPlan.signal as "LONG" | "SHORT",
        shotPlan.stopLoss!,
        shotPlan.takeProfit1!,
        shotPlan.takeProfit2!,
        shotPlan.confidence,
        shotPlan.edge,
        gating.tradeDecision  // Pass trade decision for horizon selection
      );
    }
    return;
  }

  const newTrailPrice = updateTrailingStop(position, ctx.kalmanFast, ctx.atr, config);
  if (newTrailPrice && newTrailPrice !== position.trailPrice) {
    await storage.updatePosition(position.id, { trailPrice: newTrailPrice });
  }

  const equity = portfolio.currentEquityUsdt + unrealizedPnl;
  const drawdown = portfolio.peakEquityUsdt > 0 
    ? ((portfolio.peakEquityUsdt - equity) / portfolio.peakEquityUsdt) * 100 
    : 0;

  await storage.updatePortfolio({
    unrealizedPnlUsdt: unrealizedPnl,
  });
}

let monitorIntervalId: ReturnType<typeof setInterval> | null = null;
let monitorRunning = false;

const latestCycleSignals = new Map<string, { signal: NeuralSignalData; timestamp: number }>();
const SIGNAL_STALE_MS = 30 * 60 * 1000;
let neuralPmEvalCount = 0;

export function updateCachedSignal(symbol: string, signal: NeuralSignalData): void {
  latestCycleSignals.set(symbol, { signal, timestamp: Date.now() });
}

export function getCachedSignal(symbol: string): NeuralSignalData | null {
  const entry = latestCycleSignals.get(symbol);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > SIGNAL_STALE_MS) {
    latestCycleSignals.delete(symbol);
    return null;
  }
  return entry.signal;
}

export async function monitorAllPositions(): Promise<void> {
  if (monitorRunning) return;
  monitorRunning = true;

  try {
    const openPositions = await storage.getPositions("OPEN", 100);
    if (openPositions.length === 0) return;

    const { broadcast } = await import("../ws");

    for (const position of openPositions) {
      try {
        const currentPrice = await getCurrentMarketPrice(position.symbol);
        if (!currentPrice || currentPrice === 0) continue;

        const syntheticCandle = {
          timestamp: Date.now(),
          open: currentPrice,
          high: currentPrice,
          low: currentPrice,
          close: currentPrice,
          volume: 0,
        };

        const syntheticCtx: TradeContext = {
          candle: syntheticCandle,
          markPrice: currentPrice,
          fundingRate: 0,
          atr: position.initialStopDistance ?? 100,
          kalmanFast: currentPrice,
          shotPlan: null,
        };

        const slHit = checkStopLoss(position, syntheticCandle);
        const tp1Hit = checkTp1(position, syntheticCandle);
        const tp2Hit = checkTp2(position, syntheticCandle);

        if (slHit) {
          console.log(`[Position Monitor] SL HIT: ${position.symbol} ${position.side} @ ${currentPrice} (SL: ${position.stopLoss})`);
          await closePosition(position, position.stopLoss!, "SL", syntheticCtx);
          broadcast("TRADE_CLOSE", { positionId: position.id, symbol: position.symbol, reason: "SL", exitPrice: position.stopLoss });
          continue;
        }

        if (tp1Hit && position.tp1) {
          if (tp2Hit && position.tp2) {
            console.log(`[Position Monitor] TP2 HIT: ${position.symbol} ${position.side} @ ${currentPrice} (TP2: ${position.tp2})`);
            await closePosition(position, position.tp2, "TP2", syntheticCtx);
            broadcast("TRADE_CLOSE", { positionId: position.id, symbol: position.symbol, reason: "TP2", exitPrice: position.tp2 });
          } else {
            console.log(`[Position Monitor] TP1 HIT: ${position.symbol} ${position.side} @ ${currentPrice} (TP1: ${position.tp1})`);
            await closePosition(position, position.tp1, "TP1", syntheticCtx);
            broadcast("TRADE_CLOSE", { positionId: position.id, symbol: position.symbol, reason: "TP1", exitPrice: position.tp1 });
          }
          continue;
        }

        const unrealizedPnl = calculateUnrealizedPnl(position, currentPrice);

        const currentPeakProfit = position.peakProfit ?? 0;
        const newPeakProfit = Math.max(currentPeakProfit, unrealizedPnl);
        if (newPeakProfit > currentPeakProfit) {
          await storage.updatePosition(position.id, { peakProfit: newPeakProfit });
        }

        const cachedSignal = getCachedSignal(position.symbol);
        if (cachedSignal) {
          try {
            const freshPosition = newPeakProfit > currentPeakProfit
              ? { ...position, peakProfit: newPeakProfit }
              : position;
            const signalWithLivePrice: NeuralSignalData = {
              ...cachedSignal,
              price: currentPrice,
            };
            const neuralResult = await neuralPositionManager(freshPosition, signalWithLivePrice);
            neuralPmEvalCount++;
            if (neuralResult) {
              console.log(`[Position Monitor → Neural PM] ${position.symbol}: ${neuralResult.adjustmentType} — ${neuralResult.reason}`);
              if (neuralResult.positionClosed) continue;
            }
          } catch (neuralErr: any) {
            console.error(`[Position Monitor → Neural PM] Error on ${position.symbol}:`, neuralErr.message);
          }
        }
      } catch (err) {
        console.error(`[Position Monitor] Error checking ${position.symbol}:`, err);
      }
    }
    if (neuralPmEvalCount > 0 && neuralPmEvalCount % 30 === 0) {
      console.log(`[Position Monitor → Neural PM] ${neuralPmEvalCount} evaluations completed (${latestCycleSignals.size} symbols cached)`);
    }
  } finally {
    monitorRunning = false;
  }
}

export function startPositionMonitor(intervalMs: number = 1000): void {
  if (monitorIntervalId) return;
  console.log(`[Position Monitor] Started — checking all open positions every ${intervalMs / 1000}s`);
  monitorIntervalId = setInterval(() => {
    monitorAllPositions().catch(err => console.error("[Position Monitor] Error:", err));
  }, intervalMs);
}

export function stopPositionMonitor(): void {
  if (monitorIntervalId) {
    clearInterval(monitorIntervalId);
    monitorIntervalId = null;
    console.log("[Position Monitor] Stopped");
  }
}

/**
 * Monte Carlo Simulation for Backtest Confidence Intervals
 * Research-backed technique to assess statistical significance of performance
 * Shuffles trade sequence to generate distribution of possible outcomes
 */
function runMonteCarloSimulation(
  returns: number[], 
  startingEquity: number, 
  numSimulations: number = 1000
): {
  medianFinalEquity: number;
  p5FinalEquity: number;
  p95FinalEquity: number;
  medianMaxDrawdown: number;
  p95MaxDrawdown: number;
  confidenceLevel: string;
  isStatisticallySignificant: boolean;
} | null {
  if (returns.length < 10) {
    return null; // Not enough trades for meaningful simulation
  }

  const finalEquities: number[] = [];
  const maxDrawdowns: number[] = [];

  for (let sim = 0; sim < numSimulations; sim++) {
    // Shuffle returns (Fisher-Yates shuffle)
    const shuffled = [...returns];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    // Simulate equity curve with shuffled returns
    let equity = startingEquity;
    let peak = equity;
    let maxDD = 0;

    for (const ret of shuffled) {
      equity *= (1 + ret / 100);
      if (equity > peak) peak = equity;
      const dd = (peak - equity) / peak * 100;
      if (dd > maxDD) maxDD = dd;
    }

    finalEquities.push(equity);
    maxDrawdowns.push(maxDD);
  }

  // Sort for percentile calculation
  finalEquities.sort((a, b) => a - b);
  maxDrawdowns.sort((a, b) => a - b);

  const p5Idx = Math.floor(numSimulations * 0.05);
  const p50Idx = Math.floor(numSimulations * 0.50);
  const p95Idx = Math.floor(numSimulations * 0.95);

  const medianFinalEquity = finalEquities[p50Idx];
  const p5FinalEquity = finalEquities[p5Idx];
  const p95FinalEquity = finalEquities[p95Idx];
  const medianMaxDrawdown = maxDrawdowns[p50Idx];
  const p95MaxDrawdown = maxDrawdowns[p95Idx];

  // Statistical significance: 5th percentile should still be profitable
  const isStatisticallySignificant = p5FinalEquity > startingEquity;

  // Confidence level based on percentile spread
  const spread = (p95FinalEquity - p5FinalEquity) / medianFinalEquity;
  let confidenceLevel: string;
  if (spread < 0.1) confidenceLevel = "Very High (narrow distribution)";
  else if (spread < 0.2) confidenceLevel = "High";
  else if (spread < 0.4) confidenceLevel = "Moderate";
  else confidenceLevel = "Low (wide distribution - needs more trades)";

  return {
    medianFinalEquity,
    p5FinalEquity,
    p95FinalEquity,
    medianMaxDrawdown,
    p95MaxDrawdown,
    confidenceLevel,
    isStatisticallySignificant
  };
}

export async function getPortfolioSummary() {
  const portfolio = await storage.getOrCreatePortfolio();
  const openPositions = await storage.getPositions("OPEN", 100);
  const config = getConfig();
  
  let unrealizedPnl = 0;
  let unrealizedPnlR = 0;
  let exposure = 0;
  
  for (const pos of openPositions) {
    const currentPrice = await getCurrentMarketPrice(pos.symbol);
    if (currentPrice > 0) {
      const priceDiff = pos.side === "LONG"
        ? currentPrice - pos.entryPrice
        : pos.entryPrice - currentPrice;
      const positionPnl = priceDiff * pos.qty;
      unrealizedPnl += positionPnl;
      if (pos.initialRiskUsdt) {
        unrealizedPnlR += positionPnl / pos.initialRiskUsdt;
      }
    }
    exposure += pos.notionalUsdt;
  }

  const totalTrades = portfolio.totalTrades ?? 0;
  const winningTrades = portfolio.winningTrades ?? 0;
  const losingTrades = portfolio.losingTrades ?? 0;
  const realizedPnl = portfolio.realizedPnlUsdt ?? 0;
  
  const winRate = totalTrades > 0 
    ? (winningTrades / totalTrades) * 100 
    : 0;

  const closedPositions = await storage.getPositions("CLOSED", 1000);
  
  let totalWinPct = 0;
  let totalLossPct = 0;
  let bestTrade = 0;
  let worstTrade = 0;
  let grossProfit = 0;
  let grossLoss = 0;
  const returns: number[] = [];

  for (const pos of closedPositions) {
    const pnl = pos.realizedPnlUsdt ?? 0;
    const pct = pos.notionalUsdt > 0 ? (pnl / pos.notionalUsdt) * 100 : 0;
    returns.push(pct);
    
    if (pnl > 0) {
      totalWinPct += pct;
      grossProfit += pnl;
      if (pct > bestTrade) bestTrade = pct;
    } else {
      totalLossPct += Math.abs(pct);
      grossLoss += Math.abs(pnl);
      if (pct < worstTrade) worstTrade = pct;
    }
  }

  const avgWin = winningTrades > 0 ? totalWinPct / winningTrades : 0;
  const avgLoss = losingTrades > 0 ? totalLossPct / losingTrades : 0;
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
  
  const expectancy = totalTrades > 0
    ? (winRate / 100) * avgWin - ((100 - winRate) / 100) * avgLoss
    : 0;

  let sharpe = 0;
  let sharpeWarning: string | null = null;
  if (returns.length > 1) {
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    const std = Math.sqrt(variance);
    sharpe = std > 0 ? (mean / std) * Math.sqrt(252) : 0;
    
    // SHARPE RATIO SANITY CHECK (Research-backed overfitting indicator)
    // Sharpe > 3 is extremely rare in production; signals potential overfitting
    // Real-world institutional strategies rarely exceed Sharpe 2.0-2.5
    if (sharpe > 3.0 && returns.length >= 20) {
      sharpeWarning = `WARNING: Sharpe ratio ${sharpe.toFixed(2)} exceeds 3.0 - potential overfitting detected`;
      console.warn(`[Paper Trading] ${sharpeWarning}`);
    } else if (sharpe > 2.5 && returns.length >= 20) {
      sharpeWarning = `CAUTION: Sharpe ratio ${sharpe.toFixed(2)} is unusually high - verify with out-of-sample testing`;
    }
  }

  const equity = portfolio.currentEquityUsdt + unrealizedPnl;
  const startingEquity = portfolio.startingEquityUsdt;
  const totalPnlUsdt = realizedPnl + unrealizedPnl;
  const totalPnlR = startingEquity > 0 ? totalPnlUsdt / (startingEquity * 0.01) : 0;
  const maxDrawdownR = startingEquity > 0 ? ((portfolio.maxDrawdownPct ?? 0) / 100) * startingEquity / (startingEquity * 0.01) : 0;
  const peak = portfolio.peakEquityUsdt;
  const currentDrawdown = peak > 0 ? ((peak - equity) / peak) * 100 : 0;
  const exposurePct = equity > 0 ? (exposure / equity) * 100 : 0;

  return {
    equity,
    currentEquity: equity,
    startingEquity,
    availableBalance: portfolio.availableBalanceUsdt,
    realizedPnl,
    unrealizedPnl,
    unrealizedPnlR,
    totalPnl: totalPnlUsdt,
    totalPnlUsdt,
    totalPnlR: Math.round(totalPnlR * 100) / 100,
    maxDrawdown: portfolio.maxDrawdownPct ?? 0,
    maxDrawdownR: Math.round(maxDrawdownR * 100) / 100,
    currentDrawdown,
    totalTrades,
    winningTrades,
    losingTrades,
    winRate,
    avgWin,
    avgLoss,
    sharpe,
    sharpeWarning,
    expectancy,
    bestTrade,
    worstTrade,
    profitFactor: profitFactor === Infinity ? 999.99 : profitFactor,
    exposure,
    exposurePct,
    isAutoTrading: config.isAutoTrading,
    isPaperTradingEnabled: config.paperTradingEnabled,
    openPositions: openPositions.length,
    recentAuditLogs: auditLog.slice(-10),
    // Monte Carlo confidence intervals for backtest validation
    // Filter NaN/undefined values to ensure clean data for simulation
    monteCarloStats: runMonteCarloSimulation(
      returns.filter(r => typeof r === 'number' && !isNaN(r) && isFinite(r)), 
      portfolio.startingEquityUsdt
    ),
  };
}

export async function manualClosePosition(positionId: number, exitPrice?: number): Promise<{ pnl: number; isPartial: boolean }> {
  const position = await storage.getPositionById(positionId);
  if (!position) throw new Error(`Position ${positionId} not found`);
  if (position.status !== "OPEN") throw new Error(`Position ${positionId} is already ${position.status}`);

  const marketPrice = await getCurrentMarketPrice(position.symbol);
  const price = exitPrice ?? (marketPrice > 0 ? marketPrice : position.entryPrice);
  const ctx: TradeContext = {
    candle: { timestamp: Date.now(), open: price, high: price, low: price, close: price, volume: 0 } as any,
    markPrice: price,
    fundingRate: 0,
    atr: Math.abs(position.entryPrice - (position.stopLoss ?? position.entryPrice)) / 1.5,
    kalmanFast: price,
    shotPlan: null,
  };

  const result = await closePosition(position, price, "MANUAL", ctx);
  const { broadcast } = await import("../ws");
  broadcast("TRADE_CLOSE", {
    positionId: position.id,
    symbol: position.symbol,
    side: position.side,
    entryPrice: position.entryPrice,
    exitPrice: price,
    pnl: result.pnl,
    reason: "MANUAL",
  });
  return result;
}

export async function manualPartialClose(positionId: number, percent: number): Promise<{ pnl: number; isPartial: boolean }> {
  const position = await storage.getPositionById(positionId);
  if (!position) throw new Error(`Position ${positionId} not found`);
  if (position.status !== "OPEN") throw new Error(`Position ${positionId} is already ${position.status}`);
  if (percent <= 0 || percent >= 100) throw new Error("Percent must be between 1 and 99");

  const marketPrice = await getCurrentMarketPrice(position.symbol);
  const price = marketPrice > 0 ? marketPrice : position.entryPrice;
  const partialQty = position.qty * (percent / 100);
  const ctx: TradeContext = {
    candle: { timestamp: Date.now(), open: price, high: price, low: price, close: price, volume: 0 } as any,
    markPrice: price,
    fundingRate: 0,
    atr: Math.abs(position.entryPrice - (position.stopLoss ?? position.entryPrice)) / 1.5,
    kalmanFast: price,
    shotPlan: null,
  };

  const result = await closePosition(position, price, "MANUAL", ctx, partialQty);
  const { broadcast } = await import("../ws");
  broadcast("TRADE_UPDATE", {
    positionId: position.id,
    symbol: position.symbol,
    side: position.side,
    action: "PARTIAL_CLOSE",
    percent,
    pnl: result.pnl,
  });
  return result;
}

export async function updatePositionLevels(
  positionId: number,
  updates: { stopLoss?: number; tp1?: number; tp2?: number }
): Promise<PaperPosition> {
  const position = await storage.getPositionById(positionId);
  if (!position) throw new Error(`Position ${positionId} not found`);
  if (position.status !== "OPEN") throw new Error(`Position ${positionId} is already ${position.status}`);

  if (updates.stopLoss !== undefined) {
    if (position.side === "LONG" && updates.stopLoss >= position.entryPrice) {
      throw new Error("LONG stop loss must be below entry price");
    }
    if (position.side === "SHORT" && updates.stopLoss <= position.entryPrice) {
      throw new Error("SHORT stop loss must be above entry price");
    }
  }

  if (updates.tp1 !== undefined) {
    if (position.side === "LONG" && updates.tp1 <= position.entryPrice) {
      throw new Error("LONG TP must be above entry price");
    }
    if (position.side === "SHORT" && updates.tp1 >= position.entryPrice) {
      throw new Error("SHORT TP must be below entry price");
    }
  }

  const dbUpdates: Partial<PaperPosition> = {};
  if (updates.stopLoss !== undefined) dbUpdates.stopLoss = updates.stopLoss;
  if (updates.tp1 !== undefined) dbUpdates.tp1 = updates.tp1;
  if (updates.tp2 !== undefined) dbUpdates.tp2 = updates.tp2;

  const updated = await storage.updatePosition(positionId, dbUpdates);
  const { broadcast } = await import("../ws");
  broadcast("TRADE_UPDATE", {
    positionId: updated.id,
    symbol: updated.symbol,
    side: updated.side,
    action: "LEVELS_UPDATED",
    stopLoss: updated.stopLoss,
    tp1: updated.tp1,
    tp2: updated.tp2,
  });
  return updated;
}

export function computeSignalLeverage(v5Score: number | null | undefined): number {
  const config = getConfig();
  if (!config.leverageEnabled || !v5Score || v5Score <= 0) return 1;
  
  const capped = Math.min(v5Score, 50);
  const sortedTiers = [...config.leverageTiers].sort((a, b) => b.minScore - a.minScore);
  for (const tier of sortedTiers) {
    if (capped >= tier.minScore) {
      return Math.min(tier.leverage, config.maxLeverage);
    }
  }
  return 1;
}

export async function manualOpenPosition(params: {
  symbol: string;
  side: "LONG" | "SHORT";
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  riskPercent: number;
  source?: string;
  signalConfidence?: number;
  v5Score?: number;
}): Promise<PaperPosition> {
  const { symbol, side, entryPrice, stopLoss, takeProfit, riskPercent, source = "manual", signalConfidence = null, v5Score } = params;
  const portfolio = await storage.getOrCreatePortfolio();
  const config = getConfig();

  if (side === "LONG" && stopLoss >= entryPrice) throw new Error("LONG SL must be below entry");
  if (side === "SHORT" && stopLoss <= entryPrice) throw new Error("SHORT SL must be above entry");
  if (side === "LONG" && takeProfit <= entryPrice) throw new Error("LONG TP must be above entry");
  if (side === "SHORT" && takeProfit >= entryPrice) throw new Error("SHORT TP must be below entry");

  const riskMultiplier = source === "v5_signal" ? computeSignalLeverage(v5Score) : 1;
  const stopDistance = Math.abs(entryPrice - stopLoss);
  const riskUsd = portfolio.currentEquityUsdt * (riskPercent / 100);
  const baseQty = riskUsd / stopDistance;
  const qty = baseQty * riskMultiplier;
  const notional = qty * entryPrice;
  const entryFee = calculateFee(notional, config.takerFeePct);
  // initialRiskUsdt must match the actual capital at risk (qty × stopDistance).
  // Storing riskUsd × riskMultiplier keeps R-math correct:
  //   net_R = realizedPnl / initialRiskUsdt → always ≈ ±1R at SL/TP.
  const initialRiskUsdt = riskUsd * riskMultiplier;

  const position = await storage.createPosition({
    symbol,
    side,
    status: "OPEN",
    entryTs: Date.now(),
    entryPrice,
    qty,
    notionalUsdt: notional,
    leverage: riskMultiplier,
    stopLoss,
    tp1: takeProfit,
    tp2: null,
    trailMode: "none",
    trailPrice: null,
    timeStopBars: 60,
    barsOpen: 0,
    primaryHorizon: 15,
    initialRiskUsdt,
    feesPaidUsdt: entryFee,
    fundingPaidUsdt: 0,
    exitTs: null,
    exitPrice: null,
    realizedPnlUsdt: null,
    exitReason: null,
    signalConfidence: signalConfidence,
    signalEdge: null,
    v5Score: v5Score ?? null,
    peakProfit: 0,
    initialStopDistance: stopDistance,
    regime: null,
    source: source,
  });

  const { broadcast } = await import("../ws");
  broadcast("TRADE_OPEN", {
    positionId: position.id,
    symbol,
    side,
    entryPrice,
    stopLoss,
    takeProfit,
    qty,
    riskUsd,
    source,
  });

  const label = source === "v5_signal" ? "Auto-Trade" : "Paper";
  console.log(`[${label}] ${side} ${symbol} @ ${entryPrice} | SL: ${stopLoss} | TP: ${takeProfit} | Risk: $${riskUsd.toFixed(2)}`);
  return position;
}

export interface NeuralSignalData {
  v5Score: number | null;
  pHold: number | null;
  pLong: number | null;
  pShort: number | null;
  retMu: number | null;
  mfePred: number | null;
  maePred: number | null;
  v5Side: string | null;
  price: number;
}

export interface NeuralAdjustmentResult {
  action: string;
  adjustmentType: string;
  previousSl?: number;
  newSl?: number;
  reason: string;
  positionClosed: boolean;
  exitReason?: ExitReason;
}

export async function neuralPositionManager(
  position: PaperPosition,
  signal: NeuralSignalData
): Promise<NeuralAdjustmentResult | null> {
  if (position.status !== "OPEN") return null;
  if (!position.qty || position.qty <= 0) return null;

  const { broadcast } = await import("../ws");
  const { neuralAdjustments } = await import("@shared/schema");

  const propagateToBybit = async (action: "amend" | "close", params?: { stopLoss?: number; takeProfit?: number }) => {
    try {
      const { isLiveTradingEnabled, amendLiveSLTP, closeLivePosition } = await import("../bybit/live-engine");
      if (!isLiveTradingEnabled()) return;
      if (action === "close") {
        const result = await closeLivePosition(position.symbol);
        if (result.success) {
          console.log(`[Neural PM → Bybit] Closed ${position.symbol} on exchange`);
        } else if (result.error && !result.error.includes("No open position")) {
          console.warn(`[Neural PM → Bybit] Close ${position.symbol} failed: ${result.error}`);
        }
      } else if (action === "amend" && params) {
        const result = await amendLiveSLTP(position.symbol, params);
        if (result.success) {
          console.log(`[Neural PM → Bybit] Amended ${position.symbol} SL/TP on exchange`);
        } else if (result.error && !result.error.includes("No open position")) {
          console.warn(`[Neural PM → Bybit] Amend ${position.symbol} failed: ${result.error}`);
        }
      }
    } catch (err: any) {
      console.warn(`[Neural PM → Bybit] Error propagating to exchange: ${err.message}`);
    }
  };

  const propagateToBitget = async (action: "close") => {
    try {
      const { isBitgetLiveTradingEnabled, closeBitgetLivePosition } = await import("../bitget/live-engine");
      if (!isBitgetLiveTradingEnabled()) return;
      if (action === "close") {
        const result = await closeBitgetLivePosition(position.symbol);
        if (result.success) {
          console.log(`[Neural PM → Bitget] Closed ${position.symbol} on exchange`);
        } else if (result.error && !result.error.includes("No open position")) {
          console.warn(`[Neural PM → Bitget] Close ${position.symbol} failed: ${result.error}`);
        }
      }
    } catch (err: any) {
      console.warn(`[Neural PM → Bitget] Error propagating to exchange: ${err.message}`);
    }
  };

  const currentPrice = signal.price;
  if (!currentPrice || currentPrice <= 0) return null;

  const riskUsdt = Math.max(position.initialRiskUsdt ?? 1, 0.01);
  const pnlUsdt = calculateUnrealizedPnl(position, currentPrice);
  const pnlR = pnlUsdt / riskUsdt;
  const peakProfitR = (position.peakProfit ?? 0) / riskUsdt;

  const v5Score = signal.v5Score ?? 0;
  const pHold = signal.pHold ?? 0;
  const retMu = signal.retMu ?? 0;
  const v5Side = signal.v5Side ?? "";
  const pLong = signal.pLong ?? 0;
  const pShort = signal.pShort ?? 0;

  const recordAdjustment = async (type: string, prevSl: number | null, newSl: number | null, reason: string) => {
    try {
      await db.insert(neuralAdjustments).values({
        positionId: position.id,
        symbol: position.symbol,
        timestamp: Date.now(),
        adjustmentType: type,
        previousSl: prevSl,
        newSl: newSl,
        v5Score: signal.v5Score,
        pHold: signal.pHold,
        pLong: signal.pLong,
        pShort: signal.pShort,
        retMu: signal.retMu,
        positionPnlR: Math.round(pnlR * 10000) / 10000,
        reason,
      });
    } catch (err) {
      console.error(`[Neural PM] Failed to record adjustment:`, err);
    }
  };

  const sideIsLong = position.side === "LONG";
  const modelSide = v5Side.toUpperCase();
  const directionFlipped = (sideIsLong && modelSide === "SHORT") || (!sideIsLong && modelSide === "LONG");

  const refetchAndClose = async (exitReason: ExitReason, reason: string, adjType: string): Promise<NeuralAdjustmentResult | null> => {
    const fresh = await storage.getPositionById(position.id);
    if (!fresh || fresh.status !== "OPEN") {
      console.log(`[Neural PM] ${position.symbol} — position already closed, skipping ${adjType}`);
      return null;
    }
    const syntheticCtx: TradeContext = {
      candle: { timestamp: Date.now(), open: currentPrice, high: currentPrice, low: currentPrice, close: currentPrice, volume: 0 },
      markPrice: currentPrice,
      fundingRate: 0,
      atr: position.initialStopDistance ?? 100,
      kalmanFast: currentPrice,
      shotPlan: null,
    };
    await closePosition(fresh, currentPrice, exitReason, syntheticCtx);
    broadcast("TRADE_CLOSE", { positionId: position.id, symbol: position.symbol, reason: exitReason, exitPrice: currentPrice });
    await recordAdjustment(adjType, position.stopLoss, null, reason);
    await propagateToBybit("close");
    await propagateToBitget("close");
    return { action: "CLOSE", adjustmentType: adjType, reason, positionClosed: true, exitReason };
  };

  if (directionFlipped && (pLong > 0.5 || pShort > 0.5)) {
    const oppositeProb = sideIsLong ? pShort : pLong;
    if (oppositeProb > 0.55) {
      const reason = `Direction FLIP: model now says ${modelSide} (p=${oppositeProb.toFixed(3)}) while position is ${position.side}. PnL: ${pnlR.toFixed(2)}R`;
      console.log(`[Neural PM] ${position.symbol} — ${reason}`);
      return await refetchAndClose("NEURAL_FLIP", reason, "DIRECTION_FLIP_EXIT");
    }
  }

  // NEW: Low conviction exit — model hasn't fully flipped but has lost belief; lock in any profit
  const pSide = sideIsLong ? pLong : pShort;
  if (pSide < 0.30 && pnlR >= 0.3) {
    const reason = `Low conviction exit: model p_${sideIsLong ? "long" : "short"}=${pSide.toFixed(3)} (conviction lost) while position at ${pnlR.toFixed(2)}R profit. Locking in.`;
    console.log(`[Neural PM] ${position.symbol} — ${reason}`);
    return await refetchAndClose("NEURAL_LOW_CONVICTION", reason, "LOW_CONVICTION_EXIT");
  }

  if (pnlR >= 2.0 && retMu < -0.001) {
    const reason = `MFE Protection: position at ${pnlR.toFixed(2)}R profit but model predicts negative return (ret_mu=${retMu.toFixed(5)}). Locking in profit.`;
    console.log(`[Neural PM] ${position.symbol} — ${reason}`);
    return await refetchAndClose("NEURAL_MFE", reason, "MFE_PROTECTION_EXIT");
  }

  // NEW: Aggressive MFE protection — at 3R+ profit with meaningfully reduced side conviction
  if (pnlR >= 3.0 && pSide < 0.50) {
    const reason = `Aggressive MFE exit: position at ${pnlR.toFixed(2)}R profit, model conviction dropped to p_side=${pSide.toFixed(3)}. Securing 3R+ gain.`;
    console.log(`[Neural PM] ${position.symbol} — ${reason}`);
    return await refetchAndClose("NEURAL_MFE_AGGRESSIVE", reason, "MFE_PROTECTION_EXIT");
  }

  // NEW: Neural TP extension — mfePred says there's more room than current TP allows; widen TP
  const mfePred = signal.mfePred ?? 0;
  if (mfePred > 1.5 && pSide > 0.75 && pnlR >= 0.2 && position.tp1 && position.initialStopDistance && position.qty > 0) {
    const stopDist = position.initialStopDistance;
    const currentTpDistR = Math.abs(position.tp1 - position.entryPrice) / stopDist;
    const targetTpDistR = Math.min(mfePred * 0.85, currentTpDistR * 2.5);  // extend, but cap at 2.5x original TP
    if (targetTpDistR > currentTpDistR + 0.3) {
      const newTp = sideIsLong
        ? position.entryPrice + (targetTpDistR * stopDist)
        : position.entryPrice - (targetTpDistR * stopDist);
      const prevTp = position.tp1;
      await storage.updatePosition(position.id, { tp1: newTp });
      broadcast("TRADE_UPDATE", { positionId: position.id, symbol: position.symbol, side: position.side, action: "NEURAL_ADJUST", tp1: newTp, adjustmentType: "TP_EXTENSION" });
      await recordAdjustment("TP_EXTENSION", position.stopLoss, null, `Neural TP extended: mfePred=${mfePred.toFixed(2)}R, p_side=${pSide.toFixed(3)}, old_TP=${prevTp.toFixed(4)}, new_TP=${newTp.toFixed(4)}`);
      await propagateToBybit("amend", { takeProfit: newTp });
      const reason = `Neural TP extension: mfePred=${mfePred.toFixed(2)}R > currentTP=${currentTpDistR.toFixed(2)}R, p_side=${pSide.toFixed(3)}. Extended TP from ${prevTp.toFixed(4)} → ${newTp.toFixed(4)}`;
      console.log(`[Neural PM] ${position.symbol} — ${reason}`);
      return { action: "EXTEND_TP", adjustmentType: "TP_EXTENSION", reason, positionClosed: false };
    }
  }

  if (pHold > 0.6 && pnlR >= 0.5) {
    const config = getConfig();
    const feeBuffer = position.entryPrice * getTotalCostsPct() * 1.2;
    const breakevenSl = sideIsLong
      ? position.entryPrice + feeBuffer + (position.entryPrice * 0.001)
      : position.entryPrice - feeBuffer - (position.entryPrice * 0.001);
    const currentSl = position.stopLoss ?? 0;
    const BE_EPS = 1e-4;
    const shouldTighten = sideIsLong
      ? breakevenSl > currentSl + BE_EPS
      : breakevenSl < currentSl - BE_EPS;

    if (shouldTighten) {
      const reason = `Confidence decay: p_hold=${pHold.toFixed(3)} (model says HOLD now) while position at ${pnlR.toFixed(2)}R. Moving SL to breakeven+buffer.`;
      console.log(`[Neural PM] ${position.symbol} — ${reason}`);

      const prevSl = position.stopLoss;
      await storage.updatePosition(position.id, { stopLoss: breakevenSl });
      broadcast("TRADE_UPDATE", { positionId: position.id, symbol: position.symbol, side: position.side, action: "NEURAL_ADJUST", stopLoss: breakevenSl, adjustmentType: "CONFIDENCE_DECAY" });
      await recordAdjustment("CONFIDENCE_DECAY_TIGHTEN", prevSl, breakevenSl, reason);
      await propagateToBybit("amend", { stopLoss: breakevenSl });

      return { action: "TIGHTEN_SL", adjustmentType: "CONFIDENCE_DECAY_TIGHTEN", previousSl: prevSl ?? undefined, newSl: breakevenSl, reason, positionClosed: false };
    }
  }

  if (pnlR >= 1.0 && position.stopLoss) {
    const feeBuffer = position.entryPrice * getTotalCostsPct() * 1.2;
    const breakevenPrice = sideIsLong
      ? position.entryPrice + feeBuffer
      : position.entryPrice - feeBuffer;
    const currentSl = position.stopLoss;
    const BE_EPS = 1e-4;
    const slBelowBE = sideIsLong ? currentSl < breakevenPrice - BE_EPS : currentSl > breakevenPrice + BE_EPS;

    if (slBelowBE) {
      const reason = `Breakeven move: position at ${pnlR.toFixed(2)}R profit. Moving SL to breakeven ($${breakevenPrice.toFixed(2)}).`;
      console.log(`[Neural PM] ${position.symbol} — ${reason}`);

      await storage.updatePosition(position.id, { stopLoss: breakevenPrice });
      broadcast("TRADE_UPDATE", { positionId: position.id, symbol: position.symbol, side: position.side, action: "NEURAL_ADJUST", stopLoss: breakevenPrice, adjustmentType: "BREAKEVEN" });
      await recordAdjustment("BREAKEVEN", currentSl, breakevenPrice, reason);
      await propagateToBybit("amend", { stopLoss: breakevenPrice });

      return { action: "MOVE_BE", adjustmentType: "BREAKEVEN", previousSl: currentSl, newSl: breakevenPrice, reason, positionClosed: false };
    }
  }

  if (pnlR >= 1.5 && position.stopLoss && peakProfitR > 0) {
    const givebackPct = v5Score >= 0.15 ? 0.40 : v5Score >= 0.05 ? 0.30 : 0.20;
    const stopDistance = position.initialStopDistance ?? Math.abs(position.entryPrice - (position.stopLoss ?? position.entryPrice));
    const peakPnlPrice = sideIsLong
      ? position.entryPrice + (peakProfitR * (riskUsdt / position.qty))
      : position.entryPrice - (peakProfitR * (riskUsdt / position.qty));
    const givebackAmount = Math.abs(peakPnlPrice - position.entryPrice) * givebackPct;
    const adaptiveTrail = sideIsLong
      ? peakPnlPrice - givebackAmount
      : peakPnlPrice + givebackAmount;

    const currentSl = position.stopLoss;
    const trailBetter = sideIsLong ? adaptiveTrail > currentSl : adaptiveTrail < currentSl;

    if (trailBetter) {
      const reason = `Adaptive trail (${(givebackPct*100).toFixed(0)}% giveback): v5Score=${v5Score.toFixed(3)}, peak=${peakProfitR.toFixed(2)}R, trail=$${adaptiveTrail.toFixed(2)}`;
      console.log(`[Neural PM] ${position.symbol} — ${reason}`);

      await storage.updatePosition(position.id, { stopLoss: adaptiveTrail, trailMode: "neural" });
      broadcast("TRADE_UPDATE", { positionId: position.id, symbol: position.symbol, side: position.side, action: "NEURAL_ADJUST", stopLoss: adaptiveTrail, adjustmentType: "ADAPTIVE_TRAIL" });
      await recordAdjustment("ADAPTIVE_TRAIL", currentSl, adaptiveTrail, reason);
      await propagateToBybit("amend", { stopLoss: adaptiveTrail });

      return { action: "TRAIL", adjustmentType: "ADAPTIVE_TRAIL", previousSl: currentSl, newSl: adaptiveTrail, reason, positionClosed: false };
    }
  }

  return null;
}

export interface PositionHealth {
  score: number;
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  reason: string;
  factors: {
    pnlScore: number;
    slTpRatioScore: number;
    modelConfidenceScore: number;
    timeScore: number;
    mfeTrendScore: number;
  };
  currentPnlR: number;
  peakPnlR: number;
  giveback: number;
  latestV5Score: number | null;
  latestAdjustment: string | null;
}

export async function computePositionHealth(
  position: PaperPosition,
  latestSignal?: NeuralSignalData | null
): Promise<PositionHealth> {
  const currentPrice = await getCurrentMarketPrice(position.symbol) || latestSignal?.price || 0;
  if (!currentPrice || currentPrice <= 0) {
    return { score: 50, riskLevel: "MEDIUM", reason: "Unable to determine current price", factors: { pnlScore: 50, slTpRatioScore: 50, modelConfidenceScore: 50, timeScore: 50, mfeTrendScore: 50 }, currentPnlR: 0, peakPnlR: 0, giveback: 0, latestV5Score: null, latestAdjustment: null };
  }
  const riskUsdt = Math.max(position.initialRiskUsdt ?? 1, 0.01);
  const pnlUsdt = calculateUnrealizedPnl(position, currentPrice);
  const pnlR = riskUsdt > 0 ? pnlUsdt / riskUsdt : 0;
  const storedPeakR = riskUsdt > 0 ? (position.peakProfit ?? 0) / riskUsdt : 0;
  const peakProfitR = Math.max(storedPeakR, pnlR);
  const giveback = peakProfitR > 0 ? (peakProfitR - pnlR) / peakProfitR : 0;

  let pnlScore = 50;
  if (pnlR >= 3) pnlScore = 95;
  else if (pnlR >= 2) pnlScore = 85;
  else if (pnlR >= 1) pnlScore = 70;
  else if (pnlR >= 0) pnlScore = 55;
  else if (pnlR >= -0.5) pnlScore = 35;
  else if (pnlR >= -1) pnlScore = 15;
  else pnlScore = 5;

  let slTpRatioScore = 50;
  if (position.stopLoss && position.tp1) {
    const distToSl = Math.abs(currentPrice - position.stopLoss);
    const distToTp = Math.abs(position.tp1 - currentPrice);
    const totalDist = distToSl + distToTp;
    if (totalDist > 0) {
      slTpRatioScore = Math.round((distToSl / totalDist) * 100);
    }
  }

  let modelConfidenceScore = 50;
  if (latestSignal) {
    const sideIsLong = position.side === "LONG";
    const friendlyProb = sideIsLong ? (latestSignal.pLong ?? 0) : (latestSignal.pShort ?? 0);
    const holdProb = latestSignal.pHold ?? 0;
    if (friendlyProb > 0.6) modelConfidenceScore = 90;
    else if (friendlyProb > 0.4) modelConfidenceScore = 65;
    else if (holdProb > 0.6) modelConfidenceScore = 30;
    else modelConfidenceScore = 15;

    const v5Side = (latestSignal.v5Side ?? "").toUpperCase();
    const flipped = (sideIsLong && v5Side === "SHORT") || (!sideIsLong && v5Side === "LONG");
    if (flipped) modelConfidenceScore = Math.max(0, modelConfidenceScore - 40);
  }

  let timeScore = 80;
  const holdHours = (Date.now() - position.entryTs) / (1000 * 60 * 60);
  if (holdHours > 8) timeScore = 20;
  else if (holdHours > 4) timeScore = 40;
  else if (holdHours > 2) timeScore = 60;

  let mfeTrendScore = 60;
  if (peakProfitR > 0) {
    if (giveback < 0.1) mfeTrendScore = 95;
    else if (giveback < 0.25) mfeTrendScore = 75;
    else if (giveback < 0.5) mfeTrendScore = 45;
    else mfeTrendScore = 15;
  }

  const weights = { pnl: 0.30, slTp: 0.20, model: 0.25, time: 0.10, mfe: 0.15 };
  const rawScore = Math.round(
    pnlScore * weights.pnl +
    slTpRatioScore * weights.slTp +
    modelConfidenceScore * weights.model +
    timeScore * weights.time +
    mfeTrendScore * weights.mfe
  );
  const score = Math.max(0, Math.min(100, rawScore));

  let riskLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" = "LOW";
  let reason = "";
  if (score >= 70) { riskLevel = "LOW"; reason = "Position is healthy"; }
  else if (score >= 45) { riskLevel = "MEDIUM"; reason = "Position needs monitoring"; }
  else if (score >= 25) { riskLevel = "HIGH"; reason = "Position at elevated risk"; }
  else { riskLevel = "CRITICAL"; reason = "Position in critical condition"; }

  if (giveback > 0.5 && peakProfitR > 1) reason += ` — giving back ${(giveback*100).toFixed(0)}% of peak ${peakProfitR.toFixed(1)}R profit`;
  if (pnlR < -1) reason += ` — down ${pnlR.toFixed(1)}R`;

  let latestAdjustment: string | null = null;
  try {
    const { neuralAdjustments } = await import("@shared/schema");
    const recent = await db.select()
      .from(neuralAdjustments)
      .where(eq(neuralAdjustments.positionId, position.id))
      .orderBy(desc(neuralAdjustments.timestamp))
      .limit(1);
    if (recent.length > 0) {
      latestAdjustment = recent[0].adjustmentType;
    }
  } catch {}

  return {
    score,
    riskLevel,
    reason,
    factors: {
      pnlScore,
      slTpRatioScore,
      modelConfidenceScore,
      timeScore,
      mfeTrendScore,
    },
    currentPnlR: Math.round(pnlR * 10000) / 10000,
    peakPnlR: Math.round(peakProfitR * 10000) / 10000,
    giveback: Math.round(giveback * 10000) / 10000,
    latestV5Score: (() => {
      const raw = latestSignal?.v5Score ?? position.v5Score ?? null;
      if (raw == null) return null;
      return Math.min(Math.max(raw, 0), 1);
    })(),
    latestAdjustment,
  };
}

export interface RiskAlert {
  id: string;
  type: "SL_PROXIMITY" | "DAILY_LOSS_CAP" | "DRAWDOWN" | "POSITION_DURATION" | "HIGH_EXPOSURE";
  severity: "info" | "warning" | "critical";
  symbol?: string;
  message: string;
  data: Record<string, unknown>;
}

export async function computeRiskAlerts(): Promise<RiskAlert[]> {
  const alerts: RiskAlert[] = [];
  const portfolio = await storage.getOrCreatePortfolio();
  const openPositions = await storage.getPositions("OPEN", 50);

  const priceCache = new Map<string, number>();
  for (const pos of openPositions) {
    if (!priceCache.has(pos.symbol)) {
      priceCache.set(pos.symbol, await getCurrentMarketPrice(pos.symbol));
    }
  }

  for (const pos of openPositions) {
    const currentPrice = priceCache.get(pos.symbol) || pos.entryPrice;

    if (pos.stopLoss) {
      const priceToSl = pos.side === "LONG"
        ? currentPrice - pos.stopLoss
        : pos.stopLoss - currentPrice;
      const entryToSl = Math.abs(pos.entryPrice - pos.stopLoss);
      const distanceRatio = entryToSl > 0 ? priceToSl / entryToSl : 1;

      if (distanceRatio < 0.15) {
        alerts.push({
          id: `sl-${pos.id}`,
          type: "SL_PROXIMITY",
          severity: "critical",
          symbol: pos.symbol,
          message: `${pos.symbol} ${pos.side} is very close to stop loss (${(distanceRatio * 100).toFixed(0)}% remaining)`,
          data: { positionId: pos.id, distanceRatio, stopLoss: pos.stopLoss, entryPrice: pos.entryPrice },
        });
      } else if (distanceRatio < 0.30) {
        alerts.push({
          id: `sl-${pos.id}`,
          type: "SL_PROXIMITY",
          severity: "warning",
          symbol: pos.symbol,
          message: `${pos.symbol} ${pos.side} approaching stop loss (${(distanceRatio * 100).toFixed(0)}% remaining)`,
          data: { positionId: pos.id, distanceRatio, stopLoss: pos.stopLoss, entryPrice: pos.entryPrice },
        });
      }
    }

    const durationMs = Date.now() - pos.entryTs;
    const durationHours = durationMs / (1000 * 60 * 60);
    if (durationHours > 4) {
      alerts.push({
        id: `dur-${pos.id}`,
        type: "POSITION_DURATION",
        severity: "info",
        symbol: pos.symbol,
        message: `${pos.symbol} ${pos.side} has been open for ${durationHours.toFixed(1)} hours`,
        data: { positionId: pos.id, durationHours },
      });
    }
  }

  const totalExposure = openPositions.reduce((sum, p) => sum + p.notionalUsdt, 0);
  const exposureRatio = portfolio.currentEquityUsdt > 0 ? totalExposure / portfolio.currentEquityUsdt : 0;
  if (exposureRatio > 0.5) {
    alerts.push({
      id: "exposure",
      type: "HIGH_EXPOSURE",
      severity: "warning",
      message: `Total exposure is ${(exposureRatio * 100).toFixed(0)}% of equity ($${totalExposure.toFixed(0)} / $${portfolio.currentEquityUsdt.toFixed(0)})`,
      data: { totalExposure, equity: portfolio.currentEquityUsdt, ratio: exposureRatio },
    });
  }

  const drawdownPct = portfolio.maxDrawdownPct ?? 0;
  if (drawdownPct > 10) {
    alerts.push({
      id: "drawdown",
      type: "DRAWDOWN",
      severity: "critical",
      message: `Drawdown has reached ${drawdownPct.toFixed(1)}% from peak equity`,
      data: { drawdownPct, peakEquity: portfolio.peakEquityUsdt, currentEquity: portfolio.currentEquityUsdt },
    });
  } else if (drawdownPct > 5) {
    alerts.push({
      id: "drawdown",
      type: "DRAWDOWN",
      severity: "warning",
      message: `Drawdown at ${drawdownPct.toFixed(1)}% from peak equity`,
      data: { drawdownPct, peakEquity: portfolio.peakEquityUsdt, currentEquity: portfolio.currentEquityUsdt },
    });
  }

  return alerts;
}
