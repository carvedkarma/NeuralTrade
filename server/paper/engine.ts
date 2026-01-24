import { getConfig, getTotalCostsPct, isPaperTradingEnabled, type PaperTradingConfig } from "./config";
import * as storage from "./storage";
import type { PaperPosition } from "@shared/schema";
import type { Candle } from "@shared/schema";
import type { ShotPlan } from "../signal-engine";

export type ExitReason = "SL" | "TP1" | "TP2" | "TRAIL" | "TIME" | "FLIP" | "MANUAL";

interface TradeContext {
  candle: Candle;
  markPrice: number;
  fundingRate: number;
  atr: number;
  kalmanFast: number;
  shotPlan: ShotPlan | null;
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
  exposureAfter: number;
  decision: "ALLOWED" | "BLOCKED";
  reason: string;
}

const auditLog: TradeAudit[] = [];

const EDGE_MULTIPLE_MIN = 1.5;

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
  console.log(`  Position Size: ${audit.positionSize.toFixed(6)}, Exposure After: ${(audit.exposureAfter * 100).toFixed(1)}%`);
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
  config: PaperTradingConfig
): { stopLoss: number; stopDistance: number } {
  const atrStop = atr * config.atrStopMultiplier;
  const minStopAbs = entryPrice * (config.minStopDistancePct / 100);
  const minCostStop = entryPrice * getTotalCostsPct() * 1.5;
  
  const stopDistance = Math.max(atrStop, minStopAbs, minCostStop);
  
  const stopLoss = side === "LONG" 
    ? entryPrice - stopDistance 
    : entryPrice + stopDistance;
    
  return { stopLoss, stopDistance };
}

function calculatePositionSize(
  equity: number,
  stopDistance: number,
  riskPct: number,
  maxRiskPct: number
): { qty: number; riskUsdt: number } {
  const actualRiskPct = Math.min(riskPct, maxRiskPct);
  const riskUsdt = equity * (actualRiskPct / 100);
  
  if (stopDistance <= 0) {
    return { qty: 0, riskUsdt: 0 };
  }
  
  const qty = riskUsdt / stopDistance;
  return { qty, riskUsdt };
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
  return barsOpen >= config.timeStopBars && pnlR < config.minPnlForTimeStop;
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
}

function checkShotPlanGating(shotPlan: ShotPlan | null, config: PaperTradingConfig): GatingResult {
  if (!shotPlan) {
    return { allowed: false, reason: "No shot plan available" };
  }
  
  if (shotPlan.signal === "HOLD") {
    console.log("[Paper ASSERTION] Attempted trade during HOLD signal - blocked");
    return { allowed: false, reason: "HOLD signal - no trade allowed" };
  }
  
  if (shotPlan.signal !== "LONG" && shotPlan.signal !== "SHORT") {
    return { allowed: false, reason: `Invalid signal: ${shotPlan.signal}` };
  }
  
  if (shotPlan.confidence < config.minConfidence) {
    return { 
      allowed: false, 
      reason: `Confidence ${(shotPlan.confidence * 100).toFixed(1)}% < ${(config.minConfidence * 100).toFixed(1)}% minimum` 
    };
  }
  
  const costs = shotPlan.estimatedCosts || getTotalCostsPct();
  if (shotPlan.edge <= costs) {
    return { 
      allowed: false, 
      reason: `Edge ${(shotPlan.edge * 100).toFixed(3)}% <= Costs ${(costs * 100).toFixed(3)}%` 
    };
  }
  
  const edgeMultiple = shotPlan.edgeMultiple ?? (costs > 0 ? shotPlan.edge / costs : 0);
  if (edgeMultiple < EDGE_MULTIPLE_MIN) {
    return { 
      allowed: false, 
      reason: `Edge multiple ${edgeMultiple.toFixed(2)}x < ${EDGE_MULTIPLE_MIN}x minimum (edge must be >= ${EDGE_MULTIPLE_MIN}x costs)` 
    };
  }
  
  if (shotPlan.regime === "chop") {
    return { allowed: false, reason: "Chop regime - no trades allowed" };
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
  
  if (shotPlan.vetoReasons && shotPlan.vetoReasons.length > 0) {
    return { allowed: false, reason: `Veto reasons: ${shotPlan.vetoReasons.join(", ")}` };
  }
  
  if (!shotPlan.reasons || shotPlan.reasons.length < 1) {
    return { allowed: false, reason: "No supporting reasons for trade" };
  }
  
  if (shotPlan.combinedIntelligence) {
    const ci = shotPlan.combinedIntelligence;
    
    if (ci.finalSignal === "HOLD") {
      return { 
        allowed: false, 
        reason: `Combined Intelligence: ${ci.vetoes.join("; ") || "Systems recommend HOLD"}` 
      };
    }
    
    if (ci.strategyEV <= 0) {
      return { 
        allowed: false, 
        reason: `Strategy Learner: Negative EV (${(ci.strategyEV * 100).toFixed(2)}%) for ${shotPlan.signal}` 
      };
    }
    
    if (!ci.systemsAgree && ci.patternWinRate < 0.5) {
      return { 
        allowed: false, 
        reason: `ML and Strategy disagree, pattern win rate only ${(ci.patternWinRate * 100).toFixed(0)}%` 
      };
    }
  }
  
  return { allowed: true, reason: "All gating checks passed (ML + Strategy Learner agree)" };
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
  tp1: number,
  tp2: number,
  confidence: number,
  edge: number
): Promise<PaperPosition | null> {
  const config = getConfig();
  const portfolio = await storage.getOrCreatePortfolio();
  
  const entryPrice = applySlippage(ctx.candle.open, side, config.slippageBps);
  
  const { stopLoss, stopDistance } = calculateStopDistance(
    entryPrice,
    ctx.atr,
    side,
    config
  );
  
  const { qty, riskUsdt } = calculatePositionSize(
    portfolio.currentEquityUsdt,
    stopDistance,
    config.riskPerTradePct,
    config.maxRiskPerTradePct
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
    initialRiskUsdt: riskUsdt,
    feesPaidUsdt: entryFee,
    fundingPaidUsdt: 0,
    exitTs: null,
    exitPrice: null,
    realizedPnlUsdt: null,
    exitReason: null,
    signalConfidence: confidence,
    signalEdge: edge,
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
  console.log(`  Stop: ${stopLoss.toFixed(2)} (${stopDistance.toFixed(2)} distance)`);
  console.log(`  Risk: $${riskUsdt.toFixed(2)} (${config.riskPerTradePct}% of equity)`);
  console.log(`  TP1: ${tp1.toFixed(2)}, TP2: ${tp2.toFixed(2)}`);
  
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
    await storage.updatePosition(position.id, {
      status: "CLOSED",
      exitTs: ctx.candle.timestamp,
      exitPrice: slippedExitPrice,
      realizedPnlUsdt: totalRealizedPnl,
      exitReason: reason,
      feesPaidUsdt: (position.feesPaidUsdt || 0) + exitFee,
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
  }

  return { pnl: netPnl, isPartial };
}

export async function processCandle(ctx: TradeContext): Promise<void> {
  const config = getConfig();
  
  if (!isPaperTradingEnabled()) {
    return;
  }
  
  const position = await storage.getOpenPosition();
  const portfolio = await storage.getOrCreatePortfolio();
  
  if (!position) {
    const gating = checkShotPlanGating(ctx.shotPlan, config);
    const shotPlanCosts = ctx.shotPlan?.estimatedCosts || getTotalCostsPct();
    const shotPlanEdge = ctx.shotPlan?.edge || 0;
    const shotPlanEdgeMultiple = shotPlanCosts > 0 ? shotPlanEdge / shotPlanCosts : 0;
    
    logAudit({
      timestamp: Date.now(),
      signal: ctx.shotPlan?.signal || "NONE",
      confidence: ctx.shotPlan?.confidence || 0,
      regime: ctx.shotPlan?.regime || "unknown",
      edge: shotPlanEdge,
      costs: shotPlanCosts,
      edgeVsCosts: shotPlanEdge > shotPlanCosts ? "PASS" : "FAIL",
      edgeBucket: ctx.shotPlan?.edgeBucket || "none",
      edgeMultiple: shotPlanEdgeMultiple,
      expansionConfirmed: ctx.shotPlan?.expansionGate?.confirmed || false,
      expansionDetails: ctx.shotPlan?.expansionGate?.details || "N/A",
      positionSize: 0,
      exposureAfter: 0,
      decision: gating.allowed ? "ALLOWED" : "BLOCKED",
      reason: gating.reason,
    });
    
    if (!gating.allowed) {
      return;
    }
    
    const shotPlan = ctx.shotPlan!;
    await openPosition(
      ctx,
      shotPlan.signal as "LONG" | "SHORT",
      shotPlan.stopLoss!,
      shotPlan.takeProfit1!,
      shotPlan.takeProfit2!,
      shotPlan.confidence,
      shotPlan.edge
    );
    return;
  }

  await storage.updatePosition(position.id, {
    barsOpen: (position.barsOpen || 0) + 1,
  });

  const slHit = checkStopLoss(position, ctx.candle);
  const tp1Hit = checkTp1(position, ctx.candle);
  const tp2Hit = checkTp2(position, ctx.candle);
  const trailHit = checkTrailingStop(position, ctx.candle);

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
    const gating = checkShotPlanGating(ctx.shotPlan, config);
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
        shotPlan.edge
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

export async function getPortfolioSummary() {
  const portfolio = await storage.getOrCreatePortfolio();
  const openPosition = await storage.getOpenPosition();
  const config = getConfig();
  
  let unrealizedPnl = portfolio.unrealizedPnlUsdt;
  let exposure = 0;
  
  if (openPosition) {
    exposure = openPosition.notionalUsdt;
  }

  const totalTrades = portfolio.totalTrades ?? 0;
  const winningTrades = portfolio.winningTrades ?? 0;
  const losingTrades = portfolio.losingTrades ?? 0;
  const realizedPnl = portfolio.realizedPnlUsdt ?? 0;
  const unrealized = unrealizedPnl ?? 0;
  
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
  if (returns.length > 1) {
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    const std = Math.sqrt(variance);
    sharpe = std > 0 ? (mean / std) * Math.sqrt(252) : 0;
  }

  const equity = portfolio.currentEquityUsdt + unrealized;
  const peak = portfolio.peakEquityUsdt;
  const currentDrawdown = peak > 0 ? ((peak - equity) / peak) * 100 : 0;
  const exposurePct = equity > 0 ? (exposure / equity) * 100 : 0;

  return {
    equity,
    startingEquity: portfolio.startingEquityUsdt,
    availableBalance: portfolio.availableBalanceUsdt,
    realizedPnl,
    unrealizedPnl: unrealized,
    totalPnl: realizedPnl + unrealized,
    maxDrawdown: portfolio.maxDrawdownPct ?? 0,
    currentDrawdown,
    totalTrades,
    winningTrades,
    losingTrades,
    winRate,
    avgWin,
    avgLoss,
    sharpe,
    expectancy,
    bestTrade,
    worstTrade,
    profitFactor: profitFactor === Infinity ? 999.99 : profitFactor,
    exposure,
    exposurePct,
    isAutoTrading: config.isAutoTrading,
    isPaperTradingEnabled: config.paperTradingEnabled,
    openPosition: openPosition ? {
      id: openPosition.id,
      side: openPosition.side as "LONG" | "SHORT",
      entryPrice: openPosition.entryPrice,
      qty: openPosition.qty,
      notional: openPosition.notionalUsdt,
      stopLoss: openPosition.stopLoss,
      tp1: openPosition.tp1,
      tp2: openPosition.tp2,
      barsOpen: openPosition.barsOpen,
      unrealizedPnl: unrealized,
      entryTs: openPosition.entryTs,
    } : null,
    recentAuditLogs: auditLog.slice(-10),
  };
}
