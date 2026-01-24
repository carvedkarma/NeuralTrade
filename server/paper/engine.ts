import { getConfig, type PaperTradingConfig } from "./config";
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

function applySlippage(price: number, side: "LONG" | "SHORT", slippageBps: number): number {
  const mult = side === "LONG" ? 1 + slippageBps / 10000 : 1 - slippageBps / 10000;
  return price * mult;
}

function calculateFee(notional: number, feePct: number): number {
  return notional * (feePct / 100);
}

function calculatePositionSize(
  equity: number,
  entryPrice: number,
  stopLoss: number,
  riskPct: number
): { qty: number; riskUsdt: number; notional: number } {
  const riskUsdt = equity * (riskPct / 100);
  const stopDistance = Math.abs(entryPrice - stopLoss);
  if (stopDistance === 0) {
    return { qty: 0, riskUsdt: 0, notional: 0 };
  }
  const qty = riskUsdt / stopDistance;
  const notional = qty * entryPrice;
  return { qty, riskUsdt, notional };
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
  const costs = shotPlan.estimatedCosts || 0.001;
  return shotPlan.confidence >= config.flipConfidenceThreshold && 
         shotPlan.edge > costs * config.flipEdgeMultiplier;
}

export async function openPosition(
  ctx: TradeContext,
  side: "LONG" | "SHORT",
  stopLoss: number,
  tp1: number,
  tp2: number,
  confidence: number,
  edge: number
): Promise<PaperPosition | null> {
  const config = getConfig();
  const portfolio = await storage.getOrCreatePortfolio();
  const existingPosition = await storage.getOpenPosition();
  
  if (existingPosition) {
    console.log("[Paper] Position already open, skipping new entry");
    return null;
  }

  const entryPrice = applySlippage(ctx.candle.open, side, config.slippageBps);
  const { qty, riskUsdt, notional } = calculatePositionSize(
    portfolio.currentEquityUsdt,
    entryPrice,
    stopLoss,
    config.riskPerTradePct
  );

  if (qty <= 0) {
    console.log("[Paper] Invalid position size, skipping");
    return null;
  }

  const entryFee = calculateFee(notional, config.takerFeePct);

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
    reason: `${side} entry at ${entryPrice.toFixed(2)}`,
  });

  await storage.updatePortfolio({
    availableBalanceUsdt: portfolio.availableBalanceUsdt - riskUsdt,
  });

  console.log(`[Paper] Opened ${side} position: ${qty.toFixed(6)} BTC @ ${entryPrice.toFixed(2)}`);
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
  const totalFees = (position.feesPaidUsdt || 0) * (qtyToClose / position.qty) + exitFee;
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
    reason: `${reason}: Exit @ ${slippedExitPrice.toFixed(2)}`,
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
    const maxDrawdown = Math.max(portfolio.maxDrawdownPct, drawdown);

    await storage.updatePortfolio({
      currentEquityUsdt: newEquity,
      availableBalanceUsdt: newEquity,
      realizedPnlUsdt: portfolio.realizedPnlUsdt + totalRealizedPnl,
      unrealizedPnlUsdt: 0,
      peakEquityUsdt: newPeak,
      maxDrawdownPct: maxDrawdown,
      totalTrades: portfolio.totalTrades + 1,
      winningTrades: totalRealizedPnl > 0 ? portfolio.winningTrades + 1 : portfolio.winningTrades,
      losingTrades: totalRealizedPnl <= 0 ? portfolio.losingTrades + 1 : portfolio.losingTrades,
    });

    await storage.recordEquityPoint(newEquity, drawdown);
    console.log(`[Paper] Closed ${position.side} position: PnL ${totalRealizedPnl.toFixed(2)} USDT (${reason})`);
  }

  return { pnl: netPnl, isPartial };
}

export async function processCandle(ctx: TradeContext): Promise<void> {
  const config = getConfig();
  const position = await storage.getOpenPosition();
  
  if (!position) {
    if (ctx.shotPlan && ctx.shotPlan.signal !== "HOLD" && ctx.shotPlan.vetoReasons.length === 0) {
      const { entryZone, stopLoss, takeProfit1, takeProfit2 } = ctx.shotPlan;
      if (entryZone && stopLoss && takeProfit1 && takeProfit2) {
        await openPosition(
          ctx,
          ctx.shotPlan.signal as "LONG" | "SHORT",
          stopLoss,
          takeProfit1,
          takeProfit2,
          ctx.shotPlan.confidence,
          ctx.shotPlan.edge
        );
      }
    }
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
    await closePosition(position, ctx.markPrice, "FLIP", ctx);
    if (ctx.shotPlan && ctx.shotPlan.entryZone && ctx.shotPlan.stopLoss && 
        ctx.shotPlan.takeProfit1 && ctx.shotPlan.takeProfit2) {
      await openPosition(
        ctx,
        ctx.shotPlan.signal as "LONG" | "SHORT",
        ctx.shotPlan.stopLoss,
        ctx.shotPlan.takeProfit1,
        ctx.shotPlan.takeProfit2,
        ctx.shotPlan.confidence,
        ctx.shotPlan.edge
      );
    }
    return;
  }

  const newTrailPrice = updateTrailingStop(position, ctx.kalmanFast, ctx.atr, config);
  if (newTrailPrice && newTrailPrice !== position.trailPrice) {
    await storage.updatePosition(position.id, { trailPrice: newTrailPrice });
  }

  const portfolio = await storage.getOrCreatePortfolio();
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
  
  let unrealizedPnl = portfolio.unrealizedPnlUsdt;
  let exposure = 0;
  
  if (openPosition) {
    exposure = openPosition.notionalUsdt;
  }

  const winRate = portfolio.totalTrades > 0 
    ? (portfolio.winningTrades / portfolio.totalTrades) * 100 
    : 0;

  return {
    equity: portfolio.currentEquityUsdt + unrealizedPnl,
    startingEquity: portfolio.startingEquityUsdt,
    availableBalance: portfolio.availableBalanceUsdt,
    realizedPnl: portfolio.realizedPnlUsdt,
    unrealizedPnl,
    totalPnl: portfolio.realizedPnlUsdt + unrealizedPnl,
    maxDrawdown: portfolio.maxDrawdownPct,
    totalTrades: portfolio.totalTrades,
    winningTrades: portfolio.winningTrades,
    losingTrades: portfolio.losingTrades,
    winRate,
    exposure,
    openPosition: openPosition ? {
      id: openPosition.id,
      side: openPosition.side,
      entryPrice: openPosition.entryPrice,
      qty: openPosition.qty,
      notional: openPosition.notionalUsdt,
      stopLoss: openPosition.stopLoss,
      tp1: openPosition.tp1,
      tp2: openPosition.tp2,
      barsOpen: openPosition.barsOpen,
      unrealizedPnl,
      entryTs: openPosition.entryTs,
    } : null,
  };
}
