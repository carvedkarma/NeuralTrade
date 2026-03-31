import { db } from "../db";
import { paperPortfolio, paperPositions, paperTrades, paperEquityCurve, paperTradeHistory } from "@shared/schema";
import type { PaperPortfolio, PaperPosition, PaperTrade, PaperEquityCurve } from "@shared/schema";
import { eq, desc, gte, and } from "drizzle-orm";
import { getConfig } from "./config";

export async function getOrCreatePortfolio(): Promise<PaperPortfolio> {
  const existing = await db.select().from(paperPortfolio).limit(1);
  if (existing.length > 0) {
    return existing[0];
  }
  const config = getConfig();
  const now = Date.now();
  const [created] = await db.insert(paperPortfolio).values({
    startingEquityUsdt: config.startingEquity,
    currentEquityUsdt: config.startingEquity,
    availableBalanceUsdt: config.startingEquity,
    unrealizedPnlUsdt: 0,
    realizedPnlUsdt: 0,
    maxDrawdownPct: 0,
    peakEquityUsdt: config.startingEquity,
    totalTrades: 0,
    winningTrades: 0,
    losingTrades: 0,
    updatedTs: now,
  }).returning();
  return created;
}

export async function updatePortfolio(updates: Partial<PaperPortfolio>): Promise<PaperPortfolio> {
  const portfolio = await getOrCreatePortfolio();
  const [updated] = await db.update(paperPortfolio)
    .set({ ...updates, updatedTs: Date.now() })
    .where(eq(paperPortfolio.id, portfolio.id))
    .returning();
  return updated;
}

export async function resetPortfolio(): Promise<PaperPortfolio> {
  await db.delete(paperPositions);
  await db.delete(paperTrades);
  await db.delete(paperEquityCurve);
  await db.delete(paperPortfolio);
  return getOrCreatePortfolio();
}

export async function getOpenPosition(): Promise<PaperPosition | null> {
  const positions = await db.select()
    .from(paperPositions)
    .where(eq(paperPositions.status, "OPEN"))
    .limit(1);
  return positions[0] || null;
}

export async function getPositions(status?: "OPEN" | "CLOSED", limit: number = 100): Promise<PaperPosition[]> {
  if (status) {
    return db.select()
      .from(paperPositions)
      .where(eq(paperPositions.status, status))
      .orderBy(desc(paperPositions.entryTs))
      .limit(limit);
  }
  return db.select()
    .from(paperPositions)
    .orderBy(desc(paperPositions.entryTs))
    .limit(limit);
}

export async function getPositionById(id: number): Promise<PaperPosition | null> {
  const positions = await db.select()
    .from(paperPositions)
    .where(eq(paperPositions.id, id))
    .limit(1);
  return positions[0] || null;
}

export async function createPosition(position: Omit<PaperPosition, "id">): Promise<PaperPosition> {
  const [created] = await db.insert(paperPositions).values(position).returning();
  return created;
}

export async function updatePosition(id: number, updates: Partial<PaperPosition>): Promise<PaperPosition> {
  const [updated] = await db.update(paperPositions)
    .set(updates)
    .where(eq(paperPositions.id, id))
    .returning();
  return updated;
}

export async function updatePositionPeakProfit(id: number, peakProfit: number): Promise<void> {
  await db.update(paperPositions)
    .set({ peakProfit })
    .where(eq(paperPositions.id, id));
}

export async function createTrade(trade: Omit<PaperTrade, "id">): Promise<PaperTrade> {
  const [created] = await db.insert(paperTrades).values(trade).returning();
  return created;
}

export async function getTrades(limit: number = 500): Promise<PaperTrade[]> {
  return db.select()
    .from(paperTrades)
    .orderBy(desc(paperTrades.ts))
    .limit(limit);
}

export async function getTradesByPosition(positionId: number): Promise<PaperTrade[]> {
  return db.select()
    .from(paperTrades)
    .where(eq(paperTrades.positionId, positionId))
    .orderBy(desc(paperTrades.ts));
}

export async function recordEquityPoint(equity: number, drawdownPct: number): Promise<void> {
  await db.insert(paperEquityCurve).values({
    ts: Date.now(),
    equityUsdt: equity,
    drawdownPct,
  });
}

export async function getEquityCurve(range?: "7d" | "30d" | "all"): Promise<PaperEquityCurve[]> {
  const now = Date.now();
  let startTs = 0;
  if (range === "7d") {
    startTs = now - 7 * 24 * 60 * 60 * 1000;
  } else if (range === "30d") {
    startTs = now - 30 * 24 * 60 * 60 * 1000;
  }
  if (startTs > 0) {
    return db.select()
      .from(paperEquityCurve)
      .where(gte(paperEquityCurve.ts, startTs))
      .orderBy(paperEquityCurve.ts);
  }
  return db.select()
    .from(paperEquityCurve)
    .orderBy(paperEquityCurve.ts);
}

export async function getPositionsBySymbol(symbol: string, status?: "OPEN" | "CLOSED", limit: number = 100): Promise<PaperPosition[]> {
  const conditions = [eq(paperPositions.symbol, symbol)];
  if (status) conditions.push(eq(paperPositions.status, status));
  return db.select()
    .from(paperPositions)
    .where(and(...conditions))
    .orderBy(desc(paperPositions.entryTs))
    .limit(limit);
}

export interface TradeCloseRecord {
  positionId: number;
  symbol: string;
  side: string;
  entryTs: number;
  entryPrice: number;
  exitTs: number;
  exitPrice: number;
  grossR: number;
  netR: number;
  costR: number;
  pnlUsdt: number;
  riskUsdt: number;
  barsHeld: number;
  exitReason: string;
  maxFavorableR: number;
  maxAdverseR?: number | null;
  initialSl?: number | null;
  takeProfit?: number | null;
  v5Score?: number | null;
  trailActive?: number | null;
  regime: string | null;
  signalConfidence: number | null;
  signalEdge: number | null;
}

export async function recordTradeClose(record: TradeCloseRecord): Promise<void> {
  try {
    await db.insert(paperTradeHistory).values({
      positionId: record.positionId,
      symbol: record.symbol,
      side: record.side,
      entryTs: record.entryTs,
      entryPrice: record.entryPrice,
      exitTs: record.exitTs,
      exitPrice: record.exitPrice,
      grossR: record.grossR,
      netR: record.netR,
      costR: record.costR,
      pnlUsdt: record.pnlUsdt,
      riskUsdt: record.riskUsdt,
      barsHeld: record.barsHeld,
      exitReason: record.exitReason,
      maxFavorableR: record.maxFavorableR,
      maxAdverseR: record.maxAdverseR ?? null,
      initialSl: record.initialSl ?? null,
      takeProfit: record.takeProfit ?? null,
      v5Score: record.v5Score ?? null,
      trailActive: record.trailActive ?? 0,
      regime: record.regime,
      signalConfidence: record.signalConfidence,
      signalEdge: record.signalEdge,
    });
  } catch (err) {
    console.error("[Paper Storage] Failed to record trade close:", err);
  }
}

export async function getTradeHistory(options?: { symbol?: string; limit?: number; offset?: number }): Promise<any[]> {
  const conditions = [];
  if (options?.symbol) conditions.push(eq(paperTradeHistory.symbol, options.symbol));
  return db.select()
    .from(paperTradeHistory)
    .where(conditions.length > 0 ? and(...conditions) : undefined)
    .orderBy(desc(paperTradeHistory.exitTs))
    .limit(options?.limit ?? 100)
    .offset(options?.offset ?? 0);
}

/**
 * Get consecutive losing trades at the end of the trade history
 * Used for institution-grade loss streak tracking
 */
export async function clearTradeHistory(): Promise<void> {
  await db.delete(paperTradeHistory);
}

export async function clearEquityCurve(): Promise<void> {
  await db.delete(paperEquityCurve);
}

export async function getRecentLossStreak(): Promise<number> {
  const recentTrades = await db.select()
    .from(paperTrades)
    .orderBy(desc(paperTrades.id))
    .limit(10);
  
  let lossStreak = 0;
  for (const trade of recentTrades) {
    const pnl = trade.pnlUsdt ?? 0;
    if (pnl < 0) {
      lossStreak++;
    } else {
      break;
    }
  }
  return lossStreak;
}
