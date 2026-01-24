import { db } from "../db";
import { paperPortfolio, paperPositions, paperTrades, paperEquityCurve } from "@shared/schema";
import type { PaperPortfolio, PaperPosition, PaperTrade, PaperEquityCurve } from "@shared/schema";
import { eq, desc, gte } from "drizzle-orm";
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
