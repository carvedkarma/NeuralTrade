import { z } from "zod";

export const signalTypeSchema = z.enum(["LONG", "SHORT", "HOLD"]);
export type SignalType = z.infer<typeof signalTypeSchema>;

export const regimeTypeSchema = z.enum(["trend_up", "trend_down", "chop"]);
export type RegimeType = z.infer<typeof regimeTypeSchema>;

export const riskModeSchema = z.enum(["normal", "high_vol", "no_trade"]);
export type RiskMode = z.infer<typeof riskModeSchema>;

export const candleSchema = z.object({
  timestamp: z.number(),
  open: z.number(),
  high: z.number(),
  low: z.number(),
  close: z.number(),
  volume: z.number(),
});
export type Candle = z.infer<typeof candleSchema>;

export const futuresDataSchema = z.object({
  fundingRate: z.number(),
  nextFundingTime: z.number(),
  openInterest: z.number(),
  oiChange15m: z.number(),
  oiChange1h: z.number(),
  longShortRatio: z.number(),
  liquidations15m: z.number(),
  liquidations1h: z.number(),
  markPrice: z.number(),
  indexPrice: z.number(),
  basis: z.number(),
});
export type FuturesData = z.infer<typeof futuresDataSchema>;

export const featureSchema = z.object({
  name: z.string(),
  value: z.number(),
  importance: z.number(),
  description: z.string(),
});
export type Feature = z.infer<typeof featureSchema>;

export const signalSchema = z.object({
  timestamp: z.number(),
  signal: signalTypeSchema,
  confidence: z.number(),
  probUp: z.number(),
  probDown: z.number(),
  probChop: z.number(),
  expectedMove: z.number(),
  costs: z.number(),
  edge: z.number(),
  regime: regimeTypeSchema,
  riskMode: riskModeSchema,
  topFeatures: z.array(featureSchema),
});
export type Signal = z.infer<typeof signalSchema>;

export const tradeSchema = z.object({
  id: z.string(),
  timestamp: z.number(),
  side: z.enum(["LONG", "SHORT"]),
  entryPrice: z.number(),
  exitPrice: z.number().nullable(),
  size: z.number(),
  pnl: z.number().nullable(),
  pnlPercent: z.number().nullable(),
  status: z.enum(["open", "closed"]),
  stopLoss: z.number(),
  takeProfit: z.number(),
});
export type Trade = z.infer<typeof tradeSchema>;

export const dashboardDataSchema = z.object({
  candles: z.array(candleSchema),
  currentSignal: signalSchema,
  futuresData: futuresDataSchema,
  recentTrades: z.array(tradeSchema),
  equity: z.number(),
  drawdown: z.number(),
  maxDrawdown: z.number(),
  dailyPnl: z.number(),
  winRate: z.number(),
  profitFactor: z.number(),
  totalTrades: z.number(),
  exposure: z.number(),
});
export type DashboardData = z.infer<typeof dashboardDataSchema>;
