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
  entryCandle: z.number().optional(),
  signalType: z.enum(["crossover", "retest"]).optional(),
});
export type Trade = z.infer<typeof tradeSchema>;

export const kalmanStateSchema = z.object({
  x: z.number(),
  P: z.number(),
});
export type KalmanState = z.infer<typeof kalmanStateSchema>;

export const strategySignalSchema = z.object({
  type: z.enum(["crossover", "retest", "none"]),
  direction: signalTypeSchema,
  entryZone: z.number().nullable(),
  stopLoss: z.number().nullable(),
  takeProfit1: z.number().nullable(),
  takeProfit2: z.number().nullable(),
  atr: z.number(),
  regime: z.enum(["bull", "bear"]),
  kalmanFast: z.number(),
  kalmanSlow: z.number(),
});
export type StrategySignal = z.infer<typeof strategySignalSchema>;

export const strategyStateSchema = z.object({
  isRunning: z.boolean(),
  useRetestSignals: z.boolean(),
  riskPercent: z.number(),
  atrMultiplier: z.number(),
  timeStopCandles: z.number(),
});
export type StrategyState = z.infer<typeof strategyStateSchema>;

export const aiAnalysisSchema = z.object({
  marketSummary: z.string(),
  trendExplanation: z.string(),
  signalReasoning: z.string(),
  riskAssessment: z.string(),
  recommendation: z.enum(["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]),
  confidence: z.number(),
  keyInsights: z.array(z.string()),
  warnings: z.array(z.string()),
});
export type AIAnalysis = z.infer<typeof aiAnalysisSchema>;

export const aiSignalSchema = z.object({
  direction: signalTypeSchema,
  confidence: z.number(),
  entryPrice: z.number().nullable(),
  stopLoss: z.number().nullable(),
  takeProfit1: z.number().nullable(),
  takeProfit2: z.number().nullable(),
  reasoning: z.string(),
  riskReward: z.number(),
  timeframe: z.string(),
});
export type AISignal = z.infer<typeof aiSignalSchema>;

export const technicalIndicatorSchema = z.object({
  name: z.string(),
  value: z.number(),
  signal: z.enum(["bullish", "bearish", "neutral"]),
  strength: z.number(),
  description: z.string(),
});
export type TechnicalIndicator = z.infer<typeof technicalIndicatorSchema>;

export const multiTimeframeScoreSchema = z.object({
  score: z.number(),
  direction: z.enum(["bullish", "bearish", "neutral"]),
  alignment: z.number(),
  details: z.array(z.object({
    timeframe: z.string(),
    trend: z.enum(["up", "down", "neutral"]),
    weight: z.number(),
  })),
});
export type MultiTimeframeScore = z.infer<typeof multiTimeframeScoreSchema>;

export const whaleActivitySchema = z.object({
  largeBuys: z.number(),
  largeSells: z.number(),
  netFlow: z.number(),
  whaleActivity: z.enum(["bullish", "bearish", "neutral"]),
});
export type WhaleActivity = z.infer<typeof whaleActivitySchema>;

export const performanceStatsSchema = z.object({
  totalTrades: z.number(),
  winningTrades: z.number(),
  losingTrades: z.number(),
  winRate: z.number(),
  avgWin: z.number(),
  avgLoss: z.number(),
  profitFactor: z.number(),
  sharpeRatio: z.number(),
  maxDrawdown: z.number(),
  currentDrawdown: z.number(),
  expectancy: z.number(),
  avgRMultiple: z.number(),
  bestTrade: z.number(),
  worstTrade: z.number(),
  consecutiveWins: z.number(),
  consecutiveLosses: z.number(),
});
export type PerformanceStats = z.infer<typeof performanceStatsSchema>;

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
  kalmanFast: z.array(z.number()),
  kalmanSlow: z.array(z.number()),
  strategySignal: strategySignalSchema,
  strategyState: strategyStateSchema,
  activeTrade: tradeSchema.nullable(),
  aiAnalysis: aiAnalysisSchema.optional(),
  aiSignal: aiSignalSchema.optional(),
  indicators: z.object({
    rsi: technicalIndicatorSchema,
    macd: technicalIndicatorSchema,
    bollingerBands: technicalIndicatorSchema,
    obv: technicalIndicatorSchema,
    vwap: technicalIndicatorSchema,
    atr: technicalIndicatorSchema,
    adx: technicalIndicatorSchema,
    stochastic: technicalIndicatorSchema,
  }).optional(),
  mtfScore: multiTimeframeScoreSchema.optional(),
  whaleActivity: whaleActivitySchema.optional(),
  performanceStats: performanceStatsSchema.optional(),
  isLiveData: z.boolean().optional(),
});
export type DashboardData = z.infer<typeof dashboardDataSchema>;
