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

export const shotPlanSchema = z.object({
  signal: signalTypeSchema,
  confidence: z.number(),
  regime: z.enum(["trend_up", "trend_down", "chop", "shock"]),
  strategy: z.string(),
  entryZone: z.object({ low: z.number(), high: z.number() }).nullable(),
  stopLoss: z.number().nullable(),
  takeProfit1: z.number().nullable(),
  takeProfit2: z.number().nullable(),
  trailingStop: z.number().nullable(),
  riskReward: z.number(),
  expectedHoldTime: z.string(),
  estimatedCosts: z.number(),
  edge: z.number(),
  probUp: z.number(),
  probDown: z.number(),
  probChop: z.number(),
  expectedMove: z.number(),
  reasons: z.array(z.string()),
  vetoReasons: z.array(z.string()),
  patternMatchCount: z.number().optional(),
  modelConsensus: z.number().optional(),
});
export type ShotPlan = z.infer<typeof shotPlanSchema>;

export const fearGreedSchema = z.object({
  value: z.number(),
  classification: z.string(),
  signal: z.enum(["bullish", "bearish", "neutral"]),
  description: z.string(),
});
export type FearGreed = z.infer<typeof fearGreedSchema>;

export const sentimentSchema = z.object({
  fearGreed: fearGreedSchema.nullable(),
  socialScore: z.number(),
  newsScore: z.number(),
  topNews: z.array(z.object({
    title: z.string(),
    sentiment: z.string(),
    source: z.string(),
  })),
});
export type Sentiment = z.infer<typeof sentimentSchema>;

export const dataSourceStatsSchema = z.object({
  name: z.string(),
  status: z.enum(["active", "fallback", "error", "idle"]),
  lastFetch: z.number().nullable(),
  candlesCollected: z.number(),
  successRate: z.number(),
  avgLatency: z.number(),
});
export type DataSourceStats = z.infer<typeof dataSourceStatsSchema>;

export const patternLearningStatsSchema = z.object({
  totalPatterns: z.number(),
  uniquePatterns: z.number(),
  avgSimilarity: z.number(),
  matchRate: z.number(),
  lastPatternAdded: z.number().nullable(),
  patternsByRegime: z.record(z.string(), z.number()),
  topPatternOutcomes: z.array(z.object({
    pattern: z.string(),
    winRate: z.number(),
    count: z.number(),
  })),
});
export type PatternLearningStats = z.infer<typeof patternLearningStatsSchema>;

export const featureComputationStatsSchema = z.object({
  totalFeatures: z.number(),
  featuresComputed: z.number(),
  computationTime: z.number(),
  topFeatures: z.array(z.object({
    name: z.string(),
    importance: z.number(),
    currentValue: z.number(),
  })),
  featureCategories: z.record(z.string(), z.number()),
});
export type FeatureComputationStats = z.infer<typeof featureComputationStatsSchema>;

export const modelPerformanceStatsSchema = z.object({
  modelName: z.string(),
  weight: z.number(),
  predictionsToday: z.number(),
  accuracy: z.number(),
  avgConfidence: z.number(),
  lastPrediction: z.number().nullable(),
  signalDistribution: z.object({
    long: z.number(),
    short: z.number(),
    hold: z.number(),
  }),
});
export type ModelPerformanceStats = z.infer<typeof modelPerformanceStatsSchema>;

export const learningStatsSchema = z.object({
  dataSources: z.array(dataSourceStatsSchema),
  patternLearning: patternLearningStatsSchema,
  featureComputation: featureComputationStatsSchema,
  modelPerformance: z.array(modelPerformanceStatsSchema),
  ensembleStats: z.object({
    totalPredictions: z.number(),
    consensusRate: z.number(),
    avgConfidence: z.number(),
    lastUpdate: z.number(),
  }),
  dataIngestion: z.object({
    candlesTotal: z.number(),
    timeRangeDays: z.number(),
    oldestCandle: z.number().nullable(),
    newestCandle: z.number().nullable(),
    dataGaps: z.number(),
  }),
});
export type LearningStats = z.infer<typeof learningStatsSchema>;

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
  shotPlan: shotPlanSchema.optional(),
  sentiment: sentimentSchema.optional(),
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
  learningStats: learningStatsSchema.optional(),
  isLiveData: z.boolean().optional(),
  dataSource: z.enum(["coingecko", "cryptocompare", "binance", "none"]).optional(),
  dataError: z.string().nullable().optional(),
});
export type DashboardData = z.infer<typeof dashboardDataSchema>;
