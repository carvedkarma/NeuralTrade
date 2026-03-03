import { z } from "zod";
import { pgTable, text, serial, integer, bigint, real, timestamp, jsonb, boolean, index, uniqueIndex, varchar } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";

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
  mu: z.number().optional(),
  sigma: z.number().optional(),
  positionSizePct: z.number().optional(),
  stopLossPct: z.number().optional(),
  takeProfitPct: z.number().optional(),
  urgency: z.enum(["low", "medium", "high"]).optional(),
  suggestedOrderType: z.enum(["limit", "market"]).optional(),
  expertWeights: z.record(z.number()).optional(),
  // Multihead-specific fields
  isMultihead: z.boolean().optional(),
  entryOffsetPct: z.number().optional(),  // Learned entry offset from MFE
  entryPrice: z.number().optional(),
  stopLossPrice: z.number().optional(),
  takeProfitPrice: z.number().optional(),
  quantiles: z.object({
    q10: z.number(),
    q25: z.number(),
    q50: z.number(),
    q75: z.number(),
    q90: z.number(),
  }).optional(),
  predictedCandles: z.array(z.object({
    step: z.number(),
    closeDeleta: z.number(),
    highDelta: z.number(),
    lowDelta: z.number(),
  })).optional(),
  riskRewardRatio: z.number().optional(),
  isLearnedLevels: z.boolean().optional(),
  volState: z.enum(["contraction", "neutral", "expansion"]).optional(),
  volStateProbs: z.object({
    contraction: z.number(),
    neutral: z.number(),
    expansion: z.number(),
  }).optional(),
  modelName: z.string().optional(),
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

export const expansionGateSchema = z.object({
  impulseCandle: z.boolean(),
  atrExpansion: z.boolean(),
  rangeBreak: z.boolean(),
  confirmed: z.boolean(),
  details: z.string(),
});
export type ExpansionGate = z.infer<typeof expansionGateSchema>;

export const edgeBucketSchema = z.enum(["none", "weak", "moderate", "strong"]);
export type EdgeBucket = z.infer<typeof edgeBucketSchema>;

export const combinedIntelligenceSchema = z.object({
  mlDirection: signalTypeSchema,
  mlConfidence: z.number(),
  strategyAction: signalTypeSchema,
  strategyEV: z.number(),
  patternWinRate: z.number(),
  patternSupport: z.number(),
  combinedScore: z.number(),
  systemsAgree: z.boolean(),
  finalSignal: signalTypeSchema,
  finalConfidence: z.number(),
  reasoning: z.array(z.string()),
  vetoes: z.array(z.string()),
});
export type CombinedIntelligence = z.infer<typeof combinedIntelligenceSchema>;

export const shotPlanSchema = z.object({
  signal: signalTypeSchema,
  confidence: z.number(),
  regime: z.enum(["trend_up", "trend_down", "chop", "shock", "quiet", "ranging"]),
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
  edgeBucket: edgeBucketSchema.optional(),
  edgeMultiple: z.number().optional(),
  probUp: z.number(),
  probDown: z.number(),
  probChop: z.number(),
  expectedMove: z.number(),
  reasons: z.array(z.string()),
  vetoReasons: z.array(z.string()),
  patternMatchCount: z.number().optional(),
  modelConsensus: z.number().optional(),
  expansionGate: expansionGateSchema.optional(),
  combinedIntelligence: combinedIntelligenceSchema.optional(),
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

export const similarityDistributionSchema = z.object({
  min: z.number(),
  max: z.number(),
  mean: z.number(),
  median: z.number(),
  count: z.number(),
});

export const patternLearningStatsSchema = z.object({
  totalPatterns: z.number(),
  uniquePatterns: z.number(),
  avgSimilarity: z.number(),
  matchRate: z.number(),
  lastPatternAdded: z.number().nullable(),
  patternsByRegime: z.record(z.string(), z.number()),
  similarityDistribution: similarityDistributionSchema.optional(),
  similarityHealthy: z.boolean().optional(),
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
  accuracy: z.number(), // Deprecated - use directionalAccuracy instead
  directionalAccuracy: z.number().nullable(), // Accuracy on LONG/SHORT only (excludes HOLD)
  holdRate: z.number(), // Percentage of predictions that are HOLD
  avgConfidence: z.number(),
  lastPrediction: z.number().nullable(),
  signalDistribution: z.object({
    long: z.number(),
    short: z.number(),
    hold: z.number(),
  }),
  directionalStats: z.object({
    total: z.number(), // LONG + SHORT count
    correct: z.number(), // How many were correct
  }).optional(),
});
export type ModelPerformanceStats = z.infer<typeof modelPerformanceStatsSchema>;

export const socialPlatformStatsSchema = z.object({
  platform: z.string(),
  icon: z.string(),
  status: z.enum(["active", "idle", "error"]),
  itemsRead: z.number(),
  lastFetch: z.number().nullable(),
  sentiment: z.number(),
  influence: z.number(),
});
export type SocialPlatformStats = z.infer<typeof socialPlatformStatsSchema>;

export const historicalLearningStatsSchema = z.object({
  totalHistoricalCandles: z.number(),
  yearsOfData: z.number(),
  patternsLearnedFromHistory: z.number(),
  backtestTrades: z.number(),
  historicalWinRate: z.number(),
  dataRangeStart: z.string(),
  dataRangeEnd: z.string(),
  learningProgress: z.number(),
  epochsCompleted: z.number(),
  lastTrainingTime: z.number().nullable(),
  candlesUsedForTraining: z.number(),
  candlesAvailable: z.number(),
  trainingCoverage: z.number(),
  deepLearningPass: z.number().optional(),
  deepLearningComplete: z.boolean().optional(),
});
export type HistoricalLearningStats = z.infer<typeof historicalLearningStatsSchema>;

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
  socialAwareness: z.object({
    platforms: z.array(socialPlatformStatsSchema),
    totalItemsRead: z.number(),
    globalSentiment: z.number(),
    lastGlobalUpdate: z.number().nullable(),
  }),
  historicalLearning: historicalLearningStatsSchema,
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
  dataSource: z.enum(["coingecko", "cryptocompare", "binance", "database", "database+binance", "none"]).optional(),
  dataError: z.string().nullable().optional(),
});
export type DashboardData = z.infer<typeof dashboardDataSchema>;

export const openInterestHistory = pgTable("open_interest_history", {
  id: serial("id").primaryKey(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
  period: varchar("period", { length: 10 }).notNull().default("15m"),
  sumOpenInterest: real("sum_open_interest").notNull(),
}, (table) => ({
  symbolIdx: index("oi_history_symbol_idx").on(table.symbol),
  timestampIdx: index("oi_history_timestamp_idx").on(table.timestamp),
  uniqueOi: uniqueIndex("oi_history_unique_idx").on(table.symbol, table.timestamp, table.period),
}));

export const insertOpenInterestHistorySchema = createInsertSchema(openInterestHistory).omit({ id: true });
export type InsertOpenInterestHistory = z.infer<typeof insertOpenInterestHistorySchema>;
export type OpenInterestHistory = typeof openInterestHistory.$inferSelect;

export const candles = pgTable("candles", {
  id: serial("id").primaryKey(),
  symbol: varchar("symbol", { length: 20 }).notNull().default("BTCUSDT"),
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
  timeframe: varchar("timeframe", { length: 10 }).notNull().default("15m"),
  open: real("open").notNull(),
  high: real("high").notNull(),
  low: real("low").notNull(),
  close: real("close").notNull(),
  volume: real("volume").notNull(),
}, (table) => ({
  timestampIdx: index("candles_timestamp_idx").on(table.timestamp),
  timeframeIdx: index("candles_timeframe_idx").on(table.timeframe),
  symbolIdx: index("candles_symbol_idx").on(table.symbol),
  uniqueCandle: uniqueIndex("candles_unique_idx").on(table.symbol, table.timestamp, table.timeframe),
}));

export const features = pgTable("features", {
  id: serial("id").primaryKey(),
  timestamp: integer("timestamp").notNull(),
  returns1: real("returns_1"),
  returns2: real("returns_2"),
  returns4: real("returns_4"),
  returns8: real("returns_8"),
  ema20Slope: real("ema20_slope"),
  ema50Slope: real("ema50_slope"),
  emaDistance: real("ema_distance"),
  vwapDistance: real("vwap_distance"),
  breakoutDistanceHigh: real("breakout_distance_high"),
  breakoutDistanceLow: real("breakout_distance_low"),
  efficiencyRatio: real("efficiency_ratio"),
  atr14: real("atr14"),
  volatility: real("volatility"),
  bollingerWidth: real("bollinger_width"),
  volatilityRegime: varchar("volatility_regime", { length: 20 }),
  rsi14: real("rsi14"),
  macd: real("macd"),
  macdSignal: real("macd_signal"),
  macdHist: real("macd_hist"),
  adx: real("adx"),
  plusDi: real("plus_di"),
  minusDi: real("minus_di"),
  stochK: real("stoch_k"),
  stochD: real("stoch_d"),
  obv: real("obv"),
  kalmanFast: real("kalman_fast"),
  kalmanSlow: real("kalman_slow"),
  kalmanSpread: real("kalman_spread"),
  kalmanRegime: varchar("kalman_regime", { length: 20 }),
  oiChange15m: real("oi_change_15m"),
  oiChange1h: real("oi_change_1h"),
  fundingRate: real("funding_rate"),
  fundingZscore: real("funding_zscore"),
  liquidations15m: real("liquidations_15m"),
  liquidations1h: real("liquidations_1h"),
  markIndexSpread: real("mark_index_spread"),
  orderbookSpread: real("orderbook_spread"),
  orderbookImbalance: real("orderbook_imbalance"),
  tradeDelta: real("trade_delta"),
  deltaDevergence: real("delta_divergence"),
  fearGreedIndex: real("fear_greed_index"),
  socialSentiment: real("social_sentiment"),
  newsScore: real("news_score"),
  embedding: jsonb("embedding"),
}, (table) => ({
  timestampIdx: index("features_timestamp_idx").on(table.timestamp),
}));

export const patterns = pgTable("patterns", {
  id: serial("id").primaryKey(),
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
  embedding: jsonb("embedding").notNull(),
  featureHash: varchar("feature_hash", { length: 64 }),
  forwardReturn8: real("forward_return_8"),
  forwardReturn16: real("forward_return_16"),
  forwardMaxDrawdown: real("forward_max_drawdown"),
  forwardMaxRunup: real("forward_max_runup"),
  timeToMfe: integer("time_to_mfe"),
  forwardWin: boolean("forward_win"),
  regime: varchar("regime", { length: 20 }),
  label: varchar("label", { length: 20 }),
  atrAtEntry: real("atr_at_entry"),
  dynamicThreshold: real("dynamic_threshold"),
  direction: varchar("direction", { length: 10 }),
  actualPnl: real("actual_pnl"),
  createdAt: bigint("created_at", { mode: "number" }),  // Epoch ms when pattern was stored
  trainingWindow: varchar("training_window", { length: 20 }),  // "train" or "test" split
  volatilityBucket: varchar("volatility_bucket", { length: 10 }),  // P0-3: low/medium/high/extreme
  falseFriendPenalty: real("false_friend_penalty"),  // P1-3: accumulated penalty from failed predictions
}, (table) => ({
  timestampIdx: index("patterns_timestamp_idx").on(table.timestamp),
}));

// ============================================================================
// PREDICTION EPISODES (P1-1) - Self-Learning Feedback Loop
// Every prediction is logged with matched patterns, then labeled with outcome
// ============================================================================
export const predictionEpisodes = pgTable("prediction_episodes", {
  id: serial("id").primaryKey(),
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),  // Candle being predicted
  symbol: varchar("symbol", { length: 20 }).default("BTCUSDT"),
  timeframe: varchar("timeframe", { length: 10 }).default("15m"),
  
  // Embedding used for prediction
  embedding: jsonb("embedding"),
  
  // Matched patterns (array of pattern IDs)
  matchedPatternIds: jsonb("matched_pattern_ids"),  // number[]
  matchedSimilarities: jsonb("matched_similarities"),  // number[]
  
  // Prediction output
  action: varchar("action", { length: 10 }),  // LONG, SHORT, HOLD
  entryPrice: real("entry_price"),
  suggestedSL: real("suggested_sl"),
  suggestedTP1: real("suggested_tp1"),
  suggestedTP2: real("suggested_tp2"),
  
  // Prediction metrics
  evLong: real("ev_long"),
  evShort: real("ev_short"),
  pWinLong: real("p_win_long"),
  pWinShort: real("p_win_short"),
  uncertainty: real("uncertainty"),
  confidence: real("confidence"),
  
  // Context
  regime: varchar("regime", { length: 20 }),
  volatilityBucket: varchar("volatility_bucket", { length: 10 }),
  horizon: integer("horizon").default(8),  // Forward candles for outcome
  
  // Outcome (filled later by daily feedback loop)
  outcome: varchar("outcome", { length: 20 }),  // WIN, LOSS, SCRATCH, PENDING
  outcomeTimestamp: bigint("outcome_timestamp", { mode: "number" }),
  actualReturn: real("actual_return"),
  actualMAE: real("actual_mae"),  // Max Adverse Excursion
  actualMFE: real("actual_mfe"),  // Max Favorable Excursion
  timeToOutcome: integer("time_to_outcome"),  // Candles until SL/TP hit
  hitTP: boolean("hit_tp"),
  hitSL: boolean("hit_sl"),
  
  // For hard-negative mining (P1-3)
  falsePositive: boolean("false_positive"),  // Did matched patterns mislead?
  
  createdAt: bigint("created_at", { mode: "number" }),
}, (table) => ({
  timestampIdx: index("prediction_episodes_timestamp_idx").on(table.timestamp),
  outcomeIdx: index("prediction_episodes_outcome_idx").on(table.outcome),
}));

export const signals = pgTable("signals", {
  id: serial("id").primaryKey(),
  timestamp: integer("timestamp").notNull(),
  signal: varchar("signal", { length: 10 }).notNull(),
  confidence: real("confidence").notNull(),
  probUp: real("prob_up"),
  probDown: real("prob_down"),
  probChop: real("prob_chop"),
  expectedMove: real("expected_move"),
  costs: real("costs"),
  edge: real("edge"),
  regime: varchar("regime", { length: 20 }),
  riskMode: varchar("risk_mode", { length: 20 }),
  vetoReasons: jsonb("veto_reasons"),
  supportReasons: jsonb("support_reasons"),
}, (table) => ({
  timestampIdx: index("signals_timestamp_idx").on(table.timestamp),
}));

export const sentimentData = pgTable("sentiment_data", {
  id: serial("id").primaryKey(),
  timestamp: integer("timestamp").notNull(),
  source: varchar("source", { length: 50 }).notNull(),
  value: real("value"),
  metadata: jsonb("metadata"),
}, (table) => ({
  timestampIdx: index("sentiment_timestamp_idx").on(table.timestamp),
  sourceIdx: index("sentiment_source_idx").on(table.source),
}));

export const backtestRuns = pgTable("backtest_runs", {
  id: serial("id").primaryKey(),
  startTs: bigint("start_ts", { mode: "number" }).notNull(),
  endTs: bigint("end_ts", { mode: "number" }).notNull(),
  trades: integer("trades").notNull(),
  winRate: real("win_rate"),
  totalPnlPct: real("total_pnl_pct"),
  sharpeRatio: real("sharpe_ratio"),
  maxDrawdownPct: real("max_drawdown_pct"),
  profitFactor: real("profit_factor"),
  modelVersion: varchar("model_version", { length: 50 }),
  parameters: jsonb("parameters"),
});

export const paperPortfolio = pgTable("paper_portfolio", {
  id: serial("id").primaryKey(),
  startingEquityUsdt: real("starting_equity_usdt").notNull(),
  currentEquityUsdt: real("current_equity_usdt").notNull(),
  availableBalanceUsdt: real("available_balance_usdt").notNull(),
  unrealizedPnlUsdt: real("unrealized_pnl_usdt").default(0),
  realizedPnlUsdt: real("realized_pnl_usdt").default(0),
  maxDrawdownPct: real("max_drawdown_pct").default(0),
  peakEquityUsdt: real("peak_equity_usdt").notNull(),
  totalTrades: integer("total_trades").default(0),
  winningTrades: integer("winning_trades").default(0),
  losingTrades: integer("losing_trades").default(0),
  updatedTs: bigint("updated_ts", { mode: "number" }).notNull(),
});

export const paperPositions = pgTable("paper_positions", {
  id: serial("id").primaryKey(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  side: varchar("side", { length: 10 }).notNull(),
  status: varchar("status", { length: 10 }).notNull(),
  entryTs: bigint("entry_ts", { mode: "number" }).notNull(),
  entryPrice: real("entry_price").notNull(),
  qty: real("qty").notNull(),
  notionalUsdt: real("notional_usdt").notNull(),
  leverage: real("leverage").default(1),
  stopLoss: real("stop_loss"),
  tp1: real("tp1"),
  tp2: real("tp2"),
  trailMode: varchar("trail_mode", { length: 20 }).default("none"),
  trailPrice: real("trail_price"),
  timeStopBars: integer("time_stop_bars"),
  barsOpen: integer("bars_open").default(0),
  primaryHorizon: integer("primary_horizon").default(15),
  initialRiskUsdt: real("initial_risk_usdt"),
  feesPaidUsdt: real("fees_paid_usdt").default(0),
  fundingPaidUsdt: real("funding_paid_usdt").default(0),
  exitTs: bigint("exit_ts", { mode: "number" }),
  exitPrice: real("exit_price"),
  realizedPnlUsdt: real("realized_pnl_usdt"),
  exitReason: varchar("exit_reason", { length: 20 }),
  signalConfidence: real("signal_confidence"),
  signalEdge: real("signal_edge"),
  v5Score: real("v5_score"),
  // MFE tracking and R-multiple fields
  peakProfit: real("peak_profit").default(0),           // Maximum favorable excursion (MFE) in USD
  initialStopDistance: real("initial_stop_distance"),   // Initial stop distance for R-multiple calc
  regime: varchar("regime", { length: 20 }),            // Market regime at entry
  source: varchar("source", { length: 20 }).default("manual"),
}, (table) => ({
  statusIdx: index("paper_positions_status_idx").on(table.status),
  entryTsIdx: index("paper_positions_entry_ts_idx").on(table.entryTs),
}));

export const paperTrades = pgTable("paper_trades", {
  id: serial("id").primaryKey(),
  positionId: integer("position_id").notNull(),
  ts: bigint("ts", { mode: "number" }).notNull(),
  action: varchar("action", { length: 20 }).notNull(),
  price: real("price").notNull(),
  qty: real("qty").notNull(),
  feeUsdt: real("fee_usdt").default(0),
  slippageUsdt: real("slippage_usdt").default(0),
  fundingUsdt: real("funding_usdt").default(0),
  pnlUsdt: real("pnl_usdt").default(0),
  reason: text("reason"),
}, (table) => ({
  positionIdIdx: index("paper_trades_position_id_idx").on(table.positionId),
  tsIdx: index("paper_trades_ts_idx").on(table.ts),
}));

export const paperEquityCurve = pgTable("paper_equity_curve", {
  id: serial("id").primaryKey(),
  ts: bigint("ts", { mode: "number" }).notNull(),
  equityUsdt: real("equity_usdt").notNull(),
  drawdownPct: real("drawdown_pct").default(0),
}, (table) => ({
  tsIdx: index("paper_equity_curve_ts_idx").on(table.ts),
}));

export const paperTradeHistory = pgTable("paper_trade_history", {
  id: serial("id").primaryKey(),
  positionId: integer("position_id").notNull(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  side: varchar("side", { length: 10 }).notNull(),
  entryTs: bigint("entry_ts", { mode: "number" }).notNull(),
  entryPrice: real("entry_price").notNull(),
  exitTs: bigint("exit_ts", { mode: "number" }).notNull(),
  exitPrice: real("exit_price").notNull(),
  grossR: real("gross_r"),
  netR: real("net_r"),
  costR: real("cost_r"),
  pnlUsdt: real("pnl_usdt"),
  riskUsdt: real("risk_usdt"),
  barsHeld: integer("bars_held"),
  exitReason: varchar("exit_reason", { length: 20 }),
  maxFavorableR: real("max_favorable_r"),
  regime: varchar("regime", { length: 20 }),
  signalConfidence: real("signal_confidence"),
  signalEdge: real("signal_edge"),
  createdAt: bigint("created_at", { mode: "number" }).notNull().$defaultFn(() => Date.now()),
}, (table) => ({
  symbolIdx: index("paper_trade_history_symbol_idx").on(table.symbol),
  exitTsIdx: index("paper_trade_history_exit_ts_idx").on(table.exitTs),
  positionIdIdx: index("paper_trade_history_position_id_idx").on(table.positionId),
}));

export const learningState = pgTable("learning_state", {
  id: serial("id").primaryKey(),
  key: varchar("key", { length: 50 }).notNull().unique(),
  lastTrainTs: bigint("last_train_ts", { mode: "number" }),
  lastFeatureTs: bigint("last_feature_ts", { mode: "number" }),
  lastIngestedTs: bigint("last_ingested_ts", { mode: "number" }),
  dataRangeStartTs: bigint("data_range_start_ts", { mode: "number" }),
  dataRangeEndTs: bigint("data_range_end_ts", { mode: "number" }),
  totalCandles: integer("total_candles").default(0),
  modelVersion: varchar("model_version", { length: 50 }),
  patternsVersion: varchar("patterns_version", { length: 50 }),
  trainingProgress: real("training_progress").default(0),
  epochsCompleted: integer("epochs_completed").default(0),
  isFrozen: boolean("is_frozen").default(false),
  totalPredictions: integer("total_predictions").default(0),
  historicalWinRate: real("historical_win_rate").default(0),
  backtestTrades: integer("backtest_trades").default(0),
  backfillComplete: boolean("backfill_complete").default(false),
  updatedTs: bigint("updated_ts", { mode: "number" }).notNull(),
});

export const patternClusters = pgTable("pattern_clusters", {
  id: serial("id").primaryKey(),
  clusterId: integer("cluster_id").notNull(),
  regime: varchar("regime", { length: 20 }).notNull(),
  centroid: jsonb("centroid").notNull(),
  sampleCount: integer("sample_count").default(0),
  winRate: real("win_rate").default(0),
  avgReturn: real("avg_return").default(0),
  avgMfe: real("avg_mfe").default(0),
  avgMae: real("avg_mae").default(0),
  avgTimeToMfe: real("avg_time_to_mfe").default(0),
  isMature: boolean("is_mature").default(false),
  createdTs: bigint("created_ts", { mode: "number" }).notNull(),
  updatedTs: bigint("updated_ts", { mode: "number" }).notNull(),
}, (table) => ({
  regimeIdx: index("pattern_clusters_regime_idx").on(table.regime),
  clusterIdIdx: index("pattern_clusters_cluster_id_idx").on(table.clusterId),
}));

export const socialMediaStats = pgTable("social_media_stats", {
  id: serial("id").primaryKey(),
  platform: varchar("platform", { length: 50 }).notNull(),
  itemsRead: integer("items_read").default(0),
  lastFetchTs: bigint("last_fetch_ts", { mode: "number" }),
  sentiment: real("sentiment").default(0),
  updatedTs: bigint("updated_ts", { mode: "number" }).notNull(),
}, (table) => ({
  platformIdx: index("social_media_stats_platform_idx").on(table.platform),
}));

export const backfillJobs = pgTable("backfill_jobs", {
  id: serial("id").primaryKey(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  timeframe: varchar("timeframe", { length: 10 }).notNull(),
  status: varchar("status", { length: 20 }).notNull().default("pending"),
  startTs: bigint("start_ts", { mode: "number" }),
  endTs: bigint("end_ts", { mode: "number" }),
  currentCursor: bigint("current_cursor", { mode: "number" }),
  candlesFetched: integer("candles_fetched").default(0),
  candlesExpected: integer("candles_expected").default(0),
  progressPct: real("progress_pct").default(0),
  errorMessage: text("error_message"),
  createdTs: bigint("created_ts", { mode: "number" }).notNull(),
  updatedTs: bigint("updated_ts", { mode: "number" }).notNull(),
});

export const strategyLearnerState = pgTable("strategy_learner_state", {
  id: serial("id").primaryKey(),
  epochsCompleted: integer("epochs_completed").default(0),
  totalSamples: integer("total_samples").default(0),
  modelAccuracy: real("model_accuracy").default(0),
  longWinRate: real("long_win_rate").default(0),
  shortWinRate: real("short_win_rate").default(0),
  holdWinRate: real("hold_win_rate").default(1),
  longExpectancy: real("long_expectancy").default(0),
  shortExpectancy: real("short_expectancy").default(0),
  holdExpectancy: real("hold_expectancy").default(0),
  longPnl: real("long_pnl").default(0),
  shortPnl: real("short_pnl").default(0),
  lastTrainingTs: bigint("last_training_ts", { mode: "number" }),
  trainingProgressIdx: integer("training_progress_idx").default(0),
  // Win/loss counts for proper restoration
  longWins: integer("long_wins").default(0),
  longLosses: integer("long_losses").default(0),
  shortWins: integer("short_wins").default(0),
  shortLosses: integer("short_losses").default(0),
  // Average PnL for wins and losses (to recreate distribution)
  avgWinPnl: real("avg_win_pnl").default(0),
  avgLossPnl: real("avg_loss_pnl").default(0),
  updatedTs: bigint("updated_ts", { mode: "number" }).notNull(),
});

// Shot Plan History - tracks shot plan signals and their outcomes
// ============================================================================
// SELF-LEARNING LOOP TABLES
// Continuous learning infrastructure with gated deployment
// ============================================================================

// Training runs - tracks candidate models and deployments
export const trainingRuns = pgTable("training_runs", {
  id: serial("id").primaryKey(),
  runId: varchar("run_id", { length: 64 }).notNull().unique(), // UUID
  status: varchar("status", { length: 20 }).notNull().default("pending"), // pending, training, evaluating, passed, failed, deployed, rollback
  modelType: varchar("model_type", { length: 50 }).notNull().default("transformer"),
  
  // Training parameters
  warmStartFromRunId: varchar("warm_start_from_run_id", { length: 64 }),
  learningRate: real("learning_rate"),
  epochs: integer("epochs"),
  newSamplesPct: real("new_samples_pct").default(0.3), // 30% new, 70% replay
  replaySamplesPct: real("replay_samples_pct").default(0.7),
  
  // Training data
  newSamplesCount: integer("new_samples_count").default(0),
  replaySamplesCount: integer("replay_samples_count").default(0),
  totalTrainingSamples: integer("total_training_samples").default(0),
  
  // Candidate metrics (from holdout evaluation)
  candidateExpectancy: real("candidate_expectancy"),
  candidateSharpe: real("candidate_sharpe"),
  candidateMaxDrawdown: real("candidate_max_drawdown"),
  candidateWinRate: real("candidate_win_rate"),
  candidateProfitFactor: real("candidate_profit_factor"),
  candidateTrades: integer("candidate_trades"),
  candidateStability: real("candidate_stability"), // Month-over-month consistency
  
  // Current model metrics (for comparison)
  currentExpectancy: real("current_expectancy"),
  currentSharpe: real("current_sharpe"),
  currentMaxDrawdown: real("current_max_drawdown"),
  currentWinRate: real("current_win_rate"),
  
  // Deployment decision
  beatsCurrent: boolean("beats_current").default(false),
  deploymentReason: text("deployment_reason"),
  isDeployed: boolean("is_deployed").default(false),
  deployedTs: bigint("deployed_ts", { mode: "number" }),
  
  // Checkpoint paths (on GPU trainer)
  checkpointPath: varchar("checkpoint_path", { length: 255 }),
  rollbackPath: varchar("rollback_path", { length: 255 }),
  
  startedTs: bigint("started_ts", { mode: "number" }),
  completedTs: bigint("completed_ts", { mode: "number" }),
  createdTs: bigint("created_ts", { mode: "number" }).notNull(),
  updatedTs: bigint("updated_ts", { mode: "number" }).notNull(),
}, (table) => ({
  statusIdx: index("training_runs_status_idx").on(table.status),
  deployedIdx: index("training_runs_deployed_idx").on(table.isDeployed),
}));

// Replay buffer - stores prioritized samples for training
export const replayBuffer = pgTable("replay_buffer", {
  id: serial("id").primaryKey(),
  sampleId: varchar("sample_id", { length: 64 }).notNull().unique(),
  
  // Sample data
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
  features: jsonb("features").notNull(), // Feature vector (JSON array)
  
  // Targets
  actualReturn: real("actual_return"),
  mfe: real("mfe"), // Max Favorable Excursion
  mae: real("mae"), // Max Adverse Excursion
  direction: varchar("direction", { length: 10 }), // LONG, SHORT, HOLD
  outcome: varchar("outcome", { length: 20 }), // WIN, LOSS, SCRATCH
  
  // Priority for importance sampling
  priority: real("priority").default(1.0), // Higher = more important
  timesUsed: integer("times_used").default(0), // How many times sampled for training
  lastUsedTs: bigint("last_used_ts", { mode: "number" }),
  
  // Context
  regime: varchar("regime", { length: 20 }),
  volatilityBucket: varchar("volatility_bucket", { length: 10 }),
  
  createdTs: bigint("created_ts", { mode: "number" }).notNull(),
}, (table) => ({
  priorityIdx: index("replay_buffer_priority_idx").on(table.priority),
  timestampIdx: index("replay_buffer_timestamp_idx").on(table.timestamp),
}));

// Labeled samples - samples awaiting training (horizon has matured)
export const labeledSamples = pgTable("labeled_samples", {
  id: serial("id").primaryKey(),
  sampleId: varchar("sample_id", { length: 64 }).notNull().unique(),
  
  // Candle info
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
  symbol: varchar("symbol", { length: 20 }).default("BTCUSDT"),
  timeframe: varchar("timeframe", { length: 10 }).default("15m"),
  
  // Features at prediction time
  features: jsonb("features").notNull(),
  currentPrice: real("current_price").notNull(),
  
  // Targets (filled when horizon matures)
  horizonBars: integer("horizon_bars").default(16), // How many bars forward
  actualReturn: real("actual_return"),
  mfe: real("mfe"),
  mae: real("mae"),
  direction: varchar("direction", { length: 10 }), // Actual direction (UP/DOWN/FLAT)
  
  // Labeling status
  status: varchar("status", { length: 20 }).notNull().default("pending"), // pending, labeled, used
  labeledTs: bigint("labeled_ts", { mode: "number" }),
  usedInRunId: varchar("used_in_run_id", { length: 64 }), // Which training run consumed this
  
  // Context
  regime: varchar("regime", { length: 20 }),
  
  createdTs: bigint("created_ts", { mode: "number" }).notNull(),
}, (table) => ({
  statusIdx: index("labeled_samples_status_idx").on(table.status),
  timestampIdx: index("labeled_samples_timestamp_idx").on(table.timestamp),
}));

// Learning job status - tracks the 15-minute self-learning job
export const learningJobStatus = pgTable("learning_job_status", {
  id: serial("id").primaryKey(),
  jobType: varchar("job_type", { length: 50 }).notNull().unique(), // gap_fill, labeling, training_trigger
  
  // Last run info
  lastRunTs: bigint("last_run_ts", { mode: "number" }),
  lastSuccessTs: bigint("last_success_ts", { mode: "number" }),
  lastErrorTs: bigint("last_error_ts", { mode: "number" }),
  lastError: text("last_error"),
  
  // Stats
  runsTotal: integer("runs_total").default(0),
  runsSuccess: integer("runs_success").default(0),
  runsFailed: integer("runs_failed").default(0),
  
  // Gap fill stats
  gapBarsMissing: integer("gap_bars_missing").default(0),
  lastCandleTs: bigint("last_candle_ts", { mode: "number" }),
  lastCandleClose: real("last_candle_close"),
  tickerPrice: real("ticker_price"),
  priceGapPct: real("price_gap_pct"),
  
  // Labeling stats
  pendingSamplesCount: integer("pending_samples_count").default(0),
  labeledSamplesCount: integer("labeled_samples_count").default(0),
  
  // Replay buffer stats
  replayBufferSize: integer("replay_buffer_size").default(0),
  replayBufferMaxSize: integer("replay_buffer_max_size").default(100000),
  
  // Training trigger
  samplesUntilTrain: integer("samples_until_train").default(0),
  trainThreshold: integer("train_threshold").default(1000), // Trigger training when N new samples
  
  // Deployed model info
  deployedModelId: varchar("deployed_model_id", { length: 64 }),
  deployedTs: bigint("deployed_ts", { mode: "number" }),
  
  updatedTs: bigint("updated_ts", { mode: "number" }).notNull(),
});

export const shotPlanHistory = pgTable("shot_plan_history", {
  id: serial("id").primaryKey(),
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
  signal: varchar("signal", { length: 10 }).notNull(), // LONG, SHORT, HOLD
  entryPrice: real("entry_price"),
  stopLoss: real("stop_loss"),
  takeProfit1: real("take_profit_1"),
  takeProfit2: real("take_profit_2"),
  confidence: real("confidence").notNull(),
  edge: real("edge"),
  regime: varchar("regime", { length: 50 }),
  // Outcome tracking
  outcome: varchar("outcome", { length: 20 }), // HIT_TP1, HIT_TP2, HIT_SL, EXPIRED, PENDING
  exitPrice: real("exit_price"),
  pnlPercent: real("pnl_percent"),
  exitTimestamp: bigint("exit_timestamp", { mode: "number" }),
  candlesHeld: integer("candles_held"),
  maxFavorableExcursion: real("max_favorable_excursion"),
  maxAdverseExcursion: real("max_adverse_excursion"),
});

export const insertShotPlanHistorySchema = createInsertSchema(shotPlanHistory).omit({ id: true });
export type InsertShotPlanHistory = z.infer<typeof insertShotPlanHistorySchema>;
export type ShotPlanHistoryEntry = typeof shotPlanHistory.$inferSelect;

// Cone-based trading signals - probabilistic signals from quantile predictions
export const coneSignals = pgTable("cone_signals", {
  id: serial("id").primaryKey(),
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
  
  // Signal info
  direction: varchar("direction", { length: 10 }).notNull(), // LONG, SHORT, HOLD
  
  // Entry/Exit levels
  entryPrice: real("entry_price").notNull(),
  stopLoss: real("stop_loss"),
  takeProfit: real("take_profit"),
  
  // Metrics
  mu: real("mu").notNull(), // Expected move (decimal)
  sigma: real("sigma"), // Uncertainty (decimal)
  edge: real("edge").notNull(), // abs(mu)/(q90-q10)
  riskReward: real("risk_reward"),
  
  // Quantiles (all as decimal returns)
  q10: real("q10").notNull(),
  q25: real("q25").notNull(),
  q50: real("q50").notNull(),
  q75: real("q75").notNull(),
  q90: real("q90").notNull(),
  
  // Probabilities
  probUp: real("prob_up"),
  probDown: real("prob_down"),
  probHold: real("prob_hold"),
  
  // Hold reasons (if HOLD)
  holdReasons: jsonb("hold_reasons").$type<string[]>(),
  
  // Auto-calibration
  edgeThreshold: real("edge_threshold"), // What threshold was used
  edgePercentile: real("edge_percentile"), // What percentile the threshold represents
  
  // Outcome tracking
  outcome: varchar("outcome", { length: 20 }), // HIT_TP, HIT_SL, EXPIRED, PENDING
  exitPrice: real("exit_price"),
  exitTimestamp: bigint("exit_timestamp", { mode: "number" }),
  pnlPercent: real("pnl_percent"),
  candlesHeld: integer("candles_held"),
  maxFavorableExcursion: real("max_favorable_excursion"),
  maxAdverseExcursion: real("max_adverse_excursion"),
  
  // Multi-head outputs (5-head model)
  volState: varchar("vol_state", { length: 20 }),  // contraction, neutral, expansion
  volStateProbs: jsonb("vol_state_probs"),  // { contraction: number, neutral: number, expansion: number }
  positionSizePct: real("position_size_pct"),
  isMultihead: boolean("is_multihead"),
  modelName: varchar("model_name", { length: 100 }),
  
  createdAt: bigint("created_at", { mode: "number" }).notNull(),
}, (table) => ({
  timestampIdx: index("cone_signals_timestamp_idx").on(table.timestamp),
  outcomeIdx: index("cone_signals_outcome_idx").on(table.outcome),
}));

export const multiheadPredictions = pgTable("multihead_predictions", {
  id: serial("id").primaryKey(),
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
  
  // Head 1: Classification
  action: varchar("action", { length: 10 }).notNull(),
  probLong: real("prob_long").notNull(),
  probShort: real("prob_short").notNull(),
  probHold: real("prob_hold").notNull(),
  confidence: real("confidence").notNull(),
  
  // Head 2: Quantile
  q10: real("q10"),
  q25: real("q25"),
  q50: real("q50"),
  q75: real("q75"),
  q90: real("q90"),
  
  // Head 3: Vol State
  volState: varchar("vol_state", { length: 20 }),
  volStateContraction: real("vol_state_contraction"),
  volStateNeutral: real("vol_state_neutral"),
  volStateExpansion: real("vol_state_expansion"),
  
  // Head 4: Mu (expected return)
  mu: real("mu"),
  
  // Head 5: Sigma (uncertainty)
  sigma: real("sigma"),
  
  // Derived trade levels
  edge: real("edge"),
  entryPrice: real("entry_price"),
  stopLossPrice: real("stop_loss_price"),
  takeProfitPrice: real("take_profit_price"),
  stopLossPct: real("stop_loss_pct"),
  takeProfitPct: real("take_profit_pct"),
  riskRewardRatio: real("risk_reward_ratio"),
  positionSizePct: real("position_size_pct"),
  
  // Metadata
  currentPrice: real("current_price"),
  modelName: varchar("model_name", { length: 100 }),
  isMultihead: boolean("is_multihead").default(true),
  urgency: varchar("urgency", { length: 10 }),
  suggestedOrderType: varchar("suggested_order_type", { length: 10 }),
  reasons: jsonb("reasons").$type<string[]>(),
  
  createdAt: bigint("created_at", { mode: "number" }).notNull(),
}, (table) => ({
  timestampIdx: index("multihead_predictions_timestamp_idx").on(table.timestamp),
}));

export const insertConeSignalSchema = createInsertSchema(coneSignals).omit({ id: true });
export type InsertConeSignal = z.infer<typeof insertConeSignalSchema>;
export type ConeSignal = typeof coneSignals.$inferSelect;

// Flow forecast schema for volatility regime and path projections
export const flowForecastSchema = z.object({
  volState: z.enum(["contraction", "neutral", "expansion"]),
  volStateProbs: z.object({
    contraction: z.number(),
    neutral: z.number(),
    expansion: z.number(),
  }),
  acceleration: z.number(),
  forecastMode: z.enum(["QUANTILE_PATHS", "NO_FORECAST"]),
  quantilePaths: z.object({
    q10: z.array(z.number()),
    q50: z.array(z.number()),
    q90: z.array(z.number()),
  }).optional(),
});
export type FlowForecast = z.infer<typeof flowForecastSchema>;

// Cone signal schema for API responses
export const coneSignalResponseSchema = z.object({
  direction: signalTypeSchema,
  entryPrice: z.number(),
  stopLoss: z.number().nullable(),
  takeProfit: z.number().nullable(),
  mu: z.number(),
  sigma: z.number().optional(),
  edge: z.number(),
  riskReward: z.number().nullable(),
  quantiles: z.object({
    q10: z.number(),
    q25: z.number(),
    q50: z.number(),
    q75: z.number(),
    q90: z.number(),
  }),
  probUp: z.number(),
  probDown: z.number(),
  probHold: z.number(),
  holdReasons: z.array(z.string()),
  edgeThreshold: z.number(),
  cooldownBarsRemaining: z.number(),
  timestamp: z.number(),
  flowForecast: flowForecastSchema.optional(),
});
export type ConeSignalResponse = z.infer<typeof coneSignalResponseSchema>;

// Settings table — key/value store for money config and other settings
export const settings = pgTable("settings", {
  id: serial("id").primaryKey(),
  key: varchar("key", { length: 100 }).notNull().unique(),
  valueJson: jsonb("value_json"),
  updatedAt: bigint("updated_at", { mode: "number" }).notNull(),
});

// Live trade records — pushed from GPU trainer when positions open/close
export const liveTradeRecords = pgTable("live_trade_records", {
  id: serial("id").primaryKey(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  side: varchar("side", { length: 10 }).notNull(),
  entryTime: bigint("entry_time", { mode: "number" }).notNull(),
  entryPrice: real("entry_price").notNull(),
  exitTime: bigint("exit_time", { mode: "number" }),
  exitPrice: real("exit_price"),
  stopLoss: real("stop_loss"),
  takeProfit: real("take_profit"),
  initialSl: real("initial_sl"),
  sizePct: real("size_pct"),
  pEnter: real("p_enter"),
  costsBps: real("costs_bps"),
  outcome: varchar("outcome", { length: 20 }),
  grossR: real("gross_r"),
  costR: real("cost_r"),
  netR: real("net_r"),
  sizedR: real("sized_r"),
  status: varchar("status", { length: 20 }).notNull().default("open"),
  reasons: jsonb("reasons").$type<string[]>(),
  pnlUsd: real("pnl_usd"),
  pnlUsdGross: real("pnl_usd_gross"),
  pnlUsdCost: real("pnl_usd_cost"),
  riskUsdUsed: real("risk_usd_used"),
  equitySnapshotUsd: real("equity_snapshot_usd"),
  barsHeld: integer("bars_held"),
  leverage: real("leverage"),
  modelVersion: varchar("model_version", { length: 50 }),
  notes: text("notes"),
  policy: varchar("policy", { length: 10 }),
  flowRiskMult: real("flow_risk_mult"),
  lane: varchar("lane", { length: 10 }),
  htfScore: integer("htf_score"),
  laneThresholdUsed: real("lane_threshold_used"),
  laneSizeMult: real("lane_size_mult"),
  exitReason: varchar("exit_reason", { length: 50 }),
  laneHorizon: integer("lane_horizon"),
  maxFavorableR: real("max_favorable_r"),
  maxAdverseR: real("max_adverse_r"),
  timeExit: boolean("time_exit"),
  breakevenMoved: boolean("breakeven_moved"),
  trailUpdates: integer("trail_updates").default(0),
  tmActions: jsonb("tm_actions").$type<Array<{ ts: number; action: string; reason: string; price?: number; sl?: number; ur?: number }>>(),
  createdAt: bigint("created_at", { mode: "number" }).notNull(),
}, (table) => ({
  symbolIdx: index("live_trades_symbol_idx").on(table.symbol),
  statusIdx: index("live_trades_status_idx").on(table.status),
  entryTimeIdx: index("live_trades_entry_time_idx").on(table.entryTime),
}));

// Per-symbol model learning stats — pushed after each retrain cycle
export const modelLearningStats = pgTable("model_learning_stats", {
  id: serial("id").primaryKey(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  modelVersion: varchar("model_version", { length: 50 }).notNull(),
  trainedUntilTs: bigint("trained_until_ts", { mode: "number" }),
  trainingSamples: integer("training_samples"),
  valPrAuc: real("val_pr_auc"),
  valPrecision: real("val_precision"),
  valRecall: real("val_recall"),
  valF1: real("val_f1"),
  bestPolicyThreshold: real("best_policy_threshold"),
  bestPolicyCooldown: integer("best_policy_cooldown"),
  bestPolicyTpMult: real("best_policy_tp_mult"),
  bestPolicySlMult: real("best_policy_sl_mult"),
  pfNet: real("pf_net"),
  eNet: real("e_net"),
  tradesPerDay: real("trades_per_day"),
  profitableRegimes: integer("profitable_regimes"),
  totalRegimes: integer("total_regimes"),
  promoted: boolean("promoted").default(false),
  promotionReason: varchar("promotion_reason", { length: 200 }),
  trend7d: varchar("trend_7d", { length: 20 }),
  prevPfNet: real("prev_pf_net"),
  prevENet: real("prev_e_net"),
  prevTradesPerDay: real("prev_trades_per_day"),
  createdAt: bigint("created_at", { mode: "number" }).notNull(),
}, (table) => ({
  symbolIdx: index("learning_stats_symbol_idx").on(table.symbol),
  createdAtIdx: index("learning_stats_created_at_idx").on(table.createdAt),
}));

// Live cycle logs — one row per 15m inference cycle per symbol
export const liveCycleLogs = pgTable("live_cycle_logs", {
  id: serial("id").primaryKey(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  cycleTs: bigint("cycle_ts", { mode: "number" }).notNull(),
  price: real("price"),
  pEnter: real("p_enter"),
  htfH1Trend: real("htf_h1_trend"),
  htfH4Trend: real("htf_h4_trend"),
  slopeOk: boolean("slope_ok"),
  rangeOk: boolean("range_ok"),
  direction: varchar("direction", { length: 10 }),
  thresholdUsed: real("threshold_used"),
  decision: varchar("decision", { length: 30 }).notNull(),
  reasons: jsonb("reasons").$type<string[]>(),
  policy: varchar("policy", { length: 10 }),
  coreThr: real("core_thr"),
  flowThr: real("flow_thr"),
  quotaStep: integer("quota_step"),
  flowPctUsed: real("flow_pct_used"),
  tradesTodayTotal: integer("trades_today_total"),
  tradesTodayTarget: integer("trades_today_target"),
  tradesTodayMax: integer("trades_today_max"),
  quotaFlowRiskMult: real("quota_flow_risk_mult"),
  laneSelected: varchar("lane_selected", { length: 10 }),
  htfScore: integer("htf_score"),
  scalpThr: real("scalp_thr"),
  laneSizeMult: real("lane_size_mult"),
  laneBudgetRemainingR: real("lane_budget_remaining_r"),
  holdReason: varchar("hold_reason", { length: 100 }),
  eNetPred: real("e_net_pred"),
  enterLogit: real("enter_logit"),
  temperatureUsed: real("temperature_used"),
  scalpAtrRatio: real("scalp_atr_ratio"),
  scalpTrZ: real("scalp_tr_z"),
  scalpBbZ: real("scalp_bb_z"),
  scalpEma20Slope: real("scalp_ema20_slope"),
  scalpMacdHist: real("scalp_macd_hist"),
  scalpVolRatio: real("scalp_vol_ratio"),
  scalpVolExpansionOk: boolean("scalp_vol_expansion_ok"),
  scalpMomentumOk: boolean("scalp_momentum_ok"),
  retMu: real("ret_mu"),
  mfePred: real("mfe_pred"),
  maePred: real("mae_pred"),
  pHold: real("p_hold"),
  pLong: real("p_long"),
  pShort: real("p_short"),
  v5Score: real("v5_score"),
  v5Threshold: real("v5_threshold"),
  v5Side: varchar("v5_side", { length: 10 }),
  createdAt: bigint("created_at", { mode: "number" }).notNull(),
}, (table) => ({
  symbolIdx: index("cycle_logs_symbol_idx").on(table.symbol),
  cycleTsIdx: index("cycle_logs_cycle_ts_idx").on(table.cycleTs),
}));

export const tradeEvents = pgTable("trade_events", {
  id: serial("id").primaryKey(),
  tradeId: integer("trade_id").notNull(),
  ts: bigint("ts", { mode: "number" }).notNull(),
  eventType: varchar("event_type", { length: 30 }).notNull(),
  payloadJson: jsonb("payload_json"),
}, (table) => ({
  tradeIdIdx: index("trade_events_trade_id_idx").on(table.tradeId),
  tsIdx: index("trade_events_ts_idx").on(table.ts),
}));

export const learningRuns = pgTable("learning_runs", {
  id: serial("id").primaryKey(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  startAt: bigint("start_at", { mode: "number" }).notNull(),
  endAt: bigint("end_at", { mode: "number" }),
  dataFrom: bigint("data_from", { mode: "number" }),
  dataTo: bigint("data_to", { mode: "number" }),
  newBars: integer("new_bars"),
  newTrades: integer("new_trades"),
  epochs: integer("epochs"),
  bestValLoss: real("best_val_loss"),
  prAuc: real("pr_auc"),
  pfNet: real("pf_net"),
  eNet: real("e_net"),
  profitableRegimes: integer("profitable_regimes"),
  totalRegimes: integer("total_regimes"),
  promoted: boolean("promoted").default(false),
  reason: text("reason"),
  modelVersion: varchar("model_version", { length: 50 }),
  metricsJson: jsonb("metrics_json"),
  status: varchar("status", { length: 20 }).default("running"),
}, (table) => ({
  symbolIdx: index("learning_runs_symbol_idx").on(table.symbol),
  startAtIdx: index("learning_runs_start_at_idx").on(table.startAt),
}));

export const healthStatus = pgTable("health_status", {
  id: serial("id").primaryKey(),
  ts: bigint("ts", { mode: "number" }).notNull(),
  component: varchar("component", { length: 50 }).notNull(),
  status: varchar("status", { length: 20 }).notNull(),
  message: text("message"),
}, (table) => ({
  componentIdx: index("health_component_idx").on(table.component),
  tsIdx: index("health_ts_idx").on(table.ts),
}));

export const ingestedEvents = pgTable("ingested_events", {
  id: serial("id").primaryKey(),
  eventId: varchar("event_id", { length: 100 }).notNull(),
  eventType: varchar("event_type", { length: 50 }).notNull(),
  ts: bigint("ts", { mode: "number" }).notNull(),
  payloadJson: jsonb("payload_json"),
  processedAt: bigint("processed_at", { mode: "number" }).notNull(),
}, (table) => ({
  eventIdUniqueIdx: uniqueIndex("ingested_events_event_id_unique").on(table.eventId),
  eventTypeIdx: index("ingested_events_type_idx").on(table.eventType),
  tsIdx: index("ingested_events_ts_idx").on(table.ts),
}));

export const insertSettingsSchema = createInsertSchema(settings).omit({ id: true });
export const insertCandleSchema = createInsertSchema(candles).omit({ id: true });
export const insertFeatureSchema = createInsertSchema(features).omit({ id: true });
export const insertPatternSchema = createInsertSchema(patterns).omit({ id: true });
export const insertSignalSchema = createInsertSchema(signals).omit({ id: true });
export const insertSentimentSchema = createInsertSchema(sentimentData).omit({ id: true });
export const insertPaperPortfolioSchema = createInsertSchema(paperPortfolio).omit({ id: true });
export const insertPaperPositionSchema = createInsertSchema(paperPositions).omit({ id: true });
export const insertPaperTradeSchema = createInsertSchema(paperTrades).omit({ id: true });
export const insertPaperTradeHistorySchema = createInsertSchema(paperTradeHistory).omit({ id: true });
export const insertPaperEquitySchema = createInsertSchema(paperEquityCurve).omit({ id: true });
export const insertLearningStateSchema = createInsertSchema(learningState).omit({ id: true });
export const insertPatternClusterSchema = createInsertSchema(patternClusters).omit({ id: true });
export const insertSocialMediaStatsSchema = createInsertSchema(socialMediaStats).omit({ id: true });
export const insertBackfillJobSchema = createInsertSchema(backfillJobs).omit({ id: true });
export const insertStrategyLearnerStateSchema = createInsertSchema(strategyLearnerState).omit({ id: true });
export const insertPredictionEpisodeSchema = createInsertSchema(predictionEpisodes).omit({ id: true });
export const insertTrainingRunSchema = createInsertSchema(trainingRuns).omit({ id: true });
export const insertReplayBufferSchema = createInsertSchema(replayBuffer).omit({ id: true });
export const insertLabeledSampleSchema = createInsertSchema(labeledSamples).omit({ id: true });
export const insertLearningJobStatusSchema = createInsertSchema(learningJobStatus).omit({ id: true });
export const insertMultiheadPredictionSchema = createInsertSchema(multiheadPredictions).omit({ id: true });
export const insertLiveTradeRecordSchema = createInsertSchema(liveTradeRecords).omit({ id: true });
export const insertModelLearningStatsSchema = createInsertSchema(modelLearningStats).omit({ id: true });
export const insertLiveCycleLogSchema = createInsertSchema(liveCycleLogs).omit({ id: true });
export const insertTradeEventSchema = createInsertSchema(tradeEvents).omit({ id: true });
export const insertLearningRunSchema = createInsertSchema(learningRuns).omit({ id: true });
export const insertHealthStatusSchema = createInsertSchema(healthStatus).omit({ id: true });
export const insertIngestedEventSchema = createInsertSchema(ingestedEvents).omit({ id: true });

export const ingestEventPayloadSchema = z.object({
  event_id: z.string().min(1),
  type: z.enum([
    "CYCLE_UPDATE",
    "TRADE_OPEN",
    "TRADE_UPDATE",
    "TRADE_CLOSE",
    "LEARNING_PROGRESS",
    "MODEL_PROMOTED",
    "HEALTH_STATUS",
    "SIGNAL_UPDATE",
    "TRAINING_SESSION_START",
    "TRAINING_SESSION_UPDATE",
    "TRAINING_SESSION_END",
    "TRAINING_EPOCH",
    "TRAINING_FOLD_START",
    "TRAINING_FOLD_END",
  ]),
  payload: z.record(z.unknown()),
  ts: z.number(),
});
export type IngestEventPayload = z.infer<typeof ingestEventPayloadSchema>;

export type InsertCandle = z.infer<typeof insertCandleSchema>;
export type InsertFeature = z.infer<typeof insertFeatureSchema>;
export type InsertPattern = z.infer<typeof insertPatternSchema>;
export type InsertSignal = z.infer<typeof insertSignalSchema>;
export type InsertSentiment = z.infer<typeof insertSentimentSchema>;
export type InsertPaperPortfolio = z.infer<typeof insertPaperPortfolioSchema>;
export type InsertPaperPosition = z.infer<typeof insertPaperPositionSchema>;
export type InsertPaperTrade = z.infer<typeof insertPaperTradeSchema>;
export type InsertPaperEquity = z.infer<typeof insertPaperEquitySchema>;
export type InsertLearningState = z.infer<typeof insertLearningStateSchema>;
export type InsertPatternCluster = z.infer<typeof insertPatternClusterSchema>;
export type InsertSocialMediaStats = z.infer<typeof insertSocialMediaStatsSchema>;
export type InsertBackfillJob = z.infer<typeof insertBackfillJobSchema>;
export type InsertStrategyLearnerState = z.infer<typeof insertStrategyLearnerStateSchema>;

export type DbCandle = typeof candles.$inferSelect;
export type DbFeature = typeof features.$inferSelect;
export type DbPattern = typeof patterns.$inferSelect;
export type DbSignal = typeof signals.$inferSelect;
export type DbSentimentData = typeof sentimentData.$inferSelect;
export type BacktestRun = typeof backtestRuns.$inferSelect;
export type PaperPortfolio = typeof paperPortfolio.$inferSelect;
export type PaperPosition = typeof paperPositions.$inferSelect;
export type PaperTrade = typeof paperTrades.$inferSelect;
export type PaperTradeHistory = typeof paperTradeHistory.$inferSelect;
export type PaperEquityCurve = typeof paperEquityCurve.$inferSelect;
export type LearningState = typeof learningState.$inferSelect;
export type PatternCluster = typeof patternClusters.$inferSelect;
export type SocialMediaStats = typeof socialMediaStats.$inferSelect;
export type BackfillJob = typeof backfillJobs.$inferSelect;
export type StrategyLearnerState = typeof strategyLearnerState.$inferSelect;
export type InsertPredictionEpisode = z.infer<typeof insertPredictionEpisodeSchema>;
export type PredictionEpisode = typeof predictionEpisodes.$inferSelect;
export type InsertTrainingRun = z.infer<typeof insertTrainingRunSchema>;
export type TrainingRun = typeof trainingRuns.$inferSelect;
export type InsertReplayBuffer = z.infer<typeof insertReplayBufferSchema>;
export type ReplayBufferSample = typeof replayBuffer.$inferSelect;
export type InsertLabeledSample = z.infer<typeof insertLabeledSampleSchema>;
export type LabeledSample = typeof labeledSamples.$inferSelect;
export type InsertLearningJobStatus = z.infer<typeof insertLearningJobStatusSchema>;
export type LearningJobStatus = typeof learningJobStatus.$inferSelect;
export type InsertMultiheadPrediction = z.infer<typeof insertMultiheadPredictionSchema>;
export type MultiheadPrediction = typeof multiheadPredictions.$inferSelect;
export type InsertLiveTradeRecord = z.infer<typeof insertLiveTradeRecordSchema>;
export type LiveTradeRecord = typeof liveTradeRecords.$inferSelect;
export type InsertModelLearningStats = z.infer<typeof insertModelLearningStatsSchema>;
export type ModelLearningStatsEntry = typeof modelLearningStats.$inferSelect;
export type InsertLiveCycleLog = z.infer<typeof insertLiveCycleLogSchema>;
export type LiveCycleLog = typeof liveCycleLogs.$inferSelect;
export type InsertTradeEvent = z.infer<typeof insertTradeEventSchema>;
export type TradeEventRow = typeof tradeEvents.$inferSelect;
export type InsertLearningRun = z.infer<typeof insertLearningRunSchema>;
export type LearningRunRow = typeof learningRuns.$inferSelect;
export type InsertHealthStatus = z.infer<typeof insertHealthStatusSchema>;
export type HealthStatusRow = typeof healthStatus.$inferSelect;
export type InsertIngestedEvent = z.infer<typeof insertIngestedEventSchema>;
export type IngestedEventRow = typeof ingestedEvents.$inferSelect;
export type InsertSettings = z.infer<typeof insertSettingsSchema>;
export type SettingsRow = typeof settings.$inferSelect;

export const trainingSessions = pgTable("training_sessions", {
  id: serial("id").primaryKey(),
  sessionType: varchar("session_type", { length: 30 }).notNull().default("walk_forward"),
  status: varchar("status", { length: 20 }).notNull().default("running"),
  startedAt: bigint("started_at", { mode: "number" }).notNull(),
  completedAt: bigint("completed_at", { mode: "number" }),
  totalFolds: integer("total_folds").default(0),
  completedFolds: integer("completed_folds").default(0),
  currentFold: integer("current_fold").default(0),
  totalEpochs: integer("total_epochs").default(0),
  currentEpoch: integer("current_epoch").default(0),
  symbols: text("symbols").array(),
  config: jsonb("config"),
  currentFoldMetrics: jsonb("current_fold_metrics"),
  aggregateMetrics: jsonb("aggregate_metrics"),
  gpuName: varchar("gpu_name", { length: 100 }),
  estimatedCompletionTs: bigint("estimated_completion_ts", { mode: "number" }),
  lastUpdateTs: bigint("last_update_ts", { mode: "number" }),
  trainMonths: integer("train_months"),
  testMonths: integer("test_months"),
  errorMessage: text("error_message"),
}, (table) => ({
  statusIdx: index("training_sessions_status_idx").on(table.status),
  startedAtIdx: index("training_sessions_started_at_idx").on(table.startedAt),
}));

export const trainingEpochs = pgTable("training_epochs", {
  id: serial("id").primaryKey(),
  sessionId: integer("session_id").notNull(),
  foldNum: integer("fold_num").notNull(),
  epoch: integer("epoch").notNull(),
  trainLoss: real("train_loss"),
  valLoss: real("val_loss"),
  lossBreakdown: jsonb("loss_breakdown"),
  actionAccuracy: real("action_accuracy"),
  learningRate: real("learning_rate"),
  expectancy: real("expectancy"),
  profitFactor: real("profit_factor"),
  winRate: real("win_rate"),
  maxDrawdown: real("max_drawdown"),
  tradesPerDay: real("trades_per_day"),
  threshold: real("threshold"),
  scoreDiag: jsonb("score_diag"),
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
}, (table) => ({
  sessionIdx: index("training_epochs_session_idx").on(table.sessionId),
  foldEpochIdx: index("training_epochs_fold_epoch_idx").on(table.sessionId, table.foldNum, table.epoch),
}));

export const trainingFolds = pgTable("training_folds", {
  id: serial("id").primaryKey(),
  sessionId: integer("session_id").notNull(),
  foldNum: integer("fold_num").notNull(),
  trainStart: varchar("train_start", { length: 20 }),
  trainEnd: varchar("train_end", { length: 20 }),
  testStart: varchar("test_start", { length: 20 }),
  testEnd: varchar("test_end", { length: 20 }),
  status: varchar("status", { length: 20 }).default("pending"),
  trades: integer("trades"),
  winRate: real("win_rate"),
  expectancy: real("expectancy"),
  profitFactor: real("profit_factor"),
  sharpe: real("sharpe"),
  maxDrawdown: real("max_drawdown"),
  totalR: real("total_r"),
  longShortRatio: varchar("long_short_ratio", { length: 20 }),
  perSymbol: jsonb("per_symbol"),
  startedAt: bigint("started_at", { mode: "number" }),
  completedAt: bigint("completed_at", { mode: "number" }),
  bestEpoch: integer("best_epoch"),
  finalThreshold: real("final_threshold"),
}, (table) => ({
  sessionIdx: index("training_folds_session_idx").on(table.sessionId),
  foldNumIdx: index("training_folds_fold_num_idx").on(table.sessionId, table.foldNum),
}));

export const insertTrainingSessionSchema = createInsertSchema(trainingSessions).omit({ id: true });
export const insertTrainingEpochSchema = createInsertSchema(trainingEpochs).omit({ id: true });
export const insertTrainingFoldSchema = createInsertSchema(trainingFolds).omit({ id: true });
export type InsertTrainingSession = z.infer<typeof insertTrainingSessionSchema>;
export type TrainingSession = typeof trainingSessions.$inferSelect;
export type InsertTrainingEpoch = z.infer<typeof insertTrainingEpochSchema>;
export type TrainingEpoch = typeof trainingEpochs.$inferSelect;
export type InsertTrainingFold = z.infer<typeof insertTrainingFoldSchema>;
export type TrainingFold = typeof trainingFolds.$inferSelect;

export const v5Signals = pgTable("v5_signals", {
  id: serial("id").primaryKey(),
  symbol: varchar("symbol", { length: 20 }).notNull(),
  direction: varchar("direction", { length: 10 }).notNull(),
  confidence: real("confidence").notNull(),
  score: real("score"),
  muR: real("mu_r"),
  pSide: real("p_side"),
  lane: varchar("lane", { length: 10 }),
  regime: varchar("regime", { length: 30 }),
  entryPrice: real("entry_price"),
  slPrice: real("sl_price"),
  tpPrice: real("tp_price"),
  thresholdUsed: real("threshold_used"),
  htfScore: real("htf_score"),
  sizeMultiplier: real("size_multiplier"),
  signalTs: bigint("signal_ts", { mode: "number" }).notNull(),
  createdAt: bigint("created_at", { mode: "number" }).notNull(),
}, (table) => ({
  symbolIdx: index("v5_signals_symbol_idx").on(table.symbol),
  signalTsIdx: index("v5_signals_ts_idx").on(table.signalTs),
}));

export const insertV5SignalSchema = createInsertSchema(v5Signals).omit({ id: true });
export type InsertV5Signal = z.infer<typeof insertV5SignalSchema>;
export type V5Signal = typeof v5Signals.$inferSelect;

export const moneyConfigSchema = z.object({
  account_equity_usd: z.number().min(0),
  risk_per_trade_pct: z.number().min(0).max(100).default(1.0),
  base_currency: z.string().default("USD"),
});
export type MoneyConfig = z.infer<typeof moneyConfigSchema>;
