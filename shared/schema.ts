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
  dataSource: z.enum(["coingecko", "cryptocompare", "binance", "none"]).optional(),
  dataError: z.string().nullable().optional(),
});
export type DashboardData = z.infer<typeof dashboardDataSchema>;

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
}, (table) => ({
  timestampIdx: index("patterns_timestamp_idx").on(table.timestamp),
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
  // MFE tracking and R-multiple fields
  peakProfit: real("peak_profit").default(0),           // Maximum favorable excursion (MFE) in USD
  initialStopDistance: real("initial_stop_distance"),   // Initial stop distance for R-multiple calc
  regime: varchar("regime", { length: 20 }),            // Market regime at entry
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

export const insertCandleSchema = createInsertSchema(candles).omit({ id: true });
export const insertFeatureSchema = createInsertSchema(features).omit({ id: true });
export const insertPatternSchema = createInsertSchema(patterns).omit({ id: true });
export const insertSignalSchema = createInsertSchema(signals).omit({ id: true });
export const insertSentimentSchema = createInsertSchema(sentimentData).omit({ id: true });
export const insertPaperPortfolioSchema = createInsertSchema(paperPortfolio).omit({ id: true });
export const insertPaperPositionSchema = createInsertSchema(paperPositions).omit({ id: true });
export const insertPaperTradeSchema = createInsertSchema(paperTrades).omit({ id: true });
export const insertPaperEquitySchema = createInsertSchema(paperEquityCurve).omit({ id: true });
export const insertLearningStateSchema = createInsertSchema(learningState).omit({ id: true });
export const insertPatternClusterSchema = createInsertSchema(patternClusters).omit({ id: true });
export const insertSocialMediaStatsSchema = createInsertSchema(socialMediaStats).omit({ id: true });
export const insertBackfillJobSchema = createInsertSchema(backfillJobs).omit({ id: true });
export const insertStrategyLearnerStateSchema = createInsertSchema(strategyLearnerState).omit({ id: true });

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
export type PaperEquityCurve = typeof paperEquityCurve.$inferSelect;
export type LearningState = typeof learningState.$inferSelect;
export type PatternCluster = typeof patternClusters.$inferSelect;
export type SocialMediaStats = typeof socialMediaStats.$inferSelect;
export type BackfillJob = typeof backfillJobs.$inferSelect;
export type StrategyLearnerState = typeof strategyLearnerState.$inferSelect;
