import { pgTable, text, serial, integer, bigint, real, timestamp, jsonb, boolean, index, varchar } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const candles = pgTable("candles", {
  id: serial("id").primaryKey(),
  timestamp: integer("timestamp").notNull(),
  timeframe: varchar("timeframe", { length: 10 }).notNull().default("15m"),
  open: real("open").notNull(),
  high: real("high").notNull(),
  low: real("low").notNull(),
  close: real("close").notNull(),
  volume: real("volume").notNull(),
}, (table) => ({
  timestampIdx: index("candles_timestamp_idx").on(table.timestamp),
  timeframeIdx: index("candles_timeframe_idx").on(table.timeframe),
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
  forwardWin: boolean("forward_win"),
  regime: varchar("regime", { length: 20 }),
  label: varchar("label", { length: 20 }),
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
  strategy: varchar("strategy", { length: 50 }),
  entryLow: real("entry_low"),
  entryHigh: real("entry_high"),
  stopLoss: real("stop_loss"),
  takeProfit1: real("take_profit_1"),
  takeProfit2: real("take_profit_2"),
  riskReward: real("risk_reward"),
  expectedHoldTime: varchar("expected_hold_time", { length: 50 }),
  shotPlan: jsonb("shot_plan"),
  reasons: jsonb("reasons"),
  vetoReasons: jsonb("veto_reasons"),
  patternMatches: jsonb("pattern_matches"),
  mlPredictions: jsonb("ml_predictions"),
}, (table) => ({
  timestampIdx: index("signals_timestamp_idx").on(table.timestamp),
}));

export const sentimentData = pgTable("sentiment_data", {
  id: serial("id").primaryKey(),
  timestamp: integer("timestamp").notNull(),
  fearGreedIndex: real("fear_greed_index"),
  fearGreedLabel: varchar("fear_greed_label", { length: 50 }),
  twitterSentiment: real("twitter_sentiment"),
  redditSentiment: real("reddit_sentiment"),
  newsSentiment: real("news_sentiment"),
  socialVolume: real("social_volume"),
  topNews: jsonb("top_news"),
}, (table) => ({
  timestampIdx: index("sentiment_timestamp_idx").on(table.timestamp),
}));

export const backtestRuns = pgTable("backtest_runs", {
  id: serial("id").primaryKey(),
  createdAt: timestamp("created_at").defaultNow(),
  startDate: integer("start_date").notNull(),
  endDate: integer("end_date").notNull(),
  strategy: varchar("strategy", { length: 50 }),
  totalTrades: integer("total_trades"),
  winRate: real("win_rate"),
  profitFactor: real("profit_factor"),
  sharpeRatio: real("sharpe_ratio"),
  maxDrawdown: real("max_drawdown"),
  totalReturn: real("total_return"),
  metrics: jsonb("metrics"),
});

export const insertCandleSchema = createInsertSchema(candles).omit({ id: true });
export const insertFeatureSchema = createInsertSchema(features).omit({ id: true });
export const insertPatternSchema = createInsertSchema(patterns).omit({ id: true });
export const insertSignalSchema = createInsertSchema(signals).omit({ id: true });
export const insertSentimentSchema = createInsertSchema(sentimentData).omit({ id: true });

export type InsertCandle = z.infer<typeof insertCandleSchema>;
export type InsertFeature = z.infer<typeof insertFeatureSchema>;
export type InsertPattern = z.infer<typeof insertPatternSchema>;
export type InsertSignal = z.infer<typeof insertSignalSchema>;
export type InsertSentiment = z.infer<typeof insertSentimentSchema>;

export type Candle = typeof candles.$inferSelect;
export type Feature = typeof features.$inferSelect;
export type Pattern = typeof patterns.$inferSelect;
export type Signal = typeof signals.$inferSelect;
export type SentimentData = typeof sentimentData.$inferSelect;
export type BacktestRun = typeof backtestRuns.$inferSelect;
