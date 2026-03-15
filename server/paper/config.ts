export interface PaperTradingConfig {
  paperTradingEnabled: boolean;
  isAutoTrading: boolean;
  
  riskPerTradePct: number;
  maxRiskPerTradePct: number;
  maxAccountExposurePct: number;
  minConfidence: number;
  
  takerFeePct: number;
  makerFeePct: number;
  slippageBps: number;
  
  minStopDistancePct: number;
  atrStopMultiplier: number;
  
  timeStopBars: number;
  flipConfidenceThreshold: number;
  flipEdgeMultiplier: number;
  
  startingEquity: number;
  trailBufferAtrMultiplier: number;
  minPnlForTimeStop: number;
  
  // Regime-based ATR multipliers for stops
  trendStopMultiplier: number;      // 0.9x ATR for trend trades (tighter)
  chopStopMultiplier: number;       // 0.7x ATR for chop/mean-reversion (tightest)
  
  // Regime-based take profit targets (ATR multiples)
  trendExpansionTp1: number;        // 1.0x ATR TP1 when trend + expansion
  trendExpansionTp2: number;        // 1.8x ATR TP2 when trend + expansion
  trendNoExpansionTp1: number;      // 0.7x ATR TP1 when trend, no expansion
  chopTp1: number;                  // 0.6x ATR TP1 for chop/mean-reversion
  
  // MFE-aware trailing stops
  mfeTrailActivation: number;       // Activate trail when profit >= X * ATR_pct
  mfeGivebackPct: number;           // Exit remainder if giveback >= X% of peak
  mfeMinGiveback: number;           // Min giveback threshold (ATR multiple)
  
  // Failure stop parameters
  failureStopEnabled: boolean;      // Enable Kalman+MACD failure detection
  
  // Quality score gate
  minQualityScore: number;          // Minimum quality score to trade (0-100)
  
  // Performance metrics tracking
  trackRMultiple: boolean;          // Track R-multiple per trade
  
  // Signal-strength leverage tiers
  leverageEnabled: boolean;
  leverageTiers: { minScore: number; leverage: number }[];
  maxLeverage: number;
}

export const defaultConfig: PaperTradingConfig = {
  paperTradingEnabled: false,
  isAutoTrading: false,
  
  riskPerTradePct: 0.3,             // Base margin pct per trade — multiplied by exchange leverage (e.g. 25x × 0.3% = 7.5% effective risk)
  maxRiskPerTradePct: 0.6,          // Max base risk pct (50x × 0.6% = 30% max per trade)
  maxAccountExposurePct: 3000,      // 3000% allows up to 50x leveraged multi-symbol positions
  minConfidence: 0.35,              // Lowered from 65% to 35% - more aggressive
  
  // STANDARDIZED: 0.10% round-trip total (aligned with pattern-memory & signal-engine)
  // Formula: (takerFeePct * 2 + slippageBps * 2 / 100) / 100 = 0.0010
  takerFeePct: 0.04,    // 0.04% per side = 0.08% round trip
  makerFeePct: 0.02,    // Unused in current calculations
  slippageBps: 1,       // 0.01% per side = 0.02% round trip (adjusted from 2)
  
  minStopDistancePct: 0.15,
  atrStopMultiplier: 1.2,
  
  timeStopBars: 4,              // Exit if no progress after 4 bars
  flipConfidenceThreshold: 0.75,
  flipEdgeMultiplier: 3,
  
  startingEquity: 10000,
  trailBufferAtrMultiplier: 0.2,
  minPnlForTimeStop: 0.15,      // 0.15x ATR_pct minimum profit after time stop bars
  
  // Regime-based ATR multipliers for stops
  trendStopMultiplier: 0.9,     // Tighter stop for trend trades
  chopStopMultiplier: 0.7,      // Tightest stop for mean-reversion
  
  // Regime-based take profit targets (ATR multiples)
  // NOTE: TP must be >= Stop to ensure RR >= 1 (winners bigger than losers)
  trendExpansionTp1: 1.1,       // Trend + expansion: TP1 = 1.1x ATR (RR = 1.22 vs 0.9x stop)
  trendExpansionTp2: 2.0,       // Trend + expansion: TP2 = 2.0x ATR (let winners run)
  trendNoExpansionTp1: 1.0,     // Trend no expansion: TP1 = 1.0x ATR (RR = 1.11 vs 0.9x stop)
  chopTp1: 0.8,                 // Chop/mean-reversion: TP1 = 0.8x ATR (RR = 1.14 vs 0.7x stop)
  
  // MFE-aware trailing stops
  mfeTrailActivation: 1.5,      // Activate trail when profit >= 1.5x ATR_pct (raised from 0.6 to give trades room to breathe)
  mfeGivebackPct: 0.5,          // Exit if giveback >= 50% of TP1
  mfeMinGiveback: 0.35,         // Min giveback = 0.35x ATR_pct
  
  // Failure stop parameters
  failureStopEnabled: true,     // Enable Kalman+MACD failure detection
  
  // Quality score gate
  minQualityScore: 35,          // Lowered from 70 to 35 - more aggressive
  
  // Performance metrics tracking
  trackRMultiple: true,         // Track R-multiple per trade
  
  // Exchange leverage tiers (v5Score thresholds → exchange leverage multiplier)
  // These represent real exchange leverage (15x–50x). Combined with riskPerTradePct=0.3%:
  //   50x × 0.3% = 15% equity at risk for highest conviction signals
  //   15x × 0.3% = 4.5% equity at risk for minimum conviction signals
  // Applied to BOTH qty AND initialRiskUsdt to keep R-math correct at SL.
  leverageEnabled: true,
  leverageTiers: [
    { minScore: 10.0, leverage: 50 },
    { minScore: 7.0,  leverage: 35 },
    { minScore: 5.0,  leverage: 25 },
    { minScore: 3.0,  leverage: 20 },
    { minScore: 1.0,  leverage: 15 },
    { minScore: 0.02, leverage: 15 },
  ],
  maxLeverage: 50,
};

let currentConfig: PaperTradingConfig = { ...defaultConfig };

let analyticsClearedAfterTs: number = 0;

export function getAnalyticsClearedAfterTs(): number {
  return analyticsClearedAfterTs;
}

export async function setAnalyticsClearedAfterTs(ts: number): Promise<void> {
  analyticsClearedAfterTs = ts;
  try {
    const { db } = await import("../db");
    const { settings } = await import("@shared/schema");
    const { eq } = await import("drizzle-orm");
    const existing = await db.select().from(settings).where(eq(settings.key, "analytics_cleared_after_ts")).limit(1);
    if (existing.length > 0) {
      await db.update(settings).set({ valueJson: ts, updatedAt: Date.now() }).where(eq(settings.key, "analytics_cleared_after_ts"));
    } else {
      await db.insert(settings).values({ key: "analytics_cleared_after_ts", valueJson: ts, updatedAt: Date.now() });
    }
  } catch (err) {
    console.error("[Paper Config] Failed to persist analytics clear timestamp:", err);
  }
}

export async function loadAnalyticsClearedTs(): Promise<void> {
  try {
    const { db } = await import("../db");
    const { settings } = await import("@shared/schema");
    const { eq } = await import("drizzle-orm");
    const rows = await db.select().from(settings).where(eq(settings.key, "analytics_cleared_after_ts")).limit(1);
    if (rows.length > 0 && typeof rows[0].valueJson === "number") {
      analyticsClearedAfterTs = rows[0].valueJson;
    }
  } catch (err) {
    console.error("[Paper Config] Failed to load analytics clear timestamp:", err);
  }
}

export async function loadPaperState(): Promise<void> {
  console.log("[Paper Config] Paper trading starts disabled by default (in-memory state)");
}

export function getConfig(): PaperTradingConfig {
  return { ...currentConfig };
}

export function updateConfig(updates: Partial<PaperTradingConfig>): PaperTradingConfig {
  currentConfig = { ...currentConfig, ...updates };
  return { ...currentConfig };
}

export function resetConfig(): PaperTradingConfig {
  currentConfig = { ...defaultConfig };
  return { ...currentConfig };
}

export async function enablePaperTrading(): Promise<void> {
  currentConfig.paperTradingEnabled = true;
  console.log("[Paper Config] Paper trading enabled");
}

export async function disablePaperTrading(): Promise<void> {
  currentConfig.paperTradingEnabled = false;
  currentConfig.isAutoTrading = false;
  console.log("[Paper Config] Paper trading disabled");
}

export function isPaperTradingEnabled(): boolean {
  return currentConfig.paperTradingEnabled;
}

export async function startAutoTrading(): Promise<void> {
  currentConfig.isAutoTrading = true;
  console.log("[Paper Config] Auto-trading started");
}

export async function stopAutoTrading(): Promise<void> {
  currentConfig.isAutoTrading = false;
  console.log("[Paper Config] Auto-trading stopped");
}

export function isAutoTradingEnabled(): boolean {
  return currentConfig.isAutoTrading && currentConfig.paperTradingEnabled;
}

export function getTotalCostsPct(): number {
  return (currentConfig.takerFeePct * 2 + currentConfig.slippageBps * 2 / 100) / 100;
}
