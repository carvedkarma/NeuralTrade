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
  
  riskPerTradePct: 0.5,             // Increased from 0.25% to 0.5% risk per trade
  maxRiskPerTradePct: 1.0,          // Increased from 0.5% to 1.0% max risk
  maxAccountExposurePct: 500,       // AGGRESSIVE: Increased to 500% for 5x leverage futures
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
  mfeTrailActivation: 0.6,      // Activate trail when profit >= 0.6x ATR_pct
  mfeGivebackPct: 0.5,          // Exit if giveback >= 50% of TP1
  mfeMinGiveback: 0.35,         // Min giveback = 0.35x ATR_pct
  
  // Failure stop parameters
  failureStopEnabled: true,     // Enable Kalman+MACD failure detection
  
  // Quality score gate
  minQualityScore: 35,          // Lowered from 70 to 35 - more aggressive
  
  // Performance metrics tracking
  trackRMultiple: true,         // Track R-multiple per trade
  
  // Signal-strength leverage tiers (v5Score thresholds → leverage)
  // Calibrated to actual GPU trainer v5_score range (medians 0.44–8.99)
  leverageEnabled: true,
  leverageTiers: [
    { minScore: 8.0, leverage: 30 },
    { minScore: 5.0, leverage: 10 },
    { minScore: 2.0, leverage: 5 },
    { minScore: 0.8, leverage: 3 },
    { minScore: 0.3, leverage: 2 },
    { minScore: 0.02, leverage: 1 },
  ],
  maxLeverage: 30,
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
