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
}

export const defaultConfig: PaperTradingConfig = {
  paperTradingEnabled: false,
  isAutoTrading: false,
  
  riskPerTradePct: 0.25,
  maxRiskPerTradePct: 0.5,
  maxAccountExposurePct: 100,
  minConfidence: 0.65,
  
  takerFeePct: 0.04,
  makerFeePct: 0.02,
  slippageBps: 2,
  
  minStopDistancePct: 0.15,
  atrStopMultiplier: 1.2,
  
  timeStopBars: 3,
  flipConfidenceThreshold: 0.75,
  flipEdgeMultiplier: 3,
  
  startingEquity: 10000,
  trailBufferAtrMultiplier: 0.2,
  minPnlForTimeStop: 0.003,
};

let currentConfig: PaperTradingConfig = { ...defaultConfig };

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
