export interface PaperTradingConfig {
  riskPerTradePct: number;
  maxLeverage: number;
  takerFeePct: number;
  makerFeePct: number;
  slippageBps: number;
  timeStopBars: number;
  flipConfidenceThreshold: number;
  flipEdgeMultiplier: number;
  startingEquity: number;
  trailBufferAtrMultiplier: number;
  minPnlForTimeStop: number;
}

export const defaultConfig: PaperTradingConfig = {
  riskPerTradePct: 0.5,
  maxLeverage: 5,
  takerFeePct: 0.04,
  makerFeePct: 0.02,
  slippageBps: 2,
  timeStopBars: 3,
  flipConfidenceThreshold: 0.75,
  flipEdgeMultiplier: 3,
  startingEquity: 10000,
  trailBufferAtrMultiplier: 0.2,
  minPnlForTimeStop: 0.003,
};

let currentConfig: PaperTradingConfig = { ...defaultConfig };

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
