import type {
  Candle,
  Signal,
  Trade,
  FuturesData,
  Feature,
  DashboardData,
  SignalType,
  RegimeType,
  RiskMode,
} from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  getDashboardData(): Promise<DashboardData>;
  refreshData(): void;
}

function generateCandles(count: number): Candle[] {
  const candles: Candle[] = [];
  const now = Date.now();
  const interval = 15 * 60 * 1000; // 15 minutes
  
  let basePrice = 95000 + Math.random() * 10000; // BTC price around 95k-105k
  
  for (let i = count - 1; i >= 0; i--) {
    const timestamp = now - i * interval;
    const volatility = 0.002 + Math.random() * 0.003; // 0.2% to 0.5% volatility
    const trend = Math.random() > 0.5 ? 1 : -1;
    const movement = basePrice * volatility * trend;
    
    const open = basePrice;
    const close = basePrice + movement;
    const high = Math.max(open, close) + basePrice * volatility * Math.random() * 0.5;
    const low = Math.min(open, close) - basePrice * volatility * Math.random() * 0.5;
    const volume = 100000000 + Math.random() * 500000000;
    
    candles.push({
      timestamp,
      open: Math.round(open * 100) / 100,
      high: Math.round(high * 100) / 100,
      low: Math.round(low * 100) / 100,
      close: Math.round(close * 100) / 100,
      volume: Math.round(volume),
    });
    
    basePrice = close;
  }
  
  return candles;
}

function generateSignal(candles: Candle[]): Signal {
  const lastCandle = candles[candles.length - 1];
  const recentCandles = candles.slice(-20);
  
  // Calculate simple momentum
  const priceChange = (lastCandle.close - recentCandles[0].close) / recentCandles[0].close;
  const volatility = recentCandles.reduce((acc, c) => acc + Math.abs(c.close - c.open), 0) / recentCandles.length / lastCandle.close;
  
  // Simulate regime probabilities
  let probUp = 0.33;
  let probDown = 0.33;
  let probChop = 0.34;
  
  if (priceChange > 0.005) {
    probUp = 0.55 + Math.random() * 0.15;
    probDown = 0.15 + Math.random() * 0.1;
    probChop = 1 - probUp - probDown;
  } else if (priceChange < -0.005) {
    probDown = 0.55 + Math.random() * 0.15;
    probUp = 0.15 + Math.random() * 0.1;
    probChop = 1 - probUp - probDown;
  } else {
    probChop = 0.5 + Math.random() * 0.15;
    probUp = (1 - probChop) * 0.5;
    probDown = (1 - probChop) * 0.5;
  }
  
  // Determine signal
  let signal: SignalType = "HOLD";
  let confidence = 0.5;
  
  if (probChop < 0.45) {
    if (probUp > 0.6) {
      signal = "LONG";
      confidence = probUp;
    } else if (probDown > 0.6) {
      signal = "SHORT";
      confidence = probDown;
    }
  }
  
  // Calculate expected move and costs
  const expectedMove = volatility * 8 * (signal === "LONG" ? 1 : signal === "SHORT" ? -1 : 0);
  const costs = 0.0008 + Math.random() * 0.0002; // fees + slippage
  const edge = Math.abs(expectedMove) - costs;
  
  // Determine regime
  let regime: RegimeType = "chop";
  if (probUp > probDown && probUp > probChop) regime = "trend_up";
  else if (probDown > probUp && probDown > probChop) regime = "trend_down";
  
  // Determine risk mode
  let riskMode: RiskMode = "normal";
  if (volatility > 0.008) riskMode = "high_vol";
  if (probChop > 0.6 || volatility > 0.012) riskMode = "no_trade";
  
  // Generate top features
  const features: Feature[] = [
    {
      name: "Returns 8-bar",
      value: priceChange,
      importance: priceChange > 0 ? 0.85 : -0.85,
      description: "Price return over the last 8 candles (2 hours)",
    },
    {
      name: "OI Change",
      value: 0.02 + Math.random() * 0.04 - 0.02,
      importance: 0.6 * (Math.random() > 0.5 ? 1 : -1),
      description: "Change in open interest - rising OI with rising price suggests trend continuation",
    },
    {
      name: "Funding Rate",
      value: 0.0001 + Math.random() * 0.0003 - 0.0002,
      importance: 0.45 * (Math.random() > 0.5 ? 1 : -1),
      description: "Current funding rate - extreme values suggest potential reversal",
    },
    {
      name: "Long/Short Ratio",
      value: 0.9 + Math.random() * 0.4,
      importance: 0.35 * (Math.random() > 0.5 ? 1 : -1),
      description: "Ratio of long to short positions - contrarian indicator at extremes",
    },
    {
      name: "Volatility Regime",
      value: volatility * 100,
      importance: -0.25 * (volatility > 0.006 ? 1 : -1),
      description: "Current volatility level relative to historical average",
    },
  ];
  
  return {
    timestamp: lastCandle.timestamp,
    signal,
    confidence,
    probUp,
    probDown,
    probChop,
    expectedMove,
    costs,
    edge,
    regime,
    riskMode,
    topFeatures: features,
  };
}

function generateFuturesData(lastPrice: number): FuturesData {
  const fundingRate = 0.0001 + Math.random() * 0.0004 - 0.0002;
  const openInterest = 25000000000 + Math.random() * 10000000000;
  const oiChange15m = Math.random() * 0.04 - 0.02;
  const oiChange1h = Math.random() * 0.08 - 0.04;
  const longShortRatio = 0.85 + Math.random() * 0.3;
  const liquidations15m = 1000000 + Math.random() * 10000000;
  const liquidations1h = 5000000 + Math.random() * 30000000;
  const indexPrice = lastPrice * (1 - 0.0001);
  const markPrice = lastPrice;
  const basis = (markPrice - indexPrice) / indexPrice;
  
  return {
    fundingRate,
    nextFundingTime: Date.now() + (8 - (new Date().getHours() % 8)) * 3600000,
    openInterest,
    oiChange15m,
    oiChange1h,
    longShortRatio,
    liquidations15m,
    liquidations1h,
    markPrice,
    indexPrice,
    basis,
  };
}

function generateTrades(): Trade[] {
  const trades: Trade[] = [];
  const now = Date.now();
  
  // Generate some historical trades
  for (let i = 0; i < 8; i++) {
    const timestamp = now - (i + 1) * 4 * 60 * 60 * 1000; // Every 4 hours
    const isLong = Math.random() > 0.5;
    const entryPrice = 95000 + Math.random() * 10000;
    const profitLoss = Math.random() > 0.55 ? 1 : -1;
    const movePercent = 0.002 + Math.random() * 0.008;
    const exitPrice = entryPrice * (1 + (isLong ? profitLoss : -profitLoss) * movePercent);
    const pnlPercent = (isLong ? (exitPrice - entryPrice) : (entryPrice - exitPrice)) / entryPrice * 100;
    
    trades.push({
      id: randomUUID(),
      timestamp,
      side: isLong ? "LONG" : "SHORT",
      entryPrice: Math.round(entryPrice * 100) / 100,
      exitPrice: Math.round(exitPrice * 100) / 100,
      size: 0.1 + Math.random() * 0.2,
      pnl: pnlPercent * 100,
      pnlPercent: Math.round(pnlPercent * 100) / 100,
      status: "closed",
      stopLoss: Math.round((entryPrice * (isLong ? 0.985 : 1.015)) * 100) / 100,
      takeProfit: Math.round((entryPrice * (isLong ? 1.02 : 0.98)) * 100) / 100,
    });
  }
  
  // Sometimes add an open trade
  if (Math.random() > 0.5) {
    const isLong = Math.random() > 0.5;
    const entryPrice = 95000 + Math.random() * 10000;
    
    trades.unshift({
      id: randomUUID(),
      timestamp: now - 30 * 60 * 1000,
      side: isLong ? "LONG" : "SHORT",
      entryPrice: Math.round(entryPrice * 100) / 100,
      exitPrice: null,
      size: 0.15,
      pnl: null,
      pnlPercent: null,
      status: "open",
      stopLoss: Math.round((entryPrice * (isLong ? 0.985 : 1.015)) * 100) / 100,
      takeProfit: Math.round((entryPrice * (isLong ? 1.02 : 0.98)) * 100) / 100,
    });
  }
  
  return trades;
}

export class MemStorage implements IStorage {
  private candles: Candle[] = [];
  private trades: Trade[] = [];
  private equity = 10000;
  private lastRefresh = 0;

  constructor() {
    this.refreshData();
  }

  refreshData(): void {
    this.candles = generateCandles(96); // 24 hours of 15m candles
    this.trades = generateTrades();
    this.equity = 10000 + Math.random() * 2000 - 500;
    this.lastRefresh = Date.now();
  }

  async getDashboardData(): Promise<DashboardData> {
    // Refresh data if it's older than 15 seconds
    if (Date.now() - this.lastRefresh > 15000) {
      this.refreshData();
    }

    const signal = generateSignal(this.candles);
    const lastPrice = this.candles[this.candles.length - 1].close;
    const futuresData = generateFuturesData(lastPrice);
    
    // Calculate stats
    const closedTrades = this.trades.filter(t => t.status === "closed");
    const winningTrades = closedTrades.filter(t => (t.pnlPercent ?? 0) > 0);
    const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;
    const totalProfit = winningTrades.reduce((acc, t) => acc + (t.pnlPercent ?? 0), 0);
    const totalLoss = Math.abs(closedTrades.filter(t => (t.pnlPercent ?? 0) < 0).reduce((acc, t) => acc + (t.pnlPercent ?? 0), 0));
    const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;
    
    const drawdown = -0.01 - Math.random() * 0.03;
    const maxDrawdown = -0.03 - Math.random() * 0.04;
    const dailyPnl = -0.5 + Math.random() * 2;
    const exposure = this.trades.some(t => t.status === "open") ? 0.15 + Math.random() * 0.1 : 0;

    return {
      candles: this.candles.slice(-48), // Last 12 hours
      currentSignal: signal,
      futuresData,
      recentTrades: this.trades,
      equity: this.equity,
      drawdown,
      maxDrawdown,
      dailyPnl,
      winRate,
      profitFactor,
      totalTrades: closedTrades.length,
      exposure,
    };
  }
}

export const storage = new MemStorage();
