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
  KalmanState,
  StrategySignal,
  StrategyState,
} from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  getDashboardData(): Promise<DashboardData>;
  refreshData(): void;
  startStrategy(): void;
  stopStrategy(): void;
  updateStrategySettings(settings: Partial<StrategyState>): void;
  getStrategyState(): StrategyState;
}

class KalmanFilter {
  private x: number;
  private P: number;
  private Q: number;
  private R: number;

  constructor(len: number, initialValue: number) {
    this.x = initialValue;
    this.P = 1;
    this.R = 1;
    this.Q = 2 / (len + 1);
  }

  update(measurement: number): number {
    this.P = this.P + this.Q;
    const K = this.P / (this.P + this.R);
    this.x = this.x + K * (measurement - this.x);
    this.P = (1 - K) * this.P;
    return this.x;
  }

  getValue(): number {
    return this.x;
  }

  getState(): KalmanState {
    return { x: this.x, P: this.P };
  }
}

function calculateATR(candles: Candle[], period: number = 14): number {
  if (candles.length < period + 1) return 0;
  
  let atrSum = 0;
  for (let i = candles.length - period; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1]?.close ?? candles[i].open;
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    atrSum += tr;
  }
  return atrSum / period;
}

function generateCandles(count: number): Candle[] {
  const candles: Candle[] = [];
  const now = Date.now();
  const interval = 15 * 60 * 1000;
  
  let basePrice = 95000 + Math.random() * 10000;
  
  for (let i = count - 1; i >= 0; i--) {
    const timestamp = now - i * interval;
    const volatility = 0.002 + Math.random() * 0.003;
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

function generateSignal(candles: Candle[], kalmanFast: number, kalmanSlow: number): Signal {
  const lastCandle = candles[candles.length - 1];
  const recentCandles = candles.slice(-20);
  
  const priceChange = (lastCandle.close - recentCandles[0].close) / recentCandles[0].close;
  const volatility = recentCandles.reduce((acc, c) => acc + Math.abs(c.close - c.open), 0) / recentCandles.length / lastCandle.close;
  
  const isBullRegime = kalmanFast > kalmanSlow;
  
  let probUp = 0.33;
  let probDown = 0.33;
  let probChop = 0.34;
  
  if (isBullRegime) {
    probUp = 0.5 + Math.random() * 0.2;
    probDown = 0.15 + Math.random() * 0.1;
    probChop = 1 - probUp - probDown;
  } else {
    probDown = 0.5 + Math.random() * 0.2;
    probUp = 0.15 + Math.random() * 0.1;
    probChop = 1 - probUp - probDown;
  }
  
  let signal: SignalType = "HOLD";
  let confidence = 0.5;
  
  if (probChop < 0.45) {
    if (probUp > 0.55) {
      signal = "LONG";
      confidence = probUp;
    } else if (probDown > 0.55) {
      signal = "SHORT";
      confidence = probDown;
    }
  }
  
  const expectedMove = volatility * 8 * (signal === "LONG" ? 1 : signal === "SHORT" ? -1 : 0);
  const costs = 0.0008 + Math.random() * 0.0002;
  const edge = Math.abs(expectedMove) - costs;
  
  let regime: RegimeType = "chop";
  if (isBullRegime) regime = "trend_up";
  else regime = "trend_down";
  
  let riskMode: RiskMode = "normal";
  if (volatility > 0.008) riskMode = "high_vol";
  if (probChop > 0.6 || volatility > 0.012) riskMode = "no_trade";
  
  const features: Feature[] = [
    {
      name: "Kalman Fast",
      value: kalmanFast,
      importance: isBullRegime ? 0.9 : -0.9,
      description: "Fast Kalman filter (70-period) - quick trend indicator",
    },
    {
      name: "Kalman Slow",
      value: kalmanSlow,
      importance: isBullRegime ? 0.7 : -0.7,
      description: "Slow Kalman filter (250-period) - trend baseline",
    },
    {
      name: "Kalman Spread",
      value: ((kalmanFast - kalmanSlow) / kalmanSlow) * 100,
      importance: isBullRegime ? 0.85 : -0.85,
      description: "Difference between fast and slow Kalman as % - trend strength",
    },
    {
      name: "ATR(14)",
      value: calculateATR(candles, 14),
      importance: volatility > 0.006 ? -0.4 : 0.4,
      description: "Average True Range - volatility measure for stops",
    },
    {
      name: "Price vs Fast",
      value: ((lastCandle.close - kalmanFast) / kalmanFast) * 100,
      importance: lastCandle.close > kalmanFast ? 0.5 : -0.5,
      description: "Price position relative to fast Kalman",
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

export class MemStorage implements IStorage {
  private candles: Candle[] = [];
  private trades: Trade[] = [];
  private equity = 10000;
  private lastRefresh = 0;
  private kalmanFastFilter: KalmanFilter | null = null;
  private kalmanSlowFilter: KalmanFilter | null = null;
  private kalmanFastValues: number[] = [];
  private kalmanSlowValues: number[] = [];
  private activeTrade: Trade | null = null;
  private prevKalmanFast = 0;
  private prevKalmanSlow = 0;
  private waitingForRetest = false;
  private retestDirection: "LONG" | "SHORT" | null = null;
  
  private strategyState: StrategyState = {
    isRunning: false,
    useRetestSignals: true,
    riskPercent: 1,
    atrMultiplier: 1.3,
    timeStopCandles: 3,
  };

  constructor() {
    this.refreshData();
  }

  startStrategy(): void {
    this.strategyState.isRunning = true;
    this.waitingForRetest = false;
    this.retestDirection = null;
  }

  stopStrategy(): void {
    this.strategyState.isRunning = false;
    if (this.activeTrade) {
      this.closeTrade(this.candles[this.candles.length - 1].close, "manual_stop");
    }
  }

  updateStrategySettings(settings: Partial<StrategyState>): void {
    this.strategyState = { ...this.strategyState, ...settings };
  }

  getStrategyState(): StrategyState {
    return this.strategyState;
  }

  private initializeKalmanFilters(): void {
    if (this.candles.length === 0) return;
    
    const initialPrice = this.candles[0].close;
    this.kalmanFastFilter = new KalmanFilter(70, initialPrice);
    this.kalmanSlowFilter = new KalmanFilter(250, initialPrice);
    
    this.kalmanFastValues = [];
    this.kalmanSlowValues = [];
    
    for (const candle of this.candles) {
      const fastVal = this.kalmanFastFilter.update(candle.close);
      const slowVal = this.kalmanSlowFilter.update(candle.close);
      this.kalmanFastValues.push(fastVal);
      this.kalmanSlowValues.push(slowVal);
    }
    
    if (this.kalmanFastValues.length >= 2) {
      this.prevKalmanFast = this.kalmanFastValues[this.kalmanFastValues.length - 2];
      this.prevKalmanSlow = this.kalmanSlowValues[this.kalmanSlowValues.length - 2];
    }
  }

  private detectCrossover(): { crossed: boolean; direction: "LONG" | "SHORT" | null } {
    if (this.kalmanFastValues.length < 2) {
      return { crossed: false, direction: null };
    }
    
    const currFast = this.kalmanFastValues[this.kalmanFastValues.length - 1];
    const currSlow = this.kalmanSlowValues[this.kalmanSlowValues.length - 1];
    
    const prevAbove = this.prevKalmanFast > this.prevKalmanSlow;
    const currAbove = currFast > currSlow;
    
    if (!prevAbove && currAbove) {
      return { crossed: true, direction: "LONG" };
    } else if (prevAbove && !currAbove) {
      return { crossed: true, direction: "SHORT" };
    }
    
    return { crossed: false, direction: null };
  }

  private detectRetest(): { retest: boolean; direction: "LONG" | "SHORT" | null } {
    if (this.candles.length < 2) {
      return { retest: false, direction: null };
    }
    
    const lastCandle = this.candles[this.candles.length - 1];
    const currFast = this.kalmanFastValues[this.kalmanFastValues.length - 1];
    const currSlow = this.kalmanSlowValues[this.kalmanSlowValues.length - 1];
    const atr = calculateATR(this.candles, 14);
    const tol = 0.25 * atr;
    
    const isBullRegime = currFast > currSlow;
    
    if (isBullRegime) {
      const touchedFast = lastCandle.low <= currFast + tol;
      const closedAbove = lastCandle.close > currFast;
      const bullishCandle = lastCandle.close > lastCandle.open;
      
      if (touchedFast && closedAbove && bullishCandle) {
        return { retest: true, direction: "LONG" };
      }
    } else {
      const touchedFast = lastCandle.high >= currFast - tol;
      const closedBelow = lastCandle.close < currFast;
      const bearishCandle = lastCandle.close < lastCandle.open;
      
      if (touchedFast && closedBelow && bearishCandle) {
        return { retest: true, direction: "SHORT" };
      }
    }
    
    return { retest: false, direction: null };
  }

  private openTrade(direction: "LONG" | "SHORT", signalType: "crossover" | "retest"): void {
    const lastCandle = this.candles[this.candles.length - 1];
    const atr = calculateATR(this.candles, 14);
    const stopDistance = atr * this.strategyState.atrMultiplier;
    
    const entryPrice = lastCandle.close;
    const stopLoss = direction === "LONG" 
      ? entryPrice - stopDistance 
      : entryPrice + stopDistance;
    const takeProfit = direction === "LONG"
      ? entryPrice + stopDistance * 2
      : entryPrice - stopDistance * 2;
    
    const riskAmount = this.equity * (this.strategyState.riskPercent / 100);
    const size = riskAmount / stopDistance;
    
    this.activeTrade = {
      id: randomUUID(),
      timestamp: lastCandle.timestamp,
      side: direction,
      entryPrice: Math.round(entryPrice * 100) / 100,
      exitPrice: null,
      size: Math.round(size * 10000) / 10000,
      pnl: null,
      pnlPercent: null,
      status: "open",
      stopLoss: Math.round(stopLoss * 100) / 100,
      takeProfit: Math.round(takeProfit * 100) / 100,
      entryCandle: this.candles.length - 1,
      signalType,
    };
  }

  private closeTrade(exitPrice: number, reason: string): void {
    if (!this.activeTrade) return;
    
    const priceDiff = this.activeTrade.side === "LONG"
      ? exitPrice - this.activeTrade.entryPrice
      : this.activeTrade.entryPrice - exitPrice;
    
    const pnl = priceDiff * this.activeTrade.size;
    const pnlPercent = (priceDiff / this.activeTrade.entryPrice) * 100;
    
    const closedTrade: Trade = {
      ...this.activeTrade,
      exitPrice: Math.round(exitPrice * 100) / 100,
      pnl: Math.round(pnl * 100) / 100,
      pnlPercent: Math.round(pnlPercent * 100) / 100,
      status: "closed",
    };
    
    this.trades.unshift(closedTrade);
    this.equity += pnl;
    this.activeTrade = null;
  }

  private checkExitConditions(): void {
    if (!this.activeTrade) return;
    
    const lastCandle = this.candles[this.candles.length - 1];
    const currentPrice = lastCandle.close;
    
    if (this.activeTrade.side === "LONG") {
      if (lastCandle.low <= this.activeTrade.stopLoss) {
        this.closeTrade(this.activeTrade.stopLoss, "stop_loss");
        return;
      }
      if (lastCandle.high >= this.activeTrade.takeProfit) {
        this.closeTrade(this.activeTrade.takeProfit, "take_profit");
        return;
      }
    } else {
      if (lastCandle.high >= this.activeTrade.stopLoss) {
        this.closeTrade(this.activeTrade.stopLoss, "stop_loss");
        return;
      }
      if (lastCandle.low <= this.activeTrade.takeProfit) {
        this.closeTrade(this.activeTrade.takeProfit, "take_profit");
        return;
      }
    }
    
    const candlesSinceEntry = this.candles.length - 1 - (this.activeTrade.entryCandle ?? 0);
    if (candlesSinceEntry >= this.strategyState.timeStopCandles) {
      const expectedR = this.activeTrade.side === "LONG"
        ? (currentPrice - this.activeTrade.entryPrice) / (this.activeTrade.entryPrice - this.activeTrade.stopLoss)
        : (this.activeTrade.entryPrice - currentPrice) / (this.activeTrade.stopLoss - this.activeTrade.entryPrice);
      
      if (expectedR < 0.6) {
        this.closeTrade(currentPrice, "time_stop");
        return;
      }
    }
  }

  private executeStrategy(): void {
    if (!this.strategyState.isRunning) return;
    
    if (this.activeTrade) {
      this.checkExitConditions();
      return;
    }
    
    const crossover = this.detectCrossover();
    
    if (crossover.crossed && crossover.direction) {
      if (this.strategyState.useRetestSignals) {
        this.waitingForRetest = true;
        this.retestDirection = crossover.direction;
      } else {
        this.openTrade(crossover.direction, "crossover");
      }
      return;
    }
    
    if (this.strategyState.useRetestSignals) {
      const retest = this.detectRetest();
      if (retest.retest && retest.direction) {
        this.openTrade(retest.direction, "retest");
        this.waitingForRetest = false;
        this.retestDirection = null;
      }
    }
  }

  refreshData(): void {
    this.candles = generateCandles(300);
    this.initializeKalmanFilters();
    this.lastRefresh = Date.now();
    
    if (this.strategyState.isRunning) {
      this.executeStrategy();
    }
  }

  private updateWithNewCandle(): void {
    const lastCandle = this.candles[this.candles.length - 1];
    const now = Date.now();
    const interval = 15 * 60 * 1000;
    
    if (now - lastCandle.timestamp >= interval) {
      const volatility = 0.002 + Math.random() * 0.003;
      const trend = Math.random() > 0.5 ? 1 : -1;
      const movement = lastCandle.close * volatility * trend;
      
      const open = lastCandle.close;
      const close = open + movement;
      const high = Math.max(open, close) + lastCandle.close * volatility * Math.random() * 0.5;
      const low = Math.min(open, close) - lastCandle.close * volatility * Math.random() * 0.5;
      
      const newCandle: Candle = {
        timestamp: now,
        open: Math.round(open * 100) / 100,
        high: Math.round(high * 100) / 100,
        low: Math.round(low * 100) / 100,
        close: Math.round(close * 100) / 100,
        volume: Math.round(100000000 + Math.random() * 500000000),
      };
      
      this.candles.push(newCandle);
      this.candles.shift();
      
      if (this.kalmanFastFilter && this.kalmanSlowFilter) {
        this.prevKalmanFast = this.kalmanFastValues[this.kalmanFastValues.length - 1];
        this.prevKalmanSlow = this.kalmanSlowValues[this.kalmanSlowValues.length - 1];
        
        const fastVal = this.kalmanFastFilter.update(close);
        const slowVal = this.kalmanSlowFilter.update(close);
        
        this.kalmanFastValues.push(fastVal);
        this.kalmanSlowValues.push(slowVal);
        this.kalmanFastValues.shift();
        this.kalmanSlowValues.shift();
      }
      
      if (this.strategyState.isRunning) {
        this.executeStrategy();
      }
    }
  }

  private getStrategySignal(): StrategySignal {
    const lastIdx = this.kalmanFastValues.length - 1;
    const kalmanFast = this.kalmanFastValues[lastIdx] ?? 0;
    const kalmanSlow = this.kalmanSlowValues[lastIdx] ?? 0;
    const atr = calculateATR(this.candles, 14);
    const isBullRegime = kalmanFast > kalmanSlow;
    
    let signalType: "crossover" | "retest" | "none" = "none";
    let direction: SignalType = "HOLD";
    let entryZone: number | null = null;
    let stopLoss: number | null = null;
    let tp1: number | null = null;
    let tp2: number | null = null;
    
    if (this.activeTrade) {
      direction = this.activeTrade.side;
      entryZone = this.activeTrade.entryPrice;
      stopLoss = this.activeTrade.stopLoss;
      tp1 = this.activeTrade.takeProfit;
      tp2 = this.activeTrade.side === "LONG"
        ? this.activeTrade.entryPrice + (this.activeTrade.entryPrice - this.activeTrade.stopLoss) * 3
        : this.activeTrade.entryPrice - (this.activeTrade.stopLoss - this.activeTrade.entryPrice) * 3;
      signalType = this.activeTrade.signalType ?? "crossover";
    } else {
      const crossover = this.detectCrossover();
      const retest = this.detectRetest();
      
      if (crossover.crossed && crossover.direction) {
        signalType = "crossover";
        direction = crossover.direction;
        entryZone = this.candles[this.candles.length - 1].close;
        stopLoss = direction === "LONG" 
          ? entryZone - atr * this.strategyState.atrMultiplier
          : entryZone + atr * this.strategyState.atrMultiplier;
        tp1 = direction === "LONG"
          ? entryZone + atr * this.strategyState.atrMultiplier * 2
          : entryZone - atr * this.strategyState.atrMultiplier * 2;
        tp2 = direction === "LONG"
          ? entryZone + atr * this.strategyState.atrMultiplier * 3
          : entryZone - atr * this.strategyState.atrMultiplier * 3;
      } else if (retest.retest && retest.direction) {
        signalType = "retest";
        direction = retest.direction;
        entryZone = this.candles[this.candles.length - 1].close;
        stopLoss = direction === "LONG"
          ? entryZone - atr * this.strategyState.atrMultiplier
          : entryZone + atr * this.strategyState.atrMultiplier;
        tp1 = direction === "LONG"
          ? entryZone + atr * this.strategyState.atrMultiplier * 2
          : entryZone - atr * this.strategyState.atrMultiplier * 2;
        tp2 = direction === "LONG"
          ? entryZone + atr * this.strategyState.atrMultiplier * 3
          : entryZone - atr * this.strategyState.atrMultiplier * 3;
      }
    }
    
    return {
      type: signalType,
      direction,
      entryZone,
      stopLoss,
      takeProfit1: tp1,
      takeProfit2: tp2,
      atr,
      regime: isBullRegime ? "bull" : "bear",
      kalmanFast,
      kalmanSlow,
    };
  }

  async getDashboardData(): Promise<DashboardData> {
    this.updateWithNewCandle();
    
    const kalmanFast = this.kalmanFastValues[this.kalmanFastValues.length - 1] ?? 0;
    const kalmanSlow = this.kalmanSlowValues[this.kalmanSlowValues.length - 1] ?? 0;
    
    const signal = generateSignal(this.candles, kalmanFast, kalmanSlow);
    const lastPrice = this.candles[this.candles.length - 1].close;
    const futuresData = generateFuturesData(lastPrice);
    
    const closedTrades = this.trades.filter(t => t.status === "closed");
    const winningTrades = closedTrades.filter(t => (t.pnlPercent ?? 0) > 0);
    const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;
    const totalProfit = winningTrades.reduce((acc, t) => acc + (t.pnlPercent ?? 0), 0);
    const totalLoss = Math.abs(closedTrades.filter(t => (t.pnlPercent ?? 0) < 0).reduce((acc, t) => acc + (t.pnlPercent ?? 0), 0));
    const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;
    
    const drawdown = -0.01 - Math.random() * 0.03;
    const maxDrawdown = -0.03 - Math.random() * 0.04;
    const dailyPnl = -0.5 + Math.random() * 2;
    const exposure = this.activeTrade ? 0.15 + Math.random() * 0.1 : 0;

    const displayCandles = this.candles.slice(-48);
    const displayKalmanFast = this.kalmanFastValues.slice(-48);
    const displayKalmanSlow = this.kalmanSlowValues.slice(-48);

    return {
      candles: displayCandles,
      currentSignal: signal,
      futuresData,
      recentTrades: this.trades.slice(0, 10),
      equity: Math.round(this.equity * 100) / 100,
      drawdown,
      maxDrawdown,
      dailyPnl,
      winRate: Math.round(winRate * 100) / 100,
      profitFactor: Math.round(profitFactor * 1000) / 1000,
      totalTrades: closedTrades.length,
      exposure,
      kalmanFast: displayKalmanFast,
      kalmanSlow: displayKalmanSlow,
      strategySignal: this.getStrategySignal(),
      strategyState: this.strategyState,
      activeTrade: this.activeTrade,
    };
  }
}

export const storage = new MemStorage();
