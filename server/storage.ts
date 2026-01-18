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
  AIAnalysis,
  AISignal,
  MultiTimeframeScore,
  WhaleActivity,
  PerformanceStats,
} from "@shared/schema";
import { randomUUID } from "crypto";
import { getKlines, getMultiTimeframeKlines, getFuturesData, detectLargeOrders } from "./binance";
import { getAllIndicators, calculateMultiTimeframeScore, type TechnicalIndicators } from "./indicators";
import { analyzeMarket, generateAISignal } from "./ai-analysis";

export interface IStorage {
  getDashboardData(): Promise<DashboardData>;
  refreshData(): void;
  startStrategy(): void;
  stopStrategy(): void;
  updateStrategySettings(settings: Partial<StrategyState>): void;
  getStrategyState(): StrategyState;
  requestAIAnalysis(): Promise<void>;
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

export class MemStorage implements IStorage {
  private candles: Candle[] = [];
  private trades: Trade[] = [];
  private equity = 10000;
  private peakEquity = 10000;
  private lastRefresh = 0;
  private lastBinanceUpdate = 0;
  private kalmanFastFilter: KalmanFilter | null = null;
  private kalmanSlowFilter: KalmanFilter | null = null;
  private kalmanFastValues: number[] = [];
  private kalmanSlowValues: number[] = [];
  private activeTrade: Trade | null = null;
  private prevKalmanFast = 0;
  private prevKalmanSlow = 0;
  private waitingForRetest = false;
  private retestDirection: "LONG" | "SHORT" | null = null;
  
  private aiAnalysis: AIAnalysis | null = null;
  private aiSignal: AISignal | null = null;
  private lastAIUpdate = 0;
  private indicators: TechnicalIndicators | null = null;
  private mtfScore: MultiTimeframeScore | null = null;
  private whaleActivity: WhaleActivity | null = null;
  private isLiveData = false;
  
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
    if (this.activeTrade && this.candles.length > 0) {
      this.closeTrade(this.candles[this.candles.length - 1].close, "manual_stop");
    }
  }

  updateStrategySettings(settings: Partial<StrategyState>): void {
    this.strategyState = { ...this.strategyState, ...settings };
  }

  getStrategyState(): StrategyState {
    return this.strategyState;
  }

  async requestAIAnalysis(): Promise<void> {
    if (this.candles.length === 0) return;
    
    const now = Date.now();
    if (now - this.lastAIUpdate < 30000) return;
    
    try {
      const lastPrice = this.candles[this.candles.length - 1]?.close ?? 0;
      if (lastPrice === 0) return;
      
      let futuresData: FuturesData;
      try {
        futuresData = await getFuturesData("BTCUSDT", lastPrice);
      } catch {
        futuresData = {
          fundingRate: 0,
          nextFundingTime: Date.now() + 8 * 3600000,
          openInterest: 0,
          oiChange15m: 0,
          oiChange1h: 0,
          longShortRatio: 1,
          liquidations15m: 0,
          liquidations1h: 0,
          markPrice: lastPrice,
          indexPrice: lastPrice,
          basis: 0,
        };
      }
      
      if (!this.indicators) {
        this.indicators = getAllIndicators(this.candles);
      }
      
      const defaultWhale = { largeBuys: 0, largeSells: 0, netFlow: 0, whaleActivity: "neutral" as const };
      const defaultMtf = { score: 0, direction: "neutral" as const, alignment: 0, details: [] };
      
      if (!this.whaleActivity) {
        try {
          const whaleData = await detectLargeOrders();
          this.whaleActivity = whaleData ?? defaultWhale;
        } catch {
          this.whaleActivity = defaultWhale;
        }
      }
      
      if (!this.mtfScore) {
        try {
          const mtfCandles = await getMultiTimeframeKlines();
          if (mtfCandles.m15.length > 0) {
            this.mtfScore = calculateMultiTimeframeScore(mtfCandles);
          } else {
            this.mtfScore = defaultMtf;
          }
        } catch {
          this.mtfScore = defaultMtf;
        }
      }
      
      const [analysis, signal] = await Promise.all([
        analyzeMarket(
          this.candles,
          this.indicators,
          futuresData,
          this.whaleActivity ?? defaultWhale,
          this.mtfScore ?? defaultMtf
        ),
        generateAISignal(
          this.candles,
          this.indicators,
          futuresData,
          this.whaleActivity ?? defaultWhale,
          this.mtfScore ?? defaultMtf
        ),
      ]);
      
      this.aiAnalysis = analysis;
      this.aiSignal = signal;
      this.lastAIUpdate = now;
    } catch (error) {
      console.error("Error requesting AI analysis:", error);
    }
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
    if (this.equity > this.peakEquity) {
      this.peakEquity = this.equity;
    }
    this.activeTrade = null;
  }

  private checkExitConditions(): void {
    if (!this.activeTrade || this.candles.length === 0) return;
    
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
      const stopDistance = Math.abs(this.activeTrade.entryPrice - this.activeTrade.stopLoss);
      const expectedR = this.activeTrade.side === "LONG"
        ? (currentPrice - this.activeTrade.entryPrice) / stopDistance
        : (this.activeTrade.entryPrice - currentPrice) / stopDistance;
      
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

  async refreshData(): Promise<void> {
    const now = Date.now();
    
    if (now - this.lastBinanceUpdate < 5000 && this.candles.length > 0) {
      return;
    }
    
    try {
      const liveCandles = await getKlines("BTCUSDT", "15m", 300);
      
      if (liveCandles.length > 0) {
        this.candles = liveCandles;
        this.isLiveData = true;
        this.initializeKalmanFilters();
        this.indicators = getAllIndicators(this.candles);
        
        const [mtfCandles, whaleData] = await Promise.all([
          getMultiTimeframeKlines(),
          detectLargeOrders(),
        ]);
        
        if (mtfCandles.m15.length > 0) {
          this.mtfScore = calculateMultiTimeframeScore(mtfCandles);
        }
        this.whaleActivity = whaleData;
        
        this.lastBinanceUpdate = now;
      } else {
        this.generateFallbackCandles();
      }
    } catch (error) {
      console.error("Error fetching Binance data:", error);
      if (this.candles.length === 0) {
        this.generateFallbackCandles();
      }
    }
    
    this.lastRefresh = now;
    
    if (this.strategyState.isRunning) {
      this.executeStrategy();
    }
  }

  private generateFallbackCandles(): void {
    const now = Date.now();
    const interval = 15 * 60 * 1000;
    const targetPrice = 95000;
    let currentPrice = targetPrice;
    
    this.candles = [];
    for (let i = 299; i >= 0; i--) {
      const timestamp = now - i * interval;
      const volatility = 0.001 + Math.random() * 0.002;
      const meanReversion = (targetPrice - currentPrice) * 0.01;
      const randomWalk = currentPrice * volatility * (Math.random() > 0.5 ? 1 : -1);
      const movement = meanReversion + randomWalk;
      
      const open = currentPrice;
      const close = currentPrice + movement;
      const high = Math.max(open, close) + currentPrice * volatility * Math.random() * 0.3;
      const low = Math.min(open, close) - currentPrice * volatility * Math.random() * 0.3;
      
      this.candles.push({
        timestamp,
        open: Math.round(open * 100) / 100,
        high: Math.round(high * 100) / 100,
        low: Math.round(low * 100) / 100,
        close: Math.round(close * 100) / 100,
        volume: Math.round(100000000 + Math.random() * 500000000),
      });
      
      currentPrice = close;
    }
    
    this.isLiveData = false;
    this.initializeKalmanFilters();
    this.indicators = getAllIndicators(this.candles);
  }

  private generateSignal(candles: Candle[], kalmanFast: number, kalmanSlow: number): Signal {
    const lastCandle = candles[candles.length - 1];
    const recentCandles = candles.slice(-20);
    
    const priceChange = (lastCandle.close - recentCandles[0].close) / recentCandles[0].close;
    const volatility = recentCandles.reduce((acc, c) => acc + Math.abs(c.close - c.open), 0) / recentCandles.length / lastCandle.close;
    
    const isBullRegime = kalmanFast > kalmanSlow;
    
    let probUp = 0.33;
    let probDown = 0.33;
    let probChop = 0.34;
    
    if (this.indicators) {
      const rsi = this.indicators.rsi.value;
      const macdSignal = this.indicators.macd.signal;
      const adxValue = this.indicators.adx.value;
      
      if (isBullRegime) {
        probUp = 0.45 + (rsi < 50 ? 0.1 : 0) + (macdSignal === "bullish" ? 0.1 : 0);
        probDown = 0.20 + (rsi > 70 ? 0.1 : 0);
        probChop = 1 - probUp - probDown;
      } else {
        probDown = 0.45 + (rsi > 50 ? 0.1 : 0) + (macdSignal === "bearish" ? 0.1 : 0);
        probUp = 0.20 + (rsi < 30 ? 0.1 : 0);
        probChop = 1 - probUp - probDown;
      }
      
      if (adxValue < 20) {
        probChop = Math.min(0.6, probChop + 0.2);
        probUp = (1 - probChop) / 2;
        probDown = (1 - probChop) / 2;
      }
    } else {
      if (isBullRegime) {
        probUp = 0.5 + Math.random() * 0.2;
        probDown = 0.15 + Math.random() * 0.1;
        probChop = 1 - probUp - probDown;
      } else {
        probDown = 0.5 + Math.random() * 0.2;
        probUp = 0.15 + Math.random() * 0.1;
        probChop = 1 - probUp - probDown;
      }
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
    
    if (this.aiSignal && this.aiSignal.confidence > 0.6) {
      signal = this.aiSignal.direction;
      confidence = this.aiSignal.confidence;
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
    
    const features: Feature[] = [];
    
    if (this.indicators) {
      features.push(
        {
          name: "RSI(14)",
          value: this.indicators.rsi.value,
          importance: this.indicators.rsi.signal === "bullish" ? 0.8 : this.indicators.rsi.signal === "bearish" ? -0.8 : 0,
          description: this.indicators.rsi.description,
        },
        {
          name: "MACD",
          value: this.indicators.macd.histogram,
          importance: this.indicators.macd.signal === "bullish" ? 0.7 : this.indicators.macd.signal === "bearish" ? -0.7 : 0,
          description: this.indicators.macd.description,
        },
        {
          name: "ADX",
          value: this.indicators.adx.value,
          importance: this.indicators.adx.value > 25 ? 0.6 : 0.2,
          description: this.indicators.adx.description,
        },
        {
          name: "Kalman Fast",
          value: kalmanFast,
          importance: isBullRegime ? 0.9 : -0.9,
          description: "Fast Kalman filter (70-period)",
        },
        {
          name: "Kalman Slow",
          value: kalmanSlow,
          importance: isBullRegime ? 0.7 : -0.7,
          description: "Slow Kalman filter (250-period)",
        }
      );
    } else {
      features.push(
        {
          name: "Kalman Fast",
          value: kalmanFast,
          importance: isBullRegime ? 0.9 : -0.9,
          description: "Fast Kalman filter (70-period)",
        },
        {
          name: "Kalman Slow",
          value: kalmanSlow,
          importance: isBullRegime ? 0.7 : -0.7,
          description: "Slow Kalman filter (250-period)",
        },
        {
          name: "ATR(14)",
          value: calculateATR(candles, 14),
          importance: volatility > 0.006 ? -0.4 : 0.4,
          description: "Average True Range",
        }
      );
    }
    
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
      const stopDistance = Math.abs(this.activeTrade.entryPrice - this.activeTrade.stopLoss);
      tp2 = this.activeTrade.side === "LONG"
        ? this.activeTrade.entryPrice + stopDistance * 3
        : this.activeTrade.entryPrice - stopDistance * 3;
      signalType = this.activeTrade.signalType ?? "crossover";
    } else if (this.aiSignal && this.aiSignal.direction !== "HOLD" && this.aiSignal.confidence > 0.6) {
      direction = this.aiSignal.direction;
      entryZone = this.aiSignal.entryPrice;
      stopLoss = this.aiSignal.stopLoss;
      tp1 = this.aiSignal.takeProfit1;
      tp2 = this.aiSignal.takeProfit2;
      signalType = "crossover";
    } else {
      const crossover = this.detectCrossover();
      const retest = this.detectRetest();
      
      if (crossover.crossed && crossover.direction) {
        signalType = "crossover";
        direction = crossover.direction;
        entryZone = this.candles[this.candles.length - 1]?.close ?? 0;
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
        entryZone = this.candles[this.candles.length - 1]?.close ?? 0;
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

  private calculatePerformanceStats(): PerformanceStats {
    const closedTrades = this.trades.filter(t => t.status === "closed");
    const winningTrades = closedTrades.filter(t => (t.pnlPercent ?? 0) > 0);
    const losingTrades = closedTrades.filter(t => (t.pnlPercent ?? 0) <= 0);
    
    const avgWin = winningTrades.length > 0 
      ? winningTrades.reduce((acc, t) => acc + (t.pnlPercent ?? 0), 0) / winningTrades.length 
      : 0;
    const avgLoss = losingTrades.length > 0 
      ? Math.abs(losingTrades.reduce((acc, t) => acc + (t.pnlPercent ?? 0), 0) / losingTrades.length)
      : 0;
    
    const totalProfit = winningTrades.reduce((acc, t) => acc + (t.pnlPercent ?? 0), 0);
    const totalLoss = Math.abs(losingTrades.reduce((acc, t) => acc + (t.pnlPercent ?? 0), 0));
    const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;
    
    const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;
    
    const returns = closedTrades.map(t => t.pnlPercent ?? 0);
    const avgReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
    const stdDev = returns.length > 1 
      ? Math.sqrt(returns.reduce((acc, r) => acc + Math.pow(r - avgReturn, 2), 0) / returns.length)
      : 0;
    const sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;
    
    const currentDrawdown = this.peakEquity > 0 ? ((this.peakEquity - this.equity) / this.peakEquity) * 100 : 0;
    
    let maxDrawdown = 0;
    let peak = 10000;
    let runningEquity = 10000;
    for (const trade of [...closedTrades].reverse()) {
      runningEquity += (trade.pnl ?? 0);
      if (runningEquity > peak) peak = runningEquity;
      const dd = ((peak - runningEquity) / peak) * 100;
      if (dd > maxDrawdown) maxDrawdown = dd;
    }
    
    const expectancy = closedTrades.length > 0
      ? (winRate / 100 * avgWin) - ((100 - winRate) / 100 * avgLoss)
      : 0;
    
    const rMultiples = closedTrades.map(t => {
      if (!t.pnlPercent) return 0;
      const stopDistance = Math.abs(t.entryPrice - t.stopLoss);
      const riskPercent = (stopDistance / t.entryPrice) * 100;
      return riskPercent > 0 ? t.pnlPercent / riskPercent : 0;
    });
    const avgRMultiple = rMultiples.length > 0 
      ? rMultiples.reduce((a, b) => a + b, 0) / rMultiples.length 
      : 0;
    
    const allPnls = closedTrades.map(t => t.pnlPercent ?? 0);
    const bestTrade = allPnls.length > 0 ? Math.max(...allPnls) : 0;
    const worstTrade = allPnls.length > 0 ? Math.min(...allPnls) : 0;
    
    let consecutiveWins = 0;
    let consecutiveLosses = 0;
    let maxConsecutiveWins = 0;
    let maxConsecutiveLosses = 0;
    
    for (const trade of closedTrades) {
      if ((trade.pnlPercent ?? 0) > 0) {
        consecutiveWins++;
        consecutiveLosses = 0;
        if (consecutiveWins > maxConsecutiveWins) maxConsecutiveWins = consecutiveWins;
      } else {
        consecutiveLosses++;
        consecutiveWins = 0;
        if (consecutiveLosses > maxConsecutiveLosses) maxConsecutiveLosses = consecutiveLosses;
      }
    }
    
    return {
      totalTrades: closedTrades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate: Math.round(winRate * 100) / 100,
      avgWin: Math.round(avgWin * 100) / 100,
      avgLoss: Math.round(avgLoss * 100) / 100,
      profitFactor: Math.round(profitFactor * 1000) / 1000,
      sharpeRatio: Math.round(sharpeRatio * 100) / 100,
      maxDrawdown: Math.round(maxDrawdown * 100) / 100,
      currentDrawdown: Math.round(currentDrawdown * 100) / 100,
      expectancy: Math.round(expectancy * 100) / 100,
      avgRMultiple: Math.round(avgRMultiple * 100) / 100,
      bestTrade: Math.round(bestTrade * 100) / 100,
      worstTrade: Math.round(worstTrade * 100) / 100,
      consecutiveWins: maxConsecutiveWins,
      consecutiveLosses: maxConsecutiveLosses,
    };
  }

  async getDashboardData(): Promise<DashboardData> {
    await this.refreshData();
    
    if (this.candles.length === 0) {
      this.generateFallbackCandles();
    }
    
    const kalmanFast = this.kalmanFastValues[this.kalmanFastValues.length - 1] ?? 0;
    const kalmanSlow = this.kalmanSlowValues[this.kalmanSlowValues.length - 1] ?? 0;
    
    const signal = this.generateSignal(this.candles, kalmanFast, kalmanSlow);
    const lastPrice = this.candles[this.candles.length - 1]?.close ?? 0;
    
    let futuresData: FuturesData;
    try {
      futuresData = await getFuturesData("BTCUSDT", lastPrice);
    } catch {
      futuresData = {
        fundingRate: 0,
        nextFundingTime: Date.now() + 8 * 3600000,
        openInterest: 0,
        oiChange15m: 0,
        oiChange1h: 0,
        longShortRatio: 1,
        liquidations15m: 0,
        liquidations1h: 0,
        markPrice: lastPrice,
        indexPrice: lastPrice,
        basis: 0,
      };
    }
    
    const performanceStats = this.calculatePerformanceStats();
    
    const currentDrawdown = this.peakEquity > 0 ? -((this.peakEquity - this.equity) / this.peakEquity) : 0;
    const exposure = this.activeTrade ? 0.15 + Math.random() * 0.1 : 0;

    const displayCandles = this.candles.slice(-48);
    const displayKalmanFast = this.kalmanFastValues.slice(-48);
    const displayKalmanSlow = this.kalmanSlowValues.slice(-48);

    const indicatorsSummary = this.indicators ? {
      rsi: { name: "RSI", value: this.indicators.rsi.value, signal: this.indicators.rsi.signal, strength: this.indicators.rsi.strength, description: this.indicators.rsi.description },
      macd: { name: "MACD", value: this.indicators.macd.value, signal: this.indicators.macd.signal, strength: this.indicators.macd.strength, description: this.indicators.macd.description },
      bollingerBands: { name: "BB", value: this.indicators.bollingerBands.value, signal: this.indicators.bollingerBands.signal, strength: this.indicators.bollingerBands.strength, description: this.indicators.bollingerBands.description },
      obv: { name: "OBV", value: this.indicators.obv.value, signal: this.indicators.obv.signal, strength: this.indicators.obv.strength, description: this.indicators.obv.description },
      vwap: { name: "VWAP", value: this.indicators.vwap.value, signal: this.indicators.vwap.signal, strength: this.indicators.vwap.strength, description: this.indicators.vwap.description },
      atr: { name: "ATR", value: this.indicators.atr.value, signal: this.indicators.atr.signal, strength: this.indicators.atr.strength, description: this.indicators.atr.description },
      adx: { name: "ADX", value: this.indicators.adx.value, signal: this.indicators.adx.signal, strength: this.indicators.adx.strength, description: this.indicators.adx.description },
      stochastic: { name: "Stoch", value: this.indicators.stochastic.value, signal: this.indicators.stochastic.signal, strength: this.indicators.stochastic.strength, description: this.indicators.stochastic.description },
    } : undefined;

    return {
      candles: displayCandles,
      currentSignal: signal,
      futuresData,
      recentTrades: this.trades.slice(0, 10),
      equity: Math.round(this.equity * 100) / 100,
      drawdown: currentDrawdown,
      maxDrawdown: -performanceStats.maxDrawdown / 100,
      dailyPnl: performanceStats.totalTrades > 0 ? performanceStats.avgWin - performanceStats.avgLoss : 0,
      winRate: performanceStats.winRate,
      profitFactor: performanceStats.profitFactor,
      totalTrades: performanceStats.totalTrades,
      exposure,
      kalmanFast: displayKalmanFast,
      kalmanSlow: displayKalmanSlow,
      strategySignal: this.getStrategySignal(),
      strategyState: this.strategyState,
      activeTrade: this.activeTrade,
      aiAnalysis: this.aiAnalysis ?? undefined,
      aiSignal: this.aiSignal ?? undefined,
      indicators: indicatorsSummary,
      mtfScore: this.mtfScore ?? undefined,
      whaleActivity: this.whaleActivity ?? undefined,
      performanceStats,
      isLiveData: this.isLiveData,
    };
  }
}

export const storage = new MemStorage();
