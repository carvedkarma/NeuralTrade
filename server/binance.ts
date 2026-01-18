import type { Candle, FuturesData } from "@shared/schema";

const BINANCE_BASE_URL = "https://fapi.binance.com";

export interface BinanceKline {
  openTime: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: number;
  quoteVolume: string;
  trades: number;
  takerBuyBaseVolume: string;
  takerBuyQuoteVolume: string;
}

export interface BinanceFundingRate {
  symbol: string;
  fundingRate: string;
  fundingTime: number;
  markPrice: string;
}

export interface BinanceOpenInterest {
  symbol: string;
  openInterest: string;
  time: number;
}

export interface BinanceLongShortRatio {
  symbol: string;
  longShortRatio: string;
  longAccount: string;
  shortAccount: string;
  timestamp: number;
}

export interface BinanceTickerPrice {
  symbol: string;
  price: string;
  time: number;
}

export interface BinanceMarkPrice {
  symbol: string;
  markPrice: string;
  indexPrice: string;
  estimatedSettlePrice: string;
  lastFundingRate: string;
  nextFundingTime: number;
  interestRate: string;
  time: number;
}

async function fetchWithRetry(url: string, retries = 3): Promise<any> {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}

export async function getKlines(
  symbol: string = "BTCUSDT",
  interval: string = "15m",
  limit: number = 300
): Promise<Candle[]> {
  try {
    const url = `${BINANCE_BASE_URL}/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
    const data = await fetchWithRetry(url);
    
    return data.map((kline: any[]) => ({
      timestamp: kline[0],
      open: parseFloat(kline[1]),
      high: parseFloat(kline[2]),
      low: parseFloat(kline[3]),
      close: parseFloat(kline[4]),
      volume: parseFloat(kline[5]),
    }));
  } catch (error) {
    console.error("Error fetching klines:", error);
    return [];
  }
}

export async function getMultiTimeframeKlines(symbol: string = "BTCUSDT"): Promise<{
  m5: Candle[];
  m15: Candle[];
  h1: Candle[];
  h4: Candle[];
}> {
  try {
    const [m5, m15, h1, h4] = await Promise.all([
      getKlines(symbol, "5m", 100),
      getKlines(symbol, "15m", 100),
      getKlines(symbol, "1h", 100),
      getKlines(symbol, "4h", 50),
    ]);
    return { m5, m15, h1, h4 };
  } catch (error) {
    console.error("Error fetching multi-timeframe klines:", error);
    return { m5: [], m15: [], h1: [], h4: [] };
  }
}

export async function getMarkPrice(symbol: string = "BTCUSDT"): Promise<BinanceMarkPrice | null> {
  try {
    const url = `${BINANCE_BASE_URL}/fapi/v1/premiumIndex?symbol=${symbol}`;
    return await fetchWithRetry(url);
  } catch (error) {
    console.error("Error fetching mark price:", error);
    return null;
  }
}

export async function getOpenInterest(symbol: string = "BTCUSDT"): Promise<number> {
  try {
    const url = `${BINANCE_BASE_URL}/fapi/v1/openInterest?symbol=${symbol}`;
    const data = await fetchWithRetry(url);
    return parseFloat(data.openInterest);
  } catch (error) {
    console.error("Error fetching open interest:", error);
    return 0;
  }
}

export async function getLongShortRatio(symbol: string = "BTCUSDT"): Promise<number> {
  try {
    const url = `${BINANCE_BASE_URL}/futures/data/globalLongShortAccountRatio?symbol=${symbol}&period=15m&limit=1`;
    const data = await fetchWithRetry(url);
    return data.length > 0 ? parseFloat(data[0].longShortRatio) : 1;
  } catch (error) {
    console.error("Error fetching long/short ratio:", error);
    return 1;
  }
}

export async function getRecentTrades(symbol: string = "BTCUSDT", limit: number = 100): Promise<any[]> {
  try {
    const url = `${BINANCE_BASE_URL}/fapi/v1/trades?symbol=${symbol}&limit=${limit}`;
    return await fetchWithRetry(url);
  } catch (error) {
    console.error("Error fetching recent trades:", error);
    return [];
  }
}

export async function getAggTrades(symbol: string = "BTCUSDT", limit: number = 500): Promise<any[]> {
  try {
    const url = `${BINANCE_BASE_URL}/fapi/v1/aggTrades?symbol=${symbol}&limit=${limit}`;
    return await fetchWithRetry(url);
  } catch (error) {
    console.error("Error fetching agg trades:", error);
    return [];
  }
}

export async function getFuturesData(symbol: string = "BTCUSDT", currentPrice: number): Promise<FuturesData> {
  try {
    const [markPriceData, openInterest, longShortRatio] = await Promise.all([
      getMarkPrice(symbol),
      getOpenInterest(symbol),
      getLongShortRatio(symbol),
    ]);

    const fundingRate = markPriceData ? parseFloat(markPriceData.lastFundingRate) : 0;
    const nextFundingTime = markPriceData ? markPriceData.nextFundingTime : Date.now() + 8 * 3600000;
    const markPrice = markPriceData ? parseFloat(markPriceData.markPrice) : currentPrice;
    const indexPrice = markPriceData ? parseFloat(markPriceData.indexPrice) : currentPrice;
    const basis = indexPrice > 0 ? (markPrice - indexPrice) / indexPrice : 0;

    const oiNotional = openInterest * currentPrice;

    return {
      fundingRate,
      nextFundingTime,
      openInterest: oiNotional,
      oiChange15m: 0,
      oiChange1h: 0,
      longShortRatio,
      liquidations15m: 0,
      liquidations1h: 0,
      markPrice,
      indexPrice,
      basis,
    };
  } catch (error) {
    console.error("Error getting futures data:", error);
    return {
      fundingRate: 0,
      nextFundingTime: Date.now() + 8 * 3600000,
      openInterest: 0,
      oiChange15m: 0,
      oiChange1h: 0,
      longShortRatio: 1,
      liquidations15m: 0,
      liquidations1h: 0,
      markPrice: currentPrice,
      indexPrice: currentPrice,
      basis: 0,
    };
  }
}

export async function detectLargeOrders(symbol: string = "BTCUSDT"): Promise<{
  largeBuys: number;
  largeSells: number;
  netFlow: number;
  whaleActivity: "bullish" | "bearish" | "neutral";
}> {
  try {
    const trades = await getAggTrades(symbol, 500);
    
    const largeOrderThreshold = 50000;
    let largeBuys = 0;
    let largeSells = 0;
    
    for (const trade of trades) {
      const value = parseFloat(trade.q) * parseFloat(trade.p);
      if (value >= largeOrderThreshold) {
        if (trade.m === false) {
          largeBuys += value;
        } else {
          largeSells += value;
        }
      }
    }
    
    const netFlow = largeBuys - largeSells;
    let whaleActivity: "bullish" | "bearish" | "neutral" = "neutral";
    
    if (netFlow > 100000) whaleActivity = "bullish";
    else if (netFlow < -100000) whaleActivity = "bearish";
    
    return { largeBuys, largeSells, netFlow, whaleActivity };
  } catch (error) {
    console.error("Error detecting large orders:", error);
    return { largeBuys: 0, largeSells: 0, netFlow: 0, whaleActivity: "neutral" };
  }
}
