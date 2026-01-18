import type { Candle } from "@shared/schema";

const COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3";

export interface CoinGeckoPrice {
  bitcoin: {
    usd: number;
    usd_24h_change: number;
    usd_24h_vol: number;
    last_updated_at: number;
  };
}

export interface CoinGeckoOHLC {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface CachedData {
  data: any;
  timestamp: number;
}

const cache: Map<string, CachedData> = new Map();
const CACHE_TTL = 300000;
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 30000;
let lastSuccessfulData: { price: CachedData | null; ohlc: CachedData | null } = { price: null, ohlc: null };

async function fetchWithRetry(url: string, retries = 1): Promise<any> {
  const cached = cache.get(url);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    console.log(`CoinGecko: Using cached data`);
    return cached.data;
  }
  
  if (cached) {
    console.log(`CoinGecko: Using stale cache to avoid rate limits`);
    return cached.data;
  }
  
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;
  if (timeSinceLastRequest < MIN_REQUEST_INTERVAL) {
    const waitTime = MIN_REQUEST_INTERVAL - timeSinceLastRequest;
    console.log(`CoinGecko: Waiting ${Math.round(waitTime/1000)}s before request...`);
    await new Promise(resolve => setTimeout(resolve, waitTime));
  }
  
  for (let i = 0; i < retries; i++) {
    try {
      lastRequestTime = Date.now();
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      const response = await fetch(url, {
        headers: {
          'Accept': 'application/json',
        },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (response.status === 429) {
        console.log("CoinGecko: Rate limited - using alternative");
        throw new Error("Rate limited");
      }
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      cache.set(url, { data, timestamp: Date.now() });
      console.log("CoinGecko: Data fetched successfully");
      return data;
    } catch (error) {
      console.error(`CoinGecko fetch failed:`, error);
      throw error;
    }
  }
}

export async function getBTCPrice(): Promise<{ price: number; change24h: number; volume24h: number } | null> {
  try {
    const url = `${COINGECKO_BASE_URL}/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_last_updated_at=true`;
    const data: CoinGeckoPrice = await fetchWithRetry(url);
    
    if (data?.bitcoin) {
      return {
        price: data.bitcoin.usd,
        change24h: data.bitcoin.usd_24h_change,
        volume24h: data.bitcoin.usd_24h_vol,
      };
    }
    return null;
  } catch (error) {
    console.error("Error fetching BTC price from CoinGecko:", error);
    return null;
  }
}

export async function getBTCOHLC(days: number = 7): Promise<Candle[]> {
  try {
    const url = `${COINGECKO_BASE_URL}/coins/bitcoin/ohlc?vs_currency=usd&days=${days}`;
    const data: number[][] = await fetchWithRetry(url);
    
    if (!data || !Array.isArray(data)) {
      return [];
    }
    
    return data.map((ohlc: number[]) => ({
      timestamp: ohlc[0],
      open: ohlc[1],
      high: ohlc[2],
      low: ohlc[3],
      close: ohlc[4],
      volume: 0,
    }));
  } catch (error) {
    console.error("Error fetching BTC OHLC from CoinGecko:", error);
    return [];
  }
}

export async function getBTCMarketChart(days: number = 1): Promise<Candle[]> {
  try {
    const url = `${COINGECKO_BASE_URL}/coins/bitcoin/market_chart?vs_currency=usd&days=${days}&interval=hourly`;
    const data = await fetchWithRetry(url);
    
    if (!data?.prices || !Array.isArray(data.prices)) {
      return [];
    }
    
    const candles: Candle[] = [];
    const prices = data.prices;
    const volumes = data.total_volumes || [];
    
    for (let i = 0; i < prices.length - 1; i++) {
      const timestamp = prices[i][0];
      const open = prices[i][1];
      const close = prices[i + 1] ? prices[i + 1][1] : open;
      const high = Math.max(open, close) * (1 + Math.random() * 0.002);
      const low = Math.min(open, close) * (1 - Math.random() * 0.002);
      const volume = volumes[i] ? volumes[i][1] : 0;
      
      candles.push({
        timestamp,
        open: Math.round(open * 100) / 100,
        high: Math.round(high * 100) / 100,
        low: Math.round(low * 100) / 100,
        close: Math.round(close * 100) / 100,
        volume: Math.round(volume),
      });
    }
    
    return candles;
  } catch (error) {
    console.error("Error fetching BTC market chart from CoinGecko:", error);
    return [];
  }
}

export async function getFullBTCData(): Promise<{
  candles: Candle[];
  currentPrice: number;
  change24h: number;
  volume24h: number;
} | null> {
  try {
    const [priceData, ohlcData] = await Promise.all([
      getBTCPrice(),
      getBTCOHLC(7),
    ]);
    
    if (!priceData || ohlcData.length === 0) {
      console.error("CoinGecko: No price or OHLC data available");
      return null;
    }
    
    const lastOHLC = ohlcData[ohlcData.length - 1];
    if (lastOHLC) {
      lastOHLC.close = priceData.price;
    }
    
    return {
      candles: ohlcData,
      currentPrice: priceData.price,
      change24h: priceData.change24h,
      volume24h: priceData.volume24h,
    };
  } catch (error) {
    console.error("Error fetching full BTC data from CoinGecko:", error);
    return null;
  }
}
