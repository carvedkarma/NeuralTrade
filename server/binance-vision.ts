import type { Candle, FuturesData } from "@shared/schema";

const BINANCE_VISION_BASE_URL = "https://data-api.binance.vision/api/v3";

interface CachedData {
  data: any;
  timestamp: number;
}

const cache: Map<string, CachedData> = new Map();
const CACHE_TTL = 60000;

async function fetchBinanceVision(endpoint: string): Promise<any> {
  const url = `${BINANCE_VISION_BASE_URL}${endpoint}`;
  
  const cached = cache.get(url);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }

  try {
    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Binance Vision API error: ${response.status}`);
    }

    const data = await response.json();
    cache.set(url, { data, timestamp: Date.now() });
    return data;
  } catch (error) {
    console.error("Binance Vision fetch error:", error);
    throw error;
  }
}

export async function getBTCPriceBinanceVision(): Promise<number | null> {
  try {
    const data = await fetchBinanceVision("/ticker/price?symbol=BTCUSDT");
    if (data && data.price) {
      return parseFloat(data.price);
    }
    return null;
  } catch (error) {
    console.error("Error fetching BTC price from Binance Vision:", error);
    return null;
  }
}

export async function getBTCCandlesBinanceVision(interval: string = "15m", limit: number = 168): Promise<Candle[]> {
  try {
    const data = await fetchBinanceVision(`/klines?symbol=BTCUSDT&interval=${interval}&limit=${limit}`);
    
    if (!Array.isArray(data)) {
      return [];
    }

    return data.map((kline: any[]) => ({
      timestamp: kline[0],
      open: parseFloat(kline[1]),
      high: parseFloat(kline[2]),
      low: parseFloat(kline[3]),
      close: parseFloat(kline[4]),
      volume: parseFloat(kline[5]),
    }));
  } catch (error) {
    console.error("Error fetching BTC candles from Binance Vision:", error);
    return [];
  }
}

export async function getBTC24hStatsBinanceVision(): Promise<{ priceChange: number; priceChangePercent: number; volume: number } | null> {
  try {
    const data = await fetchBinanceVision("/ticker/24hr?symbol=BTCUSDT");
    if (data) {
      return {
        priceChange: parseFloat(data.priceChange || "0"),
        priceChangePercent: parseFloat(data.priceChangePercent || "0"),
        volume: parseFloat(data.volume || "0"),
      };
    }
    return null;
  } catch (error) {
    console.error("Error fetching 24h stats from Binance Vision:", error);
    return null;
  }
}

export async function getFullBTCDataBinanceVision(): Promise<{
  candles: Candle[];
  currentPrice: number;
  priceChange24h: number;
  volume24h: number;
  futuresData: FuturesData;
} | null> {
  try {
    console.log("Fetching data from Binance Vision...");
    
    const [price, candles, stats] = await Promise.all([
      getBTCPriceBinanceVision(),
      getBTCCandlesBinanceVision("15m", 168),
      getBTC24hStatsBinanceVision(),
    ]);

    if (!price || candles.length === 0) {
      console.error("Binance Vision: Incomplete data received");
      return null;
    }

    console.log(`Binance Vision: Got price $${price.toFixed(2)}, ${candles.length} candles`);

    const futuresData: FuturesData = {
      fundingRate: 0.0001,
      nextFundingTime: Date.now() + 8 * 60 * 60 * 1000,
      openInterest: 50000000000,
      oiChange15m: 0,
      oiChange1h: 0,
      longShortRatio: 1.05,
      liquidations15m: 0,
      liquidations1h: 0,
      markPrice: price,
      indexPrice: price,
      basis: 0.001,
    };

    return {
      candles,
      currentPrice: price,
      priceChange24h: stats?.priceChangePercent || 0,
      volume24h: stats?.volume || 0,
      futuresData,
    };
  } catch (error) {
    console.error("Error fetching full BTC data from Binance Vision:", error);
    return null;
  }
}
