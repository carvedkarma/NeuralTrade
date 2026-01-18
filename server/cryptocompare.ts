import type { Candle } from "@shared/schema";

const CRYPTOCOMPARE_BASE_URL = "https://min-api.cryptocompare.com/data";

interface CryptoComparePrice {
  USD: number;
}

interface CryptoCompareOHLC {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volumefrom: number;
  volumeto: number;
}

interface CryptoCompareHistoResponse {
  Response: string;
  Data: {
    Data: CryptoCompareOHLC[];
  };
}

async function fetchCryptoCompare(url: string): Promise<any> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);
    
    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
      },
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("CryptoCompare fetch failed:", error);
    throw error;
  }
}

export async function getBTCPriceCryptoCompare(): Promise<{ price: number; change24h: number } | null> {
  try {
    const [priceData, histoData] = await Promise.all([
      fetchCryptoCompare(`${CRYPTOCOMPARE_BASE_URL}/price?fsym=BTC&tsyms=USD`),
      fetchCryptoCompare(`${CRYPTOCOMPARE_BASE_URL}/v2/histohour?fsym=BTC&tsym=USD&limit=24`),
    ]);
    
    if (!priceData?.USD) {
      return null;
    }
    
    const currentPrice = priceData.USD;
    let change24h = 0;
    
    if (histoData?.Data?.Data && histoData.Data.Data.length > 0) {
      const price24hAgo = histoData.Data.Data[0].close;
      change24h = ((currentPrice - price24hAgo) / price24hAgo) * 100;
    }
    
    return {
      price: currentPrice,
      change24h,
    };
  } catch (error) {
    console.error("Error fetching BTC price from CryptoCompare:", error);
    return null;
  }
}

export async function getBTCCandlesCryptoCompare(): Promise<Candle[]> {
  try {
    const response = await fetchCryptoCompare(
      `${CRYPTOCOMPARE_BASE_URL}/v2/histohour?fsym=BTC&tsym=USD&limit=168`
    );
    
    console.log("CryptoCompare response:", JSON.stringify(response).substring(0, 200));
    
    const dataArray = response?.Data?.Data || response?.Data || [];
    
    if (!Array.isArray(dataArray) || dataArray.length === 0) {
      console.error("CryptoCompare: No candle data in response");
      return [];
    }
    
    return dataArray.map((ohlc: CryptoCompareOHLC) => ({
      timestamp: ohlc.time * 1000,
      open: ohlc.open,
      high: ohlc.high,
      low: ohlc.low,
      close: ohlc.close,
      volume: ohlc.volumeto || 0,
    }));
  } catch (error) {
    console.error("Error fetching BTC candles from CryptoCompare:", error);
    return [];
  }
}

export async function getFullBTCDataCryptoCompare(): Promise<{
  candles: Candle[];
  currentPrice: number;
  change24h: number;
} | null> {
  try {
    const [priceData, candles] = await Promise.all([
      getBTCPriceCryptoCompare(),
      getBTCCandlesCryptoCompare(),
    ]);
    
    if (!priceData || candles.length === 0) {
      console.error("CryptoCompare: No price or candle data available");
      return null;
    }
    
    const lastCandle = candles[candles.length - 1];
    if (lastCandle) {
      lastCandle.close = priceData.price;
    }
    
    return {
      candles,
      currentPrice: priceData.price,
      change24h: priceData.change24h,
    };
  } catch (error) {
    console.error("Error fetching full BTC data from CryptoCompare:", error);
    return null;
  }
}
