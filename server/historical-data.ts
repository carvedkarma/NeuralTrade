import { db } from "./db";
import { candles, features, patterns } from "./db/schema";
import { eq, and, desc, gte, lte } from "drizzle-orm";

const CRYPTOCOMPARE_BASE_URL = "https://min-api.cryptocompare.com/data";

interface CryptoCompareOHLC {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volumefrom: number;
  volumeto: number;
}

export async function fetchHistoricalCandles(
  startYear: number = 2019,
  endYear: number = 2025
): Promise<number> {
  console.log(`Fetching historical candles from ${startYear} to ${endYear}...`);
  
  let totalFetched = 0;
  const batchSize = 2000;
  
  const endTimestamp = Math.floor(Date.now() / 1000);
  const startTimestamp = Math.floor(new Date(`${startYear}-01-01`).getTime() / 1000);
  
  let currentTs = endTimestamp;
  
  while (currentTs > startTimestamp) {
    try {
      const url = `${CRYPTOCOMPARE_BASE_URL}/v2/histominute?fsym=BTC&tsym=USD&limit=${batchSize}&toTs=${currentTs}&aggregate=15`;
      
      const response = await fetch(url);
      const data = await response.json();
      
      if (data.Response !== "Success" || !data.Data?.Data) {
        console.error("CryptoCompare API error:", data.Message);
        break;
      }
      
      const ohlcData: CryptoCompareOHLC[] = data.Data.Data;
      
      if (ohlcData.length === 0) break;
      
      const candlesToInsert = ohlcData.map(c => ({
        timestamp: c.time * 1000,
        timeframe: "15m",
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volumeto,
      }));
      
      for (const candle of candlesToInsert) {
        try {
          await db.insert(candles).values(candle).onConflictDoNothing();
          totalFetched++;
        } catch (err) {
        }
      }
      
      currentTs = ohlcData[0].time - 1;
      
      console.log(`Fetched ${totalFetched} candles, current date: ${new Date(currentTs * 1000).toISOString()}`);
      
      await new Promise(r => setTimeout(r, 500));
      
    } catch (error) {
      console.error("Error fetching historical data:", error);
      break;
    }
  }
  
  console.log(`Total candles fetched: ${totalFetched}`);
  return totalFetched;
}

export async function getStoredCandleCount(): Promise<number> {
  const result = await db.select().from(candles).limit(1);
  return result.length;
}

export async function getStoredCandles(limit: number = 1000): Promise<typeof candles.$inferSelect[]> {
  return await db.select().from(candles).orderBy(desc(candles.timestamp)).limit(limit);
}

export async function getCandlesInRange(startTs: number, endTs: number): Promise<typeof candles.$inferSelect[]> {
  return await db.select()
    .from(candles)
    .where(and(gte(candles.timestamp, startTs), lte(candles.timestamp, endTs)))
    .orderBy(candles.timestamp);
}
