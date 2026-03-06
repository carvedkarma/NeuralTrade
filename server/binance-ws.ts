import { broadcast, getClientCount } from "./ws";
import { TRADING_SYMBOLS } from "@shared/symbols";

const SYMBOLS = [...TRADING_SYMBOLS];
const VISION_BASE = "https://data-api.binance.vision/api/v3";

const latestPrices: Record<string, number> = {};
let pollInterval: ReturnType<typeof setInterval> | null = null;
let broadcastInterval: ReturnType<typeof setInterval> | null = null;

export function getLatestPrice(symbol: string): number | null {
  return latestPrices[symbol.toUpperCase()] ?? null;
}

export function getAllLatestPrices(): Record<string, number> {
  return { ...latestPrices };
}

async function fetchPrices() {
  try {
    const symbolsParam = encodeURIComponent(JSON.stringify(SYMBOLS));
    const res = await fetch(`${VISION_BASE}/ticker/price?symbols=${symbolsParam}`);
    if (!res.ok) return;
    const data = await res.json() as Array<{ symbol: string; price: string }>;
    for (const item of data) {
      if (item.symbol && item.price) {
        latestPrices[item.symbol] = parseFloat(item.price);
      }
    }
  } catch {}
}

function startPollLoop() {
  fetchPrices();
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(fetchPrices, 1000);
}

function startBroadcastLoop() {
  if (broadcastInterval) clearInterval(broadcastInterval);
  broadcastInterval = setInterval(() => {
    if (getClientCount() > 0 && Object.keys(latestPrices).length > 0) {
      broadcast("PRICE_TICK", latestPrices);
    }
  }, 1000);
}

export function startBinanceWs() {
  startPollLoop();
  startBroadcastLoop();
  console.log("[Price Ticker] Started (Binance Vision REST, 1s poll, 1s broadcast)");
}
