import { TRADING_SYMBOLS } from "../shared/symbols";

export interface OrderFlowSnapshot {
  symbol: string;
  ts: number;
  obImbalance: number;
  aggressorRatio: number;
  cvd: number;
  cvdSlope: number;
  liqProximityUp: number;
  liqProximityDown: number;
  composite: number;
}

export interface OrderFlowGateResult {
  passed: boolean;
  reason: string;
  snapshot: OrderFlowSnapshot | null;
}

const CACHE_TTL_MS = 60_000;
const cache: Map<string, { snapshot: OrderFlowSnapshot; fetchedAt: number }> = new Map();

function isCacheValid(symbol: string): boolean {
  const entry = cache.get(symbol);
  if (!entry) return false;
  return Date.now() - entry.fetchedAt < CACHE_TTL_MS;
}

export function getCachedSnapshot(symbol: string): OrderFlowSnapshot | null {
  const entry = cache.get(symbol);
  if (!entry) return null;
  if (Date.now() - entry.fetchedAt > CACHE_TTL_MS * 3) return null;
  return entry.snapshot;
}

export function getAllSnapshots(): Record<string, OrderFlowSnapshot | null> {
  const result: Record<string, OrderFlowSnapshot | null> = {};
  for (const sym of TRADING_SYMBOLS) {
    result[sym] = getCachedSnapshot(sym);
  }
  return result;
}

async function fetchBybitPublic(endpoint: string, params: Record<string, string>): Promise<any> {
  const qs = new URLSearchParams(params).toString();
  const url = `https://api.bybit.com${endpoint}?${qs}`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const data = await res.json();
    if (data.retCode !== 0) return null;
    return data.result;
  } catch {
    return null;
  }
}

async function fetchOrderbook(symbol: string): Promise<{ bidVol: number; askVol: number; imbalance: number }> {
  const result = await fetchBybitPublic("/v5/market/orderbook", {
    category: "linear",
    symbol,
    limit: "25",
  });
  if (!result?.b?.length || !result?.a?.length) {
    return { bidVol: 0, askVol: 0, imbalance: 0.5 };
  }
  let bidVol = 0;
  let askVol = 0;
  for (const [, qty] of result.b) bidVol += parseFloat(qty);
  for (const [, qty] of result.a) askVol += parseFloat(qty);
  const total = bidVol + askVol || 1;
  return { bidVol, askVol, imbalance: bidVol / total };
}

async function fetchRecentTrades(symbol: string): Promise<{ aggressorRatio: number; cvd: number; cvdSlope: number }> {
  const result = await fetchBybitPublic("/v5/market/recent-trade", {
    category: "linear",
    symbol,
    limit: "500",
  });
  if (!result?.list?.length) {
    return { aggressorRatio: 0.5, cvd: 0, cvdSlope: 0 };
  }

  let buyVol = 0;
  let sellVol = 0;
  let cvd = 0;
  const trades = result.list as Array<{ side: string; size: string; price: string; time: string }>;

  const sortedTrades = [...trades].sort((a, b) => parseInt(a.time) - parseInt(b.time));
  const cvdSeries: number[] = [];

  for (const t of sortedTrades) {
    const vol = parseFloat(t.size) * parseFloat(t.price);
    if (t.side === "Buy") {
      buyVol += vol;
      cvd += vol;
    } else {
      sellVol += vol;
      cvd -= vol;
    }
    cvdSeries.push(cvd);
  }

  const totalVol = buyVol + sellVol || 1;
  const aggressorRatio = buyVol / totalVol;

  let cvdSlope = 0;
  if (cvdSeries.length >= 10) {
    const half = Math.floor(cvdSeries.length / 2);
    const firstHalfAvg = cvdSeries.slice(0, half).reduce((s, v) => s + v, 0) / half;
    const secondHalfAvg = cvdSeries.slice(half).reduce((s, v) => s + v, 0) / (cvdSeries.length - half);
    cvdSlope = secondHalfAvg - firstHalfAvg;
  }

  return { aggressorRatio, cvd, cvdSlope };
}

async function fetchLiquidationProximity(symbol: string): Promise<{ liqUp: number; liqDown: number }> {
  const result = await fetchBybitPublic("/v5/market/tickers", {
    category: "linear",
    symbol,
  });
  if (!result?.list?.[0]) {
    return { liqUp: 999, liqDown: 999 };
  }
  const ticker = result.list[0];
  const lastPrice = parseFloat(ticker.lastPrice);
  const high24h = parseFloat(ticker.highPrice24h);
  const low24h = parseFloat(ticker.lowPrice24h);

  const liqUp = lastPrice > 0 ? ((high24h - lastPrice) / lastPrice) * 100 : 999;
  const liqDown = lastPrice > 0 ? ((lastPrice - low24h) / lastPrice) * 100 : 999;

  return { liqUp: Math.max(liqUp, 0), liqDown: Math.max(liqDown, 0) };
}

function computeComposite(ob: number, agg: number, cvdSlope: number, totalTradedVol: number): number {
  const obScore = (ob - 0.5) * 2;
  const aggScore = (agg - 0.5) * 2;
  const normalizer = totalTradedVol * 0.1 || 1;
  const cvdNorm = cvdSlope === 0 ? 0 : Math.sign(cvdSlope) * Math.min(Math.abs(cvdSlope) / normalizer, 1);
  return obScore * 0.4 + aggScore * 0.35 + cvdNorm * 0.25;
}

export async function fetchOrderFlowSnapshot(symbol: string): Promise<OrderFlowSnapshot> {
  if (isCacheValid(symbol)) {
    return cache.get(symbol)!.snapshot;
  }

  const [ob, trades, liq] = await Promise.all([
    fetchOrderbook(symbol),
    fetchRecentTrades(symbol),
    fetchLiquidationProximity(symbol),
  ]);

  const totalTradedVol = ob.bidVol + ob.askVol;
  const composite = computeComposite(ob.imbalance, trades.aggressorRatio, trades.cvdSlope, totalTradedVol);

  const snapshot: OrderFlowSnapshot = {
    symbol,
    ts: Date.now(),
    obImbalance: parseFloat(ob.imbalance.toFixed(4)),
    aggressorRatio: parseFloat(trades.aggressorRatio.toFixed(4)),
    cvd: parseFloat(trades.cvd.toFixed(2)),
    cvdSlope: parseFloat(trades.cvdSlope.toFixed(2)),
    liqProximityUp: parseFloat(liq.liqUp.toFixed(3)),
    liqProximityDown: parseFloat(liq.liqDown.toFixed(3)),
    composite: parseFloat(composite.toFixed(4)),
  };

  cache.set(symbol, { snapshot, fetchedAt: Date.now() });
  return snapshot;
}

export function evaluateOrderFlowGate(
  snapshot: OrderFlowSnapshot,
  side: "LONG" | "SHORT"
): OrderFlowGateResult {
  if (side === "LONG") {
    if (snapshot.obImbalance < 0.35 && snapshot.aggressorRatio < 0.40) {
      return {
        passed: false,
        reason: `OF_GATE: LONG blocked — OB imbalance=${snapshot.obImbalance.toFixed(2)} (<0.35) & aggressor=${snapshot.aggressorRatio.toFixed(2)} (<0.40)`,
        snapshot,
      };
    }
    if (snapshot.cvdSlope < 0 && snapshot.composite < -0.3) {
      return {
        passed: false,
        reason: `OF_GATE: LONG blocked — CVD falling (slope=${snapshot.cvdSlope.toFixed(0)}) & composite=${snapshot.composite.toFixed(2)} (<-0.3)`,
        snapshot,
      };
    }
  }

  if (side === "SHORT") {
    if (snapshot.obImbalance > 0.65 && snapshot.aggressorRatio > 0.60) {
      return {
        passed: false,
        reason: `OF_GATE: SHORT blocked — OB imbalance=${snapshot.obImbalance.toFixed(2)} (>0.65) & aggressor=${snapshot.aggressorRatio.toFixed(2)} (>0.60)`,
        snapshot,
      };
    }
    if (snapshot.cvdSlope > 0 && snapshot.composite > 0.3) {
      return {
        passed: false,
        reason: `OF_GATE: SHORT blocked — CVD rising (slope=${snapshot.cvdSlope.toFixed(0)}) & composite=${snapshot.composite.toFixed(2)} (>0.3)`,
        snapshot,
      };
    }
  }

  return {
    passed: true,
    reason: `OF_GATE: PASS — imbalance=${snapshot.obImbalance.toFixed(2)}, aggressor=${snapshot.aggressorRatio.toFixed(2)}, composite=${snapshot.composite.toFixed(2)}`,
    snapshot,
  };
}

export function clearOrderFlowCache(): void {
  cache.clear();
}
