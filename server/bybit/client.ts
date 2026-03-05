import crypto from "crypto";

const BYBIT_BASE_URL = "https://api.bybit.com";
const RECV_WINDOW = "5000";

function getGpuProxyUrl(): string {
  const url = process.env.GPU_TRAINER_URL || "http://localhost:8000";
  return url.replace(/\/+$/, "");
}

function getCredentials() {
  const apiKey = process.env.BYBIT_API_KEY;
  const apiSecret = process.env.BYBIT_API_SECRET;
  if (!apiKey || !apiSecret) {
    throw new Error("BYBIT_API_KEY and BYBIT_API_SECRET must be set");
  }
  return { apiKey, apiSecret };
}

function generateSignature(
  apiSecret: string,
  timestamp: string,
  apiKey: string,
  recvWindow: string,
  payload: string
): string {
  const preSign = timestamp + apiKey + recvWindow + payload;
  return crypto.createHmac("sha256", apiSecret).update(preSign).digest("hex");
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function request<T>(
  method: "GET" | "POST",
  endpoint: string,
  params?: Record<string, any>
): Promise<BybitResponse<T>> {
  const { apiKey, apiSecret } = getCredentials();
  const proxyUrl = getGpuProxyUrl();

  const maxRetries = 1;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) {
      await sleep(2000);
    }

    const timestamp = Date.now().toString();

    let queryString = "";
    let body = "";

    if (method === "GET" && params) {
      const qs = new URLSearchParams();
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null) qs.append(k, String(v));
      }
      queryString = qs.toString();
    }

    if (method === "POST") {
      body = params ? JSON.stringify(params) : "";
    }

    const signPayload = method === "GET" ? queryString : body;
    const signature = generateSignature(apiSecret, timestamp, apiKey, RECV_WINDOW, signPayload);

    const headers: Record<string, string> = {
      "X-BAPI-API-KEY": apiKey,
      "X-BAPI-SIGN": signature,
      "X-BAPI-SIGN-TYPE": "2",
      "X-BAPI-TIMESTAMP": timestamp,
      "X-BAPI-RECV-WINDOW": RECV_WINDOW,
    };
    if (method === "POST") {
      headers["Content-Type"] = "application/json";
    }

    try {
      const proxyRes = await fetch(`${proxyUrl}/bybit-proxy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          method,
          endpoint,
          queryString: queryString || undefined,
          body: body || undefined,
          headers,
          baseUrl: BYBIT_BASE_URL,
        }),
        signal: AbortSignal.timeout(15000),
      });

      if (!proxyRes.ok) {
        const text = await proxyRes.text().catch(() => "");
        if (proxyRes.status === 404) {
          throw new Error(`GPU trainer at ${proxyUrl} does not have /bybit-proxy endpoint — update your GPU trainer code`);
        }
        if (proxyRes.status === 502 || proxyRes.status === 503) {
          lastError = new Error(`GPU trainer proxy temporarily unavailable (${proxyRes.status})`);
          continue;
        }
        throw new Error(`GPU proxy error ${proxyRes.status}: ${text}`);
      }

      const proxyData = await proxyRes.json() as any;

      if (proxyData.error) {
        throw new Error(`Bybit proxy error: ${proxyData.error}`);
      }

      return proxyData as BybitResponse<T>;
    } catch (err: any) {
      if (err.message.includes("does not have /bybit-proxy") || err.message.includes("Bybit proxy error")) {
        throw err;
      }

      if (err.name === "TimeoutError" || err.message.includes("timeout")) {
        lastError = new Error(`Request to GPU trainer timed out (${proxyUrl}) — check your ngrok tunnel is running`);
        continue;
      }
      if (err.message.includes("fetch failed") || err.message.includes("ECONNREFUSED") || err.message.includes("ENOTFOUND")) {
        const isLocalhost = proxyUrl.includes("localhost") || proxyUrl.includes("127.0.0.1");
        if (isLocalhost) {
          throw new Error("GPU_TRAINER_URL is set to localhost — set it to your ngrok tunnel URL (e.g. https://abc123.ngrok-free.app)");
        }
        lastError = new Error(`Cannot reach GPU trainer at ${proxyUrl} — check your ngrok tunnel is running`);
        continue;
      }

      lastError = err;
      if (attempt < maxRetries) continue;
    }
  }

  throw lastError || new Error("Bybit proxy request failed after retries");
}

export interface BybitResponse<T> {
  retCode: number;
  retMsg: string;
  result: T;
  time: number;
}

export interface BybitTicker {
  symbol: string;
  lastPrice: string;
  indexPrice: string;
  markPrice: string;
  prevPrice24h: string;
  price24hPcnt: string;
  highPrice24h: string;
  lowPrice24h: string;
  volume24h: string;
  turnover24h: string;
  bid1Price: string;
  ask1Price: string;
}

export interface BybitPosition {
  symbol: string;
  side: "Buy" | "Sell" | "None";
  size: string;
  avgPrice: string;
  positionValue: string;
  leverage: string;
  markPrice: string;
  unrealisedPnl: string;
  cumRealisedPnl: string;
  stopLoss: string;
  takeProfit: string;
  trailingStop: string;
  positionIdx: number;
  liqPrice: string;
  bustPrice: string;
  createdTime: string;
  updatedTime: string;
  positionStatus: string;
}

export interface BybitWalletBalance {
  accountType: string;
  totalEquity: string;
  totalWalletBalance: string;
  totalAvailableBalance: string;
  totalMarginBalance: string;
  totalPerpUPL: string;
  coin: Array<{
    coin: string;
    equity: string;
    walletBalance: string;
    availableToWithdraw: string;
    unrealisedPnl: string;
    cumRealisedPnl: string;
    usdValue: string;
  }>;
}

export interface BybitOrderResult {
  orderId: string;
  orderLinkId: string;
}

export interface BybitKline {
  startTime: string;
  openPrice: string;
  highPrice: string;
  lowPrice: string;
  closePrice: string;
  volume: string;
  turnover: string;
}

export function isConfigured(): boolean {
  return !!(process.env.BYBIT_API_KEY && process.env.BYBIT_API_SECRET);
}

export function getProxyStatus(): { url: string; isLocalhost: boolean } {
  const url = getGpuProxyUrl();
  const isLocalhost = url.includes("localhost") || url.includes("127.0.0.1");
  return { url, isLocalhost };
}

export async function testConnection(): Promise<{ success: boolean; error?: string; balance?: string }> {
  try {
    if (!isConfigured()) {
      return { success: false, error: "API credentials not configured" };
    }
    const { isLocalhost } = getProxyStatus();
    if (isLocalhost) {
      return { success: false, error: "GPU_TRAINER_URL is set to localhost — set it to your ngrok tunnel URL" };
    }
    const result = await getWalletBalance("USDT");
    if (result.retCode !== 0) {
      return { success: false, error: `Bybit API error: ${result.retMsg}` };
    }
    const usdtCoin = result.result?.list?.[0]?.coin?.find((c: any) => c.coin === "USDT");
    return {
      success: true,
      balance: usdtCoin?.walletBalance || "0",
    };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

export async function getWalletBalance(coin?: string): Promise<BybitResponse<{ list: BybitWalletBalance[] }>> {
  const params: Record<string, any> = { accountType: "UNIFIED" };
  if (coin) params.coin = coin;
  return request("GET", "/v5/account/wallet-balance", params);
}

export async function getTicker(symbol: string): Promise<BybitResponse<{ list: BybitTicker[] }>> {
  return request("GET", "/v5/market/tickers", { category: "linear", symbol });
}

export async function getTickers(): Promise<BybitResponse<{ list: BybitTicker[] }>> {
  return request("GET", "/v5/market/tickers", { category: "linear" });
}

export async function getPositions(symbol?: string): Promise<BybitResponse<{ list: BybitPosition[] }>> {
  const params: Record<string, any> = { category: "linear", settleCoin: "USDT" };
  if (symbol) params.symbol = symbol;
  return request("GET", "/v5/position/list", params);
}

export async function createOrder(params: {
  symbol: string;
  side: "Buy" | "Sell";
  orderType: "Market" | "Limit";
  qty: string;
  price?: string;
  stopLoss?: string;
  takeProfit?: string;
  timeInForce?: string;
  reduceOnly?: boolean;
  positionIdx?: number;
}): Promise<BybitResponse<BybitOrderResult>> {
  const orderParams: Record<string, any> = {
    category: "linear",
    symbol: params.symbol,
    side: params.side,
    orderType: params.orderType,
    qty: params.qty,
    timeInForce: params.timeInForce || "GTC",
    positionIdx: params.positionIdx ?? 0,
  };
  if (params.price) orderParams.price = params.price;
  if (params.stopLoss) orderParams.stopLoss = params.stopLoss;
  if (params.takeProfit) orderParams.takeProfit = params.takeProfit;
  if (params.reduceOnly) orderParams.reduceOnly = true;
  return request("POST", "/v5/order/create", orderParams);
}

export async function amendOrder(params: {
  symbol: string;
  orderId?: string;
  stopLoss?: string;
  takeProfit?: string;
}): Promise<BybitResponse<BybitOrderResult>> {
  return request("POST", "/v5/order/amend", {
    category: "linear",
    ...params,
  });
}

export async function cancelOrder(params: {
  symbol: string;
  orderId: string;
}): Promise<BybitResponse<BybitOrderResult>> {
  return request("POST", "/v5/order/cancel", {
    category: "linear",
    ...params,
  });
}

export async function setLeverage(symbol: string, buyLeverage: string, sellLeverage: string): Promise<BybitResponse<any>> {
  return request("POST", "/v5/position/set-leverage", {
    category: "linear",
    symbol,
    buyLeverage,
    sellLeverage,
  });
}

export async function setPositionMode(symbol: string, mode: 0 | 3): Promise<BybitResponse<any>> {
  return request("POST", "/v5/position/switch-mode", {
    category: "linear",
    symbol,
    mode,
  });
}

export async function getKlines(symbol: string, interval: string, limit: number = 200): Promise<BybitResponse<{ list: string[][] }>> {
  return request("GET", "/v5/market/kline", {
    category: "linear",
    symbol,
    interval,
    limit,
  });
}

export async function setTradingStop(params: {
  symbol: string;
  stopLoss?: string;
  takeProfit?: string;
  trailingStop?: string;
  positionIdx?: number;
}): Promise<BybitResponse<any>> {
  return request("POST", "/v5/position/trading-stop", {
    category: "linear",
    symbol: params.symbol,
    stopLoss: params.stopLoss,
    takeProfit: params.takeProfit,
    trailingStop: params.trailingStop,
    positionIdx: params.positionIdx ?? 0,
  });
}

export async function getOrderHistory(symbol?: string, limit: number = 50): Promise<BybitResponse<{ list: any[] }>> {
  const params: Record<string, any> = { category: "linear", limit };
  if (symbol) params.symbol = symbol;
  return request("GET", "/v5/order/history", params);
}

export async function getClosedPnl(symbol?: string, limit: number = 50): Promise<BybitResponse<{ list: any[] }>> {
  const params: Record<string, any> = { category: "linear", limit };
  if (symbol) params.symbol = symbol;
  return request("GET", "/v5/position/closed-pnl", params);
}
