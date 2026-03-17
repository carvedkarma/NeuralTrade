import crypto from "crypto";
import { db } from "../db";
import { settings } from "@shared/schema";
import { eq } from "drizzle-orm";

const BITGET_BASE_URL = "https://api.bitget.com";

interface BitgetCredentials {
  apiKey: string;
  secretKey: string;
  passphrase: string;
}

let _cachedCredentials: BitgetCredentials | null = null;

export async function saveCredentials(creds: BitgetCredentials): Promise<void> {
  const existing = await db.select().from(settings).where(eq(settings.key, "bitget_credentials")).limit(1);
  const value = { apiKey: creds.apiKey, secretKey: creds.secretKey, passphrase: creds.passphrase };
  if (existing.length > 0) {
    await db.update(settings).set({ valueJson: value, updatedAt: BigInt(Date.now()) }).where(eq(settings.key, "bitget_credentials"));
  } else {
    await db.insert(settings).values({ key: "bitget_credentials", valueJson: value, updatedAt: BigInt(Date.now()) });
  }
  _cachedCredentials = creds;
}

export async function loadCredentials(): Promise<BitgetCredentials | null> {
  if (_cachedCredentials) return _cachedCredentials;
  try {
    const rows = await db.select().from(settings).where(eq(settings.key, "bitget_credentials")).limit(1);
    if (rows.length > 0 && rows[0].valueJson) {
      const saved = rows[0].valueJson as any;
      if (saved.apiKey && saved.secretKey && saved.passphrase) {
        _cachedCredentials = { apiKey: saved.apiKey, secretKey: saved.secretKey, passphrase: saved.passphrase };
        return _cachedCredentials;
      }
    }
  } catch (err: any) {
    console.error("[Bitget] Failed to load credentials:", err.message);
  }
  return null;
}

export function isConfigured(): boolean {
  return _cachedCredentials !== null;
}

function getCredentials(): BitgetCredentials {
  if (!_cachedCredentials) {
    throw new Error("Bitget API credentials not configured");
  }
  return _cachedCredentials;
}

function generateSignature(secretKey: string, timestamp: string, method: string, requestPath: string, body: string): string {
  const preSign = timestamp + method.toUpperCase() + requestPath + body;
  return crypto.createHmac("sha256", secretKey).update(preSign).digest("base64");
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface BitgetResponse<T> {
  code: string;
  msg: string;
  data: T;
  requestTime: number;
}

async function request<T>(
  method: "GET" | "POST",
  endpoint: string,
  params?: Record<string, any>
): Promise<BitgetResponse<T>> {
  const creds = getCredentials();
  const maxRetries = 1;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) await sleep(2000);

    const timestamp = Date.now().toString();
    let queryString = "";
    let body = "";
    let requestPath = endpoint;

    if (method === "GET" && params) {
      const qs = new URLSearchParams();
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null) qs.append(k, String(v));
      }
      queryString = qs.toString();
      if (queryString) requestPath = `${endpoint}?${queryString}`;
    }

    if (method === "POST" && params) {
      body = JSON.stringify(params);
    }

    const signPayload = method === "GET" ? "" : body;
    const signature = generateSignature(creds.secretKey, timestamp, method, requestPath, signPayload);

    const headers: Record<string, string> = {
      "ACCESS-KEY": creds.apiKey,
      "ACCESS-SIGN": signature,
      "ACCESS-PASSPHRASE": creds.passphrase,
      "ACCESS-TIMESTAMP": timestamp,
      "Content-Type": "application/json",
      "locale": "en-US",
    };

    const url = `${BITGET_BASE_URL}${requestPath}`;

    try {
      const res = await fetch(url, {
        method,
        headers,
        body: method === "POST" ? body : undefined,
        signal: AbortSignal.timeout(15000),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        if (res.status === 403) {
          throw new Error(`Bitget API rejected request (403 Forbidden)`);
        }
        if (res.status === 502 || res.status === 503) {
          lastError = new Error(`Bitget API temporarily unavailable (${res.status})`);
          continue;
        }
        throw new Error(`Bitget API error ${res.status}: ${text}`);
      }

      return await res.json() as BitgetResponse<T>;
    } catch (err: any) {
      if (err.message.includes("403 Forbidden")) throw err;
      if (err.name === "TimeoutError" || err.message.includes("timeout")) {
        lastError = new Error("Bitget API request timed out");
        continue;
      }
      if (err.message.includes("fetch failed") || err.message.includes("ECONNREFUSED")) {
        lastError = new Error("Cannot reach Bitget API");
        continue;
      }
      lastError = err;
      if (attempt < maxRetries) continue;
    }
  }

  throw lastError || new Error("Bitget API request failed after retries");
}

export async function testConnection(): Promise<{ success: boolean; error?: string; balance?: string }> {
  try {
    const creds = await loadCredentials();
    if (!creds) {
      return { success: false, error: "API credentials not configured" };
    }
    const result = await getAccountBalance();
    if (result.code !== "00000") {
      return { success: false, error: `Bitget API error (${result.code}): ${result.msg}` };
    }
    const usdtAccount = result.data?.find((a: any) => a.marginCoin === "USDT");
    return {
      success: true,
      balance: usdtAccount?.available || "0",
    };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

export function getMaskedApiKey(): string | null {
  if (!_cachedCredentials) return null;
  const key = _cachedCredentials.apiKey;
  if (key.length <= 8) return key.slice(0, 4) + "****";
  return key.slice(0, 8) + "****";
}

export async function getUserInfo(): Promise<{ uid?: string; email?: string; userType?: string; regisTime?: string; error?: string }> {
  try {
    const resp = await request<any>("GET", "/api/v2/user/info");
    if (resp.code === "00000" && resp.data) {
      return {
        uid: resp.data.userId || resp.data.uid,
        email: resp.data.email,
        userType: resp.data.userType,
        regisTime: resp.data.regisTime,
      };
    }
    return { error: `getUserInfo failed: ${resp.msg}` };
  } catch (err: any) {
    return { error: err.message };
  }
}

export async function getServerTime(): Promise<BitgetResponse<{ serverTime: string }>> {
  const res = await fetch(`${BITGET_BASE_URL}/api/v2/public/time`, {
    signal: AbortSignal.timeout(10000),
  });
  return await res.json();
}

export async function getAccountBalance(): Promise<BitgetResponse<any[]>> {
  return request("GET", "/api/v2/mix/account/accounts", { productType: "USDT-FUTURES" });
}

export async function getAllPositions(): Promise<BitgetResponse<any[]>> {
  return request("GET", "/api/v2/mix/position/all-position", { productType: "USDT-FUTURES" });
}

export async function placeOrder(params: {
  symbol: string;
  side: "buy" | "sell";
  tradeSide: "open" | "close";
  orderType: "market" | "limit";
  size: string;
  price?: string;
  presetStopLossPrice?: string;
  presetStopSurplusPrice?: string;
  marginCoin?: string;
}): Promise<BitgetResponse<{ orderId: string; clientOid: string }>> {
  return request("POST", "/api/v2/mix/order/place-order", {
    ...params,
    productType: "USDT-FUTURES",
    marginMode: "isolated",
    marginCoin: params.marginCoin || "USDT",
  });
}

export async function closePosition(symbol: string, holdSide: "long" | "short"): Promise<BitgetResponse<any>> {
  return request("POST", "/api/v2/mix/order/close-positions", {
    symbol,
    productType: "USDT-FUTURES",
    holdSide,
  });
}

export async function setLeverage(symbol: string, leverage: string, holdSide?: "long" | "short"): Promise<BitgetResponse<any>> {
  const params: Record<string, any> = {
    symbol,
    productType: "USDT-FUTURES",
    marginCoin: "USDT",
    leverage,
  };
  if (holdSide) params.holdSide = holdSide;
  return request("POST", "/api/v2/mix/account/set-leverage", params);
}

export async function getOrderFills(symbol?: string, limit: number = 50): Promise<BitgetResponse<any[]>> {
  const params: Record<string, any> = { productType: "USDT-FUTURES", limit };
  if (symbol) params.symbol = symbol;
  return request("GET", "/api/v2/mix/order/fills", params);
}

export async function setMarginMode(symbol: string, marginMode: "isolated" | "crossed"): Promise<BitgetResponse<any>> {
  return request("POST", "/api/v2/mix/account/set-margin-mode", {
    symbol,
    productType: "USDT-FUTURES",
    marginCoin: "USDT",
    marginMode,
  });
}

export async function setPositionTpSl(params: {
  symbol: string;
  holdSide: "long" | "short";
  stopLossPrice: string;
  stopSurplusPrice?: string;
}): Promise<BitgetResponse<any>> {
  const body: Record<string, any> = {
    symbol: params.symbol,
    productType: "USDT-FUTURES",
    marginCoin: "USDT",
    holdSide: params.holdSide,
    stopLossPrice: params.stopLossPrice,
  };
  if (params.stopSurplusPrice) body.stopSurplusPrice = params.stopSurplusPrice;
  return request("POST", "/api/v2/mix/order/place-tpsl-order", body);
}
