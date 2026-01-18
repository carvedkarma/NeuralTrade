import OpenAI from "openai";
import type { Candle, FuturesData, MultiTimeframeScore, WhaleActivity } from "@shared/schema";
import type { TechnicalIndicators } from "./indicators";

let openai: OpenAI | null = null;

try {
  const apiKey = process.env.AI_INTEGRATIONS_OPENAI_API_KEY || process.env.OPENAI_API_KEY;
  const baseURL = process.env.AI_INTEGRATIONS_OPENAI_BASE_URL;
  
  if (apiKey) {
    openai = new OpenAI({
      apiKey,
      baseURL: baseURL || undefined,
    });
    console.log("OpenAI configured - AI features enabled");
  }
} catch (e) {
  console.log("OpenAI not configured - AI features will use fallback analysis");
}

export interface AIAnalysis {
  marketSummary: string;
  trendExplanation: string;
  signalReasoning: string;
  riskAssessment: string;
  recommendation: "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL";
  confidence: number;
  keyInsights: string[];
  warnings: string[];
}

export interface AISignal {
  direction: "LONG" | "SHORT" | "HOLD";
  confidence: number;
  entryPrice: number | null;
  stopLoss: number | null;
  takeProfit1: number | null;
  takeProfit2: number | null;
  reasoning: string;
  riskReward: number;
  timeframe: string;
}

function safeNum(val: unknown, fallback = 0): number {
  return typeof val === 'number' && !isNaN(val) ? val : fallback;
}

function safeStr(val: unknown, fallback = "N/A"): string {
  return typeof val === 'string' ? val : fallback;
}

export async function analyzeMarket(
  candles: Candle[],
  indicators: TechnicalIndicators,
  futuresData: FuturesData,
  whaleActivity: WhaleActivity,
  mtfScore: Partial<MultiTimeframeScore>
): Promise<AIAnalysis> {
  if (!openai || !candles || candles.length === 0) {
    return getDefaultAnalysis(indicators, futuresData, mtfScore);
  }
  
  const lastCandle = candles[candles.length - 1];
  if (!lastCandle) {
    return getDefaultAnalysis(indicators, futuresData, mtfScore);
  }
  
  const priceChange24h = candles.length >= 96 
    ? ((lastCandle.close - candles[candles.length - 96].close) / candles[candles.length - 96].close) * 100 
    : 0;

  const rsiVal = safeNum(indicators?.rsi?.value, 50);
  const rsiDesc = safeStr(indicators?.rsi?.description, "N/A");
  const macdHist = safeNum(indicators?.macd?.histogram, 0);
  const bbPercentB = safeNum(indicators?.bollingerBands?.percentB, 0.5);
  const adxVal = safeNum(indicators?.adx?.value, 20);
  const adxDesc = safeStr(indicators?.adx?.description, "N/A");
  const stochK = safeNum(indicators?.stochastic?.k, 50);
  const stochD = safeNum(indicators?.stochastic?.d, 50);
  const vwapVal = safeNum(indicators?.vwap?.value, lastCandle.close);
  const supports = indicators?.supportResistance?.supports ?? [];
  const resistances = indicators?.supportResistance?.resistances ?? [];

  const prompt = `You are an expert crypto futures trader analyzing BTCUSDT. Provide a concise market analysis.

CURRENT MARKET DATA:
- Price: $${lastCandle.close.toLocaleString()}
- 24h Change: ${priceChange24h.toFixed(2)}%
- Volume: $${(lastCandle.volume / 1e9).toFixed(2)}B

TECHNICAL INDICATORS:
- RSI(14): ${rsiVal.toFixed(1)} - ${rsiDesc}
- MACD: ${macdHist > 0 ? "Bullish" : "Bearish"} histogram at ${macdHist.toFixed(2)}
- Bollinger %B: ${(bbPercentB * 100).toFixed(1)}%
- ADX: ${adxVal.toFixed(1)} - ${adxDesc}
- Stochastic: K=${stochK.toFixed(1)}, D=${stochD.toFixed(1)}

FUTURES DATA:
- Funding Rate: ${(safeNum(futuresData?.fundingRate) * 100).toFixed(4)}%
- Open Interest: $${(safeNum(futuresData?.openInterest) / 1e9).toFixed(2)}B
- Long/Short Ratio: ${safeNum(futuresData?.longShortRatio, 1).toFixed(2)}
- Basis: ${(safeNum(futuresData?.basis) * 100).toFixed(4)}%

WHALE ACTIVITY:
- Large Buys: $${(safeNum(whaleActivity?.largeBuys) / 1e6).toFixed(2)}M
- Large Sells: $${(safeNum(whaleActivity?.largeSells) / 1e6).toFixed(2)}M
- Net Flow: $${(safeNum(whaleActivity?.netFlow) / 1e6).toFixed(2)}M
- Whale Sentiment: ${safeStr(whaleActivity?.whaleActivity, "neutral")}

MULTI-TIMEFRAME:
- Score: ${safeNum(mtfScore?.score).toFixed(2)} (${safeStr(mtfScore?.direction, "neutral")})
- Alignment: ${(safeNum(mtfScore?.alignment) * 100).toFixed(0)}%

SUPPORT/RESISTANCE:
- Nearest Support: $${supports[0]?.toLocaleString() ?? "N/A"}
- Nearest Resistance: $${resistances[0]?.toLocaleString() ?? "N/A"}
- VWAP: $${vwapVal.toLocaleString()}

Respond in this exact JSON format:
{
  "marketSummary": "2-3 sentence summary of current market conditions",
  "trendExplanation": "Explain the current trend and what's driving it",
  "signalReasoning": "Why you recommend this action based on the data",
  "riskAssessment": "Key risks to watch for",
  "recommendation": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 0.0-1.0,
  "keyInsights": ["insight1", "insight2", "insight3"],
  "warnings": ["warning1", "warning2"]
}`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are an expert cryptocurrency trader and analyst. Provide accurate, actionable analysis in JSON format only. Be concise but insightful." },
        { role: "user", content: prompt }
      ],
      response_format: { type: "json_object" },
      temperature: 0.3,
      max_tokens: 800,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error("No response from AI");

    const analysis = JSON.parse(content) as AIAnalysis;
    return analysis;
  } catch (error) {
    console.error("AI analysis error:", error);
    return getDefaultAnalysis(indicators, futuresData, mtfScore);
  }
}

export async function generateAISignal(
  candles: Candle[],
  indicators: TechnicalIndicators,
  futuresData: FuturesData,
  whaleActivity: WhaleActivity,
  mtfScore: Partial<MultiTimeframeScore>
): Promise<AISignal> {
  if (!candles || candles.length === 0) {
    return getDefaultSignal(0, indicators, mtfScore);
  }
  
  const lastCandle = candles[candles.length - 1];
  if (!lastCandle) {
    return getDefaultSignal(0, indicators, mtfScore);
  }
  
  const atr = safeNum(indicators?.atr?.value, 500);
  const rsiVal = safeNum(indicators?.rsi?.value, 50);
  const rsiSig = safeStr(indicators?.rsi?.signal, "neutral");
  const macdSig = safeStr(indicators?.macd?.signal, "neutral");
  const macdHist = safeNum(indicators?.macd?.histogram, 0);
  const bbPercentB = safeNum(indicators?.bollingerBands?.percentB, 0.5);
  const adxVal = safeNum(indicators?.adx?.value, 20);
  const plusDI = safeNum(indicators?.adx?.plusDI, 20);
  const minusDI = safeNum(indicators?.adx?.minusDI, 20);
  const stochK = safeNum(indicators?.stochastic?.k, 50);
  const stochD = safeNum(indicators?.stochastic?.d, 50);
  const obvSig = safeStr(indicators?.obv?.signal, "neutral");
  const vwapVal = safeNum(indicators?.vwap?.value, lastCandle.close);
  const supports = indicators?.supportResistance?.supports ?? [];
  const resistances = indicators?.supportResistance?.resistances ?? [];

  if (!openai) {
    return getDefaultSignal(lastCandle.close, indicators, mtfScore);
  }

  const prompt = `You are a professional crypto futures trader. Generate a trading signal for BTCUSDT 15m timeframe.

PRICE: $${lastCandle.close.toLocaleString()}
ATR(14): $${atr.toFixed(2)}

INDICATORS:
- RSI: ${rsiVal.toFixed(1)} (${rsiSig})
- MACD: ${macdSig} (histogram: ${macdHist.toFixed(2)})
- Bollinger %B: ${(bbPercentB * 100).toFixed(1)}%
- ADX: ${adxVal.toFixed(1)} (+DI: ${plusDI.toFixed(1)}, -DI: ${minusDI.toFixed(1)})
- Stochastic: ${stochK.toFixed(1)}/${stochD.toFixed(1)}
- OBV: ${obvSig}
- VWAP: Price ${lastCandle.close > vwapVal ? "above" : "below"} at $${vwapVal.toFixed(2)}

FUTURES:
- Funding: ${(safeNum(futuresData?.fundingRate) * 100).toFixed(4)}%
- L/S Ratio: ${safeNum(futuresData?.longShortRatio, 1).toFixed(2)}

WHALE: ${safeStr(whaleActivity?.whaleActivity, "neutral")} ($${(safeNum(whaleActivity?.netFlow) / 1e6).toFixed(2)}M net)

MTF SCORE: ${safeNum(mtfScore?.score).toFixed(2)} (${safeStr(mtfScore?.direction, "neutral")}, ${(safeNum(mtfScore?.alignment) * 100).toFixed(0)}% aligned)

S/R LEVELS:
- Support: $${supports[0]?.toFixed(0) ?? "N/A"}
- Resistance: $${resistances[0]?.toFixed(0) ?? "N/A"}

Generate a signal. Use ATR for stop calculation. Target 2R minimum. Only signal if confidence > 60%.

Respond in JSON:
{
  "direction": "LONG|SHORT|HOLD",
  "confidence": 0.0-1.0,
  "entryPrice": number or null,
  "stopLoss": number or null,
  "takeProfit1": number or null,
  "takeProfit2": number or null,
  "reasoning": "1-2 sentence explanation",
  "riskReward": number,
  "timeframe": "15m"
}`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are a professional crypto futures trader. Generate precise, risk-managed trading signals. JSON only." },
        { role: "user", content: prompt }
      ],
      response_format: { type: "json_object" },
      temperature: 0.2,
      max_tokens: 400,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error("No response from AI");

    const signal = JSON.parse(content) as AISignal;
    return signal;
  } catch (error) {
    console.error("AI signal generation error:", error);
    return getDefaultSignal(lastCandle.close, indicators, mtfScore);
  }
}

export async function explainTrade(
  trade: { side: string; entryPrice: number; stopLoss: number; takeProfit: number; signalType: string },
  currentPrice: number,
  indicators: TechnicalIndicators
): Promise<string> {
  const entry = safeNum(trade?.entryPrice, 0);
  const current = safeNum(currentPrice, entry);
  const side = safeStr(trade?.side, "LONG");
  const sigType = safeStr(trade?.signalType, "retest");
  
  if (!openai) {
    const pnl = side === "LONG" 
      ? ((current - entry) / (entry || 1)) * 100
      : ((entry - current) / (entry || 1)) * 100;
    return `${side} position from $${entry.toLocaleString()}, currently ${pnl > 0 ? "+" : ""}${pnl.toFixed(2)}%. ${sigType} signal.`;
  }

  const rsiVal = safeNum(indicators?.rsi?.value, 50);
  const macdSig = safeStr(indicators?.macd?.signal, "neutral");
  const stopLoss = safeNum(trade?.stopLoss, entry);
  const takeProfit = safeNum(trade?.takeProfit, entry);

  const prompt = `Explain this BTC futures trade briefly:
- Side: ${side}
- Entry: $${entry.toLocaleString()}
- Current: $${current.toLocaleString()}
- Stop: $${stopLoss.toLocaleString()}
- Target: $${takeProfit.toLocaleString()}
- Signal: ${sigType}

Current RSI: ${rsiVal.toFixed(1)}, MACD: ${macdSig}

In 1-2 sentences, explain the trade logic and current status.`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "user", content: prompt }
      ],
      temperature: 0.3,
      max_tokens: 150,
    });

    return response.choices[0]?.message?.content ?? "Trade analysis unavailable.";
  } catch (error) {
    console.error("Trade explanation error:", error);
    const pnl = trade.side === "LONG" 
      ? ((currentPrice - trade.entryPrice) / trade.entryPrice) * 100
      : ((trade.entryPrice - currentPrice) / trade.entryPrice) * 100;
    return `${trade.side} position from $${trade.entryPrice.toLocaleString()}, currently ${pnl > 0 ? "+" : ""}${pnl.toFixed(2)}%. ${trade.signalType} signal.`;
  }
}

function getDefaultAnalysis(
  indicators: TechnicalIndicators,
  futuresData: FuturesData,
  mtfScore: Partial<MultiTimeframeScore>
): AIAnalysis {
  const rsiVal = safeNum(indicators?.rsi?.value, 50);
  const rsiDesc = safeStr(indicators?.rsi?.description, "RSI neutral");
  const macdHist = safeNum(indicators?.macd?.histogram, 0);
  const macdDesc = safeStr(indicators?.macd?.description, "MACD neutral");
  const macdSig = safeStr(indicators?.macd?.signal, "neutral");
  const adxDesc = safeStr(indicators?.adx?.description, "ADX neutral");
  const atrVal = safeNum(indicators?.atr?.value, 500);
  const vwapVal = safeNum(indicators?.vwap?.value, 1);
  const supports = indicators?.supportResistance?.supports ?? [];
  const direction = safeStr(mtfScore?.direction, "neutral");
  const fundingRate = safeNum(futuresData?.fundingRate, 0);
  const lsRatio = safeNum(futuresData?.longShortRatio, 1);
  
  let recommendation: AIAnalysis["recommendation"] = "HOLD";
  let confidence = 0.5;
  
  if (rsiVal < 30 && macdHist > 0 && direction === "bullish") {
    recommendation = "STRONG_BUY";
    confidence = 0.75;
  } else if (rsiVal < 40 && direction === "bullish") {
    recommendation = "BUY";
    confidence = 0.65;
  } else if (rsiVal > 70 && macdHist < 0 && direction === "bearish") {
    recommendation = "STRONG_SELL";
    confidence = 0.75;
  } else if (rsiVal > 60 && direction === "bearish") {
    recommendation = "SELL";
    confidence = 0.65;
  }

  return {
    marketSummary: `BTC showing ${direction} momentum with RSI at ${rsiVal.toFixed(1)}. Funding rate at ${(fundingRate * 100).toFixed(4)}% indicates ${fundingRate > 0 ? "bullish" : "bearish"} sentiment.`,
    trendExplanation: `${macdDesc}. ${adxDesc}.`,
    signalReasoning: `Technical indicators suggest ${recommendation.toLowerCase().replace("_", " ")} based on ${rsiDesc} and ${macdSig} MACD.`,
    riskAssessment: `ATR suggests volatility of ${((atrVal / vwapVal) * 100).toFixed(2)}%. Watch ${supports[0] ? `support at $${supports[0].toLocaleString()}` : "key levels"}.`,
    recommendation,
    confidence,
    keyInsights: [
      rsiDesc,
      macdDesc,
      `L/S ratio at ${lsRatio.toFixed(2)}`
    ],
    warnings: fundingRate > 0.001 
      ? ["High funding rate may indicate overleveraged longs"]
      : fundingRate < -0.001
      ? ["Negative funding suggests bearish sentiment"]
      : []
  };
}

function getDefaultSignal(
  currentPrice: number,
  indicators: TechnicalIndicators,
  mtfScore: Partial<MultiTimeframeScore>
): AISignal {
  const price = safeNum(currentPrice, 100000);
  const atr = safeNum(indicators?.atr?.value, 500);
  const rsiVal = safeNum(indicators?.rsi?.value, 50);
  const rsiDesc = safeStr(indicators?.rsi?.description, "RSI neutral");
  const mtfDirection = safeStr(mtfScore?.direction, "neutral");
  const mtfAlignment = safeNum(mtfScore?.alignment, 0);
  
  let direction: AISignal["direction"] = "HOLD";
  let confidence = 0.5;
  
  if (rsiVal < 35 && mtfDirection === "bullish" && mtfAlignment > 0.5) {
    direction = "LONG";
    confidence = 0.65;
  } else if (rsiVal > 65 && mtfDirection === "bearish" && mtfAlignment > 0.5) {
    direction = "SHORT";
    confidence = 0.65;
  }

  const entry = direction !== "HOLD" ? price : null;
  const stop = direction === "LONG" 
    ? price - atr * 1.5 
    : direction === "SHORT" 
    ? price + atr * 1.5 
    : null;
  const tp1 = direction === "LONG"
    ? price + atr * 3
    : direction === "SHORT"
    ? price - atr * 3
    : null;
  const tp2 = direction === "LONG"
    ? price + atr * 5
    : direction === "SHORT"
    ? price - atr * 5
    : null;

  return {
    direction,
    confidence,
    entryPrice: entry,
    stopLoss: stop,
    takeProfit1: tp1,
    takeProfit2: tp2,
    reasoning: direction !== "HOLD" 
      ? `${direction} signal based on ${rsiDesc} with ${mtfDirection} MTF confirmation.`
      : "No clear setup. Waiting for better conditions.",
    riskReward: 2,
    timeframe: "15m"
  };
}
