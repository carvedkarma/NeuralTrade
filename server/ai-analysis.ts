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

export async function analyzeMarket(
  candles: Candle[],
  indicators: TechnicalIndicators,
  futuresData: FuturesData,
  whaleActivity: WhaleActivity,
  mtfScore: Partial<MultiTimeframeScore>
): Promise<AIAnalysis> {
  if (!openai) {
    return getDefaultAnalysis(indicators, futuresData, mtfScore);
  }
  const lastCandle = candles[candles.length - 1];
  const priceChange24h = candles.length >= 96 
    ? ((lastCandle.close - candles[candles.length - 96].close) / candles[candles.length - 96].close) * 100 
    : 0;

  const prompt = `You are an expert crypto futures trader analyzing BTCUSDT. Provide a concise market analysis.

CURRENT MARKET DATA:
- Price: $${lastCandle.close.toLocaleString()}
- 24h Change: ${priceChange24h.toFixed(2)}%
- Volume: $${(lastCandle.volume / 1e9).toFixed(2)}B

TECHNICAL INDICATORS:
- RSI(14): ${indicators.rsi.value.toFixed(1)} - ${indicators.rsi.description}
- MACD: ${indicators.macd.histogram > 0 ? "Bullish" : "Bearish"} histogram at ${indicators.macd.histogram.toFixed(2)}
- Bollinger %B: ${(indicators.bollingerBands.percentB * 100).toFixed(1)}%
- ADX: ${indicators.adx.value.toFixed(1)} - ${indicators.adx.description}
- Stochastic: K=${indicators.stochastic.k.toFixed(1)}, D=${indicators.stochastic.d.toFixed(1)}

FUTURES DATA:
- Funding Rate: ${(futuresData.fundingRate * 100).toFixed(4)}%
- Open Interest: $${(futuresData.openInterest / 1e9).toFixed(2)}B
- Long/Short Ratio: ${futuresData.longShortRatio.toFixed(2)}
- Basis: ${(futuresData.basis * 100).toFixed(4)}%

WHALE ACTIVITY:
- Large Buys: $${(whaleActivity.largeBuys / 1e6).toFixed(2)}M
- Large Sells: $${(whaleActivity.largeSells / 1e6).toFixed(2)}M
- Net Flow: $${(whaleActivity.netFlow / 1e6).toFixed(2)}M
- Whale Sentiment: ${whaleActivity.whaleActivity}

MULTI-TIMEFRAME:
- Score: ${(mtfScore.score ?? 0).toFixed(2)} (${mtfScore.direction ?? "neutral"})
- Alignment: ${((mtfScore.alignment ?? 0) * 100).toFixed(0)}%

SUPPORT/RESISTANCE:
- Nearest Support: $${indicators.supportResistance.supports[0]?.toLocaleString() ?? "N/A"}
- Nearest Resistance: $${indicators.supportResistance.resistances[0]?.toLocaleString() ?? "N/A"}
- VWAP: $${indicators.vwap.value.toLocaleString()}

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
  const lastCandle = candles[candles.length - 1];
  const atr = indicators.atr.value;

  if (!openai) {
    return getDefaultSignal(lastCandle.close, indicators, mtfScore);
  }

  const prompt = `You are a professional crypto futures trader. Generate a trading signal for BTCUSDT 15m timeframe.

PRICE: $${lastCandle.close.toLocaleString()}
ATR(14): $${atr.toFixed(2)}

INDICATORS:
- RSI: ${indicators.rsi.value.toFixed(1)} (${indicators.rsi.signal})
- MACD: ${indicators.macd.signal} (histogram: ${indicators.macd.histogram.toFixed(2)})
- Bollinger %B: ${(indicators.bollingerBands.percentB * 100).toFixed(1)}%
- ADX: ${indicators.adx.value.toFixed(1)} (+DI: ${indicators.adx.plusDI.toFixed(1)}, -DI: ${indicators.adx.minusDI.toFixed(1)})
- Stochastic: ${indicators.stochastic.k.toFixed(1)}/${indicators.stochastic.d.toFixed(1)}
- OBV: ${indicators.obv.signal}
- VWAP: Price ${lastCandle.close > indicators.vwap.value ? "above" : "below"} at $${indicators.vwap.value.toFixed(2)}

FUTURES:
- Funding: ${(futuresData.fundingRate * 100).toFixed(4)}%
- L/S Ratio: ${futuresData.longShortRatio.toFixed(2)}

WHALE: ${whaleActivity.whaleActivity} ($${(whaleActivity.netFlow / 1e6).toFixed(2)}M net)

MTF SCORE: ${(mtfScore.score ?? 0).toFixed(2)} (${mtfScore.direction ?? "neutral"}, ${((mtfScore.alignment ?? 0) * 100).toFixed(0)}% aligned)

S/R LEVELS:
- Support: $${indicators.supportResistance.supports[0]?.toFixed(0) ?? "N/A"}
- Resistance: $${indicators.supportResistance.resistances[0]?.toFixed(0) ?? "N/A"}

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
  if (!openai) {
    const pnl = trade.side === "LONG" 
      ? ((currentPrice - trade.entryPrice) / trade.entryPrice) * 100
      : ((trade.entryPrice - currentPrice) / trade.entryPrice) * 100;
    return `${trade.side} position from $${trade.entryPrice.toLocaleString()}, currently ${pnl > 0 ? "+" : ""}${pnl.toFixed(2)}%. ${trade.signalType} signal.`;
  }

  const prompt = `Explain this BTC futures trade briefly:
- Side: ${trade.side}
- Entry: $${trade.entryPrice.toLocaleString()}
- Current: $${currentPrice.toLocaleString()}
- Stop: $${trade.stopLoss.toLocaleString()}
- Target: $${trade.takeProfit.toLocaleString()}
- Signal: ${trade.signalType}

Current RSI: ${indicators.rsi.value.toFixed(1)}, MACD: ${indicators.macd.signal}

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
  const rsi = indicators.rsi;
  const macd = indicators.macd;
  const direction = mtfScore.direction ?? "neutral";
  
  let recommendation: AIAnalysis["recommendation"] = "HOLD";
  let confidence = 0.5;
  
  if (rsi.value < 30 && macd.histogram > 0 && direction === "bullish") {
    recommendation = "STRONG_BUY";
    confidence = 0.75;
  } else if (rsi.value < 40 && direction === "bullish") {
    recommendation = "BUY";
    confidence = 0.65;
  } else if (rsi.value > 70 && macd.histogram < 0 && direction === "bearish") {
    recommendation = "STRONG_SELL";
    confidence = 0.75;
  } else if (rsi.value > 60 && direction === "bearish") {
    recommendation = "SELL";
    confidence = 0.65;
  }

  return {
    marketSummary: `BTC showing ${direction} momentum with RSI at ${rsi.value.toFixed(1)}. Funding rate at ${(futuresData.fundingRate * 100).toFixed(4)}% indicates ${futuresData.fundingRate > 0 ? "bullish" : "bearish"} sentiment.`,
    trendExplanation: `${macd.description}. ${indicators.adx.description}.`,
    signalReasoning: `Technical indicators suggest ${recommendation.toLowerCase().replace("_", " ")} based on ${rsi.description} and ${macd.signal} MACD.`,
    riskAssessment: `ATR suggests volatility of ${((indicators.atr.value / indicators.vwap.value) * 100).toFixed(2)}%. Watch ${indicators.supportResistance.supports[0] ? `support at $${indicators.supportResistance.supports[0].toLocaleString()}` : "key levels"}.`,
    recommendation,
    confidence,
    keyInsights: [
      rsi.description,
      macd.description,
      `L/S ratio at ${futuresData.longShortRatio.toFixed(2)}`
    ],
    warnings: futuresData.fundingRate > 0.001 
      ? ["High funding rate may indicate overleveraged longs"]
      : futuresData.fundingRate < -0.001
      ? ["Negative funding suggests bearish sentiment"]
      : []
  };
}

function getDefaultSignal(
  currentPrice: number,
  indicators: TechnicalIndicators,
  mtfScore: Partial<MultiTimeframeScore>
): AISignal {
  const atr = indicators.atr.value;
  const rsi = indicators.rsi;
  const mtfDirection = mtfScore.direction ?? "neutral";
  const mtfAlignment = mtfScore.alignment ?? 0;
  
  let direction: AISignal["direction"] = "HOLD";
  let confidence = 0.5;
  
  if (rsi.value < 35 && mtfDirection === "bullish" && mtfAlignment > 0.5) {
    direction = "LONG";
    confidence = 0.65;
  } else if (rsi.value > 65 && mtfDirection === "bearish" && mtfAlignment > 0.5) {
    direction = "SHORT";
    confidence = 0.65;
  }

  const entry = direction !== "HOLD" ? currentPrice : null;
  const stop = direction === "LONG" 
    ? currentPrice - atr * 1.5 
    : direction === "SHORT" 
    ? currentPrice + atr * 1.5 
    : null;
  const tp1 = direction === "LONG"
    ? currentPrice + atr * 3
    : direction === "SHORT"
    ? currentPrice - atr * 3
    : null;
  const tp2 = direction === "LONG"
    ? currentPrice + atr * 5
    : direction === "SHORT"
    ? currentPrice - atr * 5
    : null;

  return {
    direction,
    confidence,
    entryPrice: entry,
    stopLoss: stop,
    takeProfit1: tp1,
    takeProfit2: tp2,
    reasoning: direction !== "HOLD" 
      ? `${direction} signal based on ${rsi.description} with ${mtfDirection} MTF confirmation.`
      : "No clear setup. Waiting for better conditions.",
    riskReward: 2,
    timeframe: "15m"
  };
}
