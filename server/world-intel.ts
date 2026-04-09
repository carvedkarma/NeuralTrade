import OpenAI from "openai";
import { db } from "./db";
import { worldEvents, worldIntelSnapshots, macroIndicators } from "@shared/schema";
import { desc, gt, and, eq } from "drizzle-orm";

let openai: OpenAI | null = null;
try {
  const apiKey = process.env.AI_INTEGRATIONS_OPENAI_API_KEY || process.env.OPENAI_API_KEY;
  const baseURL = process.env.AI_INTEGRATIONS_OPENAI_BASE_URL;
  if (apiKey) {
    openai = new OpenAI({ apiKey, baseURL: baseURL || undefined });
  }
} catch (_) {}

// ─── RSS Feed Definitions ───────────────────────────────────────────────────

const RSS_FEEDS = [
  { url: "https://feeds.reuters.com/reuters/topNews", category: "Markets", source: "Reuters" },
  { url: "http://feeds.bbci.co.uk/news/business/rss.xml", category: "Markets", source: "BBC Business" },
  { url: "https://rsshub.app/apnews/topics/apf-finance", category: "Markets", source: "AP Finance" },
  { url: "https://www.federalreserve.gov/feeds/speeches.xml", category: "Monetary Policy", source: "Federal Reserve" },
  { url: "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&datetype=custom&owner=include&count=10&output=atom", category: "Regulatory", source: "SEC" },
  { url: "https://www.coindesk.com/arc/outboundfeeds/rss/", category: "Crypto", source: "CoinDesk" },
  { url: "https://decrypt.co/feed", category: "Crypto", source: "Decrypt" },
  { url: "https://www.theblock.co/rss.xml", category: "Crypto", source: "The Block" },
];

// ─── RSS Parsing ────────────────────────────────────────────────────────────

interface RssItem {
  title: string;
  link: string;
  description: string;
  pubDate: string;
  source: string;
  category: string;
}

function parseFeedXml(xml: string, source: string, category: string): RssItem[] {
  const items: RssItem[] = [];

  const isAtom = xml.includes("<entry") || xml.includes("xmlns=\"http://www.w3.org/2005/Atom\"");
  const tagName = isAtom ? "entry" : "item";
  const itemRegex = new RegExp(`<${tagName}[^>]*>([\\s\\S]*?)<\\/${tagName}>`, "gi");

  let match;
  while ((match = itemRegex.exec(xml)) !== null) {
    const block = match[1];
    const title = extractXmlTag(block, "title");
    let link = "";
    if (isAtom) {
      const linkMatch = /<link[^>]+href=["']([^"']+)["'][^>]*\/?>/.exec(block);
      link = linkMatch ? linkMatch[1] : extractXmlTag(block, "id");
    } else {
      link = extractXmlTag(block, "link") || extractXmlTag(block, "guid");
    }
    const desc = isAtom
      ? (extractXmlTag(block, "summary") || extractXmlTag(block, "content"))
      : extractXmlTag(block, "description");
    const pubDate = isAtom ? extractXmlTag(block, "updated") : extractXmlTag(block, "pubDate");
    if (title && title.length > 10) {
      items.push({ title: cleanText(title), link: link || "", description: cleanText(desc || ""), pubDate: pubDate || "", source, category });
    }
  }
  return items.slice(0, 15);
}

function parseRssXml(xml: string, source: string, category: string): RssItem[] {
  return parseFeedXml(xml, source, category);
}

function extractXmlTag(xml: string, tag: string): string {
  const re = new RegExp(`<${tag}[^>]*><!\\[CDATA\\[([\\s\\S]*?)\\]\\]><\\/${tag}>|<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i");
  const m = re.exec(xml);
  return m ? (m[1] || m[2] || "").trim() : "";
}

function cleanText(text: string): string {
  return text.replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/\s+/g, " ").trim();
}

async function fetchRssFeed(feed: typeof RSS_FEEDS[0]): Promise<RssItem[]> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    const res = await fetch(feed.url, {
      signal: controller.signal,
      headers: { "User-Agent": "Mozilla/5.0 (compatible; NeuralTerminal/1.0)" },
    });
    clearTimeout(timeout);
    if (!res.ok) return [];
    const xml = await res.text();
    return parseRssXml(xml, feed.source, feed.category);
  } catch (_) {
    return [];
  }
}

// ─── Macro Indicator Fetchers ────────────────────────────────────────────────

interface MacroData {
  dxy: number | null;
  sp500: number | null;
  gold: number | null;
  oil: number | null;
  btcDominance: number | null;
  fearGreedIndex: number | null;
  fearGreedLabel: string | null;
}

async function fetchFearGreed(): Promise<{ value: number; label: string } | null> {
  try {
    const res = await fetch("https://api.alternative.me/fng/?limit=1", { signal: AbortSignal.timeout(6000) });
    if (!res.ok) return null;
    const data = await res.json();
    const item = data?.data?.[0];
    if (!item) return null;
    return { value: parseInt(item.value, 10), label: item.value_classification };
  } catch (_) {
    return null;
  }
}

async function fetchCoinGeckoGlobal(): Promise<{ btcDominance: number } | null> {
  try {
    const res = await fetch("https://api.coingecko.com/api/v3/global", { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const data = await res.json();
    const btcDominance = data?.data?.market_cap_percentage?.btc ?? null;
    if (btcDominance === null) return null;
    return { btcDominance: parseFloat(btcDominance.toFixed(2)) };
  } catch (_) {
    return null;
  }
}

async function fetchYahooPrice(ticker: string): Promise<number | null> {
  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1d&range=1d`;
    const res = await fetch(url, {
      signal: AbortSignal.timeout(8000),
      headers: { "User-Agent": "Mozilla/5.0 (compatible; NeuralTerminal/1.0)" },
    });
    if (!res.ok) return null;
    const data = await res.json();
    const price = data?.chart?.result?.[0]?.meta?.regularMarketPrice ?? null;
    return price ? parseFloat(price.toFixed(2)) : null;
  } catch (_) {
    return null;
  }
}

async function fetchMacroData(): Promise<MacroData> {
  const [fearGreed, cgGlobal, dxy, sp500, gold, oil] = await Promise.allSettled([
    fetchFearGreed(),
    fetchCoinGeckoGlobal(),
    fetchYahooPrice("DX-Y.NYB"),
    fetchYahooPrice("^GSPC"),
    fetchYahooPrice("GC=F"),
    fetchYahooPrice("CL=F"),
  ]);

  return {
    fearGreedIndex: fearGreed.status === "fulfilled" ? (fearGreed.value?.value ?? null) : null,
    fearGreedLabel: fearGreed.status === "fulfilled" ? (fearGreed.value?.label ?? null) : null,
    btcDominance: cgGlobal.status === "fulfilled" ? (cgGlobal.value?.btcDominance ?? null) : null,
    dxy: dxy.status === "fulfilled" ? dxy.value : null,
    sp500: sp500.status === "fulfilled" ? sp500.value : null,
    gold: gold.status === "fulfilled" ? gold.value : null,
    oil: oil.status === "fulfilled" ? oil.value : null,
  };
}

// ─── Reddit Sentiment ────────────────────────────────────────────────────────

interface RedditChild {
  data?: { title?: string };
}

interface RedditListingResponse {
  data?: {
    children?: RedditChild[];
  };
}

async function fetchRedditTop(subreddit: string): Promise<string[]> {
  try {
    const url = `https://www.reddit.com/r/${subreddit}/hot.json?limit=10`;
    const res = await fetch(url, {
      signal: AbortSignal.timeout(6000),
      headers: { "User-Agent": "Mozilla/5.0 (compatible; NeuralTerminal/1.0)" },
    });
    if (!res.ok) return [];
    const data = (await res.json()) as RedditListingResponse;
    return (data?.data?.children ?? [])
      .map((c) => c?.data?.title ?? "")
      .filter((t) => t.length > 10)
      .slice(0, 5);
  } catch (_) {
    return [];
  }
}

async function fetchRedditSentiment(): Promise<string[]> {
  const [worldnews, economics, bitcoin, crypto] = await Promise.allSettled([
    fetchRedditTop("worldnews"),
    fetchRedditTop("economics"),
    fetchRedditTop("Bitcoin"),
    fetchRedditTop("CryptoCurrency"),
  ]);
  return [
    ...(worldnews.status === "fulfilled" ? worldnews.value : []),
    ...(economics.status === "fulfilled" ? economics.value : []),
    ...(bitcoin.status === "fulfilled" ? bitcoin.value : []),
    ...(crypto.status === "fulfilled" ? crypto.value : []),
  ].slice(0, 20);
}

// ─── OpenAI Analysis ────────────────────────────────────────────────────────

interface AiAnalysisResult {
  macroClimateScore: number;
  direction: string;
  confidence: number;
  narrative: string;
  prediction24h: string;
  heroVerdict: string;
  keyCatalysts: string[];
  categoryScores: Record<string, number>;
  events: Array<{
    title: string;
    relevanceScore: number;
    sentiment: string;
    cryptoImpact: string;
    category: string;
  }>;
}

const SYSTEM_PROMPT = `You are an institutional macro analyst specializing in crypto market impact assessment. Your role is to analyze global events, central bank policies, geopolitical developments, and market structure to predict crypto price movements.

Key historical patterns you know:
- Fed rate hikes → BTC typically −5% to −15% within 48 hours
- Fed pivots (pauses/cuts) → BTC +10% to +30% within 2 weeks
- ETF approvals/launches → strong bullish, immediate (BTC +5% to +20%)
- Banking crises (SVB-type) → short-term bearish, medium-term BTC hedge demand
- DXY above 105 → risk-off pressure on all crypto
- DXY below 100 → risk-on, positive for crypto
- Fear & Greed below 20 = extreme fear → accumulation zone, historically reversal
- Fear & Greed above 80 = extreme greed → distribution risk
- High funding rates + extreme greed → reversal risk within 7 days
- Regulatory crackdowns (SEC lawsuits, exchange shutdowns) → −10% to −30%
- Pro-crypto regulation (clear frameworks, ETF) → +15% to +40%
- Geopolitical conflict escalation → risk-off initially, then BTC as safe haven
- Inflation data above expectations → risk-off, negative crypto
- Unemployment spike → risk-off, then Fed cut speculation bullish
- China economic slowdown → negative for risk assets
- S&P500 drops → crypto follows with 1.2-1.5x leverage typically

You must respond with a valid JSON object only, no additional text.`;

function buildAnalysisPrompt(events: RssItem[], macro: MacroData, redditTopics: string[]): string {
  const eventList = events.slice(0, 30).map((e, i) =>
    `${i + 1}. [${e.category}] ${e.source}: "${e.title}"`
  ).join("\n");

  const macroStr = [
    macro.dxy ? `DXY: ${macro.dxy}` : null,
    macro.sp500 ? `S&P500: ${macro.sp500}` : null,
    macro.gold ? `Gold: ${macro.gold}/oz` : null,
    macro.oil ? `Oil: ${macro.oil}/bbl` : null,
    macro.btcDominance ? `BTC Dominance: ${macro.btcDominance}%` : null,
    macro.fearGreedIndex !== null ? `Fear & Greed: ${macro.fearGreedIndex} (${macro.fearGreedLabel})` : null,
  ].filter(Boolean).join(" | ");

  const redditStr = redditTopics.slice(0, 10).join("; ");

  return `Analyze the following world events and market data for crypto market impact.

CURRENT MACRO DATA: ${macroStr || "unavailable"}

TRENDING REDDIT TOPICS: ${redditStr || "none available"}

RECENT WORLD EVENTS (last 6 hours):
${eventList}

Respond ONLY with this exact JSON structure:
{
  "macroClimateScore": <number from -100 to +100, where +100 = extremely bullish for crypto, -100 = extremely bearish>,
  "direction": <"Bullish" | "Bearish" | "Neutral">,
  "confidence": <number 0-100>,
  "heroVerdict": "<one sentence verdict: e.g. 'Risk-off macro environment with DXY strength likely to pressure BTC toward $60k support in the next 24 hours.'>",
  "narrative": "<3-5 sentence plain English explanation of current macro situation and specific crypto implications>",
  "prediction24h": "<2-3 sentences: specific 24-hour directional prediction for BTC/crypto — what price action is likely, key levels to watch, and the primary catalyst that will drive it>",
  "keyCatalysts": ["<catalyst 1>", "<catalyst 2>", "<catalyst 3>"],
  "categoryScores": {
    "Monetary Policy": <-100 to 100>,
    "Regulatory": <-100 to 100>,
    "Geopolitical": <-100 to 100>,
    "Market Structure": <-100 to 100>,
    "Tech/Innovation": <-100 to 100>,
    "Social Sentiment": <-100 to 100>
  },
  "events": [
    {
      "title": "<event title>",
      "relevanceScore": <1-10>,
      "sentiment": <"bullish" | "bearish" | "neutral">,
      "cryptoImpact": "<one sentence explanation of specific crypto impact>",
      "category": <"Monetary Policy" | "Regulatory" | "Geopolitical" | "Markets" | "Crypto" | "Tech">
    }
  ]
}

Include up to 20 most relevant events in the events array. Score macroClimateScore based on the aggregate impact of all signals.`;
}

async function runOpenAiAnalysis(events: RssItem[], macro: MacroData, redditTopics: string[]): Promise<AiAnalysisResult> {
  if (!openai) {
    return buildFallbackAnalysis(events, macro);
  }

  const prompt = buildAnalysisPrompt(events, macro, redditTopics);
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
      response_format: { type: "json_object" },
      max_tokens: 2500,
      temperature: 0.3,
    });

    const content = response.choices[0]?.message?.content ?? "{}";
    const parsed = JSON.parse(content);

    return {
      macroClimateScore: clamp(Number(parsed.macroClimateScore ?? 0), -100, 100),
      direction: parsed.direction ?? "Neutral",
      confidence: clamp(Number(parsed.confidence ?? 50), 0, 100),
      heroVerdict: parsed.heroVerdict ?? "",
      narrative: parsed.narrative ?? "Analysis unavailable.",
      prediction24h: parsed.prediction24h ?? "",
      keyCatalysts: Array.isArray(parsed.keyCatalysts) ? parsed.keyCatalysts.slice(0, 5) : [],
      categoryScores: parsed.categoryScores ?? {},
      events: Array.isArray(parsed.events) ? parsed.events.slice(0, 25) : [],
    };
  } catch (err) {
    console.error("[WorldIntel] OpenAI error:", err);
    return buildFallbackAnalysis(events, macro);
  }
}

function buildFallbackAnalysis(events: RssItem[], macro: MacroData): AiAnalysisResult {
  const fg = macro.fearGreedIndex ?? 50;
  const dxy = macro.dxy ?? 104;
  let score = 0;
  if (fg < 25) score += 20;
  else if (fg > 75) score -= 15;
  if (dxy < 100) score += 15;
  else if (dxy > 106) score -= 20;

  const dirStr = score > 10 ? "Bullish" : score < -10 ? "Bearish" : "Neutral";
  return {
    macroClimateScore: clamp(score, -100, 100),
    direction: dirStr,
    confidence: 40,
    heroVerdict: `${dirStr} macro backdrop based on Fear & Greed ${fg} and DXY ${dxy.toFixed(1)} — AI offline, basic heuristics only.`,
    narrative: "AI analysis unavailable. Basic macro signals suggest " +
      (score > 0 ? "mildly positive" : score < 0 ? "mildly negative" : "neutral") +
      " conditions for crypto based on fear/greed and DXY readings.",
    prediction24h: "24-hour prediction unavailable — AI offline. Monitor Fear & Greed and DXY for directional signals.",
    keyCatalysts: events.slice(0, 3).map(e => e.title),
    categoryScores: { "Monetary Policy": 0, "Regulatory": 0, "Geopolitical": 0, "Market Structure": score, "Tech/Innovation": 0, "Social Sentiment": fg > 50 ? 20 : -20 },
    events: events.slice(0, 15).map(e => ({
      title: e.title,
      relevanceScore: 5,
      sentiment: "neutral",
      cryptoImpact: "Impact assessment unavailable — AI offline.",
      category: e.category,
    })),
  };
}

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

// ─── Risk Calendar ──────────────────────────────────────────────────────────

type RiskCalendarEntry = { event: string; date: string; category: string; btcImpact: string; direction: string };

function nextOccurrence(month: number, day: number): string {
  const now = new Date();
  let year = now.getFullYear();
  const candidate = new Date(year, month - 1, day);
  if (candidate < now) {
    const nextCandidate = new Date(year + 1, month - 1, day);
    return nextCandidate.toISOString().slice(0, 10);
  }
  return candidate.toISOString().slice(0, 10);
}

function nextWeekdayOfMonth(month: number, weekday: number, nth: number): string {
  const now = new Date();
  let year = now.getFullYear();
  for (let attempt = 0; attempt < 2; attempt++) {
    const d = new Date(year, month - 1, 1);
    let count = 0;
    while (d.getMonth() === month - 1) {
      if (d.getDay() === weekday) {
        count++;
        if (count === nth) {
          if (d > now) return d.toISOString().slice(0, 10);
          break;
        }
      }
      d.setDate(d.getDate() + 1);
    }
    year++;
    month = month > 12 ? 1 : month;
  }
  return new Date(now.getFullYear(), month - 1, 15).toISOString().slice(0, 10);
}

function buildRiskCalendar(): RiskCalendarEntry[] {
  const now = new Date();
  const m = now.getMonth() + 1;

  const fomcMonths = [1, 3, 5, 6, 7, 9, 11, 12];
  const nextFomcMonth = fomcMonths.find(fm => fm > m) ?? fomcMonths[0];
  const nextCpiMonth = m < 12 ? m + 1 : 1;
  const nextJobsMonth = m < 12 ? m + 1 : 1;

  return [
    {
      event: "FOMC Meeting",
      date: nextWeekdayOfMonth(nextFomcMonth, 3, 2),
      category: "Monetary Policy",
      btcImpact: "±8% avg move on rate decision",
      direction: "volatile",
    },
    {
      event: "US CPI Release",
      date: nextWeekdayOfMonth(nextCpiMonth, 3, 2),
      category: "Economic",
      btcImpact: "±5% on surprise vs expectations",
      direction: "volatile",
    },
    {
      event: "US Non-Farm Payrolls",
      date: nextWeekdayOfMonth(nextJobsMonth, 5, 1),
      category: "Economic",
      btcImpact: "±3% typically within 24h",
      direction: "volatile",
    },
    {
      event: "FOMC Minutes Release",
      date: nextWeekdayOfMonth(m < 12 ? m + 1 : 1, 3, 3),
      category: "Monetary Policy",
      btcImpact: "±3% on hawkish signals",
      direction: "volatile",
    },
    {
      event: "Bitcoin Halving Anniversary",
      date: nextOccurrence(4, 20),
      category: "Crypto",
      btcImpact: "+15-40% post-halving historically",
      direction: "bullish",
    },
  ];
}

export function getRiskCalendar(): RiskCalendarEntry[] {
  return buildRiskCalendar();
}

// ─── Main Intelligence Runner ─────────────────────────────────────────────

export let isRunning = false;
export let lastRunAt: number | null = null;
export let lastError: string | null = null;

export async function runWorldIntelCycle(): Promise<{ success: boolean; eventsStored: number; error?: string }> {
  if (isRunning) {
    return { success: false, eventsStored: 0, error: "Already running" };
  }
  isRunning = true;
  lastError = null;

  try {
    console.log("[WorldIntel] Starting intelligence cycle...");

    const [rssResults, macro, redditTopics] = await Promise.allSettled([
      Promise.all(RSS_FEEDS.map(fetchRssFeed)),
      fetchMacroData(),
      fetchRedditSentiment(),
    ]);

    const allRssItems: RssItem[] = rssResults.status === "fulfilled"
      ? rssResults.value.flat()
      : [];

    const macroData: MacroData = macro.status === "fulfilled"
      ? macro.value
      : { dxy: null, sp500: null, gold: null, oil: null, btcDominance: null, fearGreedIndex: null, fearGreedLabel: null };

    const reddit: string[] = redditTopics.status === "fulfilled"
      ? redditTopics.value
      : [];

    const deduped = deduplicateEvents(allRssItems);
    console.log(`[WorldIntel] Fetched ${allRssItems.length} events → ${deduped.length} unique`);

    const aiResult = await runOpenAiAnalysis(deduped, macroData, reddit);
    const now = Date.now();

    const [snapshotInsert, macroInsert] = await Promise.allSettled([
      db.insert(worldIntelSnapshots).values({
        macroClimateScore: aiResult.macroClimateScore,
        direction: aiResult.direction,
        confidence: aiResult.confidence,
        heroVerdict: aiResult.heroVerdict,
        narrative: aiResult.narrative,
        prediction24h: aiResult.prediction24h,
        keyCatalysts: aiResult.keyCatalysts,
        categoryScores: aiResult.categoryScores,
        createdAt: now,
      }),
      db.insert(macroIndicators).values({
        dxy: macroData.dxy ?? undefined,
        sp500: macroData.sp500 ?? undefined,
        gold: macroData.gold ?? undefined,
        oil: macroData.oil ?? undefined,
        btcDominance: macroData.btcDominance ?? undefined,
        fearGreedIndex: macroData.fearGreedIndex ?? undefined,
        fearGreedLabel: macroData.fearGreedLabel ?? undefined,
        recordedAt: now,
      }),
    ]);

    if (snapshotInsert.status === "rejected") {
      console.error("[WorldIntel] Failed to insert snapshot:", snapshotInsert.reason);
    }
    if (macroInsert.status === "rejected") {
      console.error("[WorldIntel] Failed to insert macro:", macroInsert.reason);
    }

    let eventsStored = 0;
    const eventsToStore = aiResult.events.slice(0, 30);

    for (const ev of eventsToStore) {
      const source = deduped.find(d => d.title === ev.title || ev.title.includes(d.title.slice(0, 30)));
      try {
        await db.insert(worldEvents).values({
          title: ev.title,
          source: source?.source ?? "AI Analysis",
          category: ev.category,
          url: source?.link ?? null,
          rawContent: source?.description ?? null,
          relevanceScore: ev.relevanceScore,
          sentiment: ev.sentiment,
          cryptoImpactExplanation: ev.cryptoImpact,
          publishedAt: source?.pubDate ? parseRssDate(source.pubDate) : null,
          fetchedAt: now,
        });
        eventsStored++;
      } catch (_) {}
    }

    lastRunAt = now;
    console.log(`[WorldIntel] Cycle complete. Score=${aiResult.macroClimateScore}, Events=${eventsStored}`);
    return { success: true, eventsStored };
  } catch (err: unknown) {
    lastError = err instanceof Error ? err.message : String(err);
    console.error("[WorldIntel] Cycle error:", err);
    return { success: false, eventsStored: 0, error: lastError ?? undefined };
  } finally {
    isRunning = false;
  }
}

function deduplicateEvents(items: RssItem[]): RssItem[] {
  const seen = new Set<string>();
  return items.filter(item => {
    const key = item.title.toLowerCase().replace(/\s+/g, " ").trim().slice(0, 60);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function parseRssDate(dateStr: string): number | null {
  try {
    const d = new Date(dateStr);
    return isNaN(d.getTime()) ? null : d.getTime();
  } catch (_) {
    return null;
  }
}

// ─── Scheduler ───────────────────────────────────────────────────────────────

let schedulerTimer: ReturnType<typeof setInterval> | null = null;
const REFRESH_INTERVAL_MS = 30 * 60 * 1000;

export function startWorldIntelScheduler() {
  if (schedulerTimer) return;
  runWorldIntelCycle().catch(console.error);
  schedulerTimer = setInterval(() => {
    runWorldIntelCycle().catch(console.error);
  }, REFRESH_INTERVAL_MS);
  console.log("[WorldIntel] Scheduler started (30 min interval)");
}

export function stopWorldIntelScheduler() {
  if (schedulerTimer) {
    clearInterval(schedulerTimer);
    schedulerTimer = null;
  }
}

// ─── Query Helpers ────────────────────────────────────────────────────────────

export async function getLatestSnapshot() {
  const rows = await db.select().from(worldIntelSnapshots).orderBy(desc(worldIntelSnapshots.createdAt)).limit(1);
  return rows[0] ?? null;
}

export async function getLatestMacro() {
  const rows = await db.select().from(macroIndicators).orderBy(desc(macroIndicators.recordedAt)).limit(1);
  return rows[0] ?? null;
}

export async function getRecentEvents(limit = 50, category?: string, sort: "recent" | "relevance" = "relevance") {
  const sixHoursAgo = Date.now() - 6 * 60 * 60 * 1000;
  const condition = category
    ? and(gt(worldEvents.fetchedAt, sixHoursAgo), eq(worldEvents.category, category))
    : gt(worldEvents.fetchedAt, sixHoursAgo);
  const orderCol = sort === "relevance" ? desc(worldEvents.relevanceScore) : desc(worldEvents.fetchedAt);
  return db.select().from(worldEvents)
    .where(condition)
    .orderBy(orderCol)
    .limit(limit);
}
