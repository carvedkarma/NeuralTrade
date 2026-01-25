export interface FearGreedData {
  value: number;
  valueClassification: string;
  timestamp: number;
}

export interface SentimentData {
  fearGreed: FearGreedData | null;
  socialSentiment: number;
  newsScore: number;
  topNews: { title: string; sentiment: string; source: string }[];
}

let cachedFearGreed: { data: FearGreedData | null; timestamp: number } = { data: null, timestamp: 0 };
const CACHE_TTL = 300000;

// CRITICAL FIX: Track real news API reads, not simulated data
// User requirement: NO fake or sample data - only real live data
let realNewsState = {
  articles: [] as { title: string; sentiment: string; source: string; timestamp: number }[],
  lastFetch: 0,
  totalReads: 0,
  apiAvailable: false,
};

// Cache TTL for news - 5 minutes
const NEWS_CACHE_TTL = 300000;

export function getNewsStats(): { totalReads: number; lastUpdate: number; apiAvailable: boolean } {
  return {
    totalReads: realNewsState.totalReads,
    lastUpdate: realNewsState.lastFetch,
    apiAvailable: realNewsState.apiAvailable,
  };
}

export async function getFearGreedIndex(): Promise<FearGreedData | null> {
  if (cachedFearGreed.data && Date.now() - cachedFearGreed.timestamp < CACHE_TTL) {
    return cachedFearGreed.data;
  }
  
  try {
    const response = await fetch("https://api.alternative.me/fng/?limit=1");
    const data = await response.json();
    
    if (data.data && data.data.length > 0) {
      const fng = data.data[0];
      const result: FearGreedData = {
        value: parseInt(fng.value),
        valueClassification: fng.value_classification,
        timestamp: parseInt(fng.timestamp) * 1000,
      };
      
      cachedFearGreed = { data: result, timestamp: Date.now() };
      console.log(`Fear & Greed Index: ${result.value} (${result.valueClassification})`);
      return result;
    }
    return null;
  } catch (error) {
    console.error("Error fetching Fear & Greed Index:", error);
    return cachedFearGreed.data;
  }
}

// CRITICAL FIX: Return only real news data - no simulation
// When news API is unavailable, return empty array (not fake data)
export async function getCryptoNews(): Promise<{ title: string; sentiment: string; source: string }[]> {
  const now = Date.now();
  
  // Return cached if still valid
  if (realNewsState.articles.length > 0 && now - realNewsState.lastFetch < NEWS_CACHE_TTL) {
    return realNewsState.articles.map(a => ({
      title: a.title,
      sentiment: a.sentiment,
      source: a.source,
    }));
  }
  
  // Try to fetch from CryptoPanic API (free tier)
  try {
    // Note: CryptoPanic requires API key for full access, but provides limited free access
    const response = await fetch("https://cryptopanic.com/api/v1/posts/?auth_token=public&currencies=BTC&kind=news&filter=hot", {
      headers: { 'Accept': 'application/json' },
      signal: AbortSignal.timeout(5000),
    });
    
    if (response.ok) {
      const data = await response.json();
      if (data.results && Array.isArray(data.results)) {
        realNewsState.articles = data.results.slice(0, 5).map((item: any) => ({
          title: item.title || "Untitled",
          sentiment: item.votes?.positive > item.votes?.negative ? "bullish" : 
                     item.votes?.negative > item.votes?.positive ? "bearish" : "neutral",
          source: item.domain || "CryptoPanic",
          timestamp: new Date(item.published_at).getTime(),
        }));
        realNewsState.lastFetch = now;
        realNewsState.totalReads += realNewsState.articles.length;
        realNewsState.apiAvailable = true;
        console.log(`[Sentiment] Fetched ${realNewsState.articles.length} real news articles`);
      }
    }
  } catch (error) {
    // API unavailable - this is fine, we just won't have news sentiment
    realNewsState.apiAvailable = false;
    console.log("[Sentiment] News API unavailable - using price-only signals");
  }
  
  // Return real cached data or empty array (never fake data)
  return realNewsState.articles.map(a => ({
    title: a.title,
    sentiment: a.sentiment,
    source: a.source,
  }));
}

export async function getSentimentData(): Promise<SentimentData> {
  const [fearGreed, news] = await Promise.all([
    getFearGreedIndex(),
    getCryptoNews(),
  ]);
  
  let socialSentiment = 0.5;
  if (fearGreed) {
    socialSentiment = fearGreed.value / 100;
  }
  
  let newsScore = 0;
  const sentimentMap: Record<string, number> = {
    bullish: 1,
    bearish: -1,
    neutral: 0,
    news: 0,
    media: 0,
  };
  
  if (news.length > 0) {
    newsScore = news.reduce((acc, n) => acc + (sentimentMap[n.sentiment] || 0), 0) / news.length;
  }
  
  return {
    fearGreed,
    socialSentiment,
    newsScore,
    topNews: news,
  };
}

export function interpretFearGreed(value: number): {
  signal: "bullish" | "bearish" | "neutral";
  strength: number;
  description: string;
} {
  if (value <= 25) {
    return {
      signal: "bullish",
      strength: (25 - value) / 25,
      description: "Extreme fear - contrarian buy signal",
    };
  } else if (value <= 45) {
    return {
      signal: "bullish",
      strength: 0.3,
      description: "Fear - potential accumulation zone",
    };
  } else if (value <= 55) {
    return {
      signal: "neutral",
      strength: 0.1,
      description: "Neutral sentiment",
    };
  } else if (value <= 75) {
    return {
      signal: "bearish",
      strength: 0.3,
      description: "Greed - caution advised",
    };
  } else {
    return {
      signal: "bearish",
      strength: (value - 75) / 25,
      description: "Extreme greed - contrarian sell signal",
    };
  }
}
