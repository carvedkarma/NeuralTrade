import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import paperRoutes from "./paper/routes";
import { backfillHistoricalData, getDataRangeInfo, getIntegrityReport, getActiveBackfillJob, incrementalUpdate, fillGaps, checkIncompleteBackfillJobs } from "./historical-data";
import { strategyLearner } from "./strategy-learner";
import { gpuBridge } from "./gpu-bridge";
import { getUnifiedProgressReport, initializeUnifiedLearning, resetUnifiedLearning } from "./unified-learning-controller";

export const backfillState = {
  inProgress: false,
  progress: 0,
  message: "",
};

export async function hydrateBackfillStateFromDb(): Promise<void> {
  const activeJob = await getActiveBackfillJob();
  if (activeJob) {
    const isResumable = activeJob.status === "running" || activeJob.status === "pending" || 
      (activeJob.status === "error" && activeJob.currentCursor && activeJob.progressPct && activeJob.progressPct < 100);
    
    if (isResumable) {
      backfillState.inProgress = activeJob.status === "running";
      backfillState.progress = activeJob.progressPct ?? 0;
      backfillState.message = `Job ${activeJob.id}: ${activeJob.status} (${activeJob.progressPct ?? 0}%)`;
      console.log(`[Routes] Hydrated backfill state from DB: job ${activeJob.id}, status=${activeJob.status}, ${activeJob.progressPct}%`);
    }
  }
}

export async function initializeStrategyLearner(): Promise<void> {
  try {
    await strategyLearner.loadStateFromDb();
  } catch (err) {
    console.error("[Routes] Failed to initialize Strategy Learner:", err);
  }
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  app.use("/api/paper", paperRoutes);
  app.get("/api/dashboard", async (req, res) => {
    try {
      const data = await storage.getDashboardData();
      res.json(data);
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
      res.status(500).json({ error: "Failed to fetch dashboard data" });
    }
  });

  app.post("/api/refresh", async (req, res) => {
    try {
      storage.refreshData();
      const data = await storage.getDashboardData();
      res.json(data);
    } catch (error) {
      console.error("Error refreshing data:", error);
      res.status(500).json({ error: "Failed to refresh data" });
    }
  });

  app.post("/api/ai/analyze", async (req, res) => {
    try {
      await storage.requestAIAnalysis();
      const data = await storage.getDashboardData();
      res.json(data);
    } catch (error) {
      console.error("Error requesting AI analysis:", error);
      res.status(500).json({ error: "Failed to get AI analysis" });
    }
  });

  app.post("/api/strategy/start", async (req, res) => {
    try {
      storage.startStrategy();
      const data = await storage.getDashboardData();
      res.json(data);
    } catch (error) {
      console.error("Error starting strategy:", error);
      res.status(500).json({ error: "Failed to start strategy" });
    }
  });

  app.post("/api/strategy/stop", async (req, res) => {
    try {
      storage.stopStrategy();
      const data = await storage.getDashboardData();
      res.json(data);
    } catch (error) {
      console.error("Error stopping strategy:", error);
      res.status(500).json({ error: "Failed to stop strategy" });
    }
  });

  app.patch("/api/strategy/settings", async (req, res) => {
    try {
      storage.updateStrategySettings(req.body);
      const data = await storage.getDashboardData();
      res.json(data);
    } catch (error) {
      console.error("Error updating strategy settings:", error);
      res.status(500).json({ error: "Failed to update settings" });
    }
  });

  app.get("/api/persistence/status", async (req, res) => {
    try {
      const status = await storage.getPersistenceStatus();
      res.json(status);
    } catch (error) {
      console.error("Error getting persistence status:", error);
      res.status(500).json({ error: "Failed to get persistence status" });
    }
  });

  app.get("/api/strategy-learner", async (req, res) => {
    try {
      const candles = storage.getCandles();
      const dashboardData = await storage.getDashboardData();
      const currentSignal = dashboardData.currentSignal.signal;
      const currentConfidence = dashboardData.currentSignal.confidence;
      
      const data = strategyLearner.getData(candles, currentSignal, currentConfidence);
      res.json(data);
    } catch (error) {
      console.error("Error getting strategy learner data:", error);
      res.status(500).json({ error: "Failed to get strategy learner data" });
    }
  });

  // Reset all learning data and strategy learner
  app.post("/api/learning/reset", async (req, res) => {
    try {
      // Reset strategy learner
      strategyLearner.reset();
      
      // Reset storage learning state
      await storage.resetLearningState();
      
      console.log("[API] All learning data and strategy learner have been reset");
      res.json({ success: true, message: "Learning data reset successfully" });
    } catch (error) {
      console.error("Error resetting learning data:", error);
      res.status(500).json({ error: "Failed to reset learning data" });
    }
  });

  app.get("/api/historical/status", async (req, res) => {
    try {
      const rangeInfo = await getDataRangeInfo();
      
      res.json({
        ...rangeInfo,
        startDate: rangeInfo.startTs ? new Date(rangeInfo.startTs).toISOString().split('T')[0] : null,
        endDate: rangeInfo.endTs ? new Date(rangeInfo.endTs).toISOString().split('T')[0] : null,
      });
    } catch (error) {
      console.error("Error getting historical status:", error);
      res.status(500).json({ error: "Failed to get historical status" });
    }
  });

  app.get("/api/historical/integrity", async (req, res) => {
    try {
      const report = await getIntegrityReport();
      res.json(report);
    } catch (error) {
      console.error("Error getting integrity report:", error);
      res.status(500).json({ error: "Failed to get integrity report" });
    }
  });

  app.post("/api/historical/backfill", async (req, res) => {
    if (backfillState.inProgress) {
      return res.status(409).json({ 
        error: "Backfill already in progress", 
        progress: backfillState.progress,
        message: backfillState.message 
      });
    }

    const days = req.body.days || 370;
    backfillState.inProgress = true;
    backfillState.progress = 0;
    backfillState.message = "Starting backfill...";

    res.json({ status: "started", days });

    backfillHistoricalData("BTCUSDT", "15m", days, (progress, message) => {
      backfillState.progress = progress;
      backfillState.message = message;
    }).then(result => {
      console.log("[Historical] Backfill finished:", result);
      backfillState.inProgress = false;
      backfillState.progress = 100;
      backfillState.message = `Complete! ${result.totalCandles} candles stored.`;
      
      storage.reloadHistoricalCandles();
    }).catch(error => {
      console.error("[Historical] Backfill error:", error);
      backfillState.inProgress = false;
      backfillState.message = `Error: ${error.message}`;
    });
  });

  app.get("/api/historical/backfill/progress", async (req, res) => {
    const activeJob = await getActiveBackfillJob();
    
    const jobProgress = activeJob?.progressPct ?? 0;
    const isResumable = activeJob && 
      activeJob.currentCursor && 
      jobProgress < 100 &&
      (activeJob.status === "running" || activeJob.status === "pending" || activeJob.status === "error");
    
    const isRunning = backfillState.inProgress || activeJob?.status === "running";
    const needsResume = !isRunning && !!isResumable;
    const currentProgress = backfillState.inProgress ? backfillState.progress : jobProgress;
    const currentMessage = backfillState.inProgress ? backfillState.message : 
      (activeJob 
        ? `Job ${activeJob.id}: ${activeJob.status} (${jobProgress}%)${needsResume ? " - will auto-resume" : ""}` 
        : "No active job");
    
    res.json({
      inProgress: isRunning,
      needsResume,
      progress: currentProgress,
      message: currentMessage,
      job: activeJob ? {
        id: activeJob.id,
        status: activeJob.status,
        candlesFetched: activeJob.candlesFetched,
        candlesExpected: activeJob.candlesExpected,
        progressPct: activeJob.progressPct,
        currentCursor: activeJob.currentCursor,
        startTs: activeJob.startTs,
        resumable: !!isResumable,
      } : null,
    });
  });

  app.post("/api/historical/update", async (req, res) => {
    try {
      const result = await incrementalUpdate();
      await storage.reloadHistoricalCandles();
      res.json(result);
    } catch (error) {
      console.error("Error in incremental update:", error);
      res.status(500).json({ error: "Failed to update historical data" });
    }
  });

  app.post("/api/historical/fill-gaps", async (req, res) => {
    try {
      const filled = await fillGaps();
      if (filled > 0) {
        await storage.reloadHistoricalCandles();
      }
      res.json({ filled });
    } catch (error) {
      console.error("Error filling gaps:", error);
      res.status(500).json({ error: "Failed to fill gaps" });
    }
  });

  app.get("/api/historical/incomplete-jobs", async (req, res) => {
    try {
      const result = await checkIncompleteBackfillJobs();
      res.json(result);
    } catch (error) {
      console.error("Error checking incomplete jobs:", error);
      res.status(500).json({ error: "Failed to check incomplete jobs" });
    }
  });

  app.get("/api/unified-learning/progress", async (req, res) => {
    try {
      const progress = getUnifiedProgressReport();
      res.json(progress);
    } catch (error) {
      console.error("Error getting unified learning progress:", error);
      res.status(500).json({ error: "Failed to get unified learning progress" });
    }
  });

  app.post("/api/unified-learning/reset", async (req, res) => {
    try {
      resetUnifiedLearning();
      await storage.resetLearningState();
      await strategyLearner.reset();
      res.json({ success: true, message: "All learning systems reset for synchronized training" });
    } catch (error) {
      console.error("Error resetting unified learning:", error);
      res.status(500).json({ error: "Failed to reset learning" });
    }
  });

  app.get("/api/gpu/status", async (req, res) => {
    try {
      const metrics = await gpuBridge.getGPUMetrics();
      res.json({
        connected: metrics !== null,
        metrics: metrics || {
          gpuAvailable: false,
          gpuName: null,
          gpuMemoryUsed: null,
          gpuMemoryTotal: null,
          gpuMemoryPercent: 0,
          modelsLoaded: [],
          uptime: 0,
          isTraining: false,
          trainingProgress: 0,
          currentModel: null,
          trainingMetrics: {}
        }
      });
    } catch (error) {
      console.error("Error fetching GPU status:", error);
      res.json({ connected: false, metrics: null });
    }
  });

  app.post("/api/gpu/train", async (req, res) => {
    try {
      const { modelType, epochs = 100 } = req.body;
      if (!modelType) {
        return res.status(400).json({ error: "modelType is required" });
      }
      const started = await gpuBridge.startTraining(modelType, epochs);
      res.json({ success: started });
    } catch (error) {
      console.error("Error starting GPU training:", error);
      res.status(500).json({ error: "Failed to start training" });
    }
  });

  app.get("/api/gpu/health", async (req, res) => {
    try {
      const health = await gpuBridge.checkHealth();
      res.json({ available: health !== null, health });
    } catch (error) {
      res.json({ available: false, health: null });
    }
  });

  // GPU Status Push Endpoint - Receives status updates from local GPU trainer
  // This allows the dashboard to know when the GPU is connected and training
  app.post("/api/gpu/push-status", (req, res) => {
    try {
      const status = req.body;
      gpuBridge.updatePushedStatus({
        connected: true,
        lastPush: Date.now(),
        gpuAvailable: status.gpuAvailable ?? false,
        gpuName: status.gpuName ?? null,
        gpuMemoryUsed: status.gpuMemoryUsed ?? null,
        gpuMemoryTotal: status.gpuMemoryTotal ?? null,
        isTraining: status.isTraining ?? false,
        trainingProgress: status.trainingProgress ?? 0,
        currentModel: status.currentModel ?? null,
        currentEpoch: status.currentEpoch ?? 0,
        totalEpochs: status.totalEpochs ?? 0,
        trainLoss: status.trainLoss ?? null,
        valLoss: status.valLoss ?? null,
        modelsLoaded: status.modelsLoaded ?? [],
        modelsCompleted: status.modelsCompleted ?? []
      });
      console.log(`[GPU Push] Received status update - GPU: ${status.gpuName}, Training: ${status.isTraining}`);
      res.json({ success: true, received: Date.now() });
    } catch (error) {
      console.error("[GPU Push] Error:", error);
      res.status(500).json({ error: "Failed to process status update" });
    }
  });

  // Get pushed GPU status (for dashboard to poll)
  app.get("/api/gpu/pushed-status", (req, res) => {
    const status = gpuBridge.getPushedStatus();
    const isStale = status.lastPush ? Date.now() - status.lastPush > 30000 : true;
    res.json({
      ...status,
      connected: status.connected && !isStale,
      isStale
    });
  });

  // Data Proxy Endpoints - Allow local GPU trainer to fetch Binance data through Replit
  const BINANCE_VISION_URL = "https://data-api.binance.vision/api/v3";
  
  app.get("/api/data/klines", async (req, res) => {
    try {
      const { symbol = "BTCUSDT", interval = "15m", limit = "1000", startTime, endTime } = req.query;
      
      const params = new URLSearchParams({
        symbol: String(symbol),
        interval: String(interval),
        limit: String(Math.min(Number(limit), 1000))
      });
      
      if (startTime) params.append("startTime", String(startTime));
      if (endTime) params.append("endTime", String(endTime));
      
      const url = `${BINANCE_VISION_URL}/klines?${params.toString()}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        console.error(`[Data Proxy] Binance error: ${response.status}`);
        return res.status(response.status).json({ error: `Binance returned ${response.status}` });
      }
      
      const data = await response.json();
      
      // Transform to cleaner format
      const candles = data.map((k: any[]) => ({
        timestamp: k[0],
        open: parseFloat(k[1]),
        high: parseFloat(k[2]),
        low: parseFloat(k[3]),
        close: parseFloat(k[4]),
        volume: parseFloat(k[5]),
        closeTime: k[6],
        quoteVolume: parseFloat(k[7]),
        trades: k[8],
        takerBuyBase: parseFloat(k[9]),
        takerBuyQuote: parseFloat(k[10])
      }));
      
      console.log(`[Data Proxy] Fetched ${candles.length} candles for ${symbol} ${interval}`);
      res.json({ candles, count: candles.length, symbol, interval });
    } catch (error) {
      console.error("[Data Proxy] Error fetching klines:", error);
      res.status(500).json({ error: "Failed to fetch klines from Binance" });
    }
  });
  
  app.get("/api/data/orderbook", async (req, res) => {
    try {
      const { symbol = "BTCUSDT", limit = "100" } = req.query;
      
      const url = `${BINANCE_VISION_URL}/depth?symbol=${symbol}&limit=${Math.min(Number(limit), 1000)}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        return res.status(response.status).json({ error: `Binance returned ${response.status}` });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error("[Data Proxy] Error fetching orderbook:", error);
      res.status(500).json({ error: "Failed to fetch orderbook" });
    }
  });
  
  app.get("/api/data/ticker", async (req, res) => {
    try {
      const { symbol = "BTCUSDT" } = req.query;
      
      const url = `${BINANCE_VISION_URL}/ticker/24hr?symbol=${symbol}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        return res.status(response.status).json({ error: `Binance returned ${response.status}` });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error("[Data Proxy] Error fetching ticker:", error);
      res.status(500).json({ error: "Failed to fetch ticker" });
    }
  });

  // Cross-asset data endpoint with real correlation calculations
  app.get("/api/cross-asset", async (req, res) => {
    try {
      const symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"];
      
      // Fetch 24hr ticker data for all symbols
      const tickerPromises = symbols.map(async (symbol) => {
        try {
          const url = `${BINANCE_VISION_URL}/ticker/24hr?symbol=${symbol}`;
          const response = await fetch(url);
          if (response.ok) {
            const data = await response.json();
            return {
              symbol,
              price: parseFloat(data.lastPrice),
              change24h: parseFloat(data.priceChangePercent),
              volume24h: parseFloat(data.volume),
              lastUpdate: Date.now()
            };
          }
        } catch (e) {
          console.error(`[Cross-Asset] Error fetching ${symbol}:`, e);
        }
        return null;
      });
      
      const tickerResults = await Promise.all(tickerPromises);
      const validSymbols = tickerResults.filter(Boolean) as Array<{
        symbol: string;
        price: number;
        change24h: number;
        volume24h: number;
        lastUpdate: number;
      }>;
      
      if (validSymbols.length < 2) {
        return res.json({ 
          symbols: validSymbols,
          correlations: [],
          marketMomentum: { allUp: false, allDown: false, mixed: true, avgChange: 0, btcDominance: 50 },
          relativeStrength: [],
          priceHistory: []
        });
      }
      
      // Calculate market momentum
      const changes = validSymbols.map(s => s.change24h);
      const avgChange = changes.reduce((a, b) => a + b, 0) / changes.length;
      const allUp = changes.every(c => c > 0);
      const allDown = changes.every(c => c < 0);
      
      // Calculate relative strength vs BTC
      const btcData = validSymbols.find(s => s.symbol === "BTCUSDT");
      const btcChange = btcData?.change24h ?? 0;
      const relativeStrength = validSymbols
        .filter(s => s.symbol !== "BTCUSDT")
        .map(s => ({
          symbol: s.symbol,
          rsVsBtc: s.change24h - btcChange
        }));
      
      // Calculate real correlations from recent price data
      // Fetch 100 recent candles for each symbol to calculate correlations
      const klinePromises = symbols.map(async (symbol) => {
        try {
          const url = `${BINANCE_VISION_URL}/klines?symbol=${symbol}&interval=15m&limit=100`;
          const response = await fetch(url);
          if (response.ok) {
            const data = await response.json();
            return {
              symbol,
              prices: data.map((k: any[]) => parseFloat(k[4])), // Close prices
              timestamps: data.map((k: any[]) => k[0])
            };
          }
        } catch (e) {
          console.error(`[Cross-Asset] Error fetching klines for ${symbol}:`, e);
        }
        return null;
      });
      
      const klineResults = await Promise.all(klinePromises);
      const validKlines = klineResults.filter(Boolean) as Array<{
        symbol: string;
        prices: number[];
        timestamps: number[];
      }>;
      
      // Calculate returns for correlation
      const calculateReturns = (prices: number[]) => {
        const returns = [];
        for (let i = 1; i < prices.length; i++) {
          returns.push((prices[i] - prices[i-1]) / prices[i-1]);
        }
        return returns;
      };
      
      // Calculate Pearson correlation between two arrays
      const pearsonCorrelation = (x: number[], y: number[]): number => {
        const n = Math.min(x.length, y.length);
        if (n < 5) return 0;
        
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        for (let i = 0; i < n; i++) {
          sumX += x[i];
          sumY += y[i];
          sumXY += x[i] * y[i];
          sumX2 += x[i] * x[i];
          sumY2 += y[i] * y[i];
        }
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return denominator === 0 ? 0 : numerator / denominator;
      };
      
      // Calculate lead/lag by checking if lagged returns improve correlation
      const detectLeadLag = (btcReturns: number[], altReturns: number[], symbol: string): string => {
        const noLag = Math.abs(pearsonCorrelation(btcReturns, altReturns));
        const btcLeads = Math.abs(pearsonCorrelation(btcReturns.slice(0, -1), altReturns.slice(1)));
        const altLeads = Math.abs(pearsonCorrelation(btcReturns.slice(1), altReturns.slice(0, -1)));
        
        if (btcLeads > noLag && btcLeads > altLeads) return "BTC leads";
        if (altLeads > noLag && altLeads > btcLeads) return `${symbol.replace("USDT", "")} leads`;
        return "Synchronized";
      };
      
      const btcKlines = validKlines.find(k => k.symbol === "BTCUSDT");
      const btcReturns = btcKlines ? calculateReturns(btcKlines.prices) : [];
      
      const correlations = validKlines
        .filter(k => k.symbol !== "BTCUSDT")
        .map(k => {
          const altReturns = calculateReturns(k.prices);
          const corr20 = btcReturns.length >= 20 && altReturns.length >= 20
            ? pearsonCorrelation(btcReturns.slice(-20), altReturns.slice(-20))
            : 0;
          const corr60 = btcReturns.length >= 60 && altReturns.length >= 60
            ? pearsonCorrelation(btcReturns.slice(-60), altReturns.slice(-60))
            : pearsonCorrelation(btcReturns, altReturns);
          
          return {
            pair: `BTC/${k.symbol.replace("USDT", "")}`,
            correlation20: parseFloat(corr20.toFixed(4)),
            correlation60: parseFloat(corr60.toFixed(4)),
            leadLag: detectLeadLag(btcReturns, altReturns, k.symbol)
          };
        });
      
      // Calculate BTC dominance approximation
      const totalVolume = validSymbols.reduce((sum, s) => sum + s.volume24h * s.price, 0);
      const btcVolume = btcData ? btcData.volume24h * btcData.price : 0;
      const btcDominance = totalVolume > 0 ? (btcVolume / totalVolume) * 100 : 50;
      
      // Build price history for charts (normalized)
      const priceHistory = btcKlines?.timestamps.map((ts, idx) => {
        const point: any = { timestamp: ts };
        for (const k of validKlines) {
          const shortName = k.symbol.replace("USDT", "");
          point[shortName] = k.prices[idx] || 0;
        }
        return point;
      }) || [];
      
      res.json({
        symbols: validSymbols,
        correlations,
        marketMomentum: {
          allUp,
          allDown,
          mixed: !allUp && !allDown,
          avgChange,
          btcDominance
        },
        relativeStrength,
        priceHistory: priceHistory.slice(-50) // Last 50 data points
      });
    } catch (error) {
      console.error("[Cross-Asset] Error:", error);
      res.status(500).json({ error: "Failed to fetch cross-asset data" });
    }
  });

  return httpServer;
}
