import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import paperRoutes from "./paper/routes";
import { backfillHistoricalData, getDataRangeInfo, getIntegrityReport, getActiveBackfillJob, incrementalUpdate, fillGaps, checkIncompleteBackfillJobs } from "./historical-data";
import { strategyLearner } from "./strategy-learner";

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

  return httpServer;
}
