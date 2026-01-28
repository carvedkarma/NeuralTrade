import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import paperRoutes from "./paper/routes";
import { db } from "./db";
import { candles } from "@shared/schema";
import { and, eq, gte, lte, asc } from "drizzle-orm";
import { backfillHistoricalData, getDataRangeInfo, getIntegrityReport, getActiveBackfillJob, incrementalUpdate, fillGaps, checkIncompleteBackfillJobs, getNNDataSummary, downloadNNData, getNNDownloadProgress, exportNNData, getNNTimeframes, clearNNData, cancelNNDownload, getResumableStatus, resumeNNDataDownload, getDownloadETA, streamNNDataBulk } from "./historical-data";
import zlib from "zlib";
import { strategyLearner } from "./strategy-learner";
import { gpuBridge } from "./gpu-bridge";
import { getUnifiedProgressReport, initializeUnifiedLearning, resetUnifiedLearning, loadCandleTimestamps } from "./unified-learning-controller";
import { getLatestFeatures } from "./feature-engine";
import { recalculatePatternLabels } from "./pattern-memory";
import { edgeTracker } from "./edge-tracker";
import { 
  getAvailableTimeframes, 
  getDataRange, 
  exportMultiTFCandles, 
  exportCrossAssetAligned,
  FEATURE_SPECS,
  getGPUTrainerConfig,
  generateWalkForwardFolds,
  computeRobustScalers,
  getEnhancedLabels,
  TRADING_COSTS,
  HORIZON_CONFIG,
  NO_TRADE_CONDITIONS,
  validateDataIntegrity
} from "./gpu-data-export";

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

  // Recalculate pattern labels using corrected return-based logic
  // This fixes the 71% false negative rate from the old regime-based direction bug
  app.post("/api/patterns/recalculate", async (req, res) => {
    try {
      console.log("[API] Pattern label recalculation requested");
      const result = await recalculatePatternLabels();
      res.json({ 
        success: true, 
        message: `Recalculated ${result.updated} patterns (${result.errors} errors)`,
        updated: result.updated,
        errors: result.errors
      });
    } catch (error) {
      console.error("Error recalculating patterns:", error);
      res.status(500).json({ error: "Failed to recalculate pattern labels" });
    }
  });

  // Manual start for Strategy Learner
  app.post("/api/strategy-learner/start", async (req, res) => {
    try {
      console.log("[API] Manual Strategy Learner training requested");
      const result = await strategyLearner.startManual();
      res.json(result);
    } catch (error) {
      console.error("Error starting strategy learner:", error);
      res.status(500).json({ success: false, message: "Failed to start strategy learner" });
    }
  });

  // Manual start for Deep Learning / Pattern Memory
  app.post("/api/deep-learning/start", async (req, res) => {
    try {
      console.log("[API] Manual Deep Learning training requested");
      const result = await storage.startDeepLearningManual();
      res.json(result);
    } catch (error) {
      console.error("Error starting deep learning:", error);
      res.status(500).json({ success: false, message: "Failed to start deep learning" });
    }
  });

  // Get training status for both systems
  app.get("/api/training/status", async (req, res) => {
    try {
      const storageStatus = storage.getTrainingStatus();
      const strategyStarted = strategyLearner.hasStartedTraining();
      
      res.json({
        deepLearning: {
          started: storageStatus.deepLearningStarted,
          progress: storageStatus.deepLearningProgress,
        },
        strategyLearner: {
          started: strategyStarted,
          epochs: storageStatus.strategyLearnerEpochs,
        },
      });
    } catch (error) {
      console.error("Error getting training status:", error);
      res.status(500).json({ error: "Failed to get training status" });
    }
  });

  // Edge Tracking Metrics - Track signal performance
  app.get("/api/edge-metrics", async (req, res) => {
    try {
      const metrics = edgeTracker.computeMetrics();
      res.json({
        success: true,
        metrics,
        report: edgeTracker.getEdgeReport()
      });
    } catch (error) {
      console.error("Error getting edge metrics:", error);
      res.status(500).json({ error: "Failed to get edge metrics" });
    }
  });

  app.post("/api/edge-metrics/clear", async (req, res) => {
    try {
      edgeTracker.clearResults();
      res.json({ success: true, message: "Edge tracking results cleared" });
    } catch (error) {
      console.error("Error clearing edge metrics:", error);
      res.status(500).json({ error: "Failed to clear edge metrics" });
    }
  });

  // Neural Network multi-timeframe data endpoints
  app.get("/api/nn-data/summary", async (req, res) => {
    try {
      const summary = await getNNDataSummary();
      res.json(summary);
    } catch (error) {
      console.error("Error getting NN data summary:", error);
      res.status(500).json({ error: "Failed to get NN data summary" });
    }
  });

  app.get("/api/nn-data/progress", async (req, res) => {
    try {
      const progress = getNNDownloadProgress();
      const eta = getDownloadETA();
      res.json({ progress, eta });
    } catch (error) {
      console.error("Error getting NN download progress:", error);
      res.status(500).json({ error: "Failed to get NN download progress" });
    }
  });

  app.post("/api/nn-data/download", async (req, res) => {
    const years = req.body.years || 3;
    
    res.json({ started: true, message: `Starting download for ${years} year(s) of multi-timeframe data (1m, 5m, 15m, 1h, 4h)` });
    
    downloadNNData(years, (symbol, timeframe, progress) => {
      console.log(`[NN Data] ${symbol} ${timeframe}: ${progress.toFixed(1)}%`);
    }).then(result => {
      console.log(`[NN Data] Download complete: ${result.totalCandles} candles`);
    }).catch(error => {
      console.error("[NN Data] Download error:", error);
    });
  });

  app.get("/api/nn-data/export", async (req, res) => {
    try {
      const data = await exportNNData();
      res.json(data);
    } catch (error) {
      console.error("Error exporting NN data:", error);
      res.status(500).json({ error: "Failed to export NN data" });
    }
  });

  // Bulk export endpoint for GPU trainer - streams gzipped NDJSON
  app.get("/api/nn-data/bulk-export", async (req, res) => {
    const timeframe = req.query.timeframe as string | undefined;
    const symbol = req.query.symbol as string | undefined;
    
    console.log(`[Bulk Export] Request received - timeframe: ${timeframe || 'all'}, symbol: ${symbol || 'all'}`);
    
    // Set headers for streaming gzipped response
    res.setHeader('Content-Type', 'application/x-ndjson');
    res.setHeader('Content-Encoding', 'gzip');
    res.setHeader('Transfer-Encoding', 'chunked');
    res.setHeader('Content-Disposition', 'attachment; filename="gpu-training-data.ndjson.gz"');
    
    const gzip = zlib.createGzip({ level: 6 });
    gzip.pipe(res);
    
    try {
      for await (const line of streamNNDataBulk(timeframe, symbol)) {
        gzip.write(line);
      }
      gzip.end();
    } catch (error) {
      console.error("[Bulk Export] Error:", error);
      gzip.destroy();
      if (!res.headersSent) {
        res.status(500).json({ error: "Failed to export bulk data" });
      }
    }
  });

  app.get("/api/nn-data/timeframes", async (req, res) => {
    res.json({ timeframes: getNNTimeframes() });
  });

  // Clear all NN data
  app.post("/api/nn-data/clear", async (req, res) => {
    try {
      console.log("[API] Clear NN data requested");
      const result = await clearNNData();
      res.json(result);
    } catch (error) {
      console.error("Error clearing NN data:", error);
      res.status(500).json({ success: false, message: "Failed to clear NN data" });
    }
  });

  // Cancel ongoing NN download
  app.post("/api/nn-data/cancel", async (req, res) => {
    try {
      console.log("[API] Cancel NN download requested");
      const result = cancelNNDownload();
      res.json(result);
    } catch (error) {
      console.error("Error cancelling NN download:", error);
      res.status(500).json({ success: false, message: "Failed to cancel download" });
    }
  });

  // Get resumable download status
  app.get("/api/nn-data/resumable", async (req, res) => {
    try {
      const status = await getResumableStatus();
      res.json(status);
    } catch (error) {
      console.error("Error checking resumable status:", error);
      res.status(500).json({ canResume: false, details: [] });
    }
  });

  // Resume NN data download from where it left off
  app.post("/api/nn-data/resume", async (req, res) => {
    try {
      const years = req.body.years || 3; // Default to 3 years for resume
      console.log(`[API] Resume NN download requested (${years} years)`);
      // Start the resume download in background
      resumeNNDataDownload(years).then(result => {
        console.log(`[API] Resume download completed: ${result.totalCandles} new candles`);
      }).catch(err => {
        console.error("[API] Resume download error:", err);
      });
      res.json({ success: true, message: `Resume download started (${years} year${years > 1 ? 's' : ''})` });
    } catch (error) {
      console.error("Error resuming NN download:", error);
      res.status(500).json({ success: false, message: "Failed to resume download" });
    }
  });

  // Manual start for Strategy Learning (continuous learning with pattern memory)
  app.post("/api/strategy-learning/start", async (req, res) => {
    try {
      console.log("[API] Manual Strategy Learning start requested");
      const result = await storage.startStrategyLearningManual();
      res.json(result);
    } catch (error) {
      console.error("Error starting strategy learning:", error);
      res.status(500).json({ success: false, message: "Failed to start strategy learning" });
    }
  });

  // Stop Strategy Learning
  app.post("/api/strategy-learning/stop", async (req, res) => {
    try {
      console.log("[API] Stop Strategy Learning requested");
      const result = storage.stopStrategyLearning();
      res.json(result);
    } catch (error) {
      console.error("Error stopping strategy learning:", error);
      res.status(500).json({ success: false, message: "Failed to stop strategy learning" });
    }
  });

  // Get manual learning status
  app.get("/api/learning/manual-status", async (req, res) => {
    try {
      const status = storage.getManualLearningStatus();
      res.json(status);
    } catch (error) {
      console.error("Error getting manual learning status:", error);
      res.status(500).json({ error: "Failed to get manual learning status" });
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

  // Multi-asset data management routes
  app.get("/api/data/summary", async (req, res) => {
    try {
      const { getMultiAssetDataSummary } = await import("./historical-data");
      const summary = await getMultiAssetDataSummary();
      res.json(summary);
    } catch (error) {
      console.error("Error getting data summary:", error);
      res.status(500).json({ error: "Failed to get data summary" });
    }
  });

  app.get("/api/data/download/status", async (req, res) => {
    try {
      const { getBulkDownloadStatus } = await import("./historical-data");
      const status = getBulkDownloadStatus();
      res.json(status);
    } catch (error) {
      console.error("Error getting download status:", error);
      res.status(500).json({ error: "Failed to get download status" });
    }
  });

  app.post("/api/data/download", async (req, res) => {
    try {
      const { years = 1, assets } = req.body;
      
      if (years < 1 || years > 15) {
        return res.status(400).json({ error: "Years must be between 1 and 15" });
      }
      
      const { downloadMultiAssetData, getSupportedAssets } = await import("./historical-data");
      const supportedAssets = getSupportedAssets();
      const assetsToDownload = assets || supportedAssets;
      
      // Start download in background
      res.json({ 
        message: `Started downloading ${years} years of data for ${assetsToDownload.length} assets`,
        assets: assetsToDownload,
        estimatedCandles: Math.floor(years * 365 * 24 * 4) * assetsToDownload.length,
      });
      
      // Run download async, reload data when complete
      downloadMultiAssetData(years, assetsToDownload).then(async (result) => {
        console.log("[Data Download] Complete:", result);
        // Reload historical candles into memory
        await storage.reloadHistoricalCandles();
        console.log("[Data Download] Reloaded candles into memory, all learning systems now use new data");
      }).catch(err => {
        console.error("[Data Download] Error:", err);
      });
      
    } catch (error) {
      console.error("Error starting download:", error);
      res.status(500).json({ error: "Failed to start download" });
    }
  });

  app.post("/api/data/clear", async (req, res) => {
    try {
      const { clearAllAssetData } = await import("./historical-data");
      const result = await clearAllAssetData();
      
      // Reset all learning systems
      resetUnifiedLearning();
      await storage.resetLearningState();
      await strategyLearner.reset();
      
      // Reload in-memory candles (will be empty after clear)
      await storage.reloadHistoricalCandles();
      
      res.json({ 
        success: result.success,
        candlesDeleted: result.candlesDeleted,
        message: `Cleared ${result.candlesDeleted} candles and reset all learning systems`
      });
    } catch (error) {
      console.error("Error clearing data:", error);
      res.status(500).json({ error: "Failed to clear data" });
    }
  });

  app.get("/api/data/assets", async (req, res) => {
    try {
      const { getSupportedAssets } = await import("./historical-data");
      res.json({ assets: getSupportedAssets() });
    } catch (error) {
      res.status(500).json({ error: "Failed to get supported assets" });
    }
  });

  // GPU data export - returns all stored candles for training
  app.get("/api/data/export/:symbol", async (req, res) => {
    try {
      const { symbol } = req.params;
      const { loadAssetCandlesFromDb } = await import("./historical-data");
      const candles = await loadAssetCandlesFromDb(symbol.toUpperCase());
      
      res.json({
        symbol: symbol.toUpperCase(),
        candles,
        count: candles.length,
        source: "database",
      });
    } catch (error) {
      console.error("Error exporting data:", error);
      res.status(500).json({ error: "Failed to export data" });
    }
  });

  // Export all assets at once for GPU trainer
  app.get("/api/data/export-all", async (req, res) => {
    try {
      const { loadAssetCandlesFromDb, getSupportedAssets, getMultiAssetDataSummary } = await import("./historical-data");
      const assets = getSupportedAssets();
      const summary = await getMultiAssetDataSummary();
      
      const exportData: Record<string, any> = {};
      for (const symbol of assets) {
        const candles = await loadAssetCandlesFromDb(symbol);
        exportData[symbol] = {
          candles,
          count: candles.length,
        };
      }
      
      res.json({
        assets: exportData,
        summary,
        source: "database",
        exportedAt: new Date().toISOString(),
      });
    } catch (error) {
      console.error("Error exporting all data:", error);
      res.status(500).json({ error: "Failed to export all data" });
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
        modelsCompleted: status.modelsCompleted ?? [],
        modelStatus: status.modelStatus ?? undefined
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

  // Get GPU trainer connection settings
  app.get("/api/gpu/settings", (req, res) => {
    res.json({
      url: gpuBridge.getUrl(),
      defaultUrl: "http://localhost:8000"
    });
  });

  // Update GPU trainer URL
  app.post("/api/gpu/settings", async (req, res) => {
    try {
      const { url } = req.body;
      
      if (!url || typeof url !== "string") {
        return res.status(400).json({ error: "URL is required" });
      }
      
      // Validate URL format
      try {
        new URL(url);
      } catch {
        return res.status(400).json({ error: "Invalid URL format" });
      }
      
      // Update the GPU bridge URL
      gpuBridge.setUrl(url);
      
      // Test connection to the new URL
      const health = await gpuBridge.checkHealth();
      
      res.json({
        success: true,
        url,
        connected: health !== null,
        health
      });
    } catch (error) {
      console.error("[GPU Settings] Update error:", error);
      res.status(500).json({ error: "Failed to update GPU trainer URL" });
    }
  });

  // Test GPU trainer connection
  app.post("/api/gpu/test-connection", async (req, res) => {
    try {
      const { url } = req.body;
      
      if (!url || typeof url !== "string") {
        return res.status(400).json({ error: "URL is required" });
      }
      
      // Test connection without changing the current URL
      try {
        const response = await fetch(`${url}/health`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
          signal: AbortSignal.timeout(5000)
        });
        
        if (response.ok) {
          const health = await response.json();
          res.json({
            connected: true,
            health,
            message: "Connection successful"
          });
        } else {
          res.json({
            connected: false,
            message: `Server responded with status ${response.status}`
          });
        }
      } catch (error) {
        res.json({
          connected: false,
          message: error instanceof Error ? error.message : "Connection failed"
        });
      }
    } catch (error) {
      console.error("[GPU Test] Connection test error:", error);
      res.status(500).json({ error: "Connection test failed" });
    }
  });

  // Get ensemble prediction status
  app.get("/api/gpu/ensemble/status", async (req, res) => {
    try {
      const status = await gpuBridge.getEnsembleStatus();
      res.json({ available: status !== null, status });
    } catch (error) {
      res.json({ available: false, status: null });
    }
  });

  // Get ensemble prediction from GPU neural networks
  app.post("/api/gpu/ensemble/predict", async (req, res) => {
    try {
      const { features } = req.body;
      
      if (!features || !Array.isArray(features)) {
        return res.status(400).json({ error: "features array required" });
      }
      
      const prediction = await gpuBridge.predictEnsemble(features);
      
      if (!prediction) {
        return res.json({ 
          available: false, 
          prediction: null,
          message: "GPU ensemble predictor not available"
        });
      }
      
      res.json({ available: true, prediction });
    } catch (error) {
      console.error("[GPU Ensemble] Prediction error:", error);
      res.status(500).json({ error: "Ensemble prediction failed" });
    }
  });

  // Get current ensemble prediction using latest market data
  app.get("/api/gpu/ensemble/current", async (req, res) => {
    try {
      // Check if GPU is available
      const health = await gpuBridge.checkHealth();
      if (!health) {
        return res.json({ 
          available: false, 
          prediction: null,
          message: "GPU trainer not connected"
        });
      }
      
      // Get current candles and compute features
      const candles = storage.getCandles();
      
      if (!candles || candles.length < 150) {
        return res.json({ 
          available: false, 
          prediction: null,
          message: "Not enough candle data available"
        });
      }
      
      // Get the last 150 candles for prediction
      const recentCandles = candles.slice(-150);
      
      // Compute features for each candle window
      const featureArrays: number[][] = [];
      for (let i = 100; i < recentCandles.length; i++) {
        const windowCandles = recentCandles.slice(i - 100, i + 1);
        const feature = getLatestFeatures(windowCandles);
        if (feature) {
          featureArrays.push(gpuBridge.featureVectorToArray(feature));
        }
      }
      
      if (featureArrays.length < 10) {
        return res.json({ 
          available: false, 
          prediction: null,
          message: "Could not compute enough features"
        });
      }
      
      const prediction = await gpuBridge.predictEnsemble(featureArrays);
      
      if (!prediction) {
        return res.json({ 
          available: false, 
          prediction: null,
          message: "Ensemble prediction failed"
        });
      }
      
      // Pass through GPU response directly - matches EnsemblePrediction interface
      res.json({ available: true, prediction });
    } catch (error) {
      console.error("[GPU Ensemble] Current prediction error:", error);
      res.json({ 
        available: false, 
        prediction: null, 
        message: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Neural Network Quantile Prediction endpoint - returns Entry/SL/TP derived from quantiles
  app.get("/api/gpu/nn-prediction", async (req, res) => {
    try {
      // Check if GPU is available
      const health = await gpuBridge.checkHealth();
      if (!health) {
        return res.json({ 
          available: false, 
          prediction: null,
          predictedCandles: [],
          error: "GPU trainer not connected"
        });
      }
      
      // Get current candles and compute features
      const candles = storage.getCandles();
      
      if (!candles || candles.length < 150) {
        return res.json({ 
          available: false, 
          prediction: null,
          predictedCandles: [],
          error: "Not enough candle data available"
        });
      }
      
      // Get the last 150 candles for prediction
      const recentCandles = candles.slice(-150);
      
      // Compute features for the most recent window
      const windowCandles = recentCandles.slice(-101);
      const feature = getLatestFeatures(windowCandles);
      
      if (!feature) {
        return res.json({ 
          available: false, 
          prediction: null,
          predictedCandles: [],
          error: "Could not compute features"
        });
      }
      
      const featureArray = gpuBridge.featureVectorToArray(feature);
      
      // Call GPU trainer for quantile prediction
      const nnResult = await gpuBridge.predictQuantile(featureArray);
      
      if (!nnResult) {
        return res.json({ 
          available: false, 
          prediction: null,
          predictedCandles: [],
          error: "Neural network prediction failed"
        });
      }
      
      const currentPrice = recentCandles[recentCandles.length - 1].close;
      const lastTimestamp = recentCandles[recentCandles.length - 1].timestamp;
      
      // Derive Entry/SL/TP from quantiles
      // Entry = current price
      // For LONG: SL = price * (1 + q10), TP = price * (1 + q90)
      // For SHORT: SL = price * (1 + q90), TP = price * (1 + q10)
      // For HOLD: No trade
      const probs = nnResult.direction_probs;
      const maxProb = Math.max(probs.LONG, probs.SHORT, probs.HOLD);
      
      // Determine direction - respect HOLD if it's the highest
      let direction: "LONG" | "SHORT" | "HOLD";
      if (probs.HOLD === maxProb && probs.HOLD > 0.4) {
        direction = "HOLD";
      } else {
        direction = probs.LONG > probs.SHORT ? "LONG" : "SHORT";
      }
      
      const isLong = direction === "LONG";
      const isHold = direction === "HOLD";
      
      const entry = currentPrice;
      
      // For HOLD, set neutral SL/TP based on uncertainty range
      let stopLoss: number;
      let takeProfit: number;
      
      if (isHold) {
        // For HOLD signals, use symmetric uncertainty bands
        stopLoss = currentPrice * (1 + nnResult.quantiles.q10);
        takeProfit = currentPrice * (1 + nnResult.quantiles.q90);
      } else if (isLong) {
        stopLoss = currentPrice * (1 + nnResult.quantiles.q10);  // q10 is negative for down move
        takeProfit = currentPrice * (1 + nnResult.quantiles.q90); // q90 is positive for up move
      } else {
        stopLoss = currentPrice * (1 + nnResult.quantiles.q90); // q90 is positive for up move
        takeProfit = currentPrice * (1 + nnResult.quantiles.q10); // q10 is negative for down move
      }
      
      // Risk/Reward ratio
      const risk = Math.abs(entry - stopLoss);
      const reward = Math.abs(takeProfit - entry);
      const riskReward = risk > 0 ? reward / risk : 0;
      
      // Confidence from direction probabilities
      const confidence = isHold ? probs.HOLD : Math.max(probs.LONG, probs.SHORT);
      
      // Generate predicted candles for visualization (10 bars horizon)
      const predictedCandles = [];
      const intervalMs = 15 * 60 * 1000; // 15 minutes
      
      for (let i = 1; i <= 10; i++) {
        const t = i / 10; // Progress through horizon
        
        // Interpolate quantiles for each future candle
        const q10 = currentPrice * (1 + nnResult.quantiles.q10 * t);
        const q25 = currentPrice * (1 + nnResult.quantiles.q25 * t);
        const q50 = currentPrice * (1 + nnResult.quantiles.q50 * t);
        const q75 = currentPrice * (1 + nnResult.quantiles.q75 * t);
        const q90 = currentPrice * (1 + nnResult.quantiles.q90 * t);
        
        predictedCandles.push({
          timestamp: lastTimestamp + (i * intervalMs),
          q10,
          q25,
          q50,
          q75,
          q90,
          direction: q50 >= currentPrice ? "up" : "down"
        });
      }
      
      const prediction = {
        action: direction as "LONG" | "SHORT",
        confidence,
        entry,
        stopLoss,
        takeProfit,
        riskReward,
        expectedMove: nnResult.quantiles.q50 * 100, // As percentage
        uncertainty: (nnResult.quantiles.q90 - nnResult.quantiles.q10) * 100, // Spread as percentage
        quantiles: {
          q10: nnResult.quantiles.q10 * 100,
          q25: nnResult.quantiles.q25 * 100,
          q50: nnResult.quantiles.q50 * 100,
          q75: nnResult.quantiles.q75 * 100,
          q90: nnResult.quantiles.q90 * 100,
        },
        directionProbs: nnResult.direction_probs,
        horizon: "2-3 hours (10 x 15m bars)",
        timestamp: Date.now(),
      };
      
      res.json({ available: true, prediction, predictedCandles });
    } catch (error) {
      console.error("[GPU NN] Prediction error:", error);
      res.json({ 
        available: false, 
        prediction: null,
        predictedCandles: [],
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
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

  // ============ GPU TRAINER DATA EXPORT API ============
  // These endpoints provide data for the local RTX 4070 GPU trainer
  
  app.get("/api/gpu-export/timeframes", async (req, res) => {
    try {
      const timeframes = await getAvailableTimeframes();
      res.json({ timeframes });
    } catch (error) {
      console.error("[GPU Export] Error fetching timeframes:", error);
      res.status(500).json({ error: "Failed to fetch timeframes" });
    }
  });

  app.get("/api/gpu-export/data-range", async (req, res) => {
    try {
      const symbol = (req.query.symbol as string) || "BTCUSDT";
      const timeframe = (req.query.timeframe as string) || "1m";
      const range = await getDataRange(symbol, timeframe);
      res.json(range);
    } catch (error) {
      console.error("[GPU Export] Error fetching data range:", error);
      res.status(500).json({ error: "Failed to fetch data range" });
    }
  });

  app.get("/api/gpu-export/multi-tf", async (req, res) => {
    try {
      const symbol = (req.query.symbol as string) || "BTCUSDT";
      const baseTF = (req.query.baseTF as string) || "1m";
      const startTs = parseInt(req.query.startTs as string) || Date.now() - 30 * 24 * 60 * 60 * 1000;
      const endTs = parseInt(req.query.endTs as string) || Date.now();
      const limit = parseInt(req.query.limit as string) || 100000;
      
      const data = await exportMultiTFCandles(symbol, baseTF, startTs, endTs, limit);
      res.json(data);
    } catch (error) {
      console.error("[GPU Export] Error exporting multi-TF candles:", error);
      res.status(500).json({ error: "Failed to export multi-TF candles" });
    }
  });

  app.get("/api/gpu-export/cross-asset", async (req, res) => {
    try {
      const baseTF = (req.query.baseTF as string) || "1m";
      const startTs = parseInt(req.query.startTs as string) || Date.now() - 30 * 24 * 60 * 60 * 1000;
      const endTs = parseInt(req.query.endTs as string) || Date.now();
      const limit = parseInt(req.query.limit as string) || 100000;
      
      const data = await exportCrossAssetAligned(baseTF, startTs, endTs, limit);
      res.json(data);
    } catch (error) {
      console.error("[GPU Export] Error exporting cross-asset data:", error);
      res.status(500).json({ error: "Failed to export cross-asset data" });
    }
  });

  app.get("/api/gpu-export/feature-specs", (req, res) => {
    const categorySet = new Set(FEATURE_SPECS.map(f => f.category));
    res.json({ 
      features: FEATURE_SPECS,
      totalFeatures: FEATURE_SPECS.length,
      categories: Array.from(categorySet)
    });
  });

  app.get("/api/gpu-export/trainer-config", (req, res) => {
    res.json(getGPUTrainerConfig());
  });

  app.get("/api/gpu-export/trading-costs", (req, res) => {
    res.json({
      costs: TRADING_COSTS,
      description: {
        makerFee: "Fee for limit orders (0.02%)",
        takerFee: "Fee for market orders (0.04%)",
        slippage: "Estimated slippage (0.01%)",
        spreadEstimate: "Bid-ask spread estimate (0.02%)",
        totalRoundTrip: "Total cost for open+close trade (0.09%)",
      },
      usage: "Subtract totalRoundTrip from raw returns to get tradable edge"
    });
  });

  // Institution-grade horizon configuration endpoint
  app.get("/api/gpu-export/horizon-config", (req, res) => {
    res.json({
      horizons: HORIZON_CONFIG,
      noTradeConditions: NO_TRADE_CONDITIONS,
      tradingCosts: TRADING_COSTS,
      usage: {
        horizons: "Use minEdge and minConfidence per horizon for trade filtering",
        noTrade: "Apply vetoes: dead zone, uncertainty spike, horizon disagreement, loss streak",
        decisionLogic: "15 & 60 bars are primary trading horizons, 240 is trend confirmation only"
      }
    });
  });

  // Data integrity validation endpoint - ensures no cross-contamination between symbol/timeframe combinations
  app.get("/api/gpu-export/validate-integrity", async (req, res) => {
    try {
      const symbol = (req.query.symbol as string) || "BTCUSDT";
      const timeframe = (req.query.timeframe as string) || "1m";
      const startTs = req.query.startTs ? parseInt(req.query.startTs as string) : undefined;
      const endTs = req.query.endTs ? parseInt(req.query.endTs as string) : undefined;
      
      console.log(`[GPU Export] Validating data integrity for ${symbol} ${timeframe}...`);
      const report = await validateDataIntegrity(symbol, timeframe, startTs, endTs);
      
      if (!report.valid) {
        console.error(`[GPU Export] DATA INTEGRITY FAILED for ${symbol} ${timeframe}:`, report.warnings);
      } else {
        console.log(`[GPU Export] Data integrity OK for ${symbol} ${timeframe}: ${report.totalRecords} records`);
      }
      
      res.json(report);
    } catch (error) {
      console.error("[GPU Export] Error validating data integrity:", error);
      res.status(500).json({ error: "Failed to validate data integrity" });
    }
  });

  app.get("/api/gpu-export/enhanced-labels", async (req, res) => {
    try {
      const symbol = (req.query.symbol as string) || "BTCUSDT";
      const timeframe = (req.query.timeframe as string) || "15m";
      const startTs = parseInt(req.query.startTs as string) || Date.now() - 30 * 24 * 60 * 60 * 1000;
      const endTs = parseInt(req.query.endTs as string) || Date.now();
      const limit = parseInt(req.query.limit as string) || 10000;
      
      const candleRows = await db
        .select()
        .from(candles)
        .where(
          and(
            eq(candles.symbol, symbol),
            eq(candles.timeframe, timeframe),
            gte(candles.timestamp, startTs),
            lte(candles.timestamp, endTs)
          )
        )
        .orderBy(asc(candles.timestamp))
        .limit(limit);
      
      if (candleRows.length < 300) {
        return res.status(400).json({ 
          error: "Insufficient data", 
          found: candleRows.length,
          required: 300
        });
      }
      
      const closes = candleRows.map(c => c.close);
      const highs = candleRows.map(c => c.high);
      const lows = candleRows.map(c => c.low);
      
      const horizons = [15, 60, 240];
      const labels: any[] = [];
      
      for (let i = 256; i < candleRows.length - 240; i++) {
        const log_ret = Math.log(closes[i] / closes[i-1]);
        const vol_20 = Math.sqrt(
          Array.from({length: 20}, (_, j) => {
            const r = Math.log(closes[i-j] / closes[i-j-1]);
            return r * r;
          }).reduce((a, b) => a + b, 0) / 20
        );
        
        const enhanced = getEnhancedLabels(closes, highs, lows, i, horizons, vol_20);
        
        labels.push({
          timestamp: candleRows[i].timestamp,
          rawReturns: enhanced.rawReturns,
          costAdjustedEdges: enhanced.costAdjustedEdges,
          directions: enhanced.directions,
          tradeWorthy: enhanced.tradeWorthy,
          sampleWeight: enhanced.sampleWeight,
        });
      }
      
      res.json({
        symbol,
        timeframe,
        horizons,
        tradingCosts: TRADING_COSTS,
        labels,
        count: labels.length,
        dateRange: {
          start: candleRows[0]?.timestamp,
          end: candleRows[candleRows.length - 1]?.timestamp
        }
      });
    } catch (error) {
      console.error("[GPU Export] Error computing enhanced labels:", error);
      res.status(500).json({ error: "Failed to compute enhanced labels" });
    }
  });

  app.get("/api/gpu-export/walk-forward-folds", async (req, res) => {
    try {
      const symbol = (req.query.symbol as string) || "BTCUSDT";
      const timeframe = (req.query.timeframe as string) || "1m";
      const trainMonths = parseInt(req.query.trainMonths as string) || 12;
      const valMonths = parseInt(req.query.valMonths as string) || 2;
      const testMonths = parseInt(req.query.testMonths as string) || 2;
      
      const range = await getDataRange(symbol, timeframe);
      const folds = generateWalkForwardFolds(
        range.startTs, 
        range.endTs, 
        trainMonths, 
        valMonths, 
        testMonths
      );
      
      res.json({ 
        dataRange: range,
        folds,
        totalFolds: folds.length
      });
    } catch (error) {
      console.error("[GPU Export] Error generating walk-forward folds:", error);
      res.status(500).json({ error: "Failed to generate walk-forward folds" });
    }
  });

  // Endpoint for GPU trainer to push predictions back
  app.post("/api/gpu-export/predictions", async (req, res) => {
    try {
      const { predictions, modelId, timestamp } = req.body;
      
      if (!predictions || !Array.isArray(predictions)) {
        return res.status(400).json({ error: "predictions array required" });
      }
      
      // Store predictions in memory for ensemble integration
      console.log(`[GPU Export] Received ${predictions.length} predictions from model ${modelId}`);
      
      // Import and use updateGPUPrediction to store in ml-predictor cache
      const { updateGPUPrediction } = await import("./ml-predictor");
      
      let stored = 0;
      for (const pred of predictions) {
        // Validate prediction has required fields
        if (pred.symbol && pred.returnH1 !== undefined && pred.directionalProb !== undefined) {
          updateGPUPrediction({
            symbol: pred.symbol,
            timestamp: pred.timestamp || timestamp || Date.now(),
            returnH1: pred.returnH1,
            returnH2: pred.returnH2 || 0,
            returnH3: pred.returnH3 || 0,
            quantile10: pred.quantile10 || pred.returnH1 * 0.5,
            quantile50: pred.quantile50 || pred.returnH1,
            quantile90: pred.quantile90 || pred.returnH1 * 1.5,
            directionalProb: pred.directionalProb,
            modelId: modelId || "gpu_transformer",
            confidence: pred.confidence || 0.5,
          });
          stored++;
        }
      }
      
      res.json({ 
        success: true, 
        received: predictions.length,
        stored,
        modelId,
        timestamp 
      });
    } catch (error) {
      console.error("[GPU Export] Error receiving predictions:", error);
      res.status(500).json({ error: "Failed to receive predictions" });
    }
  });

  return httpServer;
}
