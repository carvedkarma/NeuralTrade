import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import paperRoutes from "./paper/routes";
import ingestRouter from "./ingest";
import { db } from "./db";
import { candles, insertShotPlanHistorySchema, liveCycleLogs, liveTradeRecords, learningRuns, healthStatus, tradeEvents, settings, moneyConfigSchema } from "@shared/schema";
import type { ModelLearningStatsEntry, MoneyConfig } from "@shared/schema";
import { and, eq, gte, lte, asc, desc, sql, count } from "drizzle-orm";
import { z } from "zod";
import { backfillHistoricalData, getDataRangeInfo, getIntegrityReport, getActiveBackfillJob, incrementalUpdate, fillGaps, checkIncompleteBackfillJobs, getNNDataSummary, downloadNNData, getNNDownloadProgress, exportNNData, getNNTimeframes, clearNNData, cancelNNDownload, getResumableStatus, resumeNNDataDownload, getDownloadETA, streamNNDataBulk } from "./historical-data";
import zlib from "zlib";
import * as crypto from "crypto";
import { getMultiTimeframeKlines } from "./binance";
import { strategyLearner } from "./strategy-learner";
import { gpuBridge } from "./gpu-bridge";
import { getUnifiedProgressReport, initializeUnifiedLearning, resetUnifiedLearning, loadCandleTimestamps } from "./unified-learning-controller";
import { getLatestFeatures } from "./feature-engine";
import { recalculatePatternLabels } from "./pattern-memory";
import { edgeTracker } from "./edge-tracker";
import { 
  syncLatest15mCandles, 
  startLiveCandleSync, 
  stopLiveCandleSync, 
  getSyncStatus, 
  checkDataFreshness 
} from "./live-candle-sync";
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
  // Start live candle sync service automatically
  console.log("[Server] Starting live 15m candle sync service...");
  startLiveCandleSync();
  
  app.use("/api/paper", paperRoutes);
  app.use("/api", ingestRouter);

  app.get("/api/pro/cycles", async (req, res) => {
    try {
      const from = Number(req.query.from) || (Date.now() - 24 * 60 * 60 * 1000);
      const to = Number(req.query.to) || Date.now();
      const symbol = req.query.symbol as string | undefined;

      const conditions = [gte(liveCycleLogs.cycleTs, from), lte(liveCycleLogs.cycleTs, to)];
      if (symbol) conditions.push(eq(liveCycleLogs.symbol, symbol));

      const rows = await db
        .select()
        .from(liveCycleLogs)
        .where(and(...conditions))
        .orderBy(desc(liveCycleLogs.cycleTs))
        .limit(500);

      res.json(rows);
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get("/api/pro/trades", async (req, res) => {
    try {
      const from = Number(req.query.from) || 0;
      const to = Number(req.query.to) || Date.now();
      const symbol = req.query.symbol as string | undefined;
      const outcome = req.query.outcome as string | undefined;
      const format = req.query.format as string | undefined;

      const conditions = [gte(liveTradeRecords.entryTime, from), lte(liveTradeRecords.entryTime, to)];
      if (symbol) conditions.push(eq(liveTradeRecords.symbol, symbol));
      if (outcome) conditions.push(eq(liveTradeRecords.outcome, outcome));

      const rows = await db
        .select()
        .from(liveTradeRecords)
        .where(and(...conditions))
        .orderBy(desc(liveTradeRecords.entryTime))
        .limit(1000);

      if (format === "csv") {
        const headers = [
          "id", "symbol", "side", "entry_time", "entry_price", "exit_time", "exit_price",
          "stop_loss", "take_profit", "size_pct", "p_enter", "costs_bps",
          "outcome", "gross_r", "cost_r", "net_r", "sized_r", "status",
          "pnl_usd", "pnl_usd_gross", "pnl_usd_cost", "risk_usd_used",
          "bars_held", "leverage", "model_version",
        ];
        const csvRows = rows.map((r) =>
          [
            r.id, r.symbol, r.side, r.entryTime, r.entryPrice, r.exitTime ?? "",
            r.exitPrice ?? "", r.stopLoss ?? "", r.takeProfit ?? "", r.sizePct ?? "",
            r.pEnter ?? "", r.costsBps ?? "", r.outcome ?? "", r.grossR ?? "",
            r.costR ?? "", r.netR ?? "", r.sizedR ?? "", r.status,
            r.pnlUsd ?? "", r.pnlUsdGross ?? "", r.pnlUsdCost ?? "", r.riskUsdUsed ?? "",
            r.barsHeld ?? "", r.leverage ?? "", r.modelVersion ?? "",
          ].join(",")
        );
        const csv = [headers.join(","), ...csvRows].join("\n");
        res.setHeader("Content-Type", "text/csv");
        res.setHeader("Content-Disposition", "attachment; filename=trades.csv");
        return res.send(csv);
      }

      res.json(rows);
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get("/api/pro/trades/:id/events", async (req, res) => {
    try {
      const tradeId = Number(req.params.id);
      const rows = await db
        .select()
        .from(tradeEvents)
        .where(eq(tradeEvents.tradeId, tradeId))
        .orderBy(asc(tradeEvents.ts));
      res.json(rows);
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get("/api/pro/learning-runs", async (req, res) => {
    try {
      const from = Number(req.query.from) || 0;
      const to = Number(req.query.to) || Date.now();
      const symbol = req.query.symbol as string | undefined;

      const conditions = [gte(learningRuns.startAt, from), lte(learningRuns.startAt, to)];
      if (symbol) conditions.push(eq(learningRuns.symbol, symbol));

      const rows = await db
        .select()
        .from(learningRuns)
        .where(and(...conditions))
        .orderBy(desc(learningRuns.startAt))
        .limit(100);

      res.json(rows);
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get("/api/pro/health", async (req, res) => {
    try {
      const rows = await db
        .select()
        .from(healthStatus)
        .orderBy(desc(healthStatus.ts))
        .limit(50);
      res.json(rows);
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get("/api/pro/summary", async (req, res) => {
    try {
      const window = (req.query.window as string) || "24h";
      const windowMs: Record<string, number> = {
        "24h": 24 * 60 * 60 * 1000,
        "7d": 7 * 24 * 60 * 60 * 1000,
        "30d": 30 * 24 * 60 * 60 * 1000,
      };
      const since = Date.now() - (windowMs[window] || windowMs["24h"]);

      const trades = await db
        .select()
        .from(liveTradeRecords)
        .where(gte(liveTradeRecords.entryTime, since))
        .orderBy(asc(liveTradeRecords.entryTime));

      const openTrades = trades.filter((t) => t.status === "open");
      const closedTrades = trades.filter((t) => t.status === "closed");
      const wins = closedTrades.filter((t) => (t.netR ?? 0) > 0);
      const netRValues = closedTrades.map((t) => t.netR ?? 0);
      const totalNetR = netRValues.reduce((s, v) => s + v, 0);
      const avgR = closedTrades.length > 0 ? totalNetR / closedTrades.length : 0;
      const bestTrade = netRValues.length > 0 ? Math.max(...netRValues) : 0;
      const worstTrade = netRValues.length > 0 ? Math.min(...netRValues) : 0;

      let maxDrawdown = 0;
      let peak = 0;
      let cumR = 0;
      for (const r of netRValues) {
        cumR += r;
        if (cumR > peak) peak = cumR;
        const dd = peak - cumR;
        if (dd > maxDrawdown) maxDrawdown = dd;
      }

      const cycles = await db
        .select()
        .from(liveCycleLogs)
        .where(gte(liveCycleLogs.cycleTs, since));

      const holdCycles = cycles.filter((c) => c.decision !== "ENTER");
      const holdReasons: Record<string, number> = {};
      for (const c of holdCycles) {
        const reasons = (c.reasons as string[]) ?? [c.decision];
        for (const r of reasons) {
          holdReasons[r] = (holdReasons[r] ?? 0) + 1;
        }
      }

      const symbolStats: Record<string, { trades: number; wins: number; netR: number; pnlUsd: number }> = {};
      for (const t of closedTrades) {
        if (!symbolStats[t.symbol]) symbolStats[t.symbol] = { trades: 0, wins: 0, netR: 0, pnlUsd: 0 };
        symbolStats[t.symbol].trades++;
        if ((t.netR ?? 0) > 0) symbolStats[t.symbol].wins++;
        symbolStats[t.symbol].netR += t.netR ?? 0;
        symbolStats[t.symbol].pnlUsd += t.pnlUsd ?? 0;
      }

      const moneyRow = await db.select().from(settings).where(eq(settings.key, "money_config")).limit(1);
      const moneyConfig = moneyRow.length > 0 ? moneyRow[0].valueJson as any : { account_equity_usd: 1500, risk_per_trade_pct: 1.0 };
      const riskUsd = moneyConfig.account_equity_usd * (moneyConfig.risk_per_trade_pct / 100);
      const totalPnlUsd = closedTrades.reduce((s, t) => s + (t.pnlUsd ?? (t.netR ?? 0) * riskUsd), 0);
      const avgPnlUsd = closedTrades.length > 0 ? totalPnlUsd / closedTrades.length : 0;

      let maxDrawdownUsd = 0;
      let peakUsd = 0;
      let cumUsd = 0;
      for (const t of closedTrades) {
        cumUsd += t.pnlUsd ?? (t.netR ?? 0) * riskUsd;
        if (cumUsd > peakUsd) peakUsd = cumUsd;
        const ddUsd = peakUsd - cumUsd;
        if (ddUsd > maxDrawdownUsd) maxDrawdownUsd = ddUsd;
      }

      const todayStart = new Date();
      todayStart.setHours(0, 0, 0, 0);
      const todayTs = todayStart.getTime();
      const todayTrades = closedTrades.filter((t) => (t.exitTime ?? t.entryTime) >= todayTs);
      const todayNetR = todayTrades.reduce((s, t) => s + (t.netR ?? 0), 0);
      const todayPnlUsd = todayTrades.reduce((s, t) => s + (t.pnlUsd ?? (t.netR ?? 0) * riskUsd), 0);

      res.json({
        window,
        since,
        openPositions: openTrades.length,
        closedTrades: closedTrades.length,
        winRate: closedTrades.length > 0 ? wins.length / closedTrades.length : 0,
        totalNetR,
        avgR,
        bestTrade,
        worstTrade,
        maxDrawdown,
        totalPnlUsd,
        avgPnlUsd,
        maxDrawdownUsd,
        currentEquityUsd: moneyConfig.account_equity_usd + totalPnlUsd,
        riskUsd,
        todayTrades: todayTrades.length,
        todayNetR,
        todayPnlUsd,
        totalCycles: cycles.length,
        holdReasons,
        symbolStats,
        equityCurve: closedTrades.map((t) => ({
          ts: t.exitTime ?? t.entryTime,
          netR: t.netR ?? 0,
          pnlUsd: t.pnlUsd ?? (t.netR ?? 0) * riskUsd,
          symbol: t.symbol,
        })),
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // === Money Config API ===
  app.get("/api/config/money", async (req, res) => {
    try {
      const row = await db.select().from(settings).where(eq(settings.key, "money_config")).limit(1);
      if (row.length === 0) {
        return res.json({ account_equity_usd: 1500, risk_per_trade_pct: 1.0, base_currency: "USD" });
      }
      res.json(row[0].valueJson);
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.post("/api/config/money", async (req, res) => {
    try {
      const parsed = moneyConfigSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ error: "Invalid money config", details: parsed.error.flatten() });
      }
      const config = parsed.data;
      const riskUsd = config.account_equity_usd * (config.risk_per_trade_pct / 100);

      const existing = await db.select().from(settings).where(eq(settings.key, "money_config")).limit(1);
      if (existing.length > 0) {
        await db.update(settings).set({ valueJson: config, updatedAt: Date.now() }).where(eq(settings.key, "money_config"));
      } else {
        await db.insert(settings).values({ key: "money_config", valueJson: config, updatedAt: Date.now() });
      }

      console.log(`[Money Config] Updated: equity=$${config.account_equity_usd}, risk_pct=${config.risk_per_trade_pct}%, risk_usd=$${riskUsd.toFixed(2)}`);
      res.json({ ...config, risk_usd: riskUsd });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // === Trade Detail + Replay ===
  app.get("/api/pro/trades/:id", async (req, res) => {
    try {
      const tradeId = Number(req.params.id);
      const [trade] = await db.select().from(liveTradeRecords).where(eq(liveTradeRecords.id, tradeId)).limit(1);
      if (!trade) return res.status(404).json({ error: "Trade not found" });

      const events = await db.select().from(tradeEvents).where(eq(tradeEvents.tradeId, tradeId)).orderBy(asc(tradeEvents.ts));

      console.log(`[Trade Detail] Loaded trade detail trade_id=${tradeId}`);
      res.json({ trade, events });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get("/api/pro/trades/:id/replay", async (req, res) => {
    try {
      const tradeId = Number(req.params.id);
      const preBars = Math.min(Math.max(Number(req.query.preBars) || 50, 10), 200);
      const postBars = Math.min(Math.max(Number(req.query.postBars) || 10, 5), 50);

      const [trade] = await db.select().from(liveTradeRecords).where(eq(liveTradeRecords.id, tradeId)).limit(1);
      if (!trade) return res.status(404).json({ error: "Trade not found" });

      const barMs = 15 * 60 * 1000;
      const startTs = trade.entryTime - preBars * barMs;
      const endTs = (trade.exitTime ?? Date.now()) + postBars * barMs;

      const symbol = trade.symbol || "BTCUSDT";
      const replayCandles = await db
        .select()
        .from(candles)
        .where(
          and(
            eq(candles.symbol, symbol),
            eq(candles.timeframe, "15m"),
            gte(candles.timestamp, startTs),
            lte(candles.timestamp, endTs)
          )
        )
        .orderBy(asc(candles.timestamp))
        .limit(300);

      const isOpen = !trade.exitTime;

      const markers: Array<{ t: number; price: number; type: string; side?: string; outcome?: string }> = [];
      markers.push({
        t: trade.entryTime,
        price: trade.entryPrice,
        type: "ENTRY",
        side: trade.side || "LONG",
      });
      if (trade.exitTime && trade.exitPrice) {
        markers.push({
          t: trade.exitTime,
          price: trade.exitPrice,
          type: "EXIT",
          outcome: trade.outcome || "EXPIRE",
        });
      }

      const lines: Array<{ name: string; price: number }> = [];
      lines.push({ name: "ENTRY", price: trade.entryPrice });
      if (trade.stopLoss) lines.push({ name: "SL", price: trade.stopLoss });
      if (trade.takeProfit) lines.push({ name: "TP", price: trade.takeProfit });

      const lastCandleTs = replayCandles.length > 0
        ? replayCandles[replayCandles.length - 1].timestamp
        : trade.entryTime;

      const bands = [{
        from: trade.entryTime,
        to: isOpen ? lastCandleTs : (trade.exitTime ?? trade.entryTime),
        label: isOpen ? "OPEN" : "Trade Window",
      }];

      console.log(`[Replay] Loaded ${replayCandles.length} candles for trade_id=${tradeId} symbol=${symbol} range=${new Date(startTs).toISOString()}..${new Date(endTs).toISOString()}`);

      res.json({
        trade: {
          id: trade.id,
          symbol: trade.symbol,
          side: trade.side,
          entry_at: trade.entryTime,
          exit_at: trade.exitTime,
          entry_price: trade.entryPrice,
          exit_price: trade.exitPrice,
          sl_price: trade.stopLoss,
          tp_price: trade.takeProfit,
          outcome: trade.outcome,
          net_r: trade.netR,
          pnl_usd: trade.pnlUsd,
          bars_held: trade.barsHeld,
          status: trade.status,
        },
        timeframe: "15m",
        candles: replayCandles.map((c) => ({
          t: c.timestamp,
          o: c.open,
          h: c.high,
          l: c.low,
          c: c.close,
          v: c.volume,
        })),
        markers,
        lines,
        bands,
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // === Trade Notes ===
  app.patch("/api/pro/trades/:id/notes", async (req, res) => {
    try {
      const tradeId = Number(req.params.id);
      const { notes } = req.body;
      await db.update(liveTradeRecords).set({ notes: notes ?? null }).where(eq(liveTradeRecords.id, tradeId));
      res.json({ ok: true });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

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

  // Shot Plan History endpoints
  app.get("/api/shot-plan/history", async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 50;
      
      const history = await storage.getShotPlanHistory(limit);
      
      // Calculate stats
      const completedTrades = history.filter(h => h.outcome && h.outcome !== "PENDING");
      const tp1Hits = completedTrades.filter(h => h.outcome === "HIT_TP1").length;
      const tp2Hits = completedTrades.filter(h => h.outcome === "HIT_TP2").length;
      const slHits = completedTrades.filter(h => h.outcome === "HIT_SL").length;
      const expired = completedTrades.filter(h => h.outcome === "EXPIRED").length;
      const wins = tp1Hits + tp2Hits;
      const totalCompleted = completedTrades.length;
      
      const pnls = completedTrades.map(h => h.pnlPercent || 0);
      const avgPnl = pnls.length > 0 ? pnls.reduce((a, b) => a + b, 0) / pnls.length : 0;
      const bestTrade = pnls.length > 0 ? Math.max(...pnls) : 0;
      const worstTrade = pnls.length > 0 ? Math.min(...pnls) : 0;
      
      res.json({
        history,
        stats: {
          totalTrades: totalCompleted,
          winRate: totalCompleted > 0 ? (wins / totalCompleted) * 100 : 0,
          avgPnl,
          bestTrade,
          worstTrade,
          tp1Hits,
          tp2Hits,
          slHits,
          expired
        }
      });
    } catch (error) {
      console.error("Error getting shot plan history:", error);
      res.status(500).json({ error: "Failed to get shot plan history" });
    }
  });

  const recordShotPlanSchema = insertShotPlanHistorySchema.extend({
    outcome: z.string().optional().default("PENDING"),
  });

  app.post("/api/shot-plan/record", async (req, res) => {
    try {
      const validated = recordShotPlanSchema.parse({
        ...req.body,
        timestamp: Date.now(),
      });
      
      const entry = await storage.recordShotPlan(validated);
      res.json({ success: true, entry });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Invalid request body", details: error.errors });
        return;
      }
      console.error("Error recording shot plan:", error);
      res.status(500).json({ error: "Failed to record shot plan" });
    }
  });

  const updateOutcomeSchema = z.object({
    id: z.number(),
    outcome: z.string(),
    exitPrice: z.number().optional(),
    pnlPercent: z.number().optional(),
    candlesHeld: z.number().optional(),
    mfe: z.number().optional(),
    mae: z.number().optional(),
  });

  app.post("/api/shot-plan/update-outcome", async (req, res) => {
    try {
      const { id, outcome, exitPrice, pnlPercent, candlesHeld, mfe, mae } = updateOutcomeSchema.parse(req.body);
      
      await storage.updateShotPlanOutcome(id, {
        outcome,
        exitPrice,
        pnlPercent,
        exitTimestamp: Date.now(),
        candlesHeld,
        maxFavorableExcursion: mfe,
        maxAdverseExcursion: mae,
      });
      
      res.json({ success: true });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Invalid request body", details: error.errors });
        return;
      }
      console.error("Error updating shot plan outcome:", error);
      res.status(500).json({ error: "Failed to update outcome" });
    }
  });

  // Cone-based signal endpoints
  app.get("/api/cone-signals", async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 50;
      const signals = await storage.getConeSignals(limit);
      
      // Calculate stats
      const completedSignals = signals.filter(s => s.outcome && s.outcome !== "PENDING");
      const tpHits = completedSignals.filter(s => s.outcome === "HIT_TP").length;
      const slHits = completedSignals.filter(s => s.outcome === "HIT_SL").length;
      const expired = completedSignals.filter(s => s.outcome === "EXPIRED").length;
      const totalCompleted = completedSignals.length;
      
      const pnls = completedSignals.map(s => s.pnlPercent || 0);
      const avgPnl = pnls.length > 0 ? pnls.reduce((a, b) => a + b, 0) / pnls.length : 0;
      const bestTrade = pnls.length > 0 ? Math.max(...pnls) : 0;
      const worstTrade = pnls.length > 0 ? Math.min(...pnls) : 0;
      
      res.json({
        signals,
        stats: {
          totalSignals: signals.length,
          totalCompleted,
          winRate: totalCompleted > 0 ? (tpHits / totalCompleted) * 100 : 0,
          avgPnl,
          bestTrade,
          worstTrade,
          tpHits,
          slHits,
          expired,
          pending: signals.filter(s => s.outcome === "PENDING").length,
        },
      });
    } catch (error) {
      console.error("Error getting cone signals:", error);
      res.status(500).json({ error: "Failed to get cone signals" });
    }
  });

  const recordConeSignalSchema = z.object({
    timestamp: z.number(),
    direction: z.enum(["LONG", "SHORT", "HOLD"]),
    entryPrice: z.number(),
    stopLoss: z.number().nullable().optional(),
    takeProfit: z.number().nullable().optional(),
    mu: z.number(),
    sigma: z.number().optional(),
    edge: z.number(),
    riskReward: z.number().nullable().optional(),
    q10: z.number(),
    q25: z.number(),
    q50: z.number(),
    q75: z.number(),
    q90: z.number(),
    probUp: z.number().optional(),
    probDown: z.number().optional(),
    probHold: z.number().optional(),
    holdReasons: z.array(z.string()).optional(),
    edgeThreshold: z.number().optional(),
    edgePercentile: z.number().optional(),
    outcome: z.string().default("PENDING"),
  });

  app.post("/api/cone-signals/record", async (req, res) => {
    try {
      const validated = recordConeSignalSchema.parse(req.body);
      
      const entry = await storage.recordConeSignal({
        ...validated,
        createdAt: Date.now(),
      });
      res.json({ success: true, entry });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Invalid request body", details: error.errors });
        return;
      }
      console.error("Error recording cone signal:", error);
      res.status(500).json({ error: "Failed to record cone signal" });
    }
  });

  const updateConeOutcomeSchema = z.object({
    id: z.number(),
    outcome: z.string(),
    exitPrice: z.number().optional(),
    pnlPercent: z.number().optional(),
    candlesHeld: z.number().optional(),
    mfe: z.number().optional(),
    mae: z.number().optional(),
  });

  app.post("/api/cone-signals/update-outcome", async (req, res) => {
    try {
      const { id, outcome, exitPrice, pnlPercent, candlesHeld, mfe, mae } = updateConeOutcomeSchema.parse(req.body);
      
      await storage.updateConeSignalOutcome(id, {
        outcome,
        exitPrice,
        exitTimestamp: BigInt(Date.now()) as any,
        pnlPercent,
        candlesHeld,
        maxFavorableExcursion: mfe,
        maxAdverseExcursion: mae,
      });
      
      res.json({ success: true });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: "Invalid request body", details: error.errors });
        return;
      }
      console.error("Error updating cone signal outcome:", error);
      res.status(500).json({ error: "Failed to update outcome" });
    }
  });

  // Cone signal generator endpoint
  app.get("/api/cone-signal/current", async (req, res) => {
    try {
      const { coneSignalGenerator } = await import("./cone-signal-generator");
      const { gpuBridge } = await import("./gpu-bridge");
      
      // Get latest candle price
      const candles = storage.getCandles();
      if (candles.length === 0) {
        res.json({ available: false, reason: "No candle data available" });
        return;
      }
      
      const lastCandle = candles[candles.length - 1];
      const currentPrice = lastCandle.close;
      
      // Get quantiles from GPU or fallback
      const gpuStatus = gpuBridge.getPushedStatus();
      let quantiles: { q10: number; q25: number; q50: number; q75: number; q90: number } | null = null;
      let probs = { probUp: 0.33, probDown: 0.33, probHold: 0.34 };
      let mu = 0;
      let sigma: number | undefined;
      
      // Flow forecast from GPU trainer
      let flowForecast: {
        volState: "contraction" | "neutral" | "expansion";
        volStateProbs: { contraction: number; neutral: number; expansion: number };
        acceleration: number;
        forecastMode: "QUANTILE_PATHS" | "NO_FORECAST";
        quantilePaths?: { q10: number[]; q50: number[]; q90: number[] };
      } | undefined;
      
      // Try GPU prediction first
      if (gpuStatus.connected && gpuStatus.modelsLoaded.length > 0) {
        try {
          const ensembleStatus = await gpuBridge.getEnsembleStatus();
          if (ensembleStatus?.initialized) {
            // Use predictEnsembleFromCandles to get quantile predictions
            const prediction = await gpuBridge.predictEnsembleFromCandles(candles.slice(-100));
            if (prediction?.quantiles) {
              quantiles = prediction.quantiles;
              probs = {
                probUp: prediction.ensemble_probs?.LONG || 0.33,
                probDown: prediction.ensemble_probs?.SHORT || 0.33,
                probHold: prediction.ensemble_probs?.HOLD || 0.34,
              };
              mu = prediction.mu || 0;
              sigma = prediction.sigma;
              
              // Extract flow forecast data if available
              if (prediction.vol_state && prediction.forecast_mode) {
                flowForecast = {
                  volState: prediction.vol_state,
                  volStateProbs: prediction.vol_state_probs || { contraction: 0.33, neutral: 0.34, expansion: 0.33 },
                  acceleration: prediction.acceleration || 0,
                  forecastMode: prediction.forecast_mode,
                  quantilePaths: prediction.quantile_paths,
                };
              }
            }
          }
        } catch (e) {
          console.log("[ConeSignal] GPU prediction failed, using fallback");
        }
      }
      
      // Fallback: generate simple quantiles from recent volatility
      if (!quantiles) {
        const recentCandles = candles.slice(-100);
        const returns = recentCandles.slice(1).map((c, i) => 
          (c.close - recentCandles[i].close) / recentCandles[i].close
        );
        const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const stdReturn = Math.sqrt(returns.map(r => Math.pow(r - avgReturn, 2)).reduce((a, b) => a + b, 0) / returns.length);
        
        // 16-bar horizon multiplier
        const horizonMultiplier = Math.sqrt(16);
        const projectedStd = stdReturn * horizonMultiplier;
        
        quantiles = {
          q10: avgReturn * 16 - 1.28 * projectedStd,
          q25: avgReturn * 16 - 0.67 * projectedStd,
          q50: avgReturn * 16,
          q75: avgReturn * 16 + 0.67 * projectedStd,
          q90: avgReturn * 16 + 1.28 * projectedStd,
        };
        mu = avgReturn * 16;
        sigma = projectedStd;
      }
      
      // Generate cone signal with optional flow forecast
      const signal = coneSignalGenerator.generateSignal({
        currentPrice,
        quantiles,
        probs,
        mu,
        sigma,
        timestamp: lastCandle.timestamp,
        flowForecast,
      });
      
      // Get generator stats
      const stats = coneSignalGenerator.getStats();
      
      // AUTO-SAVE: If LONG or SHORT signal, automatically record to database
      if (signal.direction !== "HOLD") {
        try {
          // Check if this timestamp was already recorded to avoid duplicates
          const existingSignal = await storage.getConeSignalByTimestamp(signal.timestamp);
          
          if (!existingSignal) {
            await storage.recordConeSignal({
              timestamp: signal.timestamp,
              direction: signal.direction,
              entryPrice: signal.entryPrice,
              stopLoss: signal.stopLoss,
              takeProfit: signal.takeProfit,
              mu: signal.mu,
              sigma: signal.sigma,
              edge: signal.edge,
              riskReward: signal.riskReward,
              q10: signal.quantiles.q10,
              q25: signal.quantiles.q25,
              q50: signal.quantiles.q50,
              q75: signal.quantiles.q75,
              q90: signal.quantiles.q90,
              probUp: signal.probUp,
              probDown: signal.probDown,
              probHold: signal.probHold,
              holdReasons: signal.holdReasons || [],
              edgeThreshold: signal.edgeThreshold,
              outcome: "PENDING",
              createdAt: Date.now(),
            });
            console.log(`[ConeSignal] Auto-saved ${signal.direction} signal at ${new Date(signal.timestamp).toISOString()}`);
          }
        } catch (saveError) {
          console.error("[ConeSignal] Failed to auto-save signal:", saveError);
        }
      }
      
      res.json({
        available: true,
        signal,
        stats,
        source: gpuStatus.connected ? "gpu" : "fallback",
      });
    } catch (error) {
      console.error("Error getting cone signal:", error);
      res.status(500).json({ error: "Failed to get cone signal" });
    }
  });

  // Production Signal API with full gating pipeline (2024 State-of-the-Art)
  // Gate order: spread → confidence → uncertainty → cooldown → kelly_position_sizing → final
  app.get("/api/signal/production", async (req, res) => {
    try {
      const { gpuBridge } = await import("./gpu-bridge");
      
      // Get latest candles
      const candles = storage.getCandles();
      if (candles.length < 100) {
        res.json({ 
          trade: false, 
          reason: "INSUFFICIENT_DATA",
          message: `Only ${candles.length} candles available, need 100`,
          gates_passed: []
        });
        return;
      }
      
      const lastCandle = candles[candles.length - 1];
      const currentPrice = lastCandle.close;
      
      // Check GPU connection
      const gpuStatus = gpuBridge.getPushedStatus();
      if (!gpuStatus.connected || gpuStatus.modelsLoaded.length === 0) {
        res.json({ 
          trade: false, 
          reason: "GPU_DISCONNECTED",
          message: "GPU trainer not connected or no models loaded",
          gates_passed: []
        });
        return;
      }
      
      // Get ensemble prediction
      const prediction = await gpuBridge.predictEnsembleFromCandles(candles.slice(-100));
      if (!prediction) {
        res.json({ 
          trade: false, 
          reason: "PREDICTION_FAILED",
          message: "Failed to get prediction from GPU trainer",
          gates_passed: []
        });
        return;
      }
      
      const gatesPassed: string[] = [];
      const gatesFailed: { gate: string; reason: string }[] = [];
      
      // === GATE 1: Spread Check ===
      const spreadPct = 0.0005; // 0.05% assumed spread
      const expectedMove = prediction.mu || 0;
      const spreadMultiplier = 3.0;
      
      if (Math.abs(expectedMove) < spreadPct * spreadMultiplier) {
        gatesFailed.push({ gate: "SPREAD", reason: `Expected move ${(expectedMove*100).toFixed(3)}% < ${(spreadPct*spreadMultiplier*100).toFixed(3)}% threshold` });
      } else {
        gatesPassed.push("SPREAD");
      }
      
      // === GATE 2: Confidence Threshold ===
      const confidence = prediction.confidence || 0;
      const minConfidence = 0.15;
      
      if (confidence < minConfidence) {
        gatesFailed.push({ gate: "CONFIDENCE", reason: `Confidence ${(confidence*100).toFixed(1)}% < ${(minConfidence*100).toFixed(1)}% threshold` });
      } else {
        gatesPassed.push("CONFIDENCE");
      }
      
      // === GATE 3: Direction Check (not HOLD) ===
      const action = prediction.action;
      
      if (action === "HOLD") {
        gatesFailed.push({ gate: "DIRECTION", reason: "Signal is HOLD - no trade" });
      } else {
        gatesPassed.push("DIRECTION");
      }
      
      // === GATE 4: Uncertainty Check (MC Dropout) ===
      // Cast prediction to any to access optional epistemic_uncertainty field
      const epistemic = (prediction as any).epistemic_uncertainty as number | undefined;
      const maxUncertainty = 0.02;
      
      // MANDATORY GATE: If uncertainty unavailable, assume worst case (0.5)
      const effectiveUncertainty = epistemic ?? 0.5; // Default to high uncertainty if missing
      
      if (effectiveUncertainty > maxUncertainty) {
        const reason = epistemic === undefined 
          ? `Uncertainty data unavailable (assumed ${effectiveUncertainty}) > ${maxUncertainty} threshold`
          : `Epistemic uncertainty ${effectiveUncertainty.toFixed(4)} > ${maxUncertainty} threshold`;
        gatesFailed.push({ gate: "UNCERTAINTY", reason });
      } else {
        gatesPassed.push("UNCERTAINTY");
      }
      
      // === GATE 5: Cooldown Check ===
      // Use storage-based cooldown tracker for persistence across restarts
      const cooldownState = storage.getCooldownState ? storage.getCooldownState() : null;
      const lastTradeTime = cooldownState?.lastTradeTime ?? 0;
      const cooldownBars = 8;
      const barDurationMs = 15 * 60 * 1000; // 15 minutes
      const cooldownMs = cooldownBars * barDurationMs;
      const timeSinceLastTrade = Date.now() - lastTradeTime;
      
      if (lastTradeTime > 0 && timeSinceLastTrade < cooldownMs) {
        const barsRemaining = Math.ceil((cooldownMs - timeSinceLastTrade) / barDurationMs);
        gatesFailed.push({ gate: "COOLDOWN", reason: `${barsRemaining} bars cooldown remaining` });
      } else {
        gatesPassed.push("COOLDOWN");
      }
      
      // Update last trade time if all gates pass (done at the end)
      
      // === Kelly Criterion Position Sizing ===
      const winProb = action === "LONG" ? (prediction.ensemble_probs?.LONG || 0.5) : 
                      action === "SHORT" ? (prediction.ensemble_probs?.SHORT || 0.5) : 0.5;
      
      // Expected win/loss from quantiles
      const quantiles = prediction.quantiles || { q10: -0.01, q25: -0.005, q50: 0, q75: 0.005, q90: 0.01 };
      const expectedWin = action === "LONG" ? (quantiles.q75 - 0) : (0 - quantiles.q25);
      const expectedLoss = action === "LONG" ? (0 - quantiles.q25) : (quantiles.q75 - 0);
      
      // Kelly formula: f* = (p * b - q) / b where b = expected_win / expected_loss
      const odds = expectedWin / Math.max(expectedLoss, 0.001);
      const lossProb = 1 - winProb;
      const fullKelly = odds > 0 ? (winProb * odds - lossProb) / odds : 0;
      const halfKelly = fullKelly * 0.5; // Conservative Half-Kelly
      const maxPosition = 0.25; // 25% max
      const positionSize = Math.max(0, Math.min(maxPosition, halfKelly));
      
      // Calculate edge
      const fixedCost = 0.0009; // 9 bps total costs
      const grossEdge = winProb * expectedWin - lossProb * expectedLoss;
      const netEdge = grossEdge - fixedCost;
      
      // === GATE 6: Edge Check ===
      const minEdge = 0.001; // 0.1% minimum edge
      
      if (netEdge < minEdge) {
        gatesFailed.push({ gate: "EDGE", reason: `Net edge ${(netEdge*100).toFixed(3)}% < ${(minEdge*100).toFixed(3)}% minimum` });
      } else {
        gatesPassed.push("EDGE");
      }
      
      // === Final Decision ===
      const allGatesPassed = gatesFailed.length === 0;
      
      // Update cooldown tracker if trade is triggered (persisted via storage)
      if (allGatesPassed && storage.setCooldownState) {
        storage.setCooldownState({ lastTradeTime: Date.now() });
      }
      
      // Derive SL/TP from quantiles
      const slDistance = action === "LONG" ? Math.abs(quantiles.q10) : quantiles.q90;
      const tpDistance = action === "LONG" ? quantiles.q90 : Math.abs(quantiles.q10);
      
      const stopLossPrice = action === "LONG" ? currentPrice * (1 - slDistance) : currentPrice * (1 + slDistance);
      const takeProfitPrice = action === "LONG" ? currentPrice * (1 + tpDistance) : currentPrice * (1 - tpDistance);
      
      res.json({
        trade: allGatesPassed,
        action: allGatesPassed ? action : null,
        confidence,
        
        // Position sizing
        position_size_pct: allGatesPassed ? positionSize * 100 : 0,
        kelly_fraction: halfKelly,
        full_kelly: fullKelly,
        
        // Entry/Exit prices
        entry_price: currentPrice,
        stop_loss_price: stopLossPrice,
        take_profit_price: takeProfitPrice,
        sl_distance_pct: slDistance * 100,
        tp_distance_pct: tpDistance * 100,
        
        // Edge and costs
        gross_edge_pct: grossEdge * 100,
        net_edge_pct: netEdge * 100,
        fixed_cost_pct: fixedCost * 100,
        
        // Probabilities
        win_probability: winProb,
        odds_ratio: odds,
        
        // Quantiles
        quantiles: {
          q10: quantiles.q10 * 100,
          q25: quantiles.q25 * 100,
          q50: quantiles.q50 * 100,
          q75: quantiles.q75 * 100,
          q90: quantiles.q90 * 100,
        },
        
        // Uncertainty
        epistemic_uncertainty: epistemic,
        
        // Regime
        market_regime: prediction.market_regime,
        risk_regime: prediction.risk_regime,
        
        // Flow forecast
        vol_state: prediction.vol_state,
        forecast_mode: prediction.forecast_mode,
        
        // Gate results
        gates_passed: gatesPassed,
        gates_failed: gatesFailed,
        reason: allGatesPassed ? "ALL_GATES_PASSED" : gatesFailed[0]?.gate,
        
        // Metadata
        timestamp: Date.now(),
        source: "GPU_ENSEMBLE"
      });
    } catch (error) {
      console.error("Error getting production signal:", error);
      res.status(500).json({ error: "Failed to get production signal" });
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
    const timeframe = req.body.timeframe || "all";
    
    const timeframeLabel = timeframe === "all" ? "all timeframes (1m, 5m, 15m, 1h, 4h)" : timeframe;
    res.json({ started: true, message: `Starting download for ${years} year(s) of ${timeframeLabel} data` });
    
    downloadNNData(years, (symbol, tf, progress) => {
      console.log(`[NN Data] ${symbol} ${tf}: ${progress.toFixed(1)}%`);
    }, timeframe).then(result => {
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

  // Self-learning loop endpoints
  app.get("/api/self-learning/status", async (req, res) => {
    try {
      const { getLearningStatus } = await import("./self-learning-loop");
      const status = await getLearningStatus();
      res.json(status);
    } catch (error) {
      console.error("Error getting self-learning status:", error);
      res.status(500).json({ error: "Failed to get self-learning status" });
    }
  });

  app.post("/api/self-learning/enable", async (req, res) => {
    try {
      const { enabled } = req.body;
      const { setSelfLearningEnabled } = await import("./self-learning-loop");
      setSelfLearningEnabled(enabled ?? true);
      res.json({ success: true, enabled: enabled ?? true });
    } catch (error) {
      console.error("Error enabling self-learning:", error);
      res.status(500).json({ error: "Failed to enable self-learning" });
    }
  });

  app.post("/api/self-learning/run-now", async (req, res) => {
    try {
      const { startSelfLearningLoop, getSelfLearningConfig } = await import("./self-learning-loop");
      const config = getSelfLearningConfig();
      if (!config.enabled) {
        return res.status(400).json({ error: "Self-learning is not enabled. Enable it first." });
      }
      startSelfLearningLoop();
      res.json({ success: true, message: "Self-learning job triggered" });
    } catch (error) {
      console.error("Error running self-learning job:", error);
      res.status(500).json({ error: "Failed to run self-learning job" });
    }
  });

  app.post("/api/self-learning/training-callback", async (req, res) => {
    try {
      const { runId, status, metrics } = req.body;
      if (!runId || !status) {
        return res.status(400).json({ error: "runId and status are required" });
      }
      const { handleTrainingCallback } = await import("./self-learning-loop");
      const result = await handleTrainingCallback(runId, status, metrics);
      res.json(result);
    } catch (error) {
      console.error("Error handling training callback:", error);
      res.status(500).json({ error: "Failed to handle training callback" });
    }
  });

  app.post("/api/self-learning/rollback", async (req, res) => {
    try {
      const { rollbackToPrevious } = await import("./self-learning-loop");
      const result = await rollbackToPrevious();
      res.json(result);
    } catch (error) {
      console.error("Error rolling back model:", error);
      res.status(500).json({ error: "Failed to rollback model" });
    }
  });

  app.post("/api/self-learning/add-sample", async (req, res) => {
    try {
      const { timestamp, features, currentPrice, regime } = req.body;
      if (!timestamp || !features || !currentPrice) {
        return res.status(400).json({ error: "timestamp, features, and currentPrice are required" });
      }
      const { addPendingSample } = await import("./self-learning-loop");
      await addPendingSample(timestamp, features, currentPrice, regime);
      res.json({ success: true });
    } catch (error) {
      console.error("Error adding sample:", error);
      res.status(500).json({ error: "Failed to add sample" });
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
      const { years = 1, assets, timeframe = "15m" } = req.body;
      
      if (years < 1 || years > 15) {
        return res.status(400).json({ error: "Years must be between 1 and 15" });
      }
      
      // Validate timeframe
      const validTimeframes = ["1m", "5m", "15m", "1h", "4h", "all"];
      if (!validTimeframes.includes(timeframe)) {
        return res.status(400).json({ error: `Invalid timeframe. Must be one of: ${validTimeframes.join(", ")}` });
      }
      
      const { downloadMultiAssetData, getSupportedAssets } = await import("./historical-data");
      const supportedAssets = getSupportedAssets();
      const assetsToDownload = assets || supportedAssets;
      
      // Determine which timeframes to download
      const timeframesToDownload = timeframe === "all" 
        ? ["1m", "5m", "15m", "1h", "4h"] 
        : [timeframe];
      
      // Calculate estimated candles based on timeframe
      const candlesPerDay: Record<string, number> = {
        "1m": 24 * 60,      // 1440 candles per day
        "5m": 24 * 12,      // 288 candles per day
        "15m": 24 * 4,      // 96 candles per day
        "1h": 24,           // 24 candles per day
        "4h": 6,            // 6 candles per day
      };
      
      const estimatedCandles = timeframesToDownload.reduce((total, tf) => {
        return total + Math.floor(years * 365 * (candlesPerDay[tf] || 96)) * assetsToDownload.length;
      }, 0);
      
      // Start download in background
      res.json({ 
        message: `Started downloading ${years} years of ${timeframe === "all" ? "all timeframes" : timeframe} data for ${assetsToDownload.length} assets`,
        assets: assetsToDownload,
        timeframes: timeframesToDownload,
        estimatedCandles,
      });
      
      // Run download async for each timeframe, reload data when complete
      (async () => {
        for (const tf of timeframesToDownload) {
          try {
            console.log(`[Data Download] Starting ${tf} timeframe...`);
            const result = await downloadMultiAssetData(years, assetsToDownload, undefined, tf);
            console.log(`[Data Download] ${tf} Complete:`, result);
          } catch (err) {
            console.error(`[Data Download] Error for ${tf}:`, err);
          }
        }
        // Reload historical candles into memory after all downloads
        await storage.reloadHistoricalCandles();
        console.log("[Data Download] All downloads complete, reloaded candles into memory");
      })();
      
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

  app.get("/api/data/export-csv", async (req, res) => {
    try {
      const symbol = (req.query.symbol as string || "BTCUSDT").toUpperCase();
      const timeframe = (req.query.timeframe as string) || "15m";

      const validSymbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"];
      const validTimeframes = ["1m", "5m", "15m", "1h", "4h", "1d"];
      if (!validSymbols.includes(symbol)) {
        return res.status(400).json({ error: `Invalid symbol. Allowed: ${validSymbols.join(", ")}` });
      }
      if (!validTimeframes.includes(timeframe)) {
        return res.status(400).json({ error: `Invalid timeframe. Allowed: ${validTimeframes.join(", ")}` });
      }
      
      const rows = await db
        .select({
          timestamp: candles.timestamp,
          open: candles.open,
          high: candles.high,
          low: candles.low,
          close: candles.close,
          volume: candles.volume,
        })
        .from(candles)
        .where(and(eq(candles.symbol, symbol), eq(candles.timeframe, timeframe)))
        .orderBy(asc(candles.timestamp));

      if (rows.length === 0) {
        return res.status(404).json({ error: `No ${timeframe} data found for ${symbol}` });
      }

      res.setHeader("Content-Type", "text/csv");
      res.setHeader("Content-Disposition", `attachment; filename=${symbol}_${timeframe}.csv`);
      
      res.write("timestamp,open,high,low,close,volume\n");
      for (const r of rows) {
        res.write(`${r.timestamp},${r.open},${r.high},${r.low},${r.close},${r.volume}\n`);
      }
      res.end();
    } catch (error) {
      console.error("Error exporting CSV:", error);
      res.status(500).json({ error: "Failed to export CSV data" });
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
        modelStatus: status.modelStatus ?? undefined,
        // Training mode detection from GPU trainer (snake_case from Python API)
        trainingMode: status.training_mode ?? status.trainingMode ?? null,
        trainingModeDescription: status.training_mode_description ?? status.trainingModeDescription ?? null,
        inputDim: status.input_dim ?? status.inputDim ?? null
      });
      console.log(`[GPU Push] Received status update - GPU: ${status.gpuName}, Training: ${status.isTraining}, Mode: ${status.training_mode ?? status.trainingMode ?? 'unknown'}`);
      res.json({ success: true, received: Date.now() });
    } catch (error) {
      console.error("[GPU Push] Error:", error);
      res.status(500).json({ error: "Failed to process status update" });
    }
  });

  // Multi-head prediction push endpoint - Receives full 5-head predictions from local GPU trainer
  app.post("/api/gpu/push-prediction", async (req, res) => {
    try {
      const pred = req.body;
      
      if (!pred || typeof pred !== "object") {
        return res.status(400).json({ error: "Request body must be a JSON object" });
      }
      if (!pred.action || !["LONG", "SHORT", "HOLD"].includes(pred.action)) {
        return res.status(400).json({ error: "action must be LONG, SHORT, or HOLD" });
      }
      if (pred.confidence === undefined || typeof pred.confidence !== "number" || pred.confidence < 0 || pred.confidence > 1) {
        return res.status(400).json({ error: "confidence must be a number between 0 and 1" });
      }
      
      const record = await storage.recordMultiheadPrediction({
        timestamp: pred.timestamp || Date.now(),
        action: pred.action,
        probLong: pred.direction_probs?.LONG ?? pred.prob_long ?? 0.33,
        probShort: pred.direction_probs?.SHORT ?? pred.prob_short ?? 0.33,
        probHold: pred.direction_probs?.HOLD ?? pred.prob_hold ?? 0.34,
        confidence: pred.confidence,
        q10: pred.quantiles?.q10 ?? pred.q10 ?? null,
        q25: pred.quantiles?.q25 ?? pred.q25 ?? null,
        q50: pred.quantiles?.q50 ?? pred.q50 ?? null,
        q75: pred.quantiles?.q75 ?? pred.q75 ?? null,
        q90: pred.quantiles?.q90 ?? pred.q90 ?? null,
        volState: pred.vol_state ?? pred.volState ?? null,
        volStateContraction: pred.vol_state_probs?.contraction ?? null,
        volStateNeutral: pred.vol_state_probs?.neutral ?? null,
        volStateExpansion: pred.vol_state_probs?.expansion ?? null,
        mu: pred.expected_return ?? pred.mu ?? null,
        sigma: pred.uncertainty ?? pred.sigma ?? null,
        edge: pred.edge ?? null,
        entryPrice: pred.entry_price ?? pred.entryPrice ?? null,
        stopLossPrice: pred.stop_loss_price ?? pred.stopLossPrice ?? null,
        takeProfitPrice: pred.take_profit_price ?? pred.takeProfitPrice ?? null,
        stopLossPct: pred.stop_loss_pct ?? pred.stopLossPct ?? null,
        takeProfitPct: pred.take_profit_pct ?? pred.takeProfitPct ?? null,
        riskRewardRatio: pred.risk_reward_ratio ?? pred.riskRewardRatio ?? null,
        positionSizePct: pred.position_size_pct ?? pred.positionSizePct ?? null,
        currentPrice: pred.current_price ?? pred.currentPrice ?? null,
        modelName: pred.model_name ?? pred.modelName ?? "unknown",
        isMultihead: pred.is_multihead ?? true,
        urgency: pred.urgency ?? "low",
        suggestedOrderType: pred.suggested_order_type ?? "limit",
        reasons: pred.reasons ?? [],
        createdAt: Date.now(),
      });
      
      console.log(`[GPU Push] Received multi-head prediction: ${pred.action} confidence=${pred.confidence.toFixed(3)} vol_state=${pred.vol_state ?? 'N/A'}`);
      res.json({ success: true, id: record.id, received: Date.now() });
    } catch (error) {
      console.error("[GPU Push] Error recording prediction:", error);
      res.status(500).json({ error: "Failed to record prediction" });
    }
  });

  // Get latest multi-head prediction (for dashboard)
  app.get("/api/gpu/multihead/latest", async (req, res) => {
    try {
      const latest = await storage.getLatestMultiheadPrediction();
      if (!latest) {
        return res.json({ available: false, prediction: null });
      }
      
      const isStale = Date.now() - (latest.createdAt ?? 0) > 300000;
      
      res.json({
        available: !isStale,
        isStale,
        prediction: {
          action: latest.action,
          confidence: latest.confidence,
          direction_probs: {
            LONG: latest.probLong,
            SHORT: latest.probShort,
            HOLD: latest.probHold,
          },
          quantiles: latest.q10 != null ? {
            q10: latest.q10,
            q25: latest.q25,
            q50: latest.q50,
            q75: latest.q75,
            q90: latest.q90,
          } : null,
          vol_state: latest.volState,
          vol_state_probs: latest.volStateContraction != null ? {
            contraction: latest.volStateContraction,
            neutral: latest.volStateNeutral,
            expansion: latest.volStateExpansion,
          } : null,
          mu: latest.mu,
          sigma: latest.sigma,
          edge: latest.edge,
          entry_price: latest.entryPrice,
          stop_loss_price: latest.stopLossPrice,
          take_profit_price: latest.takeProfitPrice,
          stop_loss_pct: latest.stopLossPct,
          take_profit_pct: latest.takeProfitPct,
          risk_reward_ratio: latest.riskRewardRatio,
          position_size_pct: latest.positionSizePct,
          current_price: latest.currentPrice,
          model_name: latest.modelName,
          is_multihead: latest.isMultihead,
          urgency: latest.urgency,
          suggested_order_type: latest.suggestedOrderType,
          reasons: latest.reasons,
          timestamp: latest.timestamp,
          created_at: latest.createdAt,
        },
      });
    } catch (error) {
      console.error("[GPU Multihead] Error getting latest prediction:", error);
      res.json({ available: false, prediction: null, error: "Failed to fetch" });
    }
  });

  // Get multi-head prediction history
  app.get("/api/gpu/multihead/history", async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 50;
      const rawPredictions = await storage.getMultiheadPredictions(limit);
      const predictions = rawPredictions.map(p => ({
        action: p.action,
        confidence: p.confidence,
        direction_probs: {
          LONG: p.probLong ?? 0,
          SHORT: p.probShort ?? 0,
          HOLD: p.probHold ?? 0,
        },
        quantiles: p.q10 != null ? {
          q10: p.q10, q25: p.q25, q50: p.q50, q75: p.q75, q90: p.q90,
        } : null,
        vol_state: p.volState ?? null,
        vol_state_probs: p.volStateContraction != null ? {
          contraction: p.volStateContraction,
          neutral: p.volStateNeutral,
          expansion: p.volStateExpansion,
        } : null,
        mu: p.mu ?? null,
        sigma: p.sigma ?? null,
        edge: p.edge ?? null,
        entry_price: p.entryPrice ?? null,
        stop_loss_price: p.stopLossPrice ?? null,
        take_profit_price: p.takeProfitPrice ?? null,
        stop_loss_pct: p.stopLossPct ?? null,
        take_profit_pct: p.takeProfitPct ?? null,
        risk_reward_ratio: p.riskRewardRatio ?? null,
        position_size_pct: p.positionSizePct ?? null,
        current_price: p.currentPrice ?? null,
        model_name: p.modelName ?? null,
        is_multihead: p.isMultihead ?? null,
        urgency: p.urgency ?? null,
        suggested_order_type: p.suggestedOrderType ?? null,
        reasons: p.reasons ?? null,
        timestamp: p.timestamp ?? 0,
        created_at: p.createdAt ?? 0,
      }));
      res.json({ predictions, total: predictions.length });
    } catch (error) {
      console.error("[GPU Multihead] Error getting history:", error);
      res.json({ predictions: [], total: 0 });
    }
  });

  // Get pushed GPU status (for dashboard to poll)
  // Enriches with training mode from GPU trainer's /models/status if available
  app.get("/api/gpu/pushed-status", async (req, res) => {
    const status = gpuBridge.getPushedStatus();
    const isStale = status.lastPush ? Date.now() - status.lastPush > 30000 : true;
    const isConnected = status.connected && !isStale;
    
    // If connected but missing training mode, try to fetch from GPU trainer
    let trainingMode = status.trainingMode;
    let trainingModeDescription = status.trainingModeDescription;
    let inputDim = status.inputDim;
    
    if (isConnected && !trainingMode) {
      try {
        const modelsStatus = await gpuBridge.fetchModelsStatus();
        if (modelsStatus) {
          trainingMode = modelsStatus.training_mode;
          trainingModeDescription = modelsStatus.training_mode_description;
          inputDim = modelsStatus.config?.input_dim ?? null;
        }
      } catch (e) {
        // Ignore errors, use what we have
      }
    }
    
    res.json({
      ...status,
      connected: isConnected,
      isStale,
      trainingMode,
      trainingModeDescription,
      inputDim
    });
  });

  // ============ GPU DIAGNOSTIC ENDPOINTS ============
  
  // Test if GPU models respond to different inputs
  app.get("/api/gpu/diagnostics/model-sensitivity", async (req, res) => {
    try {
      const gpuUrl = process.env.GPU_TRAINER_URL || "http://localhost:8000";
      const response = await fetch(`${gpuUrl}/debug/model-sensitivity`);
      if (!response.ok) {
        throw new Error(`GPU trainer returned ${response.status}`);
      }
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error("[GPU Diagnostics] Model sensitivity test failed:", error);
      res.status(503).json({ 
        error: "GPU trainer not available for diagnostics",
        message: String(error)
      });
    }
  });
  
  // Check training label distribution
  app.get("/api/gpu/diagnostics/label-distribution", async (req, res) => {
    try {
      const gpuUrl = process.env.GPU_TRAINER_URL || "http://localhost:8000";
      const response = await fetch(`${gpuUrl}/debug/label-distribution`);
      if (!response.ok) {
        throw new Error(`GPU trainer returned ${response.status}`);
      }
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error("[GPU Diagnostics] Label distribution check failed:", error);
      res.status(503).json({ 
        error: "GPU trainer not available for diagnostics",
        message: String(error)
      });
    }
  });
  
  // Run full diagnostic suite
  app.get("/api/gpu/diagnostics/full", async (req, res) => {
    const gpuUrl = process.env.GPU_TRAINER_URL || "http://localhost:8000";
    const results: Record<string, unknown> = {
      timestamp: new Date().toISOString(),
      gpuUrl
    };
    
    // Test health
    try {
      const healthRes = await fetch(`${gpuUrl}/health`);
      results.health = healthRes.ok ? await healthRes.json() : { error: `Status ${healthRes.status}` };
    } catch (e) {
      results.health = { error: String(e) };
    }
    
    // Test model sensitivity
    try {
      const sensRes = await fetch(`${gpuUrl}/debug/model-sensitivity`);
      results.modelSensitivity = sensRes.ok ? await sensRes.json() : { error: `Status ${sensRes.status}` };
    } catch (e) {
      results.modelSensitivity = { error: String(e) };
    }
    
    // Test label distribution
    try {
      const labelRes = await fetch(`${gpuUrl}/debug/label-distribution`);
      results.labelDistribution = labelRes.ok ? await labelRes.json() : { error: `Status ${labelRes.status}` };
    } catch (e) {
      results.labelDistribution = { error: String(e) };
    }
    
    // Overall diagnosis
    const issues: string[] = [];
    if (results.health && typeof results.health === 'object' && 'error' in results.health) {
      issues.push("GPU trainer health check failed");
    }
    if (results.modelSensitivity && typeof results.modelSensitivity === 'object' && 'overall_status' in results.modelSensitivity) {
      const sens = results.modelSensitivity as { overall_status?: string };
      if (sens.overall_status?.includes("CRITICAL")) {
        issues.push("Models have collapsed to constant output - need retraining");
      }
    }
    if (results.labelDistribution && typeof results.labelDistribution === 'object' && 'is_imbalanced' in results.labelDistribution) {
      const labels = results.labelDistribution as { is_imbalanced?: boolean };
      if (labels.is_imbalanced) {
        issues.push("Training labels are heavily imbalanced toward HOLD");
      }
    }
    
    results.diagnosis = {
      issues,
      needsRetraining: issues.some(i => i.includes("collapsed") || i.includes("imbalanced")),
      recommendation: issues.length > 0 
        ? "Run fresh training with balanced class weights: POST /api/retrain/daily"
        : "Models appear healthy"
    };
    
    res.json(results);
  });
  
  // Get prediction drift report from GPU trainer
  app.get("/api/gpu/diagnostics/drift-report", async (req, res) => {
    const gpuUrl = process.env.GPU_TRAINER_URL || "http://localhost:8000";
    try {
      const response = await fetch(`${gpuUrl}/api/drift-report`);
      if (!response.ok) {
        return res.status(response.status).json({ 
          error: `GPU trainer returned ${response.status}`,
          psi: { SHORT: 0, HOLD: 0, LONG: 0, average: 0 },
          ece: 0,
          n_predictions: 0,
          status: "error"
        });
      }
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error("[Drift Report] Error fetching from GPU trainer:", error);
      res.json({ 
        error: String(error),
        psi: { SHORT: 0, HOLD: 0, LONG: 0, average: 0 },
        ece: 0,
        n_predictions: 0,
        status: "disconnected"
      });
    }
  });

  // Get comprehensive neural network diagnostics for dashboard
  app.get("/api/gpu/diagnostics/dashboard", async (req, res) => {
    const gpuUrl = process.env.GPU_TRAINER_URL || "http://localhost:8000";
    const diagnostics: Record<string, unknown> = {
      timestamp: new Date().toISOString(),
      gpuConnected: false,
      predictionDistribution: { SHORT: 0, HOLD: 0, LONG: 0 },
      calibration: { ece: 0, bins: [] as { confidence: number; accuracy: number; count: number }[] },
      uncertainty: { mean: 0, std: 0, histogram: [] as { bin: string; count: number }[] },
      trainingHealth: { status: "unknown", alerts: [] as string[], lastCheck: null as string | null },
      driftStatus: { psi: 0, ece: 0, alert: false, status: "unknown" },
      recentPredictions: [] as { timestamp: string; action: string; confidence: number; outcome?: string }[]
    };

    try {
      // Check GPU health first
      const healthRes = await fetch(`${gpuUrl}/health`, { signal: AbortSignal.timeout(3000) });
      if (healthRes.ok) {
        diagnostics.gpuConnected = true;
        const healthData = await healthRes.json();
        diagnostics.gpuHealth = healthData;
      }
    } catch (e) {
      diagnostics.gpuConnected = false;
    }

    // Get drift report
    try {
      const driftRes = await fetch(`${gpuUrl}/api/drift-report`, { signal: AbortSignal.timeout(3000) });
      if (driftRes.ok) {
        const driftData = await driftRes.json();
        diagnostics.driftStatus = {
          psi: driftData.psi?.average ?? 0,
          ece: driftData.ece ?? 0,
          alert: (driftData.psi?.average ?? 0) > 0.25 || (driftData.ece ?? 0) > 0.15,
          status: driftData.status ?? "ok",
          nPredictions: driftData.n_predictions ?? 0,
          psiPerClass: driftData.psi ?? {}
        };
        if (driftData.prediction_distribution) {
          diagnostics.predictionDistribution = driftData.prediction_distribution;
        }
        if (driftData.calibration_bins) {
          diagnostics.calibration = {
            ece: driftData.ece ?? 0,
            bins: driftData.calibration_bins
          };
        }
      }
    } catch (e) {
      // Drift endpoint unavailable
    }

    // Get label distribution (proxy for training health)
    try {
      const labelRes = await fetch(`${gpuUrl}/debug/label-distribution`, { signal: AbortSignal.timeout(3000) });
      if (labelRes.ok) {
        const labelData = await labelRes.json();
        const alerts: string[] = [];
        if (labelData.is_imbalanced) {
          alerts.push("Training labels heavily skewed toward HOLD");
        }
        if (labelData.distribution) {
          const holdPct = (labelData.distribution.HOLD ?? 0) * 100;
          if (holdPct > 80) {
            alerts.push(`HOLD class at ${holdPct.toFixed(1)}% - models may not learn direction`);
          }
        }
        diagnostics.trainingHealth = {
          status: alerts.length > 0 ? "warning" : "healthy",
          alerts,
          lastCheck: new Date().toISOString(),
          labelDistribution: labelData.distribution ?? {}
        };
      }
    } catch (e) {
      // Label distribution unavailable
    }

    // Get model sensitivity
    try {
      const sensRes = await fetch(`${gpuUrl}/debug/model-sensitivity`, { signal: AbortSignal.timeout(5000) });
      if (sensRes.ok) {
        const sensData = await sensRes.json();
        diagnostics.modelSensitivity = sensData;
        if (sensData.overall_status?.includes("CRITICAL")) {
          const alerts = (diagnostics.trainingHealth as { alerts: string[] }).alerts ?? [];
          alerts.push("Models collapsed to constant output - retraining required");
          (diagnostics.trainingHealth as { status: string; alerts: string[] }).status = "critical";
          (diagnostics.trainingHealth as { alerts: string[] }).alerts = alerts;
        }
      }
    } catch (e) {
      // Sensitivity check unavailable
    }

    res.json(diagnostics);
  });

  // ============ LIVE CANDLE SYNC ENDPOINTS ============
  
  // Get live sync status
  app.get("/api/sync/status", async (req, res) => {
    try {
      const status = getSyncStatus();
      const freshness = await checkDataFreshness();
      res.json({
        ...status,
        ...freshness,
      });
    } catch (error) {
      console.error("[Sync API] Error getting status:", error);
      res.status(500).json({ error: "Failed to get sync status" });
    }
  });
  
  // Manually trigger sync
  app.post("/api/sync/trigger", async (req, res) => {
    try {
      console.log("[Sync API] Manual sync triggered");
      const result = await syncLatest15mCandles();
      res.json(result);
    } catch (error) {
      console.error("[Sync API] Error triggering sync:", error);
      res.status(500).json({ error: "Failed to trigger sync" });
    }
  });
  
  // Start continuous live sync
  app.post("/api/sync/start", (req, res) => {
    try {
      startLiveCandleSync();
      res.json({ success: true, message: "Live sync started" });
    } catch (error) {
      console.error("[Sync API] Error starting sync:", error);
      res.status(500).json({ error: "Failed to start sync" });
    }
  });
  
  // Stop continuous live sync
  app.post("/api/sync/stop", (req, res) => {
    try {
      stopLiveCandleSync();
      res.json({ success: true, message: "Live sync stopped" });
    } catch (error) {
      console.error("[Sync API] Error stopping sync:", error);
      res.status(500).json({ error: "Failed to stop sync" });
    }
  });
  
  // ============ DAILY RETRAINING ENDPOINTS ============
  
  // Trigger daily retraining pipeline
  app.post("/api/retrain/daily", async (req, res) => {
    try {
      console.log("[Daily Retrain] Starting daily retraining pipeline...");
      
      // Step 1: Sync latest candles
      console.log("[Daily Retrain] Step 1: Syncing latest candles...");
      const syncResult = await syncLatest15mCandles();
      if (!syncResult.success) {
        return res.status(500).json({ 
          error: "Data sync failed", 
          step: "sync",
          details: syncResult.message 
        });
      }
      
      // Step 2: Check data freshness
      const freshness = await checkDataFreshness();
      if (!freshness.isFresh) {
        return res.status(400).json({
          error: "Data not fresh enough for training",
          step: "freshness_check",
          details: freshness.message,
          lastCandleAge: freshness.lastCandleAge
        });
      }
      
      // Step 3: Trigger GPU training via GPU bridge
      console.log("[Daily Retrain] Step 2: Triggering GPU training...");
      const gpuHealth = await gpuBridge.checkHealth();
      if (!gpuHealth) {
        return res.status(503).json({
          error: "GPU trainer not available",
          step: "gpu_check",
          details: "Cannot reach GPU trainer. Make sure it's running."
        });
      }
      
      // Trigger multihead training
      const trainingStarted = await gpuBridge.startTraining("multihead", 100);
      if (!trainingStarted) {
        return res.status(500).json({
          error: "Failed to start training",
          step: "training_start",
          details: "GPU trainer rejected training request"
        });
      }
      
      res.json({
        success: true,
        message: "Daily retraining pipeline started",
        steps: {
          sync: { success: true, candlesSynced: syncResult.candlesInserted },
          freshness: { success: true, lastCandleAge: freshness.lastCandleAge },
          training: { success: true, status: "started" }
        }
      });
    } catch (error) {
      console.error("[Daily Retrain] Error:", error);
      res.status(500).json({ 
        error: "Daily retraining failed",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });
  
  // Get retraining status
  app.get("/api/retrain/status", async (req, res) => {
    try {
      const syncStatus = getSyncStatus();
      const freshness = await checkDataFreshness();
      const gpuHealth = await gpuBridge.checkHealth();
      const gpuStatus = gpuBridge.getPushedStatus();
      
      res.json({
        dataSync: {
          isRunning: syncStatus.isRunning,
          lastSyncTs: syncStatus.lastSyncTs,
          dataFreshness: syncStatus.dataFreshness,
          staleDurationMinutes: syncStatus.staleDurationMinutes,
        },
        gpuTrainer: {
          connected: gpuHealth !== null,
          isTraining: gpuStatus.isTraining || false,
          trainingProgress: gpuStatus.trainingProgress || 0,
          currentModel: gpuStatus.currentModel || null,
        },
        readyForTraining: freshness.isFresh && gpuHealth !== null && !gpuStatus.isTraining,
        lastRetrainTs: null, // Could track this in DB
        nextScheduledRetrain: null, // Could implement scheduler
      });
    } catch (error) {
      console.error("[Retrain API] Error:", error);
      res.status(500).json({ error: "Failed to get retrain status" });
    }
  });
  
  // Run walk-forward validation on GPU trainer
  app.post("/api/walk-forward/evaluate", async (req, res) => {
    const gpuUrl = process.env.GPU_TRAINER_URL || "http://localhost:8000";
    try {
      const { nFolds = 5, testPeriods = 500, trainPeriods = 2000, purgePeriods = 50 } = req.body;
      
      const response = await fetch(`${gpuUrl}/api/walk-forward/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          n_folds: nFolds,
          test_periods: testPeriods,
          train_periods: trainPeriods,
          purge_periods: purgePeriods,
          min_confidence: 0.4
        }),
        signal: AbortSignal.timeout(300000) // 5 minute timeout for full evaluation
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        return res.status(response.status).json({ 
          status: "error",
          error: `GPU trainer returned ${response.status}: ${errorText}`,
          timestamp: new Date().toISOString()
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error("[Walk-Forward] Error:", error);
      res.status(500).json({ 
        status: "error",
        error: String(error),
        timestamp: new Date().toISOString()
      });
    }
  });
  
  // Get walk-forward report (cached or generate new)
  app.get("/api/walk-forward/report", async (req, res) => {
    const gpuUrl = process.env.GPU_TRAINER_URL || "http://localhost:8000";
    
    // Check if GPU trainer is available
    try {
      const healthRes = await fetch(`${gpuUrl}/health`, { signal: AbortSignal.timeout(3000) });
      if (!healthRes.ok) {
        return res.json({
          status: "gpu_unavailable",
          message: "GPU trainer is not connected",
          timestamp: new Date().toISOString()
        });
      }
    } catch (e) {
      return res.json({
        status: "gpu_unavailable",
        message: "GPU trainer is not responding",
        timestamp: new Date().toISOString()
      });
    }
    
    // Return instructions for running evaluation
    res.json({
      status: "ready",
      message: "Use POST /api/walk-forward/evaluate to run walk-forward validation",
      defaultConfig: {
        nFolds: 5,
        testPeriods: 500,
        trainPeriods: 2000,
        purgePeriods: 50
      },
      estimatedDuration: "2-5 minutes depending on model count",
      timestamp: new Date().toISOString()
    });
  });
  
  // Get detailed training progress from GPU trainer (real-time during training)
  app.get("/api/gpu/training/status", async (req, res) => {
    const gpuUrl = process.env.GPU_TRAINER_URL || "http://localhost:8000";
    try {
      const response = await fetch(`${gpuUrl}/training/status`, { 
        signal: AbortSignal.timeout(3000) 
      });
      if (!response.ok) {
        return res.json({
          is_training: false,
          current_epoch: 0,
          total_epochs: 0,
          current_model: null,
          progress: 0,
          metrics: {},
          epoch_history: [],
          start_time: null,
          eta_seconds: null,
          health_warnings: [],
          last_update: null,
          per_head_losses: {},
          learning_rate: null,
          best_val_loss: null,
          early_stop_counter: 0,
          prediction_distribution: { short: 0, hold: 0, long: 0, total: 0 },
          gradient_norm: null,
          connected: false,
          error: `GPU trainer returned ${response.status}`
        });
      }
      const data = await response.json();
      res.json({ ...data, connected: true });
    } catch (error) {
      res.json({
        is_training: false,
        current_epoch: 0,
        total_epochs: 0,
        current_model: null,
        progress: 0,
        metrics: {},
        epoch_history: [],
        start_time: null,
        eta_seconds: null,
        health_warnings: [],
        last_update: null,
        per_head_losses: {},
        learning_rate: null,
        best_val_loss: null,
        early_stop_counter: 0,
        prediction_distribution: { short: 0, hold: 0, long: 0, total: 0 },
        gradient_norm: null,
        connected: false,
        error: "GPU trainer not connected"
      });
    }
  });

  // Get GPU trainer connection settings
  app.get("/api/gpu/settings", (req, res) => {
    res.json({
      url: gpuBridge.getUrl(),
      defaultUrl: "http://localhost:8000",
      predictionMode: gpuBridge.getPredictionMode()
    });
  });

  // Update GPU trainer URL and prediction mode
  app.post("/api/gpu/settings", async (req, res) => {
    try {
      const { url, predictionMode } = req.body;
      
      if (!url || typeof url !== "string") {
        return res.status(400).json({ error: "URL is required" });
      }
      
      // Validate URL format
      try {
        new URL(url);
      } catch {
        return res.status(400).json({ error: "Invalid URL format" });
      }
      
      // Validate prediction mode
      if (predictionMode && !["stf", "mtf"].includes(predictionMode)) {
        return res.status(400).json({ error: "Invalid prediction mode. Must be 'stf' or 'mtf'" });
      }
      
      // Update the GPU bridge URL
      gpuBridge.setUrl(url);
      
      // Update prediction mode if provided
      if (predictionMode) {
        gpuBridge.setPredictionMode(predictionMode);
      }
      
      // Test connection to the new URL
      const health = await gpuBridge.checkHealth();
      
      res.json({
        success: true,
        url,
        predictionMode: gpuBridge.getPredictionMode(),
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
  // Uses MTF candles endpoint for proper feature alignment with training
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
      
      // Helper to get latest N candles from database by timeframe
      const getDbCandles = async (timeframe: string, limit: number = 200) => {
        const result = await db.select()
          .from(candles)
          .where(and(eq(candles.symbol, "BTCUSDT"), eq(candles.timeframe, timeframe)))
          .orderBy(desc(candles.timestamp))  // Get most recent first
          .limit(limit);
        return result.reverse();  // Reverse to chronological order (oldest to newest)
      };
      
      // Common candle type for both sources
      type CandleData = { timestamp: number; open: number; high: number; low: number; close: number; volume: number };
      
      // Try live API first, fallback to database if blocked (HTTP 451)
      let mtfCandles: { m5: CandleData[]; m15: CandleData[]; h1: CandleData[]; h4: CandleData[] };
      let dataSource = "live";
      
      try {
        const liveCandles = await getMultiTimeframeKlines("BTCUSDT");
        if (liveCandles.m15.length > 50) {
          mtfCandles = liveCandles;
        } else {
          throw new Error("Insufficient live data");
        }
      } catch (liveError) {
        // Fallback to database candles
        console.log("[GPU Ensemble] Live API failed, using database candles:", liveError instanceof Error ? liveError.message : "unknown");
        dataSource = "database";
        
        const [db5m, db15m, db1h, db4h] = await Promise.all([
          getDbCandles("5m", 300),
          getDbCandles("15m", 200),
          getDbCandles("1h", 100),
          getDbCandles("4h", 50)
        ]);
        
        mtfCandles = { m5: db5m, m15: db15m, h1: db1h, h4: db4h };
      }
      
      if (!mtfCandles.m15 || mtfCandles.m15.length < 50) {
        return res.json({ 
          available: false, 
          prediction: null,
          message: "Not enough 15m candle data available"
        });
      }
      
      // Format candles for MTF endpoint
      const formatCandles = (candleData: typeof mtfCandles.m15) => 
        candleData.map(c => ({
          timestamp: c.timestamp,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
          volume: c.volume
        }));
      
      const recent15m = formatCandles(mtfCandles.m15);
      const recent5m = mtfCandles.m5.length > 0 ? formatCandles(mtfCandles.m5) : undefined;
      const recent1h = mtfCandles.h1.length > 0 ? formatCandles(mtfCandles.h1) : undefined;
      const recent4h = mtfCandles.h4.length > 0 ? formatCandles(mtfCandles.h4) : undefined;
      
      console.log(`[GPU Ensemble] MTF candles (${dataSource}): 5m=${mtfCandles.m5.length}, 15m=${mtfCandles.m15.length}, 1h=${mtfCandles.h1.length}, 4h=${mtfCandles.h4.length}`);
      
      // Use the MTF endpoint that computes features server-side (matches training)
      const prediction = await gpuBridge.predictEnsembleFromCandles(
        recent15m,
        recent5m,
        recent1h,
        recent4h,
        "BTCUSDT"
      );
      
      if (!prediction) {
        return res.json({ 
          available: false, 
          prediction: null,
          message: "Ensemble prediction failed"
        });
      }
      
      // Pass through GPU response directly - matches EnsemblePrediction interface
      res.json({ available: true, prediction, dataSource });
    } catch (error) {
      console.error("[GPU Ensemble] Current prediction error:", error);
      res.json({ 
        available: false, 
        prediction: null, 
        message: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Track last closed candle timestamp for cache invalidation
  let lastClosedCandleTs: number | null = null;
  let lastInputHash: string | null = null;
  
  // Neural Network Quantile Prediction endpoint - returns Entry/SL/TP derived from quantiles
  app.get("/api/gpu/nn-prediction", async (req, res) => {
    const startTime = Date.now();
    const predictionId = crypto.randomUUID();
    const serverTs = new Date().toISOString();
    
    try {
      // Import cone signal generator for multi-step band generation
      const { coneSignalGenerator } = await import("./cone-signal-generator");
      
      // Check if GPU is available
      const health = await gpuBridge.checkHealth();
      if (!health) {
        return res.json({ 
          available: false, 
          prediction: null,
          predictedCandles: [],
          error: "GPU trainer not connected",
          trace: { prediction_id: predictionId, server_ts: serverTs }
        });
      }
      
      // ============================================================
      // USE ENSEMBLE PREDICTOR (same as /api/gpu/ensemble/current)
      // This uses the correct 66 MTF features that models were trained on
      // ============================================================
      
      // Helper to get latest N candles from database by timeframe
      const getDbCandles = async (timeframe: string, limit: number = 200) => {
        const result = await db.select()
          .from(candles)
          .where(and(eq(candles.symbol, "BTCUSDT"), eq(candles.timeframe, timeframe)))
          .orderBy(desc(candles.timestamp))  // Get most recent first
          .limit(limit);
        return result.reverse();  // Reverse to chronological order (oldest to newest)
      };
      
      type CandleData = { timestamp: number; open: number; high: number; low: number; close: number; volume: number };
      
      // Try live API first, fallback to database if blocked (HTTP 451)
      let mtfCandles: { m5: CandleData[]; m15: CandleData[]; h1: CandleData[]; h4: CandleData[] };
      
      try {
        const liveCandles = await getMultiTimeframeKlines("BTCUSDT");
        if (liveCandles.m15.length > 50) {
          mtfCandles = liveCandles;
        } else {
          throw new Error("Insufficient live data");
        }
      } catch (liveError) {
        console.log("[GPU NN] Live API failed, using database candles");
        
        const [db5m, db15m, db1h, db4h] = await Promise.all([
          getDbCandles("5m", 300),
          getDbCandles("15m", 200),
          getDbCandles("1h", 100),
          getDbCandles("4h", 50)
        ]);
        
        mtfCandles = { m5: db5m, m15: db15m, h1: db1h, h4: db4h };
      }
      
      // Validate we have enough data
      if (mtfCandles.m15.length < 50) {
        return res.json({ 
          available: false, 
          prediction: null,
          predictedCandles: [],
          error: "Not enough candle data available",
          trace: { prediction_id: predictionId, server_ts: serverTs }
        });
      }
      
      // Determine last CLOSED 15m candle (exclude current incomplete candle)
      // A candle is closed if its timestamp + 15min <= current time
      const now = Date.now();
      const candleInterval = 15 * 60 * 1000;
      const closedCandles = mtfCandles.m15.filter(c => c.timestamp + candleInterval <= now);
      
      if (closedCandles.length < 50) {
        return res.json({ 
          available: false, 
          prediction: null,
          predictedCandles: [],
          error: "Not enough closed candles available",
          trace: { prediction_id: predictionId, server_ts: serverTs }
        });
      }
      
      // Use only closed candles for prediction
      const inputCandles = closedCandles.slice(-200);
      const currentClosedTs = inputCandles[inputCandles.length - 1].timestamp;
      
      // Compute input hash for cache detection
      const inputData = inputCandles.map(c => `${c.timestamp},${c.open},${c.high},${c.low},${c.close},${c.volume}`).join("|");
      const inputHash = crypto.createHash("sha1").update(inputData).digest("hex").substring(0, 12);
      
      // Check if this is the same closed candle as last request
      const cacheHit = lastClosedCandleTs === currentClosedTs && lastInputHash === inputHash;
      const candleUnchanged = lastClosedCandleTs === currentClosedTs;
      
      // Update tracking
      lastClosedCandleTs = currentClosedTs;
      lastInputHash = inputHash;
      
      // Call ensemble predictor via GPU bridge
      const ensembleResult = await gpuBridge.predictEnsembleFromCandles(
        inputCandles,
        mtfCandles.m5.filter(c => c.timestamp + 5 * 60 * 1000 <= now).slice(-300),
        mtfCandles.h1.filter(c => c.timestamp + 60 * 60 * 1000 <= now).slice(-100),
        mtfCandles.h4.filter(c => c.timestamp + 4 * 60 * 60 * 1000 <= now).slice(-50),
        "BTCUSDT"
      );
      
      if (!ensembleResult || !ensembleResult.quantiles) {
        return res.json({ 
          available: false, 
          prediction: null,
          predictedCandles: [],
          error: "Neural network prediction failed - no quantiles available",
          trace: { prediction_id: predictionId, server_ts: serverTs, input_hash: inputHash }
        });
      }
      
      // Get current price from most recent closed candle
      const currentPrice = inputCandles[inputCandles.length - 1].close;
      const lastTimestamp = inputCandles[inputCandles.length - 1].timestamp;
      
      // Extract direction and confidence from ensemble
      const direction = ensembleResult.action as "LONG" | "SHORT" | "HOLD";
      const isLong = direction === "LONG";
      const isHold = direction === "HOLD";
      
      // Sanitize quantiles to reasonable bounds (-50% to +50%)
      const clamp = (val: number, min: number, max: number) => Math.max(min, Math.min(max, val));
      const sanitizedQuantiles = {
        q10: clamp(ensembleResult.quantiles.q10, -0.5, 0.5),
        q25: clamp(ensembleResult.quantiles.q25, -0.5, 0.5),
        q50: clamp(ensembleResult.quantiles.q50, -0.5, 0.5),
        q75: clamp(ensembleResult.quantiles.q75, -0.5, 0.5),
        q90: clamp(ensembleResult.quantiles.q90, -0.5, 0.5),
      };
      
      const entry = currentPrice;
      
      // Derive SL/TP from quantiles
      let stopLoss: number;
      let takeProfit: number;
      
      if (isHold) {
        stopLoss = currentPrice * (1 + sanitizedQuantiles.q10);
        takeProfit = currentPrice * (1 + sanitizedQuantiles.q90);
      } else if (isLong) {
        stopLoss = currentPrice * (1 + sanitizedQuantiles.q10);
        takeProfit = currentPrice * (1 + sanitizedQuantiles.q90);
      } else {
        stopLoss = currentPrice * (1 + sanitizedQuantiles.q90);
        takeProfit = currentPrice * (1 + sanitizedQuantiles.q10);
      }
      
      // Risk/Reward ratio
      const risk = Math.abs(entry - stopLoss);
      const reward = Math.abs(takeProfit - entry);
      const riskReward = risk > 0 ? reward / risk : 0;
      
      // Get confidence and probs from ensemble
      const confidence = ensembleResult.confidence;
      const probs = ensembleResult.ensemble_probs || { LONG: 0.33, SHORT: 0.33, HOLD: 0.34 };
      
      // Generate predicted candles using multi-step band (upgraded from simple cone)
      // Uses model's candle head for near-term (1-5), quantile cone for far-term (6-16)
      const intervalMs = 15 * 60 * 1000; // 15 minutes
      
      // Use multi-step band generator for more accurate forecasts
      // Alpha-shaping: expansion=1.5, neutral=1.0, contraction=0.7
      const volState = ensembleResult?.vol_state || "neutral";
      
      const bandPaths = coneSignalGenerator.generateMultiStepBand(
        currentPrice,
        sanitizedQuantiles,
        undefined,  // Candle head predictions not yet in API response
        volState as "contraction" | "neutral" | "expansion"
      );
      
      // Build predictedCandles array with multi-step band values
      const predictedCandles = [];
      for (let i = 0; i < 16; i++) {
        const q10 = bandPaths.q10[i];
        const q50 = bandPaths.q50[i];
        const q90 = bandPaths.q90[i];
        
        // Interpolate q25/q75 from q10/q50/q90
        const q25 = q10 + (q50 - q10) * 0.4;
        const q75 = q50 + (q90 - q50) * 0.6;
        
        predictedCandles.push({
          timestamp: lastTimestamp + ((i + 1) * intervalMs),
          q10,
          q25,
          q50,
          q75,
          q90,
          direction: (q50 >= currentPrice ? "up" : "down") as "up" | "down"
        });
      }
      
      // Compute output hash for tracking
      const outputData = `${sanitizedQuantiles.q10},${sanitizedQuantiles.q50},${sanitizedQuantiles.q90},${direction},${volState}`;
      const outputHash = crypto.createHash("sha1").update(outputData).digest("hex").substring(0, 12);
      
      const modelRunMs = Date.now() - startTime;
      
      const prediction = {
        action: direction,
        confidence,
        entry,
        stopLoss,
        takeProfit,
        riskReward,
        expectedMove: sanitizedQuantiles.q50,
        uncertainty: sanitizedQuantiles.q90 - sanitizedQuantiles.q10,
        quantiles: sanitizedQuantiles,
        directionProbs: probs,
        horizon: "4 hours (16 x 15m bars)",
        timestamp: Date.now(),
        currentPrice,
        units: "decimal_return" as const,
        derived_low_price: currentPrice * (1 + sanitizedQuantiles.q10),
        derived_high_price: currentPrice * (1 + sanitizedQuantiles.q90),
      };
      
      // Trace fields for debugging refresh/cache issues
      const trace = {
        prediction_id: predictionId,
        server_ts: serverTs,
        last_closed_candle_ts: new Date(currentClosedTs).toISOString(),
        window_start_ts: new Date(inputCandles[0].timestamp).toISOString(),
        window_end_ts: new Date(currentClosedTs).toISOString(),
        input_hash: inputHash,
        output_hash: outputHash,
        cache_hit: cacheHit,
        candle_unchanged: candleUnchanged,
        model_run_ms: modelRunMs,
        message: candleUnchanged ? "No new closed candle yet; prediction unchanged." : "New candle processed."
      };
      
      console.log(`[GPU NN] Prediction: ${direction} @ ${confidence.toFixed(2)} conf, q50=${(sanitizedQuantiles.q50 * 100).toFixed(2)}% | hash=${inputHash} cache=${cacheHit}`);
      
      res.json({ available: true, prediction, predictedCandles, trace });
    } catch (error) {
      console.error("[GPU NN] Prediction error:", error);
      res.json({ 
        available: false, 
        prediction: null,
        predictedCandles: [],
        error: error instanceof Error ? error.message : "Unknown error",
        trace: { prediction_id: predictionId, server_ts: serverTs }
      });
    }
  });

  // ============ MULTIHEAD PREDICTION ENDPOINT ============
  // Canonical endpoint for multi-head model inference
  // Returns ALL 6 heads: direction, μ/σ, quantiles, entry/SL/TP, candles
  app.get("/api/gpu/multihead/current", async (req, res) => {
    try {
      // Get MOST RECENT candle data from database (order desc, then reverse for chronological)
      const getDbCandles = async (timeframe: string, limit: number = 200) => {
        const result = await db.select()
          .from(candles)
          .where(and(eq(candles.symbol, "BTCUSDT"), eq(candles.timeframe, timeframe)))
          .orderBy(desc(candles.timestamp))  // Get most recent first
          .limit(limit);
        return result.reverse();  // Reverse to chronological order (oldest first)
      };
      
      const candleData = await getDbCandles("15m", 200);
      
      if (candleData.length < 100) {
        return res.json({
          available: false,
          prediction: null,
          error: "Not enough candle data (need 100+, have " + candleData.length + ")"
        });
      }
      
      // Format candles for GPU trainer (chronological order, oldest to newest)
      const formattedCandles = candleData.map((c: any) => ({
        timestamp: c.timestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume
      }));
      
      console.log(`[GPU Multihead] Using ${formattedCandles.length} candles, latest: ${new Date(formattedCandles[formattedCandles.length - 1]?.timestamp).toISOString()}`);
      
      // Call multihead endpoint
      const prediction = await gpuBridge.predictMultiheadFromCandles(formattedCandles);
      
      if (!prediction) {
        return res.json({
          available: false,
          prediction: null,
          error: "Multihead prediction failed - GPU trainer not connected or no multihead model"
        });
      }
      
      // Return multihead prediction with signal format
      const signal = {
        signal: prediction.action,
        confidence: prediction.confidence,
        expectedMove: prediction.expected_return,
        costs: 0.001,
        edge: prediction.edge,
        regime: "unknown",
        riskMode: "normal",
        topFeatures: [],
        mu: prediction.expected_return,
        sigma: prediction.uncertainty,
        positionSizePct: prediction.position_size_pct / 100,
        stopLossPct: prediction.stop_loss_pct,
        takeProfitPct: prediction.take_profit_pct,
        urgency: prediction.urgency.toLowerCase() as "low" | "medium" | "high",
        suggestedOrderType: prediction.suggested_order_type.toLowerCase() as "limit" | "market",
        isMultihead: prediction.is_multihead,
        entryOffsetPct: prediction.entry_offset_pct,
        entryPrice: prediction.entry_price,
        stopLossPrice: prediction.stop_loss_price,
        takeProfitPrice: prediction.take_profit_price,
        quantiles: prediction.quantiles,
        predictedCandles: prediction.predicted_candles,
        riskRewardRatio: prediction.risk_reward_ratio,
        isLearnedLevels: prediction.is_multihead,
        modelName: prediction.model_name,
        reasons: prediction.reasons
      };
      
      // Auto-save multihead prediction to database
      try {
        const q = prediction.quantiles;
        await storage.recordMultiheadPrediction({
          timestamp: Date.now(),
          action: prediction.action,
          probLong: prediction.direction_probs?.LONG ?? 0,
          probShort: prediction.direction_probs?.SHORT ?? 0,
          probHold: prediction.direction_probs?.HOLD ?? 0,
          confidence: prediction.confidence,
          q10: q?.q10 ?? null,
          q25: q?.q25 ?? null,
          q50: q?.q50 ?? null,
          q75: q?.q75 ?? null,
          q90: q?.q90 ?? null,
          mu: prediction.expected_return ?? null,
          sigma: prediction.uncertainty ?? null,
          edge: prediction.edge ?? null,
          entryPrice: prediction.entry_price ?? null,
          stopLossPrice: prediction.stop_loss_price ?? null,
          takeProfitPrice: prediction.take_profit_price ?? null,
          stopLossPct: prediction.stop_loss_pct ?? null,
          takeProfitPct: prediction.take_profit_pct ?? null,
          riskRewardRatio: prediction.risk_reward_ratio ?? null,
          positionSizePct: prediction.position_size_pct != null ? prediction.position_size_pct / 100 : null,
          currentPrice: prediction.current_price ?? null,
          modelName: prediction.model_name ?? null,
          isMultihead: prediction.is_multihead ?? false,
          urgency: prediction.urgency?.toLowerCase() ?? null,
          suggestedOrderType: prediction.suggested_order_type?.toLowerCase() ?? null,
          reasons: prediction.reasons ?? null,
          createdAt: Date.now(),
        });
      } catch (saveError) {
        console.error("[GPU Multihead] Failed to auto-save prediction:", saveError);
      }
      
      res.json({
        available: true,
        prediction: signal,
        currentPrice: prediction.current_price,
        isMultihead: prediction.is_multihead,
        units: "decimal_return",
        derived_low_price: prediction.current_price * (1 + prediction.quantiles.q10),
        derived_high_price: prediction.current_price * (1 + prediction.quantiles.q90)
      });
      
    } catch (error) {
      console.error("[GPU Multihead] Prediction error:", error);
      res.json({
        available: false,
        prediction: null,
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Data Proxy Endpoints - Allow local GPU trainer to fetch Binance data through Replit
  const BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3",
    "https://data-api.binance.vision/api/v3",
    "https://api1.binance.com/api/v3",
    "https://api2.binance.com/api/v3",
  ];

  async function fetchBinanceKlines(params: URLSearchParams): Promise<{ data: any[]; source: string }> {
    for (const baseUrl of BINANCE_ENDPOINTS) {
      try {
        const url = `${baseUrl}/klines?${params.toString()}`;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10000);
        const response = await fetch(url, { signal: controller.signal });
        clearTimeout(timeout);

        if (!response.ok) {
          console.warn(`[Data Proxy] ${baseUrl} returned HTTP ${response.status}`);
          continue;
        }

        const text = await response.text();
        if (!text || text.length < 3) {
          console.warn(`[Data Proxy] ${baseUrl} returned empty response (${text.length} bytes)`);
          continue;
        }

        const data = JSON.parse(text);
        if (!Array.isArray(data) || data.length === 0) {
          console.warn(`[Data Proxy] ${baseUrl} returned empty array`);
          continue;
        }

        return { data, source: baseUrl };
      } catch (e: any) {
        console.warn(`[Data Proxy] ${baseUrl} failed: ${e.message}`);
        continue;
      }
    }
    throw new Error("All Binance endpoints failed");
  }

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
      
      const { data, source } = await fetchBinanceKlines(params);
      
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
      
      console.log(`[Data Proxy] Fetched ${candles.length} candles for ${symbol} ${interval} via ${source}`);
      res.json({ candles, count: candles.length, symbol, interval });
    } catch (error: any) {
      console.error("[Data Proxy] Error fetching klines:", error.message);
      res.status(502).json({ error: "Failed to fetch klines from all Binance endpoints" });
    }
  });
  
  app.get("/api/data/orderbook", async (req, res) => {
    try {
      const { symbol = "BTCUSDT", limit = "100" } = req.query;
      
      for (const baseUrl of BINANCE_ENDPOINTS) {
        try {
          const url = `${baseUrl}/depth?symbol=${symbol}&limit=${Math.min(Number(limit), 1000)}`;
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 10000);
          const response = await fetch(url, { signal: controller.signal });
          clearTimeout(timeout);
          if (response.ok) {
            const data = await response.json();
            return res.json(data);
          }
        } catch { continue; }
      }
      res.status(502).json({ error: "Failed to fetch orderbook from all endpoints" });
    } catch (error) {
      console.error("[Data Proxy] Error fetching orderbook:", error);
      res.status(500).json({ error: "Failed to fetch orderbook" });
    }
  });
  
  app.get("/api/data/ticker", async (req, res) => {
    try {
      const { symbol = "BTCUSDT" } = req.query;
      
      for (const baseUrl of BINANCE_ENDPOINTS) {
        try {
          const url = `${baseUrl}/ticker/24hr?symbol=${symbol}`;
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 10000);
          const response = await fetch(url, { signal: controller.signal });
          clearTimeout(timeout);
          if (response.ok) {
            const data = await response.json();
            return res.json(data);
          }
        } catch { continue; }
      }
      res.status(502).json({ error: "Failed to fetch ticker from all endpoints" });
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
          const url = `${BINANCE_ENDPOINTS[0]}/ticker/24hr?symbol=${symbol}`;
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
          const url = `${BINANCE_ENDPOINTS[0]}/klines?symbol=${symbol}&interval=15m&limit=100`;
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

  // ============================================================================
  // LIVE SYSTEM ENDPOINTS — Trade records, learning stats, cycle logs
  // ============================================================================

  app.post("/api/live/trade", async (req, res) => {
    try {
      const t = req.body;
      if (!t || !t.symbol || !t.side || !t.entry_price) {
        return res.status(400).json({ error: "symbol, side, and entry_price are required" });
      }
      const record = await storage.recordLiveTradeRecord({
        symbol: t.symbol,
        side: t.side,
        entryTime: t.entry_time || Date.now(),
        entryPrice: t.entry_price,
        exitTime: t.exit_time ?? null,
        exitPrice: t.exit_price ?? null,
        stopLoss: t.stop_loss ?? null,
        takeProfit: t.take_profit ?? null,
        sizePct: t.size_pct ?? null,
        pEnter: t.p_enter ?? null,
        costsBps: t.costs_bps ?? null,
        outcome: t.outcome ?? null,
        grossR: t.gross_r ?? null,
        netR: t.net_r ?? null,
        sizedR: t.sized_r ?? null,
        status: t.status || "open",
        reasons: t.reasons ?? [],
        createdAt: Date.now(),
      });
      console.log(`[Live Trade] Recorded ${t.side} ${t.symbol} @ ${t.entry_price} (id=${record.id})`);
      res.json({ success: true, id: record.id });
    } catch (error) {
      console.error("[Live Trade] Error:", error);
      res.status(500).json({ error: "Failed to record trade" });
    }
  });

  app.patch("/api/live/trade/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const update = req.body;
      await storage.updateLiveTradeRecord(id, {
        exitTime: update.exit_time ?? undefined,
        exitPrice: update.exit_price ?? undefined,
        outcome: update.outcome ?? undefined,
        grossR: update.gross_r ?? undefined,
        netR: update.net_r ?? undefined,
        sizedR: update.sized_r ?? undefined,
        status: update.status ?? undefined,
      });
      console.log(`[Live Trade] Updated trade ${id}: ${update.outcome ?? update.status}`);
      res.json({ success: true });
    } catch (error) {
      console.error("[Live Trade] Update error:", error);
      res.status(500).json({ error: "Failed to update trade" });
    }
  });

  app.get("/api/live/trades", async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 100;
      const symbol = req.query.symbol as string | undefined;
      const trades = await storage.getLiveTradeRecords(limit, symbol);
      res.json(trades);
    } catch (error) {
      console.error("[Live Trades] Error:", error);
      res.status(500).json({ error: "Failed to get trades" });
    }
  });

  app.get("/api/live/trades/open", async (req, res) => {
    try {
      const trades = await storage.getOpenLiveTradeRecords();
      res.json(trades);
    } catch (error) {
      console.error("[Live Trades Open] Error:", error);
      res.status(500).json({ error: "Failed to get open trades" });
    }
  });

  app.post("/api/live/learning-stats", async (req, res) => {
    try {
      const s = req.body;
      if (!s || !s.symbol || !s.model_version) {
        return res.status(400).json({ error: "symbol and model_version are required" });
      }
      const record = await storage.recordModelLearningStats({
        symbol: s.symbol,
        modelVersion: s.model_version,
        trainedUntilTs: s.trained_until_ts ?? null,
        trainingSamples: s.training_samples ?? null,
        valPrAuc: s.val_pr_auc ?? null,
        valPrecision: s.val_precision ?? null,
        valRecall: s.val_recall ?? null,
        valF1: s.val_f1 ?? null,
        bestPolicyThreshold: s.best_policy_threshold ?? null,
        bestPolicyCooldown: s.best_policy_cooldown ?? null,
        bestPolicyTpMult: s.best_policy_tp_mult ?? null,
        bestPolicySlMult: s.best_policy_sl_mult ?? null,
        pfNet: s.pf_net ?? null,
        eNet: s.e_net ?? null,
        tradesPerDay: s.trades_per_day ?? null,
        profitableRegimes: s.profitable_regimes ?? null,
        totalRegimes: s.total_regimes ?? null,
        promoted: s.promoted ?? false,
        promotionReason: s.promotion_reason ?? null,
        trend7d: s.trend_7d ?? null,
        prevPfNet: s.prev_pf_net ?? null,
        prevENet: s.prev_e_net ?? null,
        prevTradesPerDay: s.prev_trades_per_day ?? null,
        createdAt: Date.now(),
      });
      console.log(`[Learning Stats] ${s.symbol} v${s.model_version} PR-AUC=${s.val_pr_auc?.toFixed(3)} PF=${s.pf_net?.toFixed(2)} promoted=${s.promoted}`);
      res.json({ success: true, id: record.id });
    } catch (error) {
      console.error("[Learning Stats] Error:", error);
      res.status(500).json({ error: "Failed to record learning stats" });
    }
  });

  app.get("/api/live/learning-stats", async (req, res) => {
    try {
      const symbol = req.query.symbol as string | undefined;
      const limit = parseInt(req.query.limit as string) || 50;
      const stats = await storage.getModelLearningStats(symbol, limit);
      res.json(stats);
    } catch (error) {
      console.error("[Learning Stats] Error:", error);
      res.status(500).json({ error: "Failed to get learning stats" });
    }
  });

  app.get("/api/live/learning-stats/latest", async (req, res) => {
    try {
      const symbols = (req.query.symbols as string || "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT").split(",");
      const results: Record<string, ModelLearningStatsEntry | null> = {};
      for (const sym of symbols) {
        results[sym.trim()] = await storage.getLatestModelLearningStats(sym.trim());
      }
      res.json(results);
    } catch (error) {
      console.error("[Learning Stats Latest] Error:", error);
      res.status(500).json({ error: "Failed to get latest stats" });
    }
  });

  app.post("/api/live/cycle-log", async (req, res) => {
    try {
      const c = req.body;
      if (!c || !c.symbol || !c.decision) {
        return res.status(400).json({ error: "symbol and decision are required" });
      }
      const record = await storage.recordLiveCycleLog({
        symbol: c.symbol,
        cycleTs: c.cycle_ts || Date.now(),
        price: c.price ?? null,
        pEnter: c.p_enter ?? null,
        htfH1Trend: c.htf_h1_trend ?? null,
        htfH4Trend: c.htf_h4_trend ?? null,
        slopeOk: c.slope_ok ?? null,
        rangeOk: c.range_ok ?? null,
        direction: c.direction ?? null,
        thresholdUsed: c.threshold_used ?? null,
        decision: c.decision,
        reasons: c.reasons ?? [],
        createdAt: Date.now(),
      });
      res.json({ success: true, id: record.id });
    } catch (error) {
      console.error("[Cycle Log] Error:", error);
      res.status(500).json({ error: "Failed to record cycle log" });
    }
  });

  app.get("/api/live/cycle-logs", async (req, res) => {
    try {
      const symbol = req.query.symbol as string | undefined;
      const limit = parseInt(req.query.limit as string) || 100;
      const logs = await storage.getLiveCycleLogs(symbol, limit);
      res.json(logs);
    } catch (error) {
      console.error("[Cycle Logs] Error:", error);
      res.status(500).json({ error: "Failed to get cycle logs" });
    }
  });

  app.get("/api/live/summary", async (req, res) => {
    try {
      const symbols = (req.query.symbols as string || "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT").split(",").map(s => s.trim());
      const openTrades = await storage.getOpenLiveTradeRecords();
      const recentTrades = await storage.getLiveTradeRecords(50);
      const learningStats: Record<string, ModelLearningStatsEntry | null> = {};
      for (const sym of symbols) {
        learningStats[sym] = await storage.getLatestModelLearningStats(sym);
      }
      const closedTrades = recentTrades.filter(t => t.status === "closed");
      const wins = closedTrades.filter(t => (t.netR ?? 0) > 0);
      const totalNetR = closedTrades.reduce((s, t) => s + (t.netR ?? 0), 0);
      res.json({
        openPositions: openTrades.length,
        openTrades: openTrades.map(t => ({ symbol: t.symbol, side: t.side, entryPrice: t.entryPrice, pEnter: t.pEnter })),
        closedTradesCount: closedTrades.length,
        winRate: closedTrades.length > 0 ? (wins.length / closedTrades.length * 100).toFixed(1) : "N/A",
        totalNetR: totalNetR.toFixed(2),
        avgNetR: closedTrades.length > 0 ? (totalNetR / closedTrades.length).toFixed(3) : "N/A",
        learningStats,
        symbols,
      });
    } catch (error) {
      console.error("[Live Summary] Error:", error);
      res.status(500).json({ error: "Failed to get summary" });
    }
  });

  return httpServer;
}
