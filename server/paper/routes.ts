import { Router } from "express";
import * as storage from "./storage";
import * as engine from "./engine";
import { 
  getConfig, 
  updateConfig, 
  resetConfig, 
  startAutoTrading, 
  stopAutoTrading, 
  enablePaperTrading,
  disablePaperTrading,
  isPaperTradingEnabled,
  isAutoTradingEnabled 
} from "./config";

const router = Router();

router.get("/portfolio", async (req, res) => {
  try {
    const summary = await engine.getPortfolioSummary();
    res.json(summary);
  } catch (error) {
    console.error("Error getting portfolio:", error);
    res.status(500).json({ error: "Failed to get portfolio" });
  }
});

router.get("/positions", async (req, res) => {
  try {
    const status = req.query.status as "OPEN" | "CLOSED" | undefined;
    const limit = parseInt(req.query.limit as string) || 100;
    const positions = await storage.getPositions(status, limit);
    res.json(positions);
  } catch (error) {
    console.error("Error getting positions:", error);
    res.status(500).json({ error: "Failed to get positions" });
  }
});

router.get("/trades", async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 500;
    const trades = await storage.getTrades(limit);
    res.json(trades);
  } catch (error) {
    console.error("Error getting trades:", error);
    res.status(500).json({ error: "Failed to get trades" });
  }
});

router.get("/equity", async (req, res) => {
  try {
    const range = req.query.range as "7d" | "30d" | "all" | undefined;
    const curve = await storage.getEquityCurve(range);
    res.json(curve);
  } catch (error) {
    console.error("Error getting equity curve:", error);
    res.status(500).json({ error: "Failed to get equity curve" });
  }
});

router.get("/config", (req, res) => {
  try {
    res.json(getConfig());
  } catch (error) {
    console.error("Error getting config:", error);
    res.status(500).json({ error: "Failed to get config" });
  }
});

router.post("/config", (req, res) => {
  try {
    const updates = req.body;
    const newConfig = updateConfig(updates);
    res.json(newConfig);
  } catch (error) {
    console.error("Error updating config:", error);
    res.status(500).json({ error: "Failed to update config" });
  }
});

router.post("/reset", async (req, res) => {
  try {
    resetConfig();
    const portfolio = await storage.resetPortfolio();
    res.json({ message: "Paper trading reset", portfolio });
  } catch (error) {
    console.error("Error resetting paper trading:", error);
    res.status(500).json({ error: "Failed to reset paper trading" });
  }
});

router.post("/enable", async (req, res) => {
  try {
    await enablePaperTrading();
    console.log("[Paper] Paper trading ENABLED - system can now execute trades");
    res.json({ 
      message: "Paper trading enabled", 
      paperTradingEnabled: true,
      isAutoTrading: isAutoTradingEnabled()
    });
  } catch (error) {
    console.error("Error enabling paper trading:", error);
    res.status(500).json({ error: "Failed to enable paper trading" });
  }
});

router.post("/disable", async (req, res) => {
  try {
    await disablePaperTrading();
    console.log("[Paper] Paper trading DISABLED - no trades will execute");
    res.json({ 
      message: "Paper trading disabled", 
      paperTradingEnabled: false,
      isAutoTrading: false
    });
  } catch (error) {
    console.error("Error disabling paper trading:", error);
    res.status(500).json({ error: "Failed to disable paper trading" });
  }
});

router.post("/start", async (req, res) => {
  try {
    if (!isPaperTradingEnabled()) {
      res.status(400).json({ 
        error: "Paper trading is not enabled. Call /api/paper/enable first.",
        paperTradingEnabled: false,
        isAutoTrading: false
      });
      return;
    }
    await startAutoTrading();
    console.log("[Paper] Auto-trading started");
    res.json({ 
      message: "Auto-trading started", 
      isAutoTrading: true,
      paperTradingEnabled: true
    });
  } catch (error) {
    console.error("Error starting auto-trading:", error);
    res.status(500).json({ error: "Failed to start auto-trading" });
  }
});

router.post("/stop", async (req, res) => {
  try {
    await stopAutoTrading();
    console.log("[Paper] Auto-trading stopped");
    res.json({ 
      message: "Auto-trading stopped", 
      isAutoTrading: false,
      paperTradingEnabled: isPaperTradingEnabled()
    });
  } catch (error) {
    console.error("Error stopping auto-trading:", error);
    res.status(500).json({ error: "Failed to stop auto-trading" });
  }
});

router.get("/audit", async (req, res) => {
  try {
    const auditLogs = engine.getAuditLog();
    res.json(auditLogs);
  } catch (error) {
    console.error("Error getting audit log:", error);
    res.status(500).json({ error: "Failed to get audit log" });
  }
});

router.get("/status", async (req, res) => {
  try {
    const config = getConfig();
    res.json({
      paperTradingEnabled: config.paperTradingEnabled,
      isAutoTrading: isAutoTradingEnabled(),
      config: {
        riskPerTradePct: config.riskPerTradePct,
        maxRiskPerTradePct: config.maxRiskPerTradePct,
        maxAccountExposurePct: config.maxAccountExposurePct,
        minConfidence: config.minConfidence,
        atrStopMultiplier: config.atrStopMultiplier,
        minStopDistancePct: config.minStopDistancePct,
      }
    });
  } catch (error) {
    console.error("Error getting status:", error);
    res.status(500).json({ error: "Failed to get status" });
  }
});

export default router;
