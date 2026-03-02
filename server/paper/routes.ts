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
    const symbol = req.query.symbol as string | undefined;
    const limit = parseInt(req.query.limit as string) || 100;
    let positions;
    if (symbol) {
      positions = await storage.getPositionsBySymbol(symbol, status, limit);
    } else {
      positions = await storage.getPositions(status, limit);
    }

    const enriched = await Promise.all(positions.map(async (pos) => {
      const currentPrice = pos.status === "OPEN" ? await engine.getMarketPrice(pos.symbol) : null;
      let pnlR = 0;
      let pnlUsdt = 0;
      if (currentPrice && pos.status === "OPEN") {
        const priceDiff = pos.side === "LONG"
          ? currentPrice - pos.entryPrice
          : pos.entryPrice - currentPrice;
        pnlUsdt = priceDiff * pos.qty;
        pnlR = pos.initialRiskUsdt ? pnlUsdt / pos.initialRiskUsdt : 0;
      }
      return {
        ...pos,
        currentPrice,
        pnlR: Math.round(pnlR * 100) / 100,
        pnlUsdt: Math.round(pnlUsdt * 100) / 100,
        takeProfit: pos.tp1,
        entryTime: pos.entryTs,
      };
    }));

    res.json(enriched);
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

router.get("/trade-history", async (req, res) => {
  try {
    const symbol = req.query.symbol as string | undefined;
    const limit = parseInt(req.query.limit as string) || 100;
    const offset = parseInt(req.query.offset as string) || 0;
    const history = await storage.getTradeHistory({ symbol, limit, offset });
    res.json(history);
  } catch (error) {
    console.error("Error getting trade history:", error);
    res.status(500).json({ error: "Failed to get trade history" });
  }
});

router.get("/equity-curve", async (req, res) => {
  try {
    const range = req.query.range as string || "all";
    let fromTs = 0;
    if (range === "7d") fromTs = Date.now() - 7 * 86400000;
    else if (range === "30d") fromTs = Date.now() - 30 * 86400000;

    const history = await storage.getTradeHistory({ limit: 10000 });
    const sorted = history
      .filter((t: any) => t.exitTs && (fromTs === 0 || t.exitTs >= fromTs))
      .sort((a: any, b: any) => (a.exitTs ?? 0) - (b.exitTs ?? 0));

    let cumR = 0;
    const curve = sorted.map((t: any) => {
      const r = t.netR ?? t.grossR ?? 0;
      cumR += r;
      return {
        ts: t.exitTs ?? t.entryTs,
        r: Math.round(cumR * 100) / 100,
        tradeR: Math.round(r * 100) / 100,
        symbol: t.symbol,
        side: t.side,
      };
    });

    res.json(curve);
  } catch (error) {
    console.error("Error getting paper equity curve:", error);
    res.status(500).json({ error: "Failed to get paper equity curve" });
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

router.post("/positions/:id/close", async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    if (isNaN(id)) { res.status(400).json({ error: "Invalid position ID" }); return; }
    const { exitPrice } = req.body || {};
    const result = await engine.manualClosePosition(id, exitPrice);
    res.json({ message: "Position closed", ...result });
  } catch (error: any) {
    console.error("Error closing position:", error);
    res.status(error.message?.includes("not found") ? 404 : 400).json({ error: error.message });
  }
});

router.post("/positions/:id/partial-close", async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    if (isNaN(id)) { res.status(400).json({ error: "Invalid position ID" }); return; }
    const { percent } = req.body || {};
    if (!percent || percent <= 0 || percent >= 100) {
      res.status(400).json({ error: "Percent must be between 1 and 99" }); return;
    }
    const result = await engine.manualPartialClose(id, percent);
    res.json({ message: "Partial close executed", ...result });
  } catch (error: any) {
    console.error("Error partial closing position:", error);
    res.status(error.message?.includes("not found") ? 404 : 400).json({ error: error.message });
  }
});

router.patch("/positions/:id/sl", async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    if (isNaN(id)) { res.status(400).json({ error: "Invalid position ID" }); return; }
    const { stopLoss } = req.body || {};
    if (stopLoss === undefined || typeof stopLoss !== "number") {
      res.status(400).json({ error: "stopLoss must be a number" }); return;
    }
    const updated = await engine.updatePositionLevels(id, { stopLoss });
    res.json(updated);
  } catch (error: any) {
    console.error("Error updating SL:", error);
    res.status(error.message?.includes("not found") ? 404 : 400).json({ error: error.message });
  }
});

router.patch("/positions/:id/tp", async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    if (isNaN(id)) { res.status(400).json({ error: "Invalid position ID" }); return; }
    const { tp1, tp2 } = req.body || {};
    if (tp1 === undefined && tp2 === undefined) {
      res.status(400).json({ error: "Provide at least tp1 or tp2" }); return;
    }
    const updates: { tp1?: number; tp2?: number } = {};
    if (tp1 !== undefined) updates.tp1 = tp1;
    if (tp2 !== undefined) updates.tp2 = tp2;
    const updated = await engine.updatePositionLevels(id, updates);
    res.json(updated);
  } catch (error: any) {
    console.error("Error updating TP:", error);
    res.status(error.message?.includes("not found") ? 404 : 400).json({ error: error.message });
  }
});

router.post("/manual-open", async (req, res) => {
  try {
    const { symbol, side, entryPrice, stopLoss, takeProfit, riskPercent } = req.body || {};
    if (!symbol || !side || !entryPrice || !stopLoss || !takeProfit || !riskPercent) {
      res.status(400).json({ error: "Missing required fields: symbol, side, entryPrice, stopLoss, takeProfit, riskPercent" }); return;
    }
    if (!["LONG", "SHORT"].includes(side)) {
      res.status(400).json({ error: "side must be LONG or SHORT" }); return;
    }
    const position = await engine.manualOpenPosition({
      symbol, side, entryPrice, stopLoss, takeProfit, riskPercent
    });
    res.json({ message: "Position opened", position });
  } catch (error: any) {
    console.error("Error opening manual position:", error);
    res.status(400).json({ error: error.message });
  }
});

router.get("/risk-alerts", async (req, res) => {
  try {
    const alerts = await engine.computeRiskAlerts();
    res.json(alerts);
  } catch (error: any) {
    console.error("Error computing risk alerts:", error);
    res.status(500).json({ error: "Failed to compute risk alerts" });
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
