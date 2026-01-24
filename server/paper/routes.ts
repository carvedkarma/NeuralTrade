import { Router } from "express";
import * as storage from "./storage";
import * as engine from "./engine";
import { getConfig, updateConfig, resetConfig, startAutoTrading, stopAutoTrading, isAutoTradingEnabled } from "./config";

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

router.post("/start", async (req, res) => {
  try {
    startAutoTrading();
    console.log("[Paper] Auto-trading started");
    res.json({ message: "Auto-trading started", isAutoTrading: true });
  } catch (error) {
    console.error("Error starting auto-trading:", error);
    res.status(500).json({ error: "Failed to start auto-trading" });
  }
});

router.post("/stop", async (req, res) => {
  try {
    stopAutoTrading();
    console.log("[Paper] Auto-trading stopped");
    res.json({ message: "Auto-trading stopped", isAutoTrading: false });
  } catch (error) {
    console.error("Error stopping auto-trading:", error);
    res.status(500).json({ error: "Failed to stop auto-trading" });
  }
});

export default router;
