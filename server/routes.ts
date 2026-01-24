import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import paperRoutes from "./paper/routes";

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

  return httpServer;
}
