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
  isAutoTradingEnabled,
  getAnalyticsClearedAfterTs,
  setAnalyticsClearedAfterTs,
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

router.get("/performance", async (req, res) => {
  try {
    const allTrades = await storage.getTradeHistory({ limit: 100000 });
    const trades = allTrades.sort((a, b) => (a.exitTs ?? 0) - (b.exitTs ?? 0));

    const getR = (t: typeof trades[0]) => t.netR ?? t.grossR ?? 0;

    const totalTrades = trades.length;
    const wins = trades.filter((t) => getR(t) > 0);
    const losses = trades.filter((t) => getR(t) <= 0);
    const rValues = trades.map(getR);
    const totalR = rValues.reduce((s, v) => s + v, 0);
    const winRate = totalTrades > 0 ? (wins.length / totalTrades) * 100 : 0;
    const avgWinR = wins.length > 0 ? wins.reduce((s, t) => s + getR(t), 0) / wins.length : 0;
    const avgLossR = losses.length > 0 ? losses.reduce((s, t) => s + getR(t), 0) / losses.length : 0;
    const grossWin = wins.reduce((s, t) => s + getR(t), 0);
    const grossLoss = Math.abs(losses.reduce((s, t) => s + getR(t), 0));
    const profitFactor = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? 999 : 0;
    const expectancy = totalTrades > 0 ? totalR / totalTrades : 0;

    const meanR = totalTrades > 0 ? totalR / totalTrades : 0;
    const variance = totalTrades > 1 ? rValues.reduce((s, v) => s + (v - meanR) ** 2, 0) / (totalTrades - 1) : 0;
    const stdDev = Math.sqrt(variance);
    const sharpeRatio = stdDev > 0 ? meanR / stdDev : 0;

    const downsideValues = rValues.filter((v) => v < 0);
    const downsideVariance = downsideValues.length > 0 ? downsideValues.reduce((s, v) => s + v ** 2, 0) / downsideValues.length : 0;
    const downsideStdDev = Math.sqrt(downsideVariance);
    const sortinoRatio = downsideStdDev > 0 ? meanR / downsideStdDev : 0;

    let maxConsecWins = 0, maxConsecLosses = 0, curWins = 0, curLosses = 0;
    const streaks: Array<{ type: "win" | "loss"; length: number; ts: number }> = [];
    let prevType: "win" | "loss" | null = null;
    let streakLen = 0;
    for (const t of trades) {
      const isWin = getR(t) > 0;
      if (isWin) { curWins++; curLosses = 0; if (curWins > maxConsecWins) maxConsecWins = curWins; }
      else { curLosses++; curWins = 0; if (curLosses > maxConsecLosses) maxConsecLosses = curLosses; }
      const curType = isWin ? "win" : "loss";
      if (curType === prevType) { streakLen++; }
      else {
        if (prevType !== null) streaks.push({ type: prevType, length: streakLen, ts: t.entryTs ?? 0 });
        streakLen = 1;
        prevType = curType;
      }
    }
    if (prevType !== null && trades.length > 0) {
      streaks.push({ type: prevType, length: streakLen, ts: trades[trades.length - 1].entryTs ?? 0 });
    }

    const symbolMap: Record<string, { trades: typeof trades; wins: number; totalR: number }> = {};
    for (const t of trades) {
      const sym = t.symbol ?? "UNKNOWN";
      if (!symbolMap[sym]) symbolMap[sym] = { trades: [], wins: 0, totalR: 0 };
      symbolMap[sym].trades.push(t);
      if (getR(t) > 0) symbolMap[sym].wins++;
      symbolMap[sym].totalR += getR(t);
    }
    const perSymbol = Object.entries(symbolMap).map(([symbol, s]) => ({
      symbol,
      trades: s.trades.length,
      wins: s.wins,
      winRate: s.trades.length > 0 ? (s.wins / s.trades.length) * 100 : 0,
      totalR: Math.round(s.totalR * 100) / 100,
      expectancy: s.trades.length > 0 ? Math.round((s.totalR / s.trades.length) * 10000) / 10000 : 0,
    }));

    const perSymbolEquity: Record<string, Array<{ ts: number; r: number; tradeR: number }>> = {};
    for (const [symbol, s] of Object.entries(symbolMap)) {
      let cum = 0;
      perSymbolEquity[symbol] = s.trades.map((t) => {
        const r = getR(t);
        cum += r;
        return { ts: t.exitTs ?? t.entryTs ?? 0, r: Math.round(cum * 100) / 100, tradeR: Math.round(r * 100) / 100 };
      });
    }

    let maxDrawdown = 0, peak = 0, cumR = 0;
    for (const t of trades) {
      cumR += getR(t);
      if (cumR > peak) peak = cumR;
      const dd = peak - cumR;
      if (dd > maxDrawdown) maxDrawdown = dd;
    }

    const bestTrade = rValues.length > 0 ? Math.max(...rValues) : 0;
    const worstTrade = rValues.length > 0 ? Math.min(...rValues) : 0;

    const totalBarsHeld = trades.reduce((s, t) => s + (t.barsHeld ?? 0), 0);
    const avgHoldBars = totalTrades > 0 ? totalBarsHeld / totalTrades : 0;
    const avgHoldMinutes = avgHoldBars * 15;

    const durationBins = [
      { label: "< 1h", min: 0, max: 4, count: 0 },
      { label: "1-3h", min: 4, max: 12, count: 0 },
      { label: "3-6h", min: 12, max: 24, count: 0 },
      { label: "6-12h", min: 24, max: 48, count: 0 },
      { label: "12-24h", min: 48, max: 96, count: 0 },
      { label: "1-3d", min: 96, max: 288, count: 0 },
      { label: "> 3d", min: 288, max: Infinity, count: 0 },
    ];
    for (const t of trades) {
      const bars = t.barsHeld ?? 0;
      for (const bin of durationBins) {
        if (bars >= bin.min && bars < bin.max) { bin.count++; break; }
      }
    }

    const hourlyPerf: Record<number, { trades: number; wins: number; totalR: number }> = {};
    for (let h = 0; h < 24; h++) hourlyPerf[h] = { trades: 0, wins: 0, totalR: 0 };
    for (const t of trades) {
      const hour = new Date(t.entryTs ?? 0).getUTCHours();
      hourlyPerf[hour].trades++;
      if (getR(t) > 0) hourlyPerf[hour].wins++;
      hourlyPerf[hour].totalR += getR(t);
    }
    const hourlyBreakdown = Object.entries(hourlyPerf).map(([hour, h]) => ({
      hour: parseInt(hour),
      trades: h.trades,
      wins: h.wins,
      winRate: h.trades > 0 ? Math.round((h.wins / h.trades) * 1000) / 10 : 0,
      totalR: Math.round(h.totalR * 100) / 100,
      avgR: h.trades > 0 ? Math.round((h.totalR / h.trades) * 1000) / 1000 : 0,
    }));

    const monthlyPnl: Record<string, { totalR: number; trades: number; wins: number }> = {};
    for (const t of trades) {
      const d = new Date(t.exitTs ?? t.entryTs ?? 0);
      const key = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
      if (!monthlyPnl[key]) monthlyPnl[key] = { totalR: 0, trades: 0, wins: 0 };
      monthlyPnl[key].totalR += getR(t);
      monthlyPnl[key].trades++;
      if (getR(t) > 0) monthlyPnl[key].wins++;
    }
    const monthlyData = Object.entries(monthlyPnl)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([month, m]) => ({
        month,
        totalR: Math.round(m.totalR * 100) / 100,
        trades: m.trades,
        wins: m.wins,
        winRate: m.trades > 0 ? Math.round((m.wins / m.trades) * 1000) / 10 : 0,
      }));

    const weeklyPnl: Record<string, { totalR: number; trades: number; wins: number }> = {};
    for (const t of trades) {
      const d = new Date(t.exitTs ?? t.entryTs ?? 0);
      const startOfWeek = new Date(d);
      startOfWeek.setUTCDate(d.getUTCDate() - d.getUTCDay());
      const key = `${startOfWeek.getUTCFullYear()}-${String(startOfWeek.getUTCMonth() + 1).padStart(2, "0")}-${String(startOfWeek.getUTCDate()).padStart(2, "0")}`;
      if (!weeklyPnl[key]) weeklyPnl[key] = { totalR: 0, trades: 0, wins: 0 };
      weeklyPnl[key].totalR += getR(t);
      weeklyPnl[key].trades++;
      if (getR(t) > 0) weeklyPnl[key].wins++;
    }
    const weeklyData = Object.entries(weeklyPnl)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([week, w]) => ({
        week,
        totalR: Math.round(w.totalR * 100) / 100,
        trades: w.trades,
        wins: w.wins,
        winRate: w.trades > 0 ? Math.round((w.wins / w.trades) * 1000) / 10 : 0,
      }));

    const now = Date.now();
    const computeRolling = (windowMs: number) => {
      const windowTrades = trades.filter((t) => (t.exitTs ?? t.entryTs ?? 0) >= now - windowMs);
      const wt = windowTrades.length;
      const wWins = windowTrades.filter((t) => getR(t) > 0).length;
      const wR = windowTrades.reduce((s, t) => s + getR(t), 0);
      return {
        trades: wt,
        wins: wWins,
        winRate: wt > 0 ? Math.round((wWins / wt) * 1000) / 10 : 0,
        totalR: Math.round(wR * 100) / 100,
        expectancy: wt > 0 ? Math.round((wR / wt) * 1000) / 1000 : 0,
      };
    };
    const rolling7d = computeRolling(7 * 86400000);
    const rolling30d = computeRolling(30 * 86400000);

    const leverageMap: Record<string, { trades: number; wins: number; totalR: number; totalPnlUsdt: number; leverageNum: number }> = {};
    for (const t of trades) {
      const leverageNum = (t as any).leverage ?? 1;
      const lev = `${leverageNum}x`;
      const key = lev;
      if (!leverageMap[key]) leverageMap[key] = { trades: 0, wins: 0, totalR: 0, totalPnlUsdt: 0, leverageNum };
      leverageMap[key].trades++;
      if (getR(t) > 0) leverageMap[key].wins++;
      leverageMap[key].totalR += getR(t);
      leverageMap[key].totalPnlUsdt += t.pnlUsdt ?? 0;
    }
    const leverageBreakdown = Object.entries(leverageMap)
      .sort((a, b) => a[1].leverageNum - b[1].leverageNum)
      .map(([tier, l]) => ({
        tier,
        leverageNum: l.leverageNum,
        trades: l.trades,
        wins: l.wins,
        winRate: l.trades > 0 ? Math.round((l.wins / l.trades) * 1000) / 10 : 0,
        totalR: Math.round(l.totalR * 100) / 100,
        totalPnlUsdt: Math.round(l.totalPnlUsdt * 100) / 100,
      }));

    res.json({
      totalTrades,
      wins: wins.length,
      losses: losses.length,
      winRate: Math.round(winRate * 100) / 100,
      totalR: Math.round(totalR * 100) / 100,
      avgWinR: Math.round(avgWinR * 10000) / 10000,
      avgLossR: Math.round(avgLossR * 10000) / 10000,
      profitFactor: Math.round(profitFactor * 100) / 100,
      maxDrawdown: Math.round(maxDrawdown * 100) / 100,
      bestTrade: Math.round(bestTrade * 10000) / 10000,
      worstTrade: Math.round(worstTrade * 10000) / 10000,
      expectancy: Math.round(expectancy * 10000) / 10000,
      sharpeRatio: Math.round(sharpeRatio * 100) / 100,
      sortinoRatio: Math.round(sortinoRatio * 100) / 100,
      maxConsecWins,
      maxConsecLosses,
      avgHoldBars: Math.round(avgHoldBars * 10) / 10,
      avgHoldMinutes: Math.round(avgHoldMinutes),
      perSymbol,
      perSymbolEquity,
      hourlyBreakdown,
      durationBins,
      streaks,
      monthlyData,
      weeklyData,
      rolling7d,
      rolling30d,
      leverageBreakdown,
      totalPnlUsdt: Math.round(trades.reduce((s, t) => s + (t.pnlUsdt ?? 0), 0) * 100) / 100,
      totalRiskUsdt: Math.round(trades.reduce((s, t) => s + (t.riskUsdt ?? 0), 0) * 100) / 100,
    });
  } catch (err: any) {
    console.error("Error computing paper performance:", err);
    res.status(500).json({ error: err.message });
  }
});

router.delete("/trade-history", async (req, res) => {
  try {
    await storage.clearTradeHistory();
    res.json({ cleared: true });
  } catch (error) {
    console.error("Error clearing trade history:", error);
    res.status(500).json({ error: "Failed to clear trade history" });
  }
});

router.delete("/equity-curve", async (req, res) => {
  try {
    await storage.clearEquityCurve();
    setAnalyticsClearedAfterTs(Date.now());
    res.json({ cleared: true });
  } catch (error) {
    console.error("Error clearing equity curve:", error);
    res.status(500).json({ error: "Failed to clear equity curve" });
  }
});

router.get("/equity-curve", async (req, res) => {
  try {
    const range = req.query.range as string || "all";
    let fromTs = 0;
    if (range === "7d") fromTs = Date.now() - 7 * 86400000;
    else if (range === "30d") fromTs = Date.now() - 30 * 86400000;

    const clearedAfter = getAnalyticsClearedAfterTs();
    if (clearedAfter > fromTs) fromTs = clearedAfter;

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

router.get("/leverage-stats", async (req, res) => {
  try {
    const config = getConfig();
    const portfolio = await storage.getOrCreatePortfolio();
    const openPositions = await storage.getPositions("OPEN", 200);
    const allTrades = await storage.getTradeHistory({ limit: 100000 });

    // Open positions leverage stats
    const openLeverages = openPositions.map(p => p.leverage ?? 1);
    const avgOpenLeverage = openLeverages.length > 0
      ? openLeverages.reduce((s, v) => s + v, 0) / openLeverages.length : 0;
    const maxOpenLeverage = openLeverages.length > 0 ? Math.max(...openLeverages) : 0;

    // Notional exposure from open positions
    const totalNotionalUsdt = openPositions.reduce((s, p) => {
      const notional = (p.qty ?? 0) * (p.entryPrice ?? 0);
      return s + notional;
    }, 0);
    const equity = portfolio.currentEquityUsdt ?? 15000;
    const exposurePct = equity > 0 ? (totalNotionalUsdt / equity) * 100 : 0;

    // Closed trades leverage breakdown
    const leverageMap: Record<number, { trades: number; wins: number; totalR: number; totalPnlUsdt: number }> = {};
    const closedLeverages: number[] = [];
    for (const t of allTrades) {
      const lev = (t as any).leverage ?? 1;
      closedLeverages.push(lev);
      if (!leverageMap[lev]) leverageMap[lev] = { trades: 0, wins: 0, totalR: 0, totalPnlUsdt: 0 };
      leverageMap[lev].trades++;
      const r = (t as any).netR ?? (t as any).grossR ?? 0;
      if (r > 0) leverageMap[lev].wins++;
      leverageMap[lev].totalR += r;
      leverageMap[lev].totalPnlUsdt += (t as any).pnlUsdt ?? 0;
    }
    const avgClosedLeverage = closedLeverages.length > 0
      ? closedLeverages.reduce((s, v) => s + v, 0) / closedLeverages.length : 0;
    const maxClosedLeverage = closedLeverages.length > 0 ? Math.max(...closedLeverages) : 0;

    const byTier = Object.entries(leverageMap)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([lev, l]) => ({
        tier: `${lev}x`,
        leverageNum: Number(lev),
        trades: l.trades,
        wins: l.wins,
        winRate: l.trades > 0 ? Math.round((l.wins / l.trades) * 1000) / 10 : 0,
        totalR: Math.round(l.totalR * 100) / 100,
        totalPnlUsdt: Math.round(l.totalPnlUsdt * 100) / 100,
      }));

    res.json({
      open: {
        count: openPositions.length,
        avgLeverage: Math.round(avgOpenLeverage * 10) / 10,
        maxLeverage: maxOpenLeverage,
        totalNotionalUsdt: Math.round(totalNotionalUsdt * 100) / 100,
        exposurePct: Math.round(exposurePct * 10) / 10,
      },
      closed: {
        count: allTrades.length,
        avgLeverage: Math.round(avgClosedLeverage * 100) / 100,
        maxLeverage: maxClosedLeverage,
        byTier,
      },
      configTiers: config.leverageTiers,
      maxConfigLeverage: config.maxLeverage,
      leverageEnabled: config.leverageEnabled,
    });
  } catch (err: any) {
    console.error("Error getting leverage stats:", err);
    res.status(500).json({ error: err.message });
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
