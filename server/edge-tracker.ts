/**
 * Edge Tracking Metrics for Signal Evaluation
 * 
 * Tracks the actual performance of trading signals:
 * - Average return on signals (net of costs)
 * - Hit rate after costs
 * - Per-month stability
 * - Edge statistics
 * 
 * This is the proper way to evaluate signal quality - 
 * NOT overall accuracy, but performance on signaled trades only.
 */

import * as fs from "fs";
import * as path from "path";

const EDGE_TRACKER_FILE = path.join(process.cwd(), "edge_tracker_state.json");

export interface SignalResult {
  timestamp: Date;
  action: "LONG" | "SHORT";
  entryPrice: number;
  exitPrice: number;
  horizon: number;
  actualReturn: number;       // Raw return
  netReturn: number;          // Return after trading costs
  tradingCost: number;        // Applied trading cost
  confidence: number;         // Signal confidence/score
  isWin: boolean;             // netReturn > 0
}

export interface EdgeMetrics {
  // Core edge metrics
  totalSignals: number;
  avgNetReturn: number;       // Average net return per signal
  medianNetReturn: number;    // Median net return
  hitRate: number;            // % of signals with positive net return
  expectancy: number;         // Expected value per trade
  
  // Breakdown by direction
  longSignals: number;
  shortSignals: number;
  longHitRate: number;
  shortHitRate: number;
  longAvgReturn: number;
  shortAvgReturn: number;
  
  // Risk metrics
  worstReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;
  
  // Per-period stability
  monthlyEdge: { [month: string]: number };
  monthlyHitRate: { [month: string]: number };
  stabilityScore: number;     // Consistency across months
  
  // Time range
  startDate: Date | null;
  endDate: Date | null;
}

export interface EdgeTrackerState {
  results: SignalResult[];
  tradingCostMode: "taker_taker" | "maker_taker" | "maker_maker";
}

const TRADING_COSTS = {
  taker_taker: 0.0009,
  maker_taker: 0.0006,
  maker_maker: 0.0004
};

/**
 * Edge Tracker - tracks signal performance over time
 */
export class EdgeTracker {
  private results: SignalResult[] = [];
  private tradingCostMode: "taker_taker" | "maker_taker" | "maker_maker" = "taker_taker";
  
  constructor() {
    this.loadState();
  }
  
  private loadState() {
    try {
      if (fs.existsSync(EDGE_TRACKER_FILE)) {
        const data = JSON.parse(fs.readFileSync(EDGE_TRACKER_FILE, "utf-8"));
        this.results = (data.results || []).map((r: any) => ({
          ...r,
          timestamp: new Date(r.timestamp)
        }));
        this.tradingCostMode = data.tradingCostMode || "taker_taker";
        console.log(`[EdgeTracker] Loaded ${this.results.length} signal results`);
      }
    } catch (e) {
      console.log("[EdgeTracker] No saved state found, starting fresh");
    }
  }
  
  private saveState() {
    try {
      fs.writeFileSync(EDGE_TRACKER_FILE, JSON.stringify({
        results: this.results,
        tradingCostMode: this.tradingCostMode
      }, null, 2));
    } catch (e) {
      console.error("[EdgeTracker] Failed to save state:", e);
    }
  }
  
  /**
   * Set trading cost mode
   */
  setTradingCostMode(mode: "taker_taker" | "maker_taker" | "maker_maker") {
    this.tradingCostMode = mode;
    this.saveState();
  }
  
  getTradingCost(): number {
    return TRADING_COSTS[this.tradingCostMode];
  }
  
  /**
   * Record a signal result
   */
  recordSignal(
    action: "LONG" | "SHORT",
    entryPrice: number,
    exitPrice: number,
    horizon: number,
    confidence: number
  ) {
    const actualReturn = action === "LONG" 
      ? (exitPrice - entryPrice) / entryPrice
      : (entryPrice - exitPrice) / entryPrice;
    
    const tradingCost = this.getTradingCost();
    const netReturn = actualReturn - tradingCost;
    
    const result: SignalResult = {
      timestamp: new Date(),
      action,
      entryPrice,
      exitPrice,
      horizon,
      actualReturn,
      netReturn,
      tradingCost,
      confidence,
      isWin: netReturn > 0
    };
    
    this.results.push(result);
    this.saveState();
    
    return result;
  }
  
  /**
   * Compute all edge metrics
   */
  computeMetrics(): EdgeMetrics {
    const empty: EdgeMetrics = {
      totalSignals: 0,
      avgNetReturn: 0,
      medianNetReturn: 0,
      hitRate: 0,
      expectancy: 0,
      longSignals: 0,
      shortSignals: 0,
      longHitRate: 0,
      shortHitRate: 0,
      longAvgReturn: 0,
      shortAvgReturn: 0,
      worstReturn: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      monthlyEdge: {},
      monthlyHitRate: {},
      stabilityScore: 0,
      startDate: null,
      endDate: null
    };
    
    if (this.results.length === 0) return empty;
    
    const longs = this.results.filter(r => r.action === "LONG");
    const shorts = this.results.filter(r => r.action === "SHORT");
    
    const netReturns = this.results.map(r => r.netReturn);
    const sortedReturns = [...netReturns].sort((a, b) => a - b);
    
    // Core metrics
    const avgNetReturn = netReturns.reduce((a, b) => a + b, 0) / netReturns.length;
    const medianNetReturn = sortedReturns[Math.floor(sortedReturns.length / 2)];
    const wins = this.results.filter(r => r.isWin).length;
    const hitRate = wins / this.results.length;
    
    // Expectancy = (hitRate * avgWin) - ((1-hitRate) * avgLoss)
    const winReturns = netReturns.filter(r => r > 0);
    const lossReturns = netReturns.filter(r => r <= 0);
    const avgWin = winReturns.length > 0 ? winReturns.reduce((a, b) => a + b, 0) / winReturns.length : 0;
    const avgLoss = lossReturns.length > 0 ? Math.abs(lossReturns.reduce((a, b) => a + b, 0) / lossReturns.length) : 0;
    const expectancy = (hitRate * avgWin) - ((1 - hitRate) * avgLoss);
    
    // Direction breakdown
    const longWins = longs.filter(r => r.isWin).length;
    const shortWins = shorts.filter(r => r.isWin).length;
    const longAvgReturn = longs.length > 0 ? longs.reduce((a, r) => a + r.netReturn, 0) / longs.length : 0;
    const shortAvgReturn = shorts.length > 0 ? shorts.reduce((a, r) => a + r.netReturn, 0) / shorts.length : 0;
    
    // Risk metrics
    const worstReturn = Math.min(...netReturns);
    const stdDev = Math.sqrt(
      netReturns.reduce((a, r) => a + Math.pow(r - avgNetReturn, 2), 0) / netReturns.length
    );
    const sharpeRatio = stdDev > 0 ? (avgNetReturn / stdDev) * Math.sqrt(252) : 0; // Annualized
    
    // Max drawdown (cumulative)
    let peak = 0;
    let maxDrawdown = 0;
    let cumReturn = 0;
    for (const ret of netReturns) {
      cumReturn += ret;
      if (cumReturn > peak) peak = cumReturn;
      const dd = peak - cumReturn;
      if (dd > maxDrawdown) maxDrawdown = dd;
    }
    
    // Monthly breakdown
    const monthlyEdge: { [month: string]: number } = {};
    const monthlyHitRate: { [month: string]: number } = {};
    const monthlyGroups: { [month: string]: SignalResult[] } = {};
    
    for (const r of this.results) {
      const month = r.timestamp.toISOString().substring(0, 7); // YYYY-MM
      if (!monthlyGroups[month]) monthlyGroups[month] = [];
      monthlyGroups[month].push(r);
    }
    
    for (const [month, group] of Object.entries(monthlyGroups)) {
      const returns = group.map(r => r.netReturn);
      monthlyEdge[month] = returns.reduce((a, b) => a + b, 0) / returns.length;
      monthlyHitRate[month] = group.filter(r => r.isWin).length / group.length;
    }
    
    // Stability score: coefficient of variation of monthly edge
    const monthlyEdges = Object.values(monthlyEdge);
    let stabilityScore = 0;
    if (monthlyEdges.length >= 2) {
      const monthlyMean = monthlyEdges.reduce((a, b) => a + b, 0) / monthlyEdges.length;
      const monthlyStd = Math.sqrt(
        monthlyEdges.reduce((a, e) => a + Math.pow(e - monthlyMean, 2), 0) / monthlyEdges.length
      );
      // Higher = more stable (inverse of CV, capped at 10)
      stabilityScore = monthlyStd > 0 && monthlyMean !== 0 
        ? Math.min(10, Math.abs(monthlyMean / monthlyStd)) 
        : 0;
    }
    
    const dates = this.results.map(r => r.timestamp).sort((a, b) => a.getTime() - b.getTime());
    
    return {
      totalSignals: this.results.length,
      avgNetReturn,
      medianNetReturn,
      hitRate,
      expectancy,
      longSignals: longs.length,
      shortSignals: shorts.length,
      longHitRate: longs.length > 0 ? longWins / longs.length : 0,
      shortHitRate: shorts.length > 0 ? shortWins / shorts.length : 0,
      longAvgReturn,
      shortAvgReturn,
      worstReturn,
      maxDrawdown,
      sharpeRatio,
      monthlyEdge,
      monthlyHitRate,
      stabilityScore,
      startDate: dates.length > 0 ? dates[0] : null,
      endDate: dates.length > 0 ? dates[dates.length - 1] : null
    };
  }
  
  /**
   * Get formatted edge report
   */
  getEdgeReport(): string {
    const m = this.computeMetrics();
    
    if (m.totalSignals === 0) {
      return "No signals tracked yet.";
    }
    
    return `
=== EDGE TRACKING REPORT ===
Signals: ${m.totalSignals} (${m.longSignals} long, ${m.shortSignals} short)
Period: ${m.startDate?.toISOString().substring(0, 10)} to ${m.endDate?.toISOString().substring(0, 10)}

EDGE ON SIGNALS:
  Avg Net Return: ${(m.avgNetReturn * 100).toFixed(3)}%
  Median Return:  ${(m.medianNetReturn * 100).toFixed(3)}%
  Hit Rate:       ${(m.hitRate * 100).toFixed(1)}%
  Expectancy:     ${(m.expectancy * 100).toFixed(3)}% per trade

DIRECTION BREAKDOWN:
  Long Hit Rate:  ${(m.longHitRate * 100).toFixed(1)}% (avg ${(m.longAvgReturn * 100).toFixed(3)}%)
  Short Hit Rate: ${(m.shortHitRate * 100).toFixed(1)}% (avg ${(m.shortAvgReturn * 100).toFixed(3)}%)

RISK METRICS:
  Worst Trade:    ${(m.worstReturn * 100).toFixed(3)}%
  Max Drawdown:   ${(m.maxDrawdown * 100).toFixed(3)}%
  Sharpe Ratio:   ${m.sharpeRatio.toFixed(2)}

MONTHLY STABILITY:
${Object.entries(m.monthlyEdge).map(([month, edge]) => 
  `  ${month}: ${(edge * 100).toFixed(3)}% edge, ${(m.monthlyHitRate[month] * 100).toFixed(1)}% hit rate`
).join('\n')}
`.trim();
  }
  
  /**
   * Clear all results (for testing)
   */
  clearResults() {
    this.results = [];
    this.saveState();
  }
}

export const edgeTracker = new EdgeTracker();
