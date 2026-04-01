import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
  BarChart, Bar, Cell, LineChart, Line, ScatterChart, Scatter, ZAxis,
} from "recharts";
import {
  TrendingUp, TrendingDown, Hash, Percent, Target, ArrowDown, Trophy, Skull,
  Scale, CheckCircle, XCircle, Minus, BarChart3, ArrowUpDown, Clock, Zap,
  Activity, Calendar, Timer, Crosshair, Brain, AlertTriangle, Sliders, Shield,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { useState } from "react";

interface PerformanceData {
  totalTrades: number;
  wins: number;
  losses: number;
  winRate: number;
  totalR: number;
  avgWinR: number;
  avgLossR: number;
  profitFactor: number;
  maxDrawdown: number;
  bestTrade: number;
  worstTrade: number;
  expectancy: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxConsecWins: number;
  maxConsecLosses: number;
  avgHoldBars: number;
  avgHoldMinutes: number;
  perSymbol: Array<{
    symbol: string;
    trades: number;
    wins: number;
    winRate: number;
    totalR: number;
    expectancy: number;
  }>;
  perSymbolEquity: Record<string, Array<{ ts: number; r: number; tradeR: number }>>;
  hourlyBreakdown: Array<{
    hour: number;
    trades: number;
    wins: number;
    winRate: number;
    totalR: number;
    avgR: number;
  }>;
  durationBins: Array<{ label: string; min: number; max: number; count: number }>;
  streaks: Array<{ type: "win" | "loss"; length: number; ts: number }>;
  monthlyData: Array<{ month: string; totalR: number; trades: number; wins: number; winRate: number }>;
  weeklyData: Array<{ week: string; totalR: number; trades: number; wins: number; winRate: number }>;
  rolling7d: { trades: number; wins: number; winRate: number; totalR: number; expectancy: number };
  rolling30d: { trades: number; wins: number; winRate: number; totalR: number; expectancy: number };
  leverageBreakdown?: Array<{ tier: string; leverageNum?: number; trades: number; wins: number; winRate: number; totalR: number; totalPnlUsdt: number }>;
  totalPnlUsdt?: number;
  totalRiskUsdt?: number;
}

interface EquityPoint {
  ts: number;
  r: number;
  tradeR: number;
  symbol: string;
  side: string;
}

interface MfeMaeData {
  mfeVsResult: Array<{
    symbol: string;
    maxFavorableR: number;
    netR: number;
    exitReason: string;
    side: string;
  }>;
  captureRatio: number;
  wastedEdge: number;
  highMfeLosses: number;
  totalLosses: number;
  optimalExitSim: Array<{ exitPct: number; totalR: number }>;
}

interface NeuralAdjustmentItem {
  id: number;
  positionId: number;
  symbol: string;
  timestamp: number;
  adjustmentType: string;
  previousSl: number | null;
  newSl: number | null;
  v5Score: number | null;
  pHold: number | null;
  pLong: number | null;
  pShort: number | null;
  retMu: number | null;
  positionPnlR: number | null;
  reason: string | null;
}

interface NeuralAdjustmentsData {
  adjustments: NeuralAdjustmentItem[];
  summary: {
    total: number;
    byType: Record<string, number>;
    avgPnlR: number;
  };
}

function formatDate(ts: number) {
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function getWinRateColor(wr: number) {
  if (wr > 50) return "text-emerald-400";
  if (wr > 40) return "text-amber-400";
  return "text-red-400";
}

function getPfColor(pf: number) {
  if (pf > 1.5) return "text-emerald-400";
  if (pf > 1) return "text-amber-400";
  return "text-red-400";
}

function getRatioColor(r: number) {
  if (r > 1) return "text-emerald-400";
  if (r > 0) return "text-amber-400";
  return "text-red-400";
}

function formatDuration(minutes: number) {
  if (minutes < 60) return `${minutes}m`;
  if (minutes < 1440) return `${(minutes / 60).toFixed(1)}h`;
  return `${(minutes / 1440).toFixed(1)}d`;
}

function StatCard({ icon: Icon, label, value, colorClass, glow, sub }: {
  icon: typeof TrendingUp;
  label: string;
  value: string;
  colorClass: string;
  glow?: string;
  sub?: string;
}) {
  return (
    <div className={`glass-card rounded-md p-3 ${glow ?? ""}`} data-testid={`stat-${label.toLowerCase().replace(/[\s\/]/g, "-")}`}>
      <Icon className="w-4 h-4 text-muted-foreground mb-1" />
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`text-xl number-mono font-bold ${colorClass}`}>{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
    </div>
  );
}

function HourlyHeatmap({ data }: { data: PerformanceData["hourlyBreakdown"] }) {
  const maxR = Math.max(...data.map((d) => Math.abs(d.totalR)), 0.01);
  return (
    <div className="glass-card rounded-md p-4" data-testid="hourly-heatmap">
      <p className="text-sm font-semibold mb-3">
        <Clock className="w-4 h-4 inline mr-1" />
        Hourly Performance Heatmap (UTC)
      </p>
      <div className="grid grid-cols-12 gap-1">
        {data.map((h) => {
          const intensity = maxR > 0 ? Math.abs(h.totalR) / maxR : 0;
          const isPositive = h.totalR >= 0;
          const bg = h.trades === 0
            ? "bg-muted/30"
            : isPositive
              ? `bg-emerald-500`
              : `bg-red-500`;
          const opacity = h.trades === 0 ? 1 : Math.max(0.15, intensity);
          return (
            <div
              key={h.hour}
              className={`${bg} rounded-sm p-1.5 text-center transition-colors`}
              style={{ opacity }}
              title={`${h.hour}:00 UTC — ${h.trades} trades, ${h.totalR >= 0 ? "+" : ""}${h.totalR.toFixed(2)}R, WR: ${h.winRate}%`}
              data-testid={`heatmap-hour-${h.hour}`}
            >
              <p className="text-[10px] font-medium text-white">{h.hour}</p>
              <p className="text-[9px] text-white/80 number-mono">{h.trades > 0 ? `${h.totalR >= 0 ? "+" : ""}${h.totalR.toFixed(1)}` : "-"}</p>
            </div>
          );
        })}
      </div>
      <div className="flex items-center justify-between gap-2 mt-2 text-[10px] text-muted-foreground">
        <span>0:00 UTC</span>
        <span>12:00 UTC</span>
        <span>23:00 UTC</span>
      </div>
    </div>
  );
}

function StreaksChart({ streaks }: { streaks: PerformanceData["streaks"] }) {
  const displayStreaks = streaks.slice(-30);
  return (
    <div className="glass-card rounded-md p-4" data-testid="streaks-chart">
      <p className="text-sm font-semibold mb-3">
        <Activity className="w-4 h-4 inline mr-1" />
        Win/Loss Streaks
      </p>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={displayStreaks}>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
            <XAxis
              dataKey="ts"
              tickFormatter={formatDate}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              allowDecimals={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: "12px",
              }}
              labelFormatter={(ts: number) => new Date(ts).toLocaleDateString()}
              formatter={(value: number, _: string, entry: any) => {
                const type = entry.payload.type;
                return [value, type === "win" ? "Win Streak" : "Loss Streak"];
              }}
            />
            <Bar dataKey="length" radius={[3, 3, 0, 0]}>
              {displayStreaks.map((s, i) => (
                <Cell key={i} fill={s.type === "win" ? "#34d399" : "#f87171"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function PerSymbolEquityCurves({ data }: { data: PerformanceData["perSymbolEquity"] }) {
  const symbols = Object.keys(data);
  const colors = ["#34d399", "#60a5fa", "#f59e0b", "#f87171", "#a78bfa", "#06b6d4"];
  if (symbols.length === 0) return null;
  return (
    <div className="glass-card rounded-md p-4" data-testid="per-symbol-equity">
      <p className="text-sm font-semibold mb-3">
        <TrendingUp className="w-4 h-4 inline mr-1" />
        Per-Symbol Equity Curves
      </p>
      <div className="grid grid-cols-2 gap-3">
        {symbols.map((sym, idx) => {
          const points = data[sym];
          if (!points || points.length === 0) return null;
          const finalR = points[points.length - 1]?.r ?? 0;
          return (
            <div key={sym} className="rounded-md border border-border/30 p-2" data-testid={`equity-curve-${sym}`}>
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="text-xs font-semibold">{sym}</span>
                <span className={`text-xs number-mono font-bold ${finalR >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {finalR >= 0 ? "+" : ""}{finalR.toFixed(2)}R
                </span>
              </div>
              <div className="h-24">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={points}>
                    <defs>
                      <linearGradient id={`grad-${sym}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={colors[idx % colors.length]} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={colors[idx % colors.length]} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeOpacity={0.2} />
                    <Area
                      type="monotone"
                      dataKey="r"
                      stroke={colors[idx % colors.length]}
                      strokeWidth={1.5}
                      fill={`url(#grad-${sym})`}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DurationHistogram({ bins }: { bins: PerformanceData["durationBins"] }) {
  return (
    <div className="glass-card rounded-md p-4" data-testid="duration-histogram">
      <p className="text-sm font-semibold mb-3">
        <Timer className="w-4 h-4 inline mr-1" />
        Trade Duration Distribution
      </p>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins}>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              allowDecimals={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: "12px",
              }}
            />
            <Bar dataKey="count" fill="#60a5fa" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function MonthlyPnlTable({ data, type }: { data: Array<{ period: string; totalR: number; trades: number; wins: number; winRate: number }>; type: "monthly" | "weekly" }) {
  return (
    <div className="glass-card rounded-md p-4" data-testid={`${type}-pnl-table`}>
      <p className="text-sm font-semibold mb-3">
        <Calendar className="w-4 h-4 inline mr-1" />
        {type === "monthly" ? "Monthly" : "Weekly"} P&L
      </p>
      <div className="max-h-64 overflow-y-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{type === "monthly" ? "Month" : "Week"}</TableHead>
              <TableHead className="text-right">Trades</TableHead>
              <TableHead className="text-right">WR%</TableHead>
              <TableHead className="text-right">P&L (R)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((row) => (
              <TableRow key={row.period}>
                <TableCell className="font-medium text-xs">{row.period}</TableCell>
                <TableCell className="text-right number-mono text-xs">{row.trades}</TableCell>
                <TableCell className={`text-right number-mono text-xs ${getWinRateColor(row.winRate)}`}>{row.winRate.toFixed(1)}%</TableCell>
                <TableCell className={`text-right number-mono text-xs font-semibold ${row.totalR >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {row.totalR >= 0 ? "+" : ""}{row.totalR.toFixed(2)}R
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

function RollingMetrics({ rolling7d, rolling30d }: { rolling7d: PerformanceData["rolling7d"]; rolling30d: PerformanceData["rolling30d"] }) {
  return (
    <div className="glass-card rounded-md p-4" data-testid="rolling-metrics">
      <p className="text-sm font-semibold mb-3">
        <Zap className="w-4 h-4 inline mr-1" />
        Rolling Performance
      </p>
      <div className="space-y-3">
        <div>
          <p className="text-xs text-muted-foreground mb-1">Last 7 Days</p>
          <div className="grid grid-cols-4 gap-2">
            <div>
              <p className="text-[10px] text-muted-foreground">Trades</p>
              <p className="text-sm number-mono font-bold" data-testid="text-rolling-7d-trades">{rolling7d.trades}</p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground">Win Rate</p>
              <p className={`text-sm number-mono font-bold ${getWinRateColor(rolling7d.winRate)}`} data-testid="text-rolling-7d-winrate">{rolling7d.winRate.toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground">Total R</p>
              <p className={`text-sm number-mono font-bold ${rolling7d.totalR >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-rolling-7d-totalr">
                {rolling7d.totalR >= 0 ? "+" : ""}{rolling7d.totalR.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground">E[R]</p>
              <p className={`text-sm number-mono font-bold ${rolling7d.expectancy >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-rolling-7d-expectancy">
                {rolling7d.expectancy >= 0 ? "+" : ""}{rolling7d.expectancy.toFixed(3)}
              </p>
            </div>
          </div>
        </div>
        <div className="border-t border-border/30 pt-3">
          <p className="text-xs text-muted-foreground mb-1">Last 30 Days</p>
          <div className="grid grid-cols-4 gap-2">
            <div>
              <p className="text-[10px] text-muted-foreground">Trades</p>
              <p className="text-sm number-mono font-bold" data-testid="text-rolling-30d-trades">{rolling30d.trades}</p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground">Win Rate</p>
              <p className={`text-sm number-mono font-bold ${getWinRateColor(rolling30d.winRate)}`} data-testid="text-rolling-30d-winrate">{rolling30d.winRate.toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground">Total R</p>
              <p className={`text-sm number-mono font-bold ${rolling30d.totalR >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-rolling-30d-totalr">
                {rolling30d.totalR >= 0 ? "+" : ""}{rolling30d.totalR.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground">E[R]</p>
              <p className={`text-sm number-mono font-bold ${rolling30d.expectancy >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-rolling-30d-expectancy">
                {rolling30d.expectancy >= 0 ? "+" : ""}{rolling30d.expectancy.toFixed(3)}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DirectionalStats({ perf, side }: { perf: PerformanceData; side: string }) {
  const symbolData = perf.perSymbol || [];
  const totalTrades = perf.totalTrades;
  const totalWins = perf.wins;
  const wr = totalTrades > 0 ? (totalWins / totalTrades) * 100 : 0;
  return (
    <div className="glass-card rounded-md p-4 flex-1" data-testid={`directional-${side.toLowerCase()}`}>
      <p className={`text-sm font-semibold mb-3 ${side === "LONG" ? "text-emerald-400" : "text-red-400"}`}>
        {side} Performance
      </p>
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">Win Rate</span>
          <span className={`number-mono text-sm font-medium ${getWinRateColor(wr)}`}>{wr.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-muted rounded-full h-2 mt-2">
          <div
            className={`h-2 rounded-full ${side === "LONG" ? "bg-emerald-500" : "bg-red-500"}`}
            style={{ width: `${Math.min(wr, 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}

function MfeScatterPlot({ data }: { data: MfeMaeData }) {
  const scatterData = data.mfeVsResult.map((d, i) => ({
    ...d,
    id: i,
    isWin: d.netR > 0,
  }));
  const maxVal = Math.max(
    ...scatterData.map(d => Math.abs(d.maxFavorableR)),
    ...scatterData.map(d => Math.abs(d.netR)),
    1,
  );

  return (
    <div className="glass-card rounded-md p-4" data-testid="mfe-scatter-plot">
      <p className="text-sm font-semibold mb-3">
        <Crosshair className="w-4 h-4 inline mr-1" />
        MFE vs Final Result (Scatter)
      </p>
      <p className="text-[10px] text-muted-foreground mb-2">
        Points below the diagonal represent wasted edge — trades that were profitable but gave back gains
      </p>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
            <XAxis
              dataKey="maxFavorableR"
              type="number"
              name="Max Favorable R"
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              label={{ value: "Max Favorable R (MFE)", position: "insideBottom", offset: -5, fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              domain={[0, "auto"]}
            />
            <YAxis
              dataKey="netR"
              type="number"
              name="Net R"
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              label={{ value: "Net R (Result)", angle: -90, position: "insideLeft", fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            />
            <ZAxis range={[30, 60]} />
            <ReferenceLine
              segment={[{ x: 0, y: 0 }, { x: maxVal, y: maxVal }]}
              stroke="hsl(var(--muted-foreground))"
              strokeOpacity={0.3}
              strokeDasharray="4 4"
            />
            <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeOpacity={0.2} />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: "12px",
              }}
              formatter={(value: number, name: string) => {
                if (name === "Max Favorable R") return [`${value.toFixed(2)}R`, "MFE"];
                if (name === "Net R") return [`${value.toFixed(2)}R`, "Result"];
                return [value, name];
              }}
              labelFormatter={() => ""}
            />
            <Scatter data={scatterData.filter(d => d.isWin)} fill="#34d399" name="Winners" />
            <Scatter data={scatterData.filter(d => !d.isWin)} fill="#f87171" name="Losers" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function CaptureRatioChart({ data }: { data: MfeMaeData }) {
  const perTrade = data.mfeVsResult
    .filter(d => d.maxFavorableR > 0)
    .map((d, i) => ({
      idx: i + 1,
      symbol: d.symbol,
      capture: d.netR > 0 ? Math.min((d.netR / d.maxFavorableR) * 100, 100) : 0,
      mfe: d.maxFavorableR,
      netR: d.netR,
      isWin: d.netR > 0,
    }));

  return (
    <div className="glass-card rounded-md p-4" data-testid="capture-ratio-chart">
      <p className="text-sm font-semibold mb-1">
        <Target className="w-4 h-4 inline mr-1" />
        Capture Ratio per Trade
      </p>
      <p className="text-[10px] text-muted-foreground mb-3">
        Avg capture: <span className="number-mono font-semibold text-cyan-400">{(data.captureRatio * 100).toFixed(1)}%</span> of available MFE
      </p>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={perTrade}>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
            <XAxis
              dataKey="idx"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              label={{ value: "Trade #", position: "insideBottom", offset: -5, fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${v}%`}
              domain={[0, 100]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: "12px",
              }}
              formatter={(value: number, _: string, entry: any) => {
                const p = entry.payload;
                return [`${value.toFixed(1)}% (${p.symbol}: MFE ${p.mfe.toFixed(2)}R → ${p.netR.toFixed(2)}R)`, "Capture"];
              }}
            />
            <Bar dataKey="capture" radius={[2, 2, 0, 0]}>
              {perTrade.map((d, i) => (
                <Cell key={i} fill={d.isWin ? "#34d399" : "#f87171"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function OptimalExitSimulator({ data }: { data: MfeMaeData }) {
  const [exitPct, setExitPct] = useState(70);
  const simData = data.optimalExitSim;
  const bestPoint = simData.reduce((best, cur) => cur.totalR > best.totalR ? cur : best, simData[0]);
  const currentPoint = simData.find(d => d.exitPct === exitPct) || simData[0];

  return (
    <div className="glass-card rounded-md p-4" data-testid="optimal-exit-simulator">
      <p className="text-sm font-semibold mb-1">
        <Sliders className="w-4 h-4 inline mr-1" />
        Optimal Exit Simulator
      </p>
      <p className="text-[10px] text-muted-foreground mb-3">
        Drag the slider to see how different MFE exit thresholds affect total R
      </p>
      <div className="flex items-center gap-4 mb-4">
        <div className="flex-1">
          <Slider
            value={[exitPct]}
            onValueChange={(v) => setExitPct(v[0])}
            min={30}
            max={100}
            step={5}
            data-testid="slider-exit-pct"
          />
        </div>
        <div className="text-right min-w-[120px]">
          <p className="text-xs text-muted-foreground">Exit at</p>
          <p className="text-lg number-mono font-bold text-cyan-400" data-testid="text-exit-pct">{exitPct}% of MFE</p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-[10px] text-muted-foreground">Simulated Total R</p>
          <p className={`text-xl number-mono font-bold ${currentPoint?.totalR >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-sim-total-r">
            {currentPoint ? `${currentPoint.totalR >= 0 ? "+" : ""}${currentPoint.totalR.toFixed(2)}R` : "-"}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-muted-foreground">Optimal Exit Point</p>
          <p className="text-xl number-mono font-bold text-amber-400" data-testid="text-optimal-exit">
            {bestPoint ? `${bestPoint.exitPct}% → ${bestPoint.totalR >= 0 ? "+" : ""}${bestPoint.totalR.toFixed(2)}R` : "-"}
          </p>
        </div>
      </div>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={simData}>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
            <XAxis
              dataKey="exitPct"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${v}%`}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${v}R`}
            />
            <ReferenceLine x={exitPct} stroke="#06b6d4" strokeDasharray="4 4" strokeOpacity={0.7} />
            <ReferenceLine x={bestPoint?.exitPct} stroke="#f59e0b" strokeDasharray="4 4" strokeOpacity={0.5} />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: "12px",
              }}
              formatter={(value: number) => [`${value.toFixed(2)}R`, "Simulated Total R"]}
              labelFormatter={(v: number) => `Exit at ${v}% of MFE`}
            />
            <Line type="monotone" dataKey="totalR" stroke="#34d399" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function NeuralManagerPerformance({ data }: { data: NeuralAdjustmentsData }) {
  const typeLabels: Record<string, string> = {
    BREAKEVEN: "Breakeven Set",
    TRAIL_TIGHTEN: "Trail Tightened",
    TRAIL_WIDEN: "Trail Widened",
    DIRECTION_FLIP_EXIT: "Direction Flip Exit",
    CONFIDENCE_DECAY_EXIT: "Confidence Decay Exit",
    MFE_PROTECTION_EXIT: "MFE Protection Exit",
    ADAPTIVE_TRAIL: "Adaptive Trail",
  };

  const typeColors: Record<string, string> = {
    BREAKEVEN: "text-cyan-400 border-cyan-400/30",
    TRAIL_TIGHTEN: "text-amber-400 border-amber-400/30",
    TRAIL_WIDEN: "text-emerald-400 border-emerald-400/30",
    DIRECTION_FLIP_EXIT: "text-red-400 border-red-400/30",
    CONFIDENCE_DECAY_EXIT: "text-orange-400 border-orange-400/30",
    MFE_PROTECTION_EXIT: "text-emerald-400 border-emerald-400/30",
    ADAPTIVE_TRAIL: "text-blue-400 border-blue-400/30",
  };

  return (
    <div className="glass-card rounded-md p-4" data-testid="neural-manager-performance">
      <p className="text-sm font-semibold mb-3">
        <Brain className="w-4 h-4 inline mr-1" />
        Neural Manager Adjustments
        <Badge variant="outline" className="ml-2 text-[10px]">{data.summary.total} total</Badge>
      </p>

      {data.summary.total > 0 && (
        <div className="flex items-center flex-wrap gap-2 mb-3">
          {Object.entries(data.summary.byType).map(([type, count]) => (
            <Badge key={type} variant="outline" className={`text-[10px] ${typeColors[type] || "text-muted-foreground"}`} data-testid={`badge-adj-type-${type}`}>
              {typeLabels[type] || type}: {count}
            </Badge>
          ))}
          <Badge variant="outline" className={`text-[10px] ${data.summary.avgPnlR >= 0 ? "text-emerald-400 border-emerald-400/30" : "text-red-400 border-red-400/30"}`}>
            Avg P&L: {data.summary.avgPnlR >= 0 ? "+" : ""}{data.summary.avgPnlR.toFixed(2)}R
          </Badge>
        </div>
      )}

      <div className="max-h-64 overflow-y-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs">Time</TableHead>
              <TableHead className="text-xs">Symbol</TableHead>
              <TableHead className="text-xs">Type</TableHead>
              <TableHead className="text-xs text-right">V5 Score</TableHead>
              <TableHead className="text-xs text-right">P&L (R)</TableHead>
              <TableHead className="text-xs">Reason</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.adjustments.slice(0, 50).map((adj) => (
              <TableRow key={adj.id} data-testid={`row-adj-${adj.id}`}>
                <TableCell className="text-[11px] number-mono text-muted-foreground">{new Date(adj.timestamp).toLocaleString()}</TableCell>
                <TableCell className="text-xs font-semibold">{adj.symbol}</TableCell>
                <TableCell>
                  <Badge variant="outline" className={`text-[10px] ${typeColors[adj.adjustmentType] || "text-muted-foreground"}`}>
                    {typeLabels[adj.adjustmentType] || adj.adjustmentType}
                  </Badge>
                </TableCell>
                <TableCell className="text-xs text-right number-mono">{adj.v5Score != null ? adj.v5Score.toFixed(3) : "-"}</TableCell>
                <TableCell className={`text-xs text-right number-mono font-semibold ${(adj.positionPnlR ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {adj.positionPnlR != null ? `${adj.positionPnlR >= 0 ? "+" : ""}${adj.positionPnlR.toFixed(2)}R` : "-"}
                </TableCell>
                <TableCell className="text-[11px] text-muted-foreground max-w-[200px] truncate">{adj.reason || "-"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {data.adjustments.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-4" data-testid="text-no-adjustments">No neural adjustments recorded yet</p>
        )}
      </div>
    </div>
  );
}

function EdgeDecayAnalysis({ data }: { data: NeuralAdjustmentsData }) {
  const positionGroups: Record<number, NeuralAdjustmentItem[]> = {};
  for (const adj of data.adjustments) {
    if (!positionGroups[adj.positionId]) positionGroups[adj.positionId] = [];
    positionGroups[adj.positionId].push(adj);
  }

  const lifecycles: Array<{ step: number; avgV5Score: number; avgPnlR: number; count: number }> = [];
  const maxSteps = 10;
  for (let step = 0; step < maxSteps; step++) {
    let totalScore = 0;
    let totalPnl = 0;
    let count = 0;
    for (const posId of Object.keys(positionGroups)) {
      const sorted = positionGroups[Number(posId)].sort((a, b) => a.timestamp - b.timestamp);
      if (step < sorted.length) {
        totalScore += sorted[step].v5Score ?? 0;
        totalPnl += sorted[step].positionPnlR ?? 0;
        count++;
      }
    }
    if (count > 0) {
      lifecycles.push({
        step: step + 1,
        avgV5Score: totalScore / count,
        avgPnlR: totalPnl / count,
        count,
      });
    }
  }

  if (lifecycles.length === 0) {
    return (
      <div className="glass-card rounded-md p-4" data-testid="edge-decay-analysis">
        <p className="text-sm font-semibold mb-3">
          <Activity className="w-4 h-4 inline mr-1" />
          Edge Decay Analysis
        </p>
        <p className="text-xs text-muted-foreground text-center py-4" data-testid="text-no-edge-decay">Insufficient data for edge decay analysis</p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-md p-4" data-testid="edge-decay-analysis">
      <p className="text-sm font-semibold mb-1">
        <Activity className="w-4 h-4 inline mr-1" />
        Edge Decay Analysis
      </p>
      <p className="text-[10px] text-muted-foreground mb-3">
        How V5 score changes during position lifecycle (averaged across positions)
      </p>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={lifecycles}>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              label={{ value: "Cycle #", position: "insideBottom", offset: -5, fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            />
            <YAxis
              yAxisId="score"
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              orientation="left"
              label={{ value: "V5 Score", angle: -90, position: "insideLeft", fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            />
            <YAxis
              yAxisId="pnl"
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
              orientation="right"
              tickFormatter={(v: number) => `${v.toFixed(1)}R`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: "12px",
              }}
              formatter={(value: number, name: string) => {
                if (name === "avgV5Score") return [value.toFixed(4), "Avg V5 Score"];
                if (name === "avgPnlR") return [`${value.toFixed(2)}R`, "Avg P&L"];
                return [value, name];
              }}
              labelFormatter={(step: number) => `Cycle ${step}`}
            />
            <Line yAxisId="score" type="monotone" dataKey="avgV5Score" stroke="#60a5fa" strokeWidth={2} dot={{ r: 3 }} name="avgV5Score" />
            <Line yAxisId="pnl" type="monotone" dataKey="avgPnlR" stroke="#34d399" strokeWidth={2} dot={{ r: 3 }} name="avgPnlR" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center justify-center gap-4 mt-2 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-400 inline-block" /> V5 Score</span>
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald-400 inline-block" /> Avg P&L (R)</span>
      </div>
    </div>
  );
}

function WastedEdgeCard({ data }: { data: MfeMaeData }) {
  const pctHighMfe = data.totalLosses > 0 ? ((data.highMfeLosses / data.totalLosses) * 100).toFixed(1) : "0";

  return (
    <div className="glass-card rounded-md p-4 glow-red" data-testid="wasted-edge-card">
      <AlertTriangle className="w-5 h-5 text-amber-400 mb-2" />
      <p className="text-xs text-muted-foreground">Wasted Edge</p>
      <p className="text-2xl number-mono font-bold text-red-400" data-testid="text-wasted-edge-value">
        {data.wastedEdge.toFixed(2)}R
      </p>
      <p className="text-[10px] text-muted-foreground mt-1">
        From trades that had MFE {">"}1.5R but ended as losses
      </p>
      <div className="mt-3 space-y-1">
        <div className="flex items-center justify-between gap-2 text-xs">
          <span className="text-muted-foreground">High-MFE Losses</span>
          <span className="number-mono font-semibold text-red-400" data-testid="text-high-mfe-losses">{data.highMfeLosses} / {data.totalLosses}</span>
        </div>
        <div className="flex items-center justify-between gap-2 text-xs">
          <span className="text-muted-foreground">% of Losses</span>
          <span className="number-mono font-semibold text-amber-400" data-testid="text-pct-high-mfe">{pctHighMfe}%</span>
        </div>
        <div className="flex items-center justify-between gap-2 text-xs">
          <span className="text-muted-foreground">Avg Capture (Wins)</span>
          <span className="number-mono font-semibold text-emerald-400" data-testid="text-avg-capture">{(data.captureRatio * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

function computeHistogram(rValues: number[]) {
  const bins = [
    { label: "< -2R", min: -Infinity, max: -2, count: 0 },
    { label: "-2 to -1", min: -2, max: -1, count: 0 },
    { label: "-1 to 0", min: -1, max: 0, count: 0 },
    { label: "0 to 1", min: 0, max: 1, count: 0 },
    { label: "1 to 2", min: 1, max: 2, count: 0 },
    { label: "2 to 3", min: 2, max: 3, count: 0 },
    { label: "> 3R", min: 3, max: Infinity, count: 0 },
  ];
  for (const r of rValues) {
    for (const bin of bins) {
      if (r >= bin.min && r < bin.max) { bin.count++; break; }
    }
  }
  return bins;
}

export default function Analytics() {
  const [pnlView, setPnlView] = useState<"monthly" | "weekly">("monthly");
  const [source, setSource] = useState<"live" | "paper">("paper");

  const perfUrl = source === "paper" ? "/api/paper/performance" : "/api/v5/performance";
  const equityUrl = source === "paper" ? "/api/paper/equity-curve?range=all" : "/api/v5/equity-curve?range=all";

  const { data: perf, isLoading: perfLoading } = useQuery<PerformanceData>({
    queryKey: [perfUrl],
    refetchInterval: source === "paper" ? 30000 : undefined,
  });

  const { data: mfeMae } = useQuery<MfeMaeData>({
    queryKey: ["/api/v5/analytics/mfe-mae"],
    enabled: source === "paper",
  });

  const { data: neuralAdj } = useQuery<NeuralAdjustmentsData>({
    queryKey: ["/api/v5/analytics/neural-adjustments"],
  });

  const { data: equity, isLoading: equityLoading } = useQuery<EquityPoint[]>({
    queryKey: [equityUrl],
    queryFn: async () => {
      const res = await fetch(equityUrl);
      if (!res.ok) throw new Error("Failed to fetch equity curve");
      return res.json();
    },
    refetchInterval: source === "paper" ? 30000 : undefined,
  });

  const { data: leverageStats } = useQuery<{
    open: { count: number; avgLeverage: number; maxLeverage: number; totalNotionalUsdt: number; exposurePct: number };
    closed: { count: number; avgLeverage: number; maxLeverage: number; byTier: Array<{ tier: string; leverageNum: number; trades: number; wins: number; winRate: number; totalR: number; totalPnlUsdt: number }> };
    configTiers: Array<{ minScore: number; leverage: number }>;
    maxConfigLeverage: number;
    leverageEnabled: boolean;
  }>({
    queryKey: ["/api/paper/leverage-stats"],
    enabled: source === "paper",
    refetchInterval: 60000,
  });

  const isLoading = perfLoading || equityLoading;

  if (isLoading) {
    return (
      <div className="p-4 space-y-4" data-testid="analytics">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Loading Analytics...</h2>
          <div className="flex items-center gap-2">
            <button onClick={() => setSource("live")} className={`px-3 py-1.5 rounded-md text-xs font-medium ${source === "live" ? "bg-cyan-500/20 text-cyan-400 border border-cyan-400/30" : "text-muted-foreground"}`}>Live</button>
            <button onClick={() => setSource("paper")} className={`px-3 py-1.5 rounded-md text-xs font-medium ${source === "paper" ? "bg-emerald-500/20 text-emerald-400 border border-emerald-400/30" : "text-muted-foreground"}`}>Paper</button>
          </div>
        </div>
        <div className="grid grid-cols-5 gap-3">
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} className="glass-card rounded-md p-3 h-20 shimmer" />
          ))}
        </div>
        <div className="glass-card rounded-md h-72 shimmer" />
        <div className="glass-card rounded-md h-48 shimmer" />
      </div>
    );
  }

  const sourceToggle = (
    <div className="flex items-center gap-2" data-testid="source-toggle">
      <button
        onClick={() => setSource("live")}
        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${source === "live" ? "bg-cyan-500/20 text-cyan-400 border border-cyan-400/30" : "text-muted-foreground hover:text-foreground"}`}
        data-testid="btn-source-live"
      >
        Live
      </button>
      <button
        onClick={() => setSource("paper")}
        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${source === "paper" ? "bg-emerald-500/20 text-emerald-400 border border-emerald-400/30" : "text-muted-foreground hover:text-foreground"}`}
        data-testid="btn-source-paper"
      >
        Paper
      </button>
    </div>
  );

  if (!perf || perf.totalTrades === 0) {
    return (
      <div className="p-4 space-y-4" data-testid="analytics">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
            {source === "paper" ? "Paper Trading" : "Live Trading"} Analytics
          </h2>
          {sourceToggle}
        </div>
        <div className="flex items-center justify-center min-h-[50vh]">
          <div className="text-center space-y-3">
            <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto" />
            <p className="text-muted-foreground text-sm" data-testid="text-empty-state">
              No {source} trading data yet. {source === "paper" ? "Paper trades will appear once the V5 model generates ENTER signals." : "Signals will appear once the v5 model starts generating trades."}
            </p>
          </div>
        </div>
      </div>
    );
  }

  const winLossRatio = perf.avgLossR !== 0 ? perf.avgWinR / Math.abs(perf.avgLossR) : 0;
  const sortedSymbols = [...(perf.perSymbol || [])].sort((a, b) => b.totalR - a.totalR);

  const tradeRValues = (equity || []).map((e) => e.tradeR);
  const histogram = computeHistogram(tradeRValues);

  const pnlData = pnlView === "monthly"
    ? (perf.monthlyData || []).map((d) => ({ period: d.month, ...d }))
    : (perf.weeklyData || []).map((d) => ({ period: d.week, ...d }));

  return (
    <div className="p-4 space-y-4" data-testid="analytics">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
          {source === "paper" ? "Paper Trading" : "Live Trading"} Analytics
          <Badge variant="outline" className={`ml-2 text-[10px] ${source === "paper" ? "text-emerald-400 border-emerald-400/30" : "text-cyan-400 border-cyan-400/30"}`}>
            {perf.totalTrades} trades
          </Badge>
        </h2>
        {sourceToggle}
      </div>

      <div className="grid grid-cols-5 gap-3">
        <StatCard
          icon={TrendingUp}
          label="Total R"
          value={`${perf.totalR >= 0 ? "+" : ""}${perf.totalR.toFixed(2)}R`}
          colorClass={perf.totalR >= 0 ? "text-emerald-400" : "text-red-400"}
          glow={perf.totalR >= 0 ? "glow-green" : "glow-red"}
        />
        <StatCard
          icon={Hash}
          label="Total Trades"
          value={String(perf.totalTrades)}
          colorClass="text-cyan-500"
          sub={`${perf.wins}W / ${perf.losses}L`}
        />
        <StatCard
          icon={Percent}
          label="Win Rate"
          value={`${perf.winRate.toFixed(1)}%`}
          colorClass={getWinRateColor(perf.winRate)}
        />
        <StatCard
          icon={Target}
          label="Profit Factor"
          value={perf.profitFactor.toFixed(2)}
          colorClass={getPfColor(perf.profitFactor)}
        />
        <StatCard
          icon={Zap}
          label="Expectancy"
          value={`${perf.expectancy >= 0 ? "+" : ""}${perf.expectancy.toFixed(3)}R`}
          colorClass={perf.expectancy >= 0 ? "text-emerald-400" : "text-red-400"}
        />
        <StatCard
          icon={Activity}
          label="Sharpe Ratio"
          value={perf.sharpeRatio.toFixed(2)}
          colorClass={getRatioColor(perf.sharpeRatio)}
        />
        <StatCard
          icon={TrendingUp}
          label="Sortino Ratio"
          value={perf.sortinoRatio.toFixed(2)}
          colorClass={getRatioColor(perf.sortinoRatio)}
        />
        <StatCard
          icon={ArrowDown}
          label="Max Drawdown"
          value={`${perf.maxDrawdown > 0 ? "-" : ""}${Math.abs(perf.maxDrawdown).toFixed(2)}R`}
          colorClass="text-red-400"
          glow="glow-red"
        />
        <StatCard
          icon={Trophy}
          label="Max Consec. Wins"
          value={String(perf.maxConsecWins)}
          colorClass="text-emerald-400"
          sub={`Best: +${perf.bestTrade.toFixed(2)}R`}
        />
        <StatCard
          icon={Clock}
          label="Avg Hold Time"
          value={formatDuration(perf.avgHoldMinutes)}
          colorClass="text-cyan-500"
          sub={`${perf.avgHoldBars.toFixed(1)} bars`}
        />
      </div>

      <div className="glass-card rounded-md p-4" data-testid="equity-curve">
        <p className="text-sm font-semibold mb-3">Equity Curve (R)</p>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equity || []}>
              <defs>
                <linearGradient id="greenGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
              <XAxis
                dataKey="ts"
                tickFormatter={formatDate}
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `${v}R`}
              />
              <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeOpacity={0.3} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
                labelFormatter={(ts: number) => new Date(ts).toLocaleDateString()}
                formatter={(value: number, name: string) => {
                  if (name === "r") return [`${value.toFixed(2)}R`, "Cumulative R"];
                  return [value, name];
                }}
              />
              <Area
                type="monotone"
                dataKey="r"
                stroke="#34d399"
                strokeWidth={2}
                fill="url(#greenGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {perf.rolling7d && perf.rolling30d && (
        <div className="grid grid-cols-2 gap-3">
          <RollingMetrics rolling7d={perf.rolling7d} rolling30d={perf.rolling30d} />
          <HourlyHeatmap data={perf.hourlyBreakdown || []} />
        </div>
      )}

      {perf.perSymbolEquity && Object.keys(perf.perSymbolEquity).length > 0 && (
        <PerSymbolEquityCurves data={perf.perSymbolEquity} />
      )}

      <div className="grid grid-cols-2 gap-3">
        {perf.streaks && perf.streaks.length > 0 && (
          <StreaksChart streaks={perf.streaks} />
        )}
        {perf.durationBins && (
          <DurationHistogram bins={perf.durationBins} />
        )}
      </div>

      <div className="glass-card rounded-md p-4" data-testid="per-symbol-table">
        <p className="text-sm font-semibold mb-3">Per-Symbol Performance</p>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead className="text-right">Trades</TableHead>
              <TableHead className="text-right">Wins</TableHead>
              <TableHead className="text-right">Losses</TableHead>
              <TableHead className="text-right">WR%</TableHead>
              <TableHead className="text-right">E[R]</TableHead>
              <TableHead className="text-right">Total R</TableHead>
              <TableHead className="text-right">Edge Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedSymbols.map((s) => (
              <TableRow key={s.symbol} data-testid={`row-symbol-${s.symbol}`}>
                <TableCell className="font-bold">{s.symbol}</TableCell>
                <TableCell className="text-right number-mono">{s.trades}</TableCell>
                <TableCell className="text-right number-mono">{s.wins}</TableCell>
                <TableCell className="text-right number-mono">{s.trades - s.wins}</TableCell>
                <TableCell className={`text-right number-mono ${getWinRateColor(s.winRate)}`}>
                  {s.winRate.toFixed(1)}%
                </TableCell>
                <TableCell className="text-right number-mono">{s.expectancy.toFixed(3)}</TableCell>
                <TableCell className={`text-right number-mono font-semibold ${s.totalR >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {s.totalR >= 0 ? "+" : ""}{s.totalR.toFixed(2)}R
                </TableCell>
                <TableCell className="text-right">
                  {s.totalR > 0 && s.trades >= 10 ? (
                    <Badge variant="outline" className="text-emerald-400 border-emerald-500/30">
                      <CheckCircle className="w-3 h-3 mr-1" />EDGE
                    </Badge>
                  ) : s.totalR <= 0 ? (
                    <Badge variant="outline" className="text-red-400 border-red-500/30">
                      <XCircle className="w-3 h-3 mr-1" />NO EDGE
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="text-muted-foreground border-muted-foreground/30">
                      <Minus className="w-3 h-3 mr-1" />INSUFFICIENT
                    </Badge>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="glass-card rounded-md p-4" data-testid="r-distribution">
          <p className="text-sm font-semibold mb-3">R-Multiple Distribution</p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogram}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
                <XAxis
                  dataKey="label"
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                  allowDecimals={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "6px",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {histogram.map((bin, i) => (
                    <Cell key={i} fill={bin.max <= 0 ? "#f87171" : "#34d399"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2 mb-2">
            <button
              onClick={() => setPnlView("monthly")}
              className={`text-xs px-2 py-1 rounded-md transition-colors ${pnlView === "monthly" ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}
              data-testid="button-monthly-pnl"
            >
              Monthly
            </button>
            <button
              onClick={() => setPnlView("weekly")}
              className={`text-xs px-2 py-1 rounded-md transition-colors ${pnlView === "weekly" ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}
              data-testid="button-weekly-pnl"
            >
              Weekly
            </button>
          </div>
          <MonthlyPnlTable data={pnlData} type={pnlView} />
        </div>
      </div>

      {source === "paper" && perf.totalPnlUsdt != null && (
        <div className="grid grid-cols-3 gap-3">
          <div className="glass-card rounded-md p-3" data-testid="stat-total-pnl-usdt">
            <Zap className="w-4 h-4 text-muted-foreground mb-1" />
            <p className="text-xs text-muted-foreground">Total P&L (USDT)</p>
            <p className={`text-xl number-mono font-bold ${(perf.totalPnlUsdt ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {(perf.totalPnlUsdt ?? 0) >= 0 ? "+" : ""}{(perf.totalPnlUsdt ?? 0).toFixed(2)}
            </p>
          </div>
          <div className="glass-card rounded-md p-3" data-testid="stat-total-risk-usdt">
            <ArrowDown className="w-4 h-4 text-muted-foreground mb-1" />
            <p className="text-xs text-muted-foreground">Total Risk (USDT)</p>
            <p className="text-xl number-mono font-bold text-amber-400">
              {(perf.totalRiskUsdt ?? 0).toFixed(2)}
            </p>
          </div>
          <div className="glass-card rounded-md p-3" data-testid="stat-roi">
            <Target className="w-4 h-4 text-muted-foreground mb-1" />
            <p className="text-xs text-muted-foreground">ROI on Risk</p>
            <p className={`text-xl number-mono font-bold ${(perf.totalRiskUsdt ?? 0) > 0 && (perf.totalPnlUsdt ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {(perf.totalRiskUsdt ?? 0) > 0 ? (((perf.totalPnlUsdt ?? 0) / (perf.totalRiskUsdt ?? 1)) * 100).toFixed(1) : "0"}%
            </p>
          </div>
        </div>
      )}

      {source === "paper" && mfeMae && (
        <>
          <div className="grid grid-cols-3 gap-3">
            <div className="col-span-2">
              <MfeScatterPlot data={mfeMae} />
            </div>
            <WastedEdgeCard data={mfeMae} />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <CaptureRatioChart data={mfeMae} />
            <OptimalExitSimulator data={mfeMae} />
          </div>
        </>
      )}

      {neuralAdj && (
        <div className="grid grid-cols-2 gap-3">
          <NeuralManagerPerformance data={neuralAdj} />
          <EdgeDecayAnalysis data={neuralAdj} />
        </div>
      )}

      {source === "paper" && (perf.leverageBreakdown?.length ?? 0) + (leverageStats?.configTiers?.length ?? 0) > 0 && (
        <div className="glass-card rounded-md p-4 space-y-4" data-testid="leverage-analysis">
          <p className="text-sm font-semibold">
            <Scale className="w-4 h-4 inline mr-1" />
            Leverage Intelligence
          </p>

          {/* KPI row */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-background/30 rounded p-3">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Avg Leverage Used</p>
              <p className="number-mono text-base font-bold text-cyan-400" data-testid="text-analytics-avg-lev">
                {leverageStats ? `${leverageStats.closed.avgLeverage.toFixed(1)}x` : perf.leverageBreakdown && perf.leverageBreakdown.length > 0
                  ? `${(perf.leverageBreakdown.reduce((s, l) => s + (l.leverageNum ?? 1) * l.trades, 0) / perf.leverageBreakdown.reduce((s, l) => s + l.trades, 0)).toFixed(1)}x`
                  : "1.0x"}
              </p>
              <p className="text-[10px] text-muted-foreground/50">{leverageStats?.closed.count ?? 0} closed trades</p>
            </div>
            <div className="bg-background/30 rounded p-3">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Peak Leverage</p>
              <p className={`number-mono text-base font-bold ${(leverageStats?.closed.maxLeverage ?? 0) >= 25 ? "text-amber-400" : "text-emerald-400"}`} data-testid="text-analytics-max-lev">
                {leverageStats ? `${leverageStats.closed.maxLeverage}x` : "—"}
              </p>
              <p className="text-[10px] text-muted-foreground/50">max config: {leverageStats?.maxConfigLeverage ?? 50}x</p>
            </div>
            <div className="bg-background/30 rounded p-3">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Best Tier (R)</p>
              {(() => {
                const tiers = leverageStats?.closed.byTier ?? perf.leverageBreakdown ?? [];
                const best = tiers.length > 0 ? tiers.reduce((a, b) => b.totalR > a.totalR ? b : a) : null;
                return (
                  <>
                    <p className="number-mono text-base font-bold text-amber-400" data-testid="text-analytics-best-tier">{best?.tier ?? "—"}</p>
                    <p className="text-[10px] text-muted-foreground/50">{best ? `+${best.totalR.toFixed(2)}R (${best.trades}t)` : "no data"}</p>
                  </>
                );
              })()}
            </div>
          </div>

          {/* Bar chart: Trades by leverage tier */}
          {(leverageStats?.closed.byTier ?? perf.leverageBreakdown ?? []).length > 0 && (
            <div>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">Trades by Tier</p>
              <div className="h-28">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={leverageStats?.closed.byTier ?? perf.leverageBreakdown ?? []} margin={{ top: 2, right: 8, bottom: 2, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="tier" tick={{ fontSize: 10, fill: "#94a3b8" }} />
                    <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} />
                    <Tooltip
                      contentStyle={{ background: "hsl(var(--background))", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "6px", fontSize: 11 }}
                      formatter={(v: number, name: string) => [v, name === "trades" ? "Trades" : "Win%"]}
                    />
                    <Bar dataKey="trades" fill="#06b6d4" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Performance table */}
          {(leverageStats?.closed.byTier ?? perf.leverageBreakdown ?? []).length > 0 && (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Tier</TableHead>
                  <TableHead className="text-xs text-center">Trades</TableHead>
                  <TableHead className="text-xs text-center">Win Rate</TableHead>
                  <TableHead className="text-xs text-right">Total R</TableHead>
                  <TableHead className="text-xs text-right">P&L (USDT)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(leverageStats?.closed.byTier ?? perf.leverageBreakdown ?? []).map((l) => (
                  <TableRow key={l.tier}>
                    <TableCell className="text-sm font-mono text-amber-400">{l.tier}</TableCell>
                    <TableCell className="text-sm text-center number-mono">{l.trades}</TableCell>
                    <TableCell className={`text-sm text-center number-mono ${getWinRateColor(l.winRate)}`}>{l.winRate.toFixed(1)}%</TableCell>
                    <TableCell className={`text-sm text-right number-mono ${l.totalR >= 0 ? "text-emerald-400" : "text-red-400"}`}>{l.totalR >= 0 ? "+" : ""}{l.totalR.toFixed(2)}R</TableCell>
                    <TableCell className={`text-sm text-right number-mono ${l.totalPnlUsdt >= 0 ? "text-emerald-400" : "text-red-400"}`}>{l.totalPnlUsdt >= 0 ? "+" : ""}{l.totalPnlUsdt.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}

          {/* Config tiers reference */}
          {leverageStats?.configTiers && (
            <div>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">Score → Leverage Mapping</p>
              <div className="flex flex-wrap gap-1.5">
                {leverageStats.configTiers.map((t, i) => (
                  <span key={i} className="text-[10px] font-mono bg-background/40 border border-white/10 rounded px-2 py-0.5 text-muted-foreground">
                    ≥{t.minScore} → <span className="text-amber-400">{t.leverage}x</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
