import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
  BarChart, Bar, Cell, LineChart, Line,
} from "recharts";
import {
  TrendingUp, TrendingDown, Hash, Percent, Target, ArrowDown, Trophy, Skull,
  Scale, CheckCircle, XCircle, Minus, BarChart3, ArrowUpDown, Clock, Zap,
  Activity, Calendar, Timer,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Card } from "@/components/ui/card";
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
}

interface EquityPoint {
  ts: number;
  r: number;
  tradeR: number;
  symbol: string;
  side: string;
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

  const { data: perf, isLoading: perfLoading } = useQuery<PerformanceData>({
    queryKey: ["/api/v5/performance"],
  });

  const { data: equity, isLoading: equityLoading } = useQuery<EquityPoint[]>({
    queryKey: ["/api/v5/equity-curve", "all"],
    queryFn: async () => {
      const res = await fetch("/api/v5/equity-curve?range=all");
      if (!res.ok) throw new Error("Failed to fetch equity curve");
      return res.json();
    },
  });

  const isLoading = perfLoading || equityLoading;

  if (isLoading) {
    return (
      <div className="p-4 space-y-4" data-testid="analytics">
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

  if (!perf || perf.totalTrades === 0) {
    return (
      <div className="p-4 space-y-4 flex items-center justify-center min-h-[60vh]" data-testid="analytics">
        <div className="text-center space-y-3">
          <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto" />
          <p className="text-muted-foreground text-sm" data-testid="text-empty-state">
            No trading data yet. Signals will appear once the v5 model starts generating trades.
          </p>
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
    </div>
  );
}
