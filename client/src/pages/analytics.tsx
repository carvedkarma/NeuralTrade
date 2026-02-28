import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
  BarChart, Bar, Cell,
} from "recharts";
import {
  TrendingUp, TrendingDown, Hash, Percent, Target, ArrowDown, Trophy, Skull,
  Scale, CheckCircle, XCircle, Minus, BarChart3, ArrowUpDown,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";

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
  perSymbol: Array<{
    symbol: string;
    trades: number;
    wins: number;
    winRate: number;
    totalR: number;
    expectancy: number;
  }>;
}

interface EquityPoint {
  ts: number;
  r: number;
  tradeR: number;
  symbol: string;
  side: string;
}

interface TradeRecord {
  rMultiple: number;
  symbol: string;
  side: string;
  entryTime: number;
  exitTime: number;
  outcome: string;
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

function StatCard({ icon: Icon, label, value, colorClass, glow }: {
  icon: typeof TrendingUp;
  label: string;
  value: string;
  colorClass: string;
  glow?: string;
}) {
  return (
    <div className={`glass-card rounded-lg p-4 ${glow ?? ""}`} data-testid={`stat-${label.toLowerCase().replace(/[\s\/]/g, "-")}`}>
      <Icon className="w-4 h-4 text-muted-foreground mb-2" />
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`text-2xl number-mono font-bold ${colorClass}`}>{value}</p>
    </div>
  );
}

function computeHistogram(trades: TradeRecord[]) {
  const bins = [
    { label: "< -2R", min: -Infinity, max: -2, count: 0 },
    { label: "-2 to -1", min: -2, max: -1, count: 0 },
    { label: "-1 to 0", min: -1, max: 0, count: 0 },
    { label: "0 to 1", min: 0, max: 1, count: 0 },
    { label: "1 to 2", min: 1, max: 2, count: 0 },
    { label: "2 to 3", min: 2, max: 3, count: 0 },
    { label: "> 3R", min: 3, max: Infinity, count: 0 },
  ];
  for (const t of trades) {
    const r = t.rMultiple;
    for (const bin of bins) {
      if (r >= bin.min && r < bin.max) {
        bin.count++;
        break;
      }
    }
  }
  return bins;
}

function DirectionalStats({ trades, side }: { trades: TradeRecord[]; side: string }) {
  const filtered = trades.filter((t) => t.side === side);
  const count = filtered.length;
  const wins = filtered.filter((t) => t.outcome === "WIN" || t.rMultiple > 0).length;
  const wr = count > 0 ? (wins / count) * 100 : 0;
  const totalR = filtered.reduce((s, t) => s + t.rMultiple, 0);

  return (
    <div className="glass-card rounded-lg p-4 flex-1" data-testid={`directional-${side.toLowerCase()}`}>
      <p className={`text-sm font-semibold mb-3 ${side === "LONG" ? "text-emerald-400" : "text-red-400"}`}>
        {side} Performance
      </p>
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">Trades</span>
          <span className="number-mono text-sm font-medium">{count}</span>
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">Win Rate</span>
          <span className={`number-mono text-sm font-medium ${getWinRateColor(wr)}`}>{wr.toFixed(1)}%</span>
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">Total R</span>
          <span className={`number-mono text-sm font-medium ${totalR >= 0 ? "text-emerald-400" : "text-red-400"}`}>
            {totalR >= 0 ? "+" : ""}{totalR.toFixed(2)}R
          </span>
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

export default function Analytics() {
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

  const { data: trades, isLoading: tradesLoading } = useQuery<TradeRecord[]>({
    queryKey: ["/api/v5/trades", 500],
    queryFn: async () => {
      const res = await fetch("/api/v5/trades?limit=500");
      if (!res.ok) throw new Error("Failed to fetch trades");
      return res.json();
    },
  });

  const isLoading = perfLoading || equityLoading || tradesLoading;

  if (isLoading) {
    return (
      <div className="p-4 space-y-4" data-testid="analytics">
        <div className="grid grid-cols-4 gap-3">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="glass-card rounded-lg p-4 h-24 shimmer" />
          ))}
        </div>
        <div className="glass-card rounded-lg h-72 shimmer" />
        <div className="glass-card rounded-lg h-48 shimmer" />
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
  const histogram = trades ? computeHistogram(trades) : [];

  return (
    <div className="p-4 space-y-4" data-testid="analytics">
      <div className="grid grid-cols-4 gap-3">
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
          icon={ArrowDown}
          label="Max Drawdown"
          value={`${perf.maxDrawdown > 0 ? "-" : ""}${Math.abs(perf.maxDrawdown).toFixed(2)}R`}
          colorClass="text-red-400"
          glow="glow-red"
        />
        <StatCard
          icon={Trophy}
          label="Best Trade"
          value={`+${perf.bestTrade.toFixed(2)}R`}
          colorClass="text-emerald-400"
          glow="glow-green"
        />
        <StatCard
          icon={Skull}
          label="Worst Trade"
          value={`${perf.worstTrade.toFixed(2)}R`}
          colorClass="text-red-400"
          glow="glow-red"
        />
        <StatCard
          icon={Scale}
          label="Avg Win/Loss"
          value={winLossRatio.toFixed(2)}
          colorClass={winLossRatio >= 1.5 ? "text-emerald-400" : winLossRatio >= 1 ? "text-amber-400" : "text-red-400"}
        />
      </div>

      <div className="glass-card rounded-lg p-4" data-testid="equity-curve">
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
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                labelFormatter={(ts: number) => new Date(ts).toLocaleDateString()}
                formatter={(value: number, name: string) => {
                  if (name === "r") return [`${value.toFixed(2)}R`, "Cumulative R"];
                  if (name === "tradeR") return [`${value.toFixed(2)}R`, "Trade R"];
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

      <div className="glass-card rounded-lg p-4" data-testid="per-symbol-table">
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
        <div className="glass-card rounded-lg p-4" data-testid="r-distribution">
          <p className="text-sm font-semibold mb-3">R-Multiple Distribution</p>
          <div className="h-64">
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
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {histogram.map((bin, i) => (
                    <Cell key={i} fill={bin.max <= 0 ? "#f87171" : "#34d399"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-card rounded-lg p-4" data-testid="directional-analysis">
          <p className="text-sm font-semibold mb-3">
            <ArrowUpDown className="w-4 h-4 inline mr-1" />
            Directional Analysis
          </p>
          <div className="flex gap-3">
            <DirectionalStats trades={trades || []} side="LONG" />
            <DirectionalStats trades={trades || []} side="SHORT" />
          </div>
        </div>
      </div>
    </div>
  );
}
