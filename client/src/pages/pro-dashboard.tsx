import { useState, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { useWebSocket } from "@/hooks/use-websocket";
import { format } from "date-fns";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Clock,
  Wifi,
  WifiOff,
  Users,
  RefreshCw,
  Download,
  Target,
  Shield,
  Zap,
  Brain,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Eye,
  BookOpen,
  Lightbulb,
  Award,
  Timer,
  Database,
  Cpu,
  CircleDot,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface ProSummary {
  window: string;
  openPositions: number;
  closedTrades: number;
  winRate: number;
  totalNetR: number;
  avgR: number;
  bestTrade: number;
  worstTrade: number;
  maxDrawdown: number;
  totalCycles: number;
  holdReasons: Record<string, number>;
  symbolStats: Record<string, { trades: number; wins: number; netR: number }>;
  equityCurve: Array<{ ts: number; netR: number; symbol: string }>;
}

interface CycleEntry {
  id: number;
  symbol: string;
  cycleTs: string;
  price: number;
  pEnter: number;
  htfH1Trend: string;
  htfH4Trend: string;
  slopeOk: boolean;
  rangeOk: boolean;
  direction: string;
  thresholdUsed: number;
  decision: string;
  reasons: string[];
  createdAt: string;
}

interface TradeEntry {
  id: number;
  symbol: string;
  side: string;
  entryTime: string;
  entryPrice: number;
  exitTime: string | null;
  exitPrice: number | null;
  stopLoss: number;
  takeProfit: number;
  sizePct: number;
  pEnter: number;
  costsBps: number;
  outcome: string | null;
  grossR: number | null;
  netR: number | null;
  sizedR: number | null;
  status: string;
  reasons: string[];
}

interface TradeEvent {
  id: number;
  ts: string;
  type: string;
  message: string;
  data?: Record<string, unknown>;
}

interface LearningRun {
  id: number;
  symbol: string;
  startAt: string;
  endAt: string | null;
  dataFrom: string;
  dataTo: string;
  newBars: number;
  newTrades: number;
  epochs: number;
  bestValLoss: number | null;
  prAuc: number | null;
  pfNet: number | null;
  eNet: number | null;
  profitableRegimes: number;
  totalRegimes: number;
  promoted: boolean;
  reason: string | null;
  modelVersion: string | null;
  metricsJson: Record<string, unknown> | null;
  status: string;
}

interface HealthEntry {
  id: number;
  ts: string;
  component: string;
  status: string;
  message: string;
}

interface ModelStats {
  symbol: string;
  version: string;
  prAuc: number;
  pfNet: number;
  eNet: number;
  promotedAt: string;
  [key: string]: unknown;
}

function StatCardSkeleton() {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-4 rounded-full" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-8 w-20 mb-1" />
        <Skeleton className="h-3 w-16" />
      </CardContent>
    </Card>
  );
}

function TableSkeleton({ rows = 5, cols = 6 }: { rows?: number; cols?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex gap-4">
          {Array.from({ length: cols }).map((_, j) => (
            <Skeleton key={j} className="h-6 flex-1" />
          ))}
        </div>
      ))}
    </div>
  );
}

function formatR(val: number | null | undefined): string {
  if (val === null || val === undefined) return "—";
  return `${val >= 0 ? "+" : ""}${val.toFixed(2)}R`;
}

function formatPct(val: number | null | undefined): string {
  if (val === null || val === undefined) return "—";
  return `${(val * 100).toFixed(1)}%`;
}

function formatPrice(val: number | null | undefined): string {
  if (val === null || val === undefined) return "—";
  return val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatTs(ts: string | number | null | undefined): string {
  if (!ts) return "—";
  try {
    return format(new Date(ts), "MMM dd HH:mm");
  } catch {
    return "—";
  }
}

function decisionColor(decision: string): string {
  switch (decision?.toUpperCase()) {
    case "ENTER": return "text-emerald-400";
    case "GATE_FAIL": return "text-amber-400";
    case "HOLD": return "text-muted-foreground";
    case "COOLDOWN": return "text-blue-400";
    default: return "text-muted-foreground";
  }
}

function decisionBadgeVariant(decision: string): "default" | "secondary" | "destructive" | "outline" {
  switch (decision?.toUpperCase()) {
    case "ENTER": return "default";
    case "GATE_FAIL": return "destructive";
    default: return "secondary";
  }
}

function trendIcon(trend: string | number | null | undefined) {
  if (trend === null || trend === undefined) {
    return <Minus className="h-3.5 w-3.5 text-muted-foreground" />;
  }
  if (typeof trend === "number") {
    if (trend > 0.1) return <ArrowUpRight className="h-3.5 w-3.5 text-emerald-400" />;
    if (trend < -0.1) return <ArrowDownRight className="h-3.5 w-3.5 text-red-400" />;
    return <Minus className="h-3.5 w-3.5 text-muted-foreground" />;
  }
  const s = String(trend).toLowerCase();
  if (s.includes("up") || s.includes("bull")) {
    return <ArrowUpRight className="h-3.5 w-3.5 text-emerald-400" />;
  }
  if (s.includes("down") || s.includes("bear")) {
    return <ArrowDownRight className="h-3.5 w-3.5 text-red-400" />;
  }
  return <Minus className="h-3.5 w-3.5 text-muted-foreground" />;
}

export default function ProDashboard() {
  const { status: wsStatus, lastUpdated, clientCount } = useWebSocket();
  const [window, setWindow] = useState<string>("24h");
  const [cycleSymbolFilter, setCycleSymbolFilter] = useState<string>("all");
  const [tradeSymbolFilter, setTradeSymbolFilter] = useState<string>("all");
  const [tradeOutcomeFilter, setTradeOutcomeFilter] = useState<string>("all");
  const [selectedTradeId, setSelectedTradeId] = useState<number | null>(null);
  const [learningSymbolFilter, setLearningSymbolFilter] = useState<string>("all");

  const { data: summary, isLoading: summaryLoading } = useQuery<ProSummary>({
    queryKey: ["/api/pro/summary", `?window=${window}`],
    refetchInterval: 15000,
  });

  const { data: cycles, isLoading: cyclesLoading } = useQuery<CycleEntry[]>({
    queryKey: ["/api/pro/cycles"],
    refetchInterval: 10000,
  });

  const { data: trades, isLoading: tradesLoading } = useQuery<TradeEntry[]>({
    queryKey: ["/api/pro/trades"],
    refetchInterval: 15000,
  });

  const { data: tradeEvents } = useQuery<TradeEvent[]>({
    queryKey: ["/api/pro/trades", String(selectedTradeId), "events"],
    enabled: selectedTradeId !== null,
  });

  const { data: learningRuns, isLoading: learningLoading } = useQuery<LearningRun[]>({
    queryKey: ["/api/pro/learning-runs"],
    refetchInterval: 30000,
  });

  const { data: health } = useQuery<HealthEntry[]>({
    queryKey: ["/api/pro/health"],
    refetchInterval: 10000,
  });

  const { data: modelStats } = useQuery<ModelStats[]>({
    queryKey: ["/api/live/learning-stats/latest"],
    refetchInterval: 30000,
  });

  const filteredCycles = useMemo(() => {
    if (!cycles) return [];
    if (cycleSymbolFilter === "all") return cycles;
    return cycles.filter((c) => c.symbol === cycleSymbolFilter);
  }, [cycles, cycleSymbolFilter]);

  const filteredTrades = useMemo(() => {
    if (!trades) return [];
    let result = trades;
    if (tradeSymbolFilter !== "all") {
      result = result.filter((t) => t.symbol === tradeSymbolFilter);
    }
    if (tradeOutcomeFilter !== "all") {
      result = result.filter((t) => t.outcome === tradeOutcomeFilter);
    }
    return result;
  }, [trades, tradeSymbolFilter, tradeOutcomeFilter]);

  const filteredLearningRuns = useMemo(() => {
    if (!learningRuns) return [];
    if (learningSymbolFilter === "all") return learningRuns;
    return learningRuns.filter((r) => r.symbol === learningSymbolFilter);
  }, [learningRuns, learningSymbolFilter]);

  const uniqueSymbols = useMemo(() => {
    const symbols = new Set<string>();
    cycles?.forEach((c) => symbols.add(c.symbol));
    trades?.forEach((t) => symbols.add(t.symbol));
    if (summary?.symbolStats) {
      Object.keys(summary.symbolStats).forEach((s) => symbols.add(s));
    }
    return Array.from(symbols).sort();
  }, [cycles, trades, summary]);

  const healthMap = useMemo(() => {
    const map: Record<string, HealthEntry> = {};
    health?.forEach((h) => {
      if (!map[h.component] || new Date(h.ts) > new Date(map[h.component].ts)) {
        map[h.component] = h;
      }
    });
    return map;
  }, [health]);

  const equityCurveData = useMemo(() => {
    if (!summary?.equityCurve) return [];
    let cumR = 0;
    let peak = 0;
    return summary.equityCurve.map((pt) => {
      cumR += pt.netR;
      peak = Math.max(peak, cumR);
      const dd = peak > 0 ? ((peak - cumR) / peak) * 100 : 0;
      return {
        ts: pt.ts,
        label: formatTs(pt.ts),
        cumR: parseFloat(cumR.toFixed(3)),
        drawdown: parseFloat((-dd).toFixed(2)),
        symbol: pt.symbol,
        netR: pt.netR,
      };
    });
  }, [summary]);

  const symbolBarData = useMemo(() => {
    if (!summary?.symbolStats) return [];
    return Object.entries(summary.symbolStats).map(([symbol, stats]) => ({
      symbol,
      netR: parseFloat(stats.netR.toFixed(2)),
      trades: stats.trades,
      wins: stats.wins,
      winRate: stats.trades > 0 ? parseFloat(((stats.wins / stats.trades) * 100).toFixed(1)) : 0,
    }));
  }, [summary]);

  const netRDistribution = useMemo(() => {
    if (!trades) return [];
    const buckets: Record<string, number> = {};
    trades.forEach((t) => {
      if (t.netR === null) return;
      const bucket = (Math.round(t.netR * 2) / 2).toFixed(1);
      buckets[bucket] = (buckets[bucket] || 0) + 1;
    });
    return Object.entries(buckets)
      .map(([r, count]) => ({ r: parseFloat(r), count }))
      .sort((a, b) => a.r - b.r);
  }, [trades]);

  const scatterData = useMemo(() => {
    if (!trades) return [];
    return trades
      .filter((t) => t.pEnter !== null && t.netR !== null)
      .map((t) => ({
        pEnter: parseFloat((t.pEnter * 100).toFixed(1)),
        netR: parseFloat((t.netR ?? 0).toFixed(2)),
        outcome: t.outcome,
      }));
  }, [trades]);

  const holdingTimeData = useMemo(() => {
    if (!trades) return [];
    const buckets: Record<string, number> = {};
    trades.forEach((t) => {
      if (!t.entryTime || !t.exitTime) return;
      const hours = Math.round(
        (new Date(t.exitTime).getTime() - new Date(t.entryTime).getTime()) / (1000 * 60 * 60)
      );
      const bucket = `${hours}h`;
      buckets[bucket] = (buckets[bucket] || 0) + 1;
    });
    return Object.entries(buckets)
      .map(([h, count]) => ({ hours: h, count }))
      .sort((a, b) => parseInt(a.hours) - parseInt(b.hours))
      .slice(0, 20);
  }, [trades]);

  const handleCsvExport = useCallback(() => {
    const params = new URLSearchParams();
    params.set("format", "csv");
    if (tradeSymbolFilter !== "all") params.set("symbol", tradeSymbolFilter);
    if (tradeOutcomeFilter !== "all") params.set("outcome", tradeOutcomeFilter);
    const url = `/api/pro/trades?${params.toString()}`;
    const a = document.createElement("a");
    a.href = url;
    a.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  }, [tradeSymbolFilter, tradeOutcomeFilter]);

  const insights = useMemo(() => {
    if (!summary) return [];
    const items: Array<{ title: string; description: string; icon: typeof Lightbulb; color: string }> = [];

    if (summary.holdReasons && Object.keys(summary.holdReasons).length > 0) {
      const sorted = Object.entries(summary.holdReasons)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 3);
      const topReasons = sorted.map(([reason, count]) => `${reason} (${count}x)`).join(", ");
      items.push({
        title: "Top Hold Reasons (24h)",
        description: `The system held back primarily due to: ${topReasons}. Consider reviewing gate thresholds if these are too restrictive.`,
        icon: AlertTriangle,
        color: "text-amber-400",
      });
    }

    if (summary.symbolStats && Object.keys(summary.symbolStats).length > 0) {
      const best = Object.entries(summary.symbolStats).sort(
        ([, a], [, b]) => b.netR - a.netR
      )[0];
      if (best) {
        items.push({
          title: "Best Performing Symbol",
          description: `${best[0]} leads with ${formatR(best[1].netR)} across ${best[1].trades} trades (${((best[1].wins / Math.max(best[1].trades, 1)) * 100).toFixed(0)}% win rate).`,
          icon: Award,
          color: "text-emerald-400",
        });
      }
    }

    if (summary.winRate !== undefined) {
      const wr = summary.winRate * 100;
      const assessment =
        wr >= 60
          ? "Strong edge detected. Current strategies are well-calibrated."
          : wr >= 45
            ? "Win rate is acceptable. Ensure risk:reward remains favorable."
            : "Win rate is below threshold. Review entry criteria and gate conditions.";
      items.push({
        title: "Win Rate Analysis",
        description: `Current win rate: ${wr.toFixed(1)}%. ${assessment}`,
        icon: Target,
        color: wr >= 60 ? "text-emerald-400" : wr >= 45 ? "text-amber-400" : "text-red-400",
      });
    }

    if (summary.avgR !== undefined) {
      const commentary =
        summary.avgR > 0.5
          ? "Excellent average expectancy per trade."
          : summary.avgR > 0
            ? "Positive but modest edge. Consider tightening stops or improving entry precision."
            : "Negative average R — system is losing money per trade on average. Urgent review needed.";
      items.push({
        title: "Average R Per Trade",
        description: `${formatR(summary.avgR)} per trade. ${commentary}`,
        icon: BarChart3,
        color: summary.avgR > 0 ? "text-emerald-400" : "text-red-400",
      });
    }

    return items;
  }, [summary]);

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-background via-background to-background/80"
      data-testid="pro-dashboard-container"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/3 pointer-events-none" />

      <header
        className="sticky top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl"
        data-testid="pro-header"
      >
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-md bg-primary/10">
                <Activity className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h1 className="text-lg font-semibold" data-testid="text-pro-title">
                  Pro Dashboard
                </h1>
                <p className="text-xs text-muted-foreground">Realtime Trading Analytics</p>
              </div>
            </div>

            <div className="flex items-center gap-3 flex-wrap">
              <Badge
                variant="secondary"
                className={`text-xs flex items-center gap-1 ${
                  wsStatus === "connected"
                    ? "bg-emerald-500/20 text-emerald-400"
                    : wsStatus === "connecting"
                      ? "bg-amber-500/20 text-amber-400"
                      : "bg-red-500/20 text-red-400"
                }`}
                data-testid="badge-ws-status"
              >
                {wsStatus === "connected" ? (
                  <Wifi className="h-3 w-3" />
                ) : (
                  <WifiOff className="h-3 w-3" />
                )}
                {wsStatus === "connected"
                  ? "Connected"
                  : wsStatus === "connecting"
                    ? "Connecting..."
                    : "Disconnected"}
              </Badge>

              <Badge variant="outline" className="text-xs flex items-center gap-1" data-testid="badge-client-count">
                <Users className="h-3 w-3" />
                {clientCount} client{clientCount !== 1 ? "s" : ""}
              </Badge>

              <div className="flex items-center gap-1.5 text-xs text-muted-foreground" data-testid="text-last-updated">
                <Clock className="h-3.5 w-3.5" />
                {lastUpdated ? format(new Date(lastUpdated), "HH:mm:ss") : "—"}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="relative container mx-auto px-4 py-6">
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="mb-6 flex-wrap" data-testid="pro-tabs-list">
            <TabsTrigger value="overview" data-testid="tab-overview">
              <Eye className="h-4 w-4 mr-1.5" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="cycles" data-testid="tab-cycles">
              <RefreshCw className="h-4 w-4 mr-1.5" />
              Cycles
            </TabsTrigger>
            <TabsTrigger value="trades" data-testid="tab-trades">
              <BookOpen className="h-4 w-4 mr-1.5" />
              Trades
            </TabsTrigger>
            <TabsTrigger value="learning" data-testid="tab-learning">
              <Brain className="h-4 w-4 mr-1.5" />
              Learning
            </TabsTrigger>
            <TabsTrigger value="performance" data-testid="tab-performance">
              <BarChart3 className="h-4 w-4 mr-1.5" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="insights" data-testid="tab-insights">
              <Lightbulb className="h-4 w-4 mr-1.5" />
              Insights
            </TabsTrigger>
          </TabsList>

          {/* ===== TAB 1: OVERVIEW ===== */}
          <TabsContent value="overview" className="mt-0 space-y-6">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <h2 className="text-xl font-semibold">System Overview</h2>
              <Select value={window} onValueChange={setWindow} data-testid="select-window">
                <SelectTrigger className="w-28" data-testid="button-window-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="24h">24h</SelectItem>
                  <SelectItem value="7d">7 Days</SelectItem>
                  <SelectItem value="30d">30 Days</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {summaryLoading ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {Array.from({ length: 6 }).map((_, i) => (
                  <StatCardSkeleton key={i} />
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <Card data-testid="card-open-positions">
                  <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Open Positions
                    </CardTitle>
                    <Target className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold" data-testid="text-open-positions">
                      {summary?.openPositions ?? 0}
                    </div>
                  </CardContent>
                </Card>

                <Card data-testid="card-closed-trades">
                  <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Closed Trades
                    </CardTitle>
                    <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold" data-testid="text-closed-trades">
                      {summary?.closedTrades ?? 0}
                    </div>
                  </CardContent>
                </Card>

                <Card data-testid="card-win-rate">
                  <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Win Rate
                    </CardTitle>
                    <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div
                      className={`text-2xl font-bold ${
                        (summary?.winRate ?? 0) >= 0.5 ? "text-emerald-400" : "text-red-400"
                      }`}
                      data-testid="text-win-rate"
                    >
                      {formatPct(summary?.winRate)}
                    </div>
                  </CardContent>
                </Card>

                <Card data-testid="card-total-net-r">
                  <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Total Net R
                    </CardTitle>
                    <Zap className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div
                      className={`text-2xl font-bold ${
                        (summary?.totalNetR ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                      }`}
                      data-testid="text-total-net-r"
                    >
                      {formatR(summary?.totalNetR)}
                    </div>
                  </CardContent>
                </Card>

                <Card data-testid="card-avg-r">
                  <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Avg R/Trade
                    </CardTitle>
                    <BarChart3 className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div
                      className={`text-2xl font-bold ${
                        (summary?.avgR ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                      }`}
                      data-testid="text-avg-r"
                    >
                      {formatR(summary?.avgR)}
                    </div>
                  </CardContent>
                </Card>

                <Card data-testid="card-max-drawdown">
                  <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Max Drawdown
                    </CardTitle>
                    <TrendingDown className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-red-400" data-testid="text-max-drawdown">
                      {formatR(summary?.maxDrawdown)}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Health Status */}
            <Card data-testid="card-health-status">
              <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">System Health</CardTitle>
                <Shield className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-4">
                  {[
                    {
                      label: "Data Feed",
                      ok: healthMap["data"]?.status === "ok" || healthMap["data"]?.status === "healthy",
                      component: "data",
                    },
                    {
                      label: "Model",
                      ok: healthMap["model"]?.status === "ok" || healthMap["model"]?.status === "healthy",
                      component: "model",
                    },
                    {
                      label: "WebSocket",
                      ok: wsStatus === "connected",
                      component: "ws",
                    },
                    {
                      label: "Last Update",
                      ok: lastUpdated ? Date.now() - lastUpdated < 60000 : false,
                      component: "update",
                    },
                  ].map((item) => (
                    <div
                      key={item.component}
                      className="flex items-center gap-2"
                      data-testid={`text-health-${item.component}`}
                    >
                      <div
                        className={`h-2.5 w-2.5 rounded-full ${
                          item.ok ? "bg-emerald-400" : "bg-red-400"
                        }`}
                      />
                      <span className="text-sm text-muted-foreground">{item.label}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Hold Reasons */}
              <Card data-testid="card-hold-reasons">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Hold Reasons Breakdown</CardTitle>
                  <CardDescription>Why the system isn't trading</CardDescription>
                </CardHeader>
                <CardContent>
                  {summary?.holdReasons && Object.keys(summary.holdReasons).length > 0 ? (
                    <div className="space-y-2">
                      {Object.entries(summary.holdReasons)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 8)
                        .map(([reason, count]) => {
                          const total = Object.values(summary.holdReasons).reduce((a, b) => a + b, 0);
                          const pct = total > 0 ? (count / total) * 100 : 0;
                          return (
                            <div key={reason} className="space-y-1">
                              <div className="flex items-center justify-between gap-2 flex-wrap">
                                <span className="text-sm truncate max-w-[200px]">{reason}</span>
                                <span className="text-xs text-muted-foreground">
                                  {count}x ({pct.toFixed(0)}%)
                                </span>
                              </div>
                              <div className="h-1.5 rounded-full bg-muted">
                                <div
                                  className="h-full rounded-full bg-amber-400/60"
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No hold reasons recorded</p>
                  )}
                </CardContent>
              </Card>

              {/* Per-Symbol Performance */}
              <Card data-testid="card-symbol-performance">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Per-Symbol Performance</CardTitle>
                  <CardDescription>Net R by symbol</CardDescription>
                </CardHeader>
                <CardContent>
                  {summary?.symbolStats && Object.keys(summary.symbolStats).length > 0 ? (
                    <div className="space-y-3">
                      {Object.entries(summary.symbolStats)
                        .sort(([, a], [, b]) => b.netR - a.netR)
                        .map(([symbol, stats]) => (
                          <div
                            key={symbol}
                            className="flex items-center justify-between gap-4 flex-wrap"
                            data-testid={`text-symbol-${symbol}`}
                          >
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs">
                                {symbol}
                              </Badge>
                              <span className="text-xs text-muted-foreground">
                                {stats.trades} trades
                              </span>
                            </div>
                            <div className="flex items-center gap-3">
                              <span className="text-xs text-muted-foreground">
                                WR: {((stats.wins / Math.max(stats.trades, 1)) * 100).toFixed(0)}%
                              </span>
                              <span
                                className={`text-sm font-semibold ${
                                  stats.netR >= 0 ? "text-emerald-400" : "text-red-400"
                                }`}
                              >
                                {formatR(stats.netR)}
                              </span>
                            </div>
                          </div>
                        ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No symbol data available</p>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ===== TAB 2: CYCLE MONITOR ===== */}
          <TabsContent value="cycles" className="mt-0 space-y-4">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <h2 className="text-xl font-semibold">Cycle Monitor</h2>
              <Select value={cycleSymbolFilter} onValueChange={setCycleSymbolFilter}>
                <SelectTrigger className="w-36" data-testid="button-cycle-symbol-filter">
                  <SelectValue placeholder="All Symbols" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Symbols</SelectItem>
                  {uniqueSymbols.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Card data-testid="card-cycle-table">
              <CardContent className="p-0">
                {cyclesLoading ? (
                  <div className="p-6">
                    <TableSkeleton rows={8} cols={8} />
                  </div>
                ) : filteredCycles.length === 0 ? (
                  <div className="p-6 text-center text-muted-foreground">
                    No cycle data available
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Time</TableHead>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Price</TableHead>
                        <TableHead>p_enter</TableHead>
                        <TableHead>H1</TableHead>
                        <TableHead>H4</TableHead>
                        <TableHead>Slope</TableHead>
                        <TableHead>Range</TableHead>
                        <TableHead>Dir</TableHead>
                        <TableHead>Threshold</TableHead>
                        <TableHead>Decision</TableHead>
                        <TableHead>Reasons</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredCycles.slice(0, 50).map((cycle) => (
                        <TableRow key={cycle.id} data-testid={`row-cycle-${cycle.id}`}>
                          <TableCell className="text-xs whitespace-nowrap">
                            {formatTs(cycle.cycleTs)}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="text-xs">
                              {cycle.symbol}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {formatPrice(cycle.price)}
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {(cycle.pEnter * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell>{trendIcon(cycle.htfH1Trend)}</TableCell>
                          <TableCell>{trendIcon(cycle.htfH4Trend)}</TableCell>
                          <TableCell>
                            {cycle.slopeOk ? (
                              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                            ) : (
                              <XCircle className="h-4 w-4 text-red-400" />
                            )}
                          </TableCell>
                          <TableCell>
                            {cycle.rangeOk ? (
                              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                            ) : (
                              <XCircle className="h-4 w-4 text-red-400" />
                            )}
                          </TableCell>
                          <TableCell className="text-xs">{cycle.direction}</TableCell>
                          <TableCell className="text-xs font-mono">
                            {(cycle.thresholdUsed * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant={decisionBadgeVariant(cycle.decision)}
                              className={`text-xs ${decisionColor(cycle.decision)}`}
                            >
                              {cycle.decision}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground max-w-[200px] truncate">
                            {Array.isArray(cycle.reasons) ? cycle.reasons.join(", ") : String(cycle.reasons || "")}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* ===== TAB 3: TRADE JOURNAL ===== */}
          <TabsContent value="trades" className="mt-0 space-y-4">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <h2 className="text-xl font-semibold">Trade Journal</h2>
              <div className="flex items-center gap-2 flex-wrap">
                <Select value={tradeSymbolFilter} onValueChange={setTradeSymbolFilter}>
                  <SelectTrigger className="w-36" data-testid="button-trade-symbol-filter">
                    <SelectValue placeholder="All Symbols" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Symbols</SelectItem>
                    {uniqueSymbols.map((s) => (
                      <SelectItem key={s} value={s}>
                        {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={tradeOutcomeFilter} onValueChange={setTradeOutcomeFilter}>
                  <SelectTrigger className="w-28" data-testid="button-trade-outcome-filter">
                    <SelectValue placeholder="All Outcomes" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="TP">TP</SelectItem>
                    <SelectItem value="SL">SL</SelectItem>
                    <SelectItem value="EXPIRE">EXPIRE</SelectItem>
                  </SelectContent>
                </Select>
                <Button variant="outline" onClick={handleCsvExport} data-testid="button-csv-export">
                  <Download className="h-4 w-4 mr-1.5" />
                  CSV
                </Button>
              </div>
            </div>

            <Card data-testid="card-trade-table">
              <CardContent className="p-0">
                {tradesLoading ? (
                  <div className="p-6">
                    <TableSkeleton rows={8} cols={8} />
                  </div>
                ) : filteredTrades.length === 0 ? (
                  <div className="p-6 text-center text-muted-foreground">
                    No trades found
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Side</TableHead>
                        <TableHead>Entry</TableHead>
                        <TableHead>Exit</TableHead>
                        <TableHead>Entry $</TableHead>
                        <TableHead>Exit $</TableHead>
                        <TableHead>SL</TableHead>
                        <TableHead>TP</TableHead>
                        <TableHead>Size</TableHead>
                        <TableHead>Outcome</TableHead>
                        <TableHead>Gross R</TableHead>
                        <TableHead>Net R</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredTrades.slice(0, 100).map((trade) => (
                        <TableRow
                          key={trade.id}
                          className="cursor-pointer"
                          onClick={() => setSelectedTradeId(trade.id)}
                          data-testid={`row-trade-${trade.id}`}
                        >
                          <TableCell>
                            <Badge variant="outline" className="text-xs">
                              {trade.symbol}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant={trade.side === "LONG" ? "default" : "destructive"}
                              className="text-xs"
                            >
                              {trade.side}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-xs whitespace-nowrap">
                            {formatTs(trade.entryTime)}
                          </TableCell>
                          <TableCell className="text-xs whitespace-nowrap">
                            {formatTs(trade.exitTime)}
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {formatPrice(trade.entryPrice)}
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {formatPrice(trade.exitPrice)}
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {formatPrice(trade.stopLoss)}
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {formatPrice(trade.takeProfit)}
                          </TableCell>
                          <TableCell className="text-xs">
                            {(trade.sizePct * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell>
                            {trade.outcome && (
                              <Badge
                                variant={
                                  trade.outcome === "TP"
                                    ? "default"
                                    : trade.outcome === "SL"
                                      ? "destructive"
                                      : "secondary"
                                }
                                className="text-xs"
                              >
                                {trade.outcome}
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell
                            className={`text-xs font-mono ${
                              (trade.grossR ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                            }`}
                          >
                            {formatR(trade.grossR)}
                          </TableCell>
                          <TableCell
                            className={`text-xs font-mono font-semibold ${
                              (trade.netR ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                            }`}
                          >
                            {formatR(trade.netR)}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="text-xs">
                              {trade.status}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>

            {/* Trade Events Sheet */}
            <Sheet
              open={selectedTradeId !== null}
              onOpenChange={(open) => {
                if (!open) setSelectedTradeId(null);
              }}
            >
              <SheetContent side="right" className="w-full sm:max-w-lg overflow-y-auto">
                <SheetHeader>
                  <SheetTitle>Trade Timeline</SheetTitle>
                  <SheetDescription>
                    Events for trade #{selectedTradeId}
                  </SheetDescription>
                </SheetHeader>
                <div className="mt-6 space-y-4">
                  {tradeEvents && tradeEvents.length > 0 ? (
                    tradeEvents.map((event, idx) => (
                      <div
                        key={event.id || idx}
                        className="flex gap-3"
                        data-testid={`text-trade-event-${idx}`}
                      >
                        <div className="flex flex-col items-center">
                          <CircleDot className="h-4 w-4 text-primary" />
                          {idx < tradeEvents.length - 1 && (
                            <div className="w-px h-full bg-border" />
                          )}
                        </div>
                        <div className="pb-4">
                          <div className="flex items-center gap-2 flex-wrap">
                            <Badge variant="outline" className="text-xs">
                              {event.type}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {formatTs(event.ts)}
                            </span>
                          </div>
                          <p className="text-sm mt-1">{event.message}</p>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground">No events recorded</p>
                  )}
                </div>
              </SheetContent>
            </Sheet>
          </TabsContent>

          {/* ===== TAB 4: LEARNING PANEL ===== */}
          <TabsContent value="learning" className="mt-0 space-y-4">
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <h2 className="text-xl font-semibold">Learning Panel</h2>
              <Select value={learningSymbolFilter} onValueChange={setLearningSymbolFilter}>
                <SelectTrigger className="w-36" data-testid="button-learning-symbol-filter">
                  <SelectValue placeholder="All Symbols" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Symbols</SelectItem>
                  {uniqueSymbols.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Active Models */}
            {modelStats && Array.isArray(modelStats) && modelStats.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {modelStats.map((model, idx) => (
                  <Card key={idx} data-testid={`card-model-${model.symbol || idx}`}>
                    <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">
                        {model.symbol || "Unknown"}
                      </CardTitle>
                      <Cpu className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between gap-2 flex-wrap">
                          <span className="text-xs text-muted-foreground">Version</span>
                          <span className="text-xs font-mono">{model.version || "—"}</span>
                        </div>
                        <div className="flex items-center justify-between gap-2 flex-wrap">
                          <span className="text-xs text-muted-foreground">PR-AUC</span>
                          <span className="text-xs font-mono">
                            {model.prAuc?.toFixed(4) ?? "—"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-2 flex-wrap">
                          <span className="text-xs text-muted-foreground">PF_net</span>
                          <span className="text-xs font-mono">
                            {model.pfNet?.toFixed(2) ?? "—"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-2 flex-wrap">
                          <span className="text-xs text-muted-foreground">E_net</span>
                          <span className="text-xs font-mono">
                            {model.eNet?.toFixed(2) ?? "—"}
                          </span>
                        </div>
                        {model.promotedAt && (
                          <div className="flex items-center justify-between gap-2 flex-wrap">
                            <span className="text-xs text-muted-foreground">Promoted</span>
                            <span className="text-xs">{formatTs(model.promotedAt)}</span>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}

            {/* Learning Runs Table */}
            <Card data-testid="card-learning-runs">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Training History</CardTitle>
                <CardDescription>All learning runs with safety gate results</CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                {learningLoading ? (
                  <div className="p-6">
                    <TableSkeleton rows={6} cols={8} />
                  </div>
                ) : filteredLearningRuns.length === 0 ? (
                  <div className="p-6 text-center text-muted-foreground">
                    No learning runs found
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Started</TableHead>
                        <TableHead>Epochs</TableHead>
                        <TableHead>Val Loss</TableHead>
                        <TableHead>PR-AUC</TableHead>
                        <TableHead>PF_net</TableHead>
                        <TableHead>E_net</TableHead>
                        <TableHead>Regimes</TableHead>
                        <TableHead>Promoted</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Reason</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredLearningRuns.slice(0, 50).map((run) => (
                        <TableRow key={run.id} data-testid={`row-learning-${run.id}`}>
                          <TableCell>
                            <Badge variant="outline" className="text-xs">
                              {run.symbol}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-xs whitespace-nowrap">
                            {formatTs(run.startAt)}
                          </TableCell>
                          <TableCell className="text-xs">{run.epochs}</TableCell>
                          <TableCell className="text-xs font-mono">
                            {run.bestValLoss?.toFixed(4) ?? "—"}
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {run.prAuc?.toFixed(4) ?? "—"}
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {run.pfNet?.toFixed(2) ?? "—"}
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {run.eNet?.toFixed(2) ?? "—"}
                          </TableCell>
                          <TableCell className="text-xs">
                            {run.profitableRegimes}/{run.totalRegimes}
                          </TableCell>
                          <TableCell>
                            {run.promoted ? (
                              <Badge variant="default" className="text-xs bg-emerald-500/20 text-emerald-400">
                                Yes
                              </Badge>
                            ) : (
                              <Badge variant="secondary" className="text-xs">
                                No
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant={
                                run.status === "completed"
                                  ? "default"
                                  : run.status === "running"
                                    ? "secondary"
                                    : "destructive"
                              }
                              className="text-xs"
                            >
                              {run.status}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground max-w-[150px] truncate">
                            {run.reason || "—"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>

            {/* Safety Gate Visualization */}
            {filteredLearningRuns.length > 0 && (
              <Card data-testid="card-safety-gate">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Safety Gate Results</CardTitle>
                  <CardDescription>
                    Promotion rate across training runs
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-6 flex-wrap">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                      <div>
                        <div className="text-lg font-bold">
                          {filteredLearningRuns.filter((r) => r.promoted).length}
                        </div>
                        <div className="text-xs text-muted-foreground">Promoted</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <XCircle className="h-5 w-5 text-red-400" />
                      <div>
                        <div className="text-lg font-bold">
                          {filteredLearningRuns.filter((r) => !r.promoted && r.status === "completed").length}
                        </div>
                        <div className="text-xs text-muted-foreground">Rejected</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Timer className="h-5 w-5 text-amber-400" />
                      <div>
                        <div className="text-lg font-bold">
                          {filteredLearningRuns.filter((r) => r.status === "running").length}
                        </div>
                        <div className="text-xs text-muted-foreground">Running</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Database className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <div className="text-lg font-bold">
                          {(
                            (filteredLearningRuns.filter((r) => r.promoted).length /
                              Math.max(
                                filteredLearningRuns.filter((r) => r.status === "completed").length,
                                1
                              )) *
                            100
                          ).toFixed(0)}
                          %
                        </div>
                        <div className="text-xs text-muted-foreground">Promotion Rate</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* ===== TAB 5: PERFORMANCE ===== */}
          <TabsContent value="performance" className="mt-0 space-y-4">
            <h2 className="text-xl font-semibold">Performance Analytics</h2>

            {summaryLoading || tradesLoading ? (
              <div className="space-y-4">
                <Skeleton className="h-64 w-full" />
                <Skeleton className="h-64 w-full" />
              </div>
            ) : (
              <>
                {/* Equity Curve */}
                <Card data-testid="card-equity-curve">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Equity Curve</CardTitle>
                    <CardDescription>Cumulative net R over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {equityCurveData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={280}>
                        <AreaChart data={equityCurveData}>
                          <defs>
                            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                          <XAxis
                            dataKey="label"
                            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                            interval="preserveStartEnd"
                          />
                          <YAxis
                            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                            tickFormatter={(v) => `${v}R`}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "hsl(var(--card))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: 8,
                              fontSize: 12,
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="cumR"
                            stroke="hsl(var(--primary))"
                            fill="url(#equityGrad)"
                            strokeWidth={2}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <p className="text-sm text-muted-foreground py-8 text-center">
                        No equity data available
                      </p>
                    )}
                  </CardContent>
                </Card>

                {/* Drawdown Chart */}
                <Card data-testid="card-drawdown">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Drawdown</CardTitle>
                    <CardDescription>Drawdown from equity peak</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {equityCurveData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={200}>
                        <AreaChart data={equityCurveData}>
                          <defs>
                            <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                          <XAxis
                            dataKey="label"
                            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                            interval="preserveStartEnd"
                          />
                          <YAxis
                            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                            tickFormatter={(v) => `${v}%`}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "hsl(var(--card))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: 8,
                              fontSize: 12,
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="drawdown"
                            stroke="#ef4444"
                            fill="url(#ddGrad)"
                            strokeWidth={2}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <p className="text-sm text-muted-foreground py-8 text-center">
                        No drawdown data
                      </p>
                    )}
                  </CardContent>
                </Card>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* Per-Symbol Bar Chart */}
                  <Card data-testid="card-symbol-bar">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Net R by Symbol</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {symbolBarData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={250}>
                          <BarChart data={symbolBarData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                            <XAxis
                              dataKey="symbol"
                              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                            />
                            <YAxis
                              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                              tickFormatter={(v) => `${v}R`}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "hsl(var(--card))",
                                border: "1px solid hsl(var(--border))",
                                borderRadius: 8,
                                fontSize: 12,
                              }}
                            />
                            <Bar dataKey="netR" radius={[4, 4, 0, 0]}>
                              {symbolBarData.map((entry, index) => (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={entry.netR >= 0 ? "#34d399" : "#ef4444"}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <p className="text-sm text-muted-foreground py-8 text-center">
                          No symbol data
                        </p>
                      )}
                    </CardContent>
                  </Card>

                  {/* Net R Distribution */}
                  <Card data-testid="card-r-distribution">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Net R Distribution</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {netRDistribution.length > 0 ? (
                        <ResponsiveContainer width="100%" height={250}>
                          <BarChart data={netRDistribution}>
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                            <XAxis
                              dataKey="r"
                              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                              tickFormatter={(v) => `${v}R`}
                            />
                            <YAxis
                              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "hsl(var(--card))",
                                border: "1px solid hsl(var(--border))",
                                borderRadius: 8,
                                fontSize: 12,
                              }}
                            />
                            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                              {netRDistribution.map((entry, index) => (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={entry.r >= 0 ? "#34d399" : "#ef4444"}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <p className="text-sm text-muted-foreground py-8 text-center">
                          No distribution data
                        </p>
                      )}
                    </CardContent>
                  </Card>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* Rolling PF & E */}
                  <Card data-testid="card-rolling-metrics">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Rolling PF & Expectancy</CardTitle>
                      <CardDescription>Computed from trade sequence</CardDescription>
                    </CardHeader>
                    <CardContent>
                      {trades && trades.length > 5 ? (
                        <ResponsiveContainer width="100%" height={250}>
                          <LineChart
                            data={trades
                              .filter((t) => t.netR !== null)
                              .map((t, idx, arr) => {
                                const windowSize = 10;
                                const start = Math.max(0, idx - windowSize + 1);
                                const slice = arr.slice(start, idx + 1);
                                const wins = slice.filter((x) => (x.netR ?? 0) > 0);
                                const losses = slice.filter((x) => (x.netR ?? 0) < 0);
                                const avgWin =
                                  wins.length > 0
                                    ? wins.reduce((s, x) => s + (x.netR ?? 0), 0) / wins.length
                                    : 0;
                                const avgLoss =
                                  losses.length > 0
                                    ? Math.abs(losses.reduce((s, x) => s + (x.netR ?? 0), 0) / losses.length)
                                    : 1;
                                const pf = avgLoss > 0 ? avgWin / avgLoss : 0;
                                const wr = slice.length > 0 ? wins.length / slice.length : 0;
                                const e = wr * avgWin - (1 - wr) * avgLoss;
                                return {
                                  idx: idx + 1,
                                  pf: parseFloat(pf.toFixed(2)),
                                  expectancy: parseFloat(e.toFixed(3)),
                                };
                              })}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                            <XAxis
                              dataKey="idx"
                              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                              label={{ value: "Trade #", position: "insideBottom", offset: -5, fontSize: 10 }}
                            />
                            <YAxis
                              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "hsl(var(--card))",
                                border: "1px solid hsl(var(--border))",
                                borderRadius: 8,
                                fontSize: 12,
                              }}
                            />
                            <Line
                              type="monotone"
                              dataKey="pf"
                              stroke="#34d399"
                              strokeWidth={2}
                              dot={false}
                              name="Profit Factor"
                            />
                            <Line
                              type="monotone"
                              dataKey="expectancy"
                              stroke="#60a5fa"
                              strokeWidth={2}
                              dot={false}
                              name="Expectancy"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <p className="text-sm text-muted-foreground py-8 text-center">
                          Need more trades for rolling metrics
                        </p>
                      )}
                    </CardContent>
                  </Card>

                  {/* p_enter vs Outcome Scatter */}
                  <Card data-testid="card-calibration-scatter">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">
                        p_enter vs Outcome
                      </CardTitle>
                      <CardDescription>Calibration scatter plot</CardDescription>
                    </CardHeader>
                    <CardContent>
                      {scatterData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={250}>
                          <ScatterChart>
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                            <XAxis
                              dataKey="pEnter"
                              name="p_enter (%)"
                              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                              label={{ value: "p_enter %", position: "insideBottom", offset: -5, fontSize: 10 }}
                            />
                            <YAxis
                              dataKey="netR"
                              name="Net R"
                              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                              label={{ value: "Net R", angle: -90, position: "insideLeft", fontSize: 10 }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "hsl(var(--card))",
                                border: "1px solid hsl(var(--border))",
                                borderRadius: 8,
                                fontSize: 12,
                              }}
                              cursor={{ strokeDasharray: "3 3" }}
                            />
                            <Scatter data={scatterData} fill="#8b5cf6">
                              {scatterData.map((entry, index) => (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={entry.netR >= 0 ? "#34d399" : "#ef4444"}
                                />
                              ))}
                            </Scatter>
                          </ScatterChart>
                        </ResponsiveContainer>
                      ) : (
                        <p className="text-sm text-muted-foreground py-8 text-center">
                          No calibration data
                        </p>
                      )}
                    </CardContent>
                  </Card>
                </div>

                {/* Holding Time Histogram */}
                <Card data-testid="card-holding-time">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Holding Time Distribution</CardTitle>
                    <CardDescription>How long trades are held</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {holdingTimeData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={holdingTimeData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                          <XAxis
                            dataKey="hours"
                            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                          />
                          <YAxis
                            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "hsl(var(--card))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: 8,
                              fontSize: 12,
                            }}
                          />
                          <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <p className="text-sm text-muted-foreground py-8 text-center">
                        No holding time data
                      </p>
                    )}
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>

          {/* ===== TAB 6: AI INSIGHTS ===== */}
          <TabsContent value="insights" className="mt-0 space-y-4">
            <h2 className="text-xl font-semibold">AI Insights</h2>
            <p className="text-sm text-muted-foreground">
              Deterministic analysis computed from your trading data — no LLM involved.
            </p>

            {summaryLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Card key={i}>
                    <CardContent className="p-6">
                      <Skeleton className="h-6 w-40 mb-3" />
                      <Skeleton className="h-4 w-full mb-2" />
                      <Skeleton className="h-4 w-3/4" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : insights.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {insights.map((insight, idx) => {
                  const Icon = insight.icon;
                  return (
                    <Card key={idx} data-testid={`card-insight-${idx}`}>
                      <CardHeader className="flex flex-row items-start gap-3 space-y-0 pb-2">
                        <div className="p-2 rounded-md bg-primary/10">
                          <Icon className={`h-5 w-5 ${insight.color}`} />
                        </div>
                        <div>
                          <CardTitle className="text-sm font-medium">{insight.title}</CardTitle>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <p
                          className="text-sm text-muted-foreground leading-relaxed"
                          data-testid={`text-insight-${idx}`}
                        >
                          {insight.description}
                        </p>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            ) : (
              <Card data-testid="card-no-insights">
                <CardContent className="p-6 text-center">
                  <Lightbulb className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
                  <p className="text-sm text-muted-foreground">
                    No insights available yet. Trade data is needed to generate analysis.
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
