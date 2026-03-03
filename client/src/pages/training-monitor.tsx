import { useQuery } from "@tanstack/react-query";
import { useTradingWs } from "@/hooks/use-trading-ws";
import { queryClient } from "@/lib/queryClient";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Brain, Clock, Layers, Zap, TrendingUp, TrendingDown,
  BarChart3, Activity, ChevronDown, ChevronUp, Timer,
  CheckCircle2, XCircle, Loader2, Target, History,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatDistanceToNow, format } from "date-fns";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine, Area, AreaChart,
  Legend,
} from "recharts";

interface TrainingSession {
  id: number;
  sessionType: string;
  status: string;
  startedAt: number;
  completedAt: number | null;
  totalFolds: number;
  completedFolds: number;
  currentFold: number;
  totalEpochs: number;
  currentEpoch: number;
  symbols: string[] | null;
  config: Record<string, any> | null;
  currentFoldMetrics: Record<string, any> | null;
  aggregateMetrics: Record<string, any> | null;
  gpuName: string | null;
  estimatedCompletionTs: number | null;
  lastUpdateTs: number | null;
  trainMonths: number | null;
  testMonths: number | null;
  errorMessage: string | null;
}

interface TrainingFold {
  id: number;
  sessionId: number;
  foldNum: number;
  trainStart: string | null;
  trainEnd: string | null;
  testStart: string | null;
  testEnd: string | null;
  status: string | null;
  trades: number | null;
  winRate: number | null;
  expectancy: number | null;
  profitFactor: number | null;
  sharpe: number | null;
  maxDrawdown: number | null;
  totalR: number | null;
  longShortRatio: string | null;
  perSymbol: Record<string, any> | null;
  startedAt: number | null;
  completedAt: number | null;
  bestEpoch: number | null;
  finalThreshold: number | null;
}

interface TrainingEpoch {
  id: number;
  sessionId: number;
  foldNum: number;
  epoch: number;
  trainLoss: number | null;
  valLoss: number | null;
  lossBreakdown: Record<string, number> | null;
  actionAccuracy: number | null;
  learningRate: number | null;
  expectancy: number | null;
  profitFactor: number | null;
  winRate: number | null;
  maxDrawdown: number | null;
  tradesPerDay: number | null;
  threshold: number | null;
  scoreDiag: Record<string, number> | null;
  timestamp: number;
}

interface ActiveResponse {
  active: TrainingSession | null;
  latest?: TrainingSession | null;
  folds?: TrainingFold[];
  recentEpochs?: TrainingEpoch[];
}

interface SessionsResponse {
  sessions: TrainingSession[];
  total: number;
}

interface SessionDetailResponse {
  session: TrainingSession;
  folds: TrainingFold[];
}

interface EpochsResponse {
  epochs: TrainingEpoch[];
}

function useActiveTraining() {
  return useQuery<ActiveResponse>({
    queryKey: ["/api/training/active"],
    refetchInterval: 10000,
  });
}

function useSessions() {
  return useQuery<SessionsResponse>({
    queryKey: ["/api/training/sessions"],
    refetchInterval: 30000,
  });
}

function useSessionDetail(id: number | null) {
  return useQuery<SessionDetailResponse>({
    queryKey: ["/api/training/sessions", id],
    enabled: !!id,
  });
}

function useSessionEpochs(id: number | null, fold?: number) {
  const params = fold !== undefined ? `?fold=${fold}` : "";
  return useQuery<EpochsResponse>({
    queryKey: ["/api/training/sessions", id, "epochs", fold],
    queryFn: async () => {
      const res = await fetch(`/api/training/sessions/${id}/epochs${params}`);
      if (!res.ok) throw new Error("Failed to fetch epochs");
      return res.json();
    },
    enabled: !!id,
    refetchInterval: 10000,
  });
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

function StatusBanner({ session, isActive }: { session: TrainingSession | null; isActive: boolean }) {
  if (!session) {
    return (
      <div className="glass-card border border-border/50 p-6 mb-6" data-testid="status-banner">
        <div className="flex items-center gap-3">
          <Brain className="w-8 h-8 text-muted-foreground" />
          <div>
            <h2 className="text-lg font-semibold text-muted-foreground">No Training Sessions</h2>
            <p className="text-sm text-muted-foreground">Start training on your local GPU to see live progress here</p>
          </div>
        </div>
      </div>
    );
  }

  const elapsed = Date.now() - session.startedAt;
  const eta = session.estimatedCompletionTs ? session.estimatedCompletionTs - Date.now() : null;
  const progressPct = session.totalFolds > 0
    ? ((session.completedFolds + (session.currentEpoch / Math.max(session.totalEpochs, 1))) / session.totalFolds) * 100
    : 0;

  return (
    <div className={cn(
      "glass-card border p-6 mb-6",
      isActive ? "border-cyan-500/50 glow-cyan" : "border-border/50",
    )} data-testid="status-banner">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {isActive ? (
            <div className="relative">
              <Brain className="w-8 h-8 text-cyan-400" />
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-cyan-400 rounded-full pulse-dot" />
            </div>
          ) : (
            <Brain className={cn("w-8 h-8", session.status === "completed" ? "text-emerald-400" : "text-red-400")} />
          )}
          <div>
            <h2 className="text-lg font-semibold">
              {isActive ? "V5 TRAINING IN PROGRESS" : `Training ${session.status === "completed" ? "Complete" : "Failed"}`}
            </h2>
            <p className="text-xs text-muted-foreground">
              {session.sessionType === "walk_forward" ? "Walk-Forward Analysis" : "Single Training"} — Started {formatDistanceToNow(new Date(session.startedAt), { addSuffix: true })}
              {session.gpuName && ` on ${session.gpuName}`}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {isActive && eta && eta > 0 && (
            <Badge variant="outline" className="border-cyan-500/50 text-cyan-400" data-testid="badge-eta">
              <Timer className="w-3 h-3 mr-1" />
              ETA: {formatDuration(eta)}
            </Badge>
          )}
          <Badge
            variant={isActive ? "default" : session.status === "completed" ? "secondary" : "destructive"}
            data-testid="badge-status"
          >
            {isActive && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
            {session.status.toUpperCase()}
          </Badge>
        </div>
      </div>

      {isActive && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Fold {session.currentFold}/{session.totalFolds} — Epoch {session.currentEpoch}/{session.totalEpochs}</span>
            <span>{progressPct.toFixed(1)}%</span>
          </div>
          <Progress value={progressPct} className="h-2" data-testid="progress-bar" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Elapsed: {formatDuration(elapsed)}</span>
            {session.lastUpdateTs && (
              <span>Last update: {formatDistanceToNow(new Date(session.lastUpdateTs), { addSuffix: true })}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function OverviewCards({ session }: { session: TrainingSession }) {
  const elapsed = Date.now() - session.startedAt;
  const eta = session.estimatedCompletionTs ? Math.max(0, session.estimatedCompletionTs - Date.now()) : null;
  const agg = session.aggregateMetrics as Record<string, any> | null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-6" data-testid="overview-cards">
      <Card className="glass-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 mb-1">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span className="text-xs text-muted-foreground">Folds</span>
          </div>
          <div className="text-2xl font-bold number-mono" data-testid="text-folds">
            {session.completedFolds}<span className="text-sm text-muted-foreground">/{session.totalFolds}</span>
          </div>
        </CardContent>
      </Card>
      <Card className="glass-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 mb-1">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span className="text-xs text-muted-foreground">Epoch</span>
          </div>
          <div className="text-2xl font-bold number-mono" data-testid="text-epoch">
            {session.currentEpoch}<span className="text-sm text-muted-foreground">/{session.totalEpochs}</span>
          </div>
        </CardContent>
      </Card>
      <Card className="glass-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 mb-1">
            <Clock className="w-4 h-4 text-amber-400" />
            <span className="text-xs text-muted-foreground">Elapsed</span>
          </div>
          <div className="text-lg font-bold number-mono" data-testid="text-elapsed">
            {formatDuration(elapsed)}
          </div>
        </CardContent>
      </Card>
      <Card className="glass-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 mb-1">
            <Timer className="w-4 h-4 text-amber-400" />
            <span className="text-xs text-muted-foreground">ETA</span>
          </div>
          <div className="text-lg font-bold number-mono" data-testid="text-eta">
            {eta !== null ? formatDuration(eta) : "—"}
          </div>
        </CardContent>
      </Card>
      <Card className="glass-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            <span className="text-xs text-muted-foreground">Total R</span>
          </div>
          <div className={cn("text-2xl font-bold number-mono", (agg?.total_r ?? 0) >= 0 ? "text-emerald-400" : "text-red-400")} data-testid="text-total-r">
            {agg?.total_r !== undefined ? `${agg.total_r >= 0 ? "+" : ""}${agg.total_r.toFixed(1)}` : "—"}
          </div>
        </CardContent>
      </Card>
      <Card className="glass-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 mb-1">
            <Target className="w-4 h-4 text-emerald-400" />
            <span className="text-xs text-muted-foreground">Avg E[R]</span>
          </div>
          <div className={cn("text-2xl font-bold number-mono", (agg?.avg_expectancy_r ?? 0) >= 0 ? "text-emerald-400" : "text-red-400")} data-testid="text-avg-expectancy">
            {agg?.avg_expectancy_r !== undefined ? `${agg.avg_expectancy_r >= 0 ? "+" : ""}${agg.avg_expectancy_r.toFixed(4)}` : "—"}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function LossCurves({ epochs, folds }: { epochs: TrainingEpoch[]; folds: TrainingFold[] }) {
  const [showComponents, setShowComponents] = useState(false);

  if (!epochs.length) {
    return (
      <Card className="glass-card mb-6">
        <CardHeader><CardTitle className="text-sm flex items-center gap-2"><BarChart3 className="w-4 h-4 text-cyan-400" />Loss Curves</CardTitle></CardHeader>
        <CardContent><p className="text-sm text-muted-foreground text-center py-8">Waiting for epoch data...</p></CardContent>
      </Card>
    );
  }

  const chartData = epochs.map((e) => ({
    label: `F${e.foldNum}E${e.epoch}`,
    epoch: e.epoch,
    fold: e.foldNum,
    trainLoss: e.trainLoss,
    valLoss: e.valLoss,
    accuracy: e.actionAccuracy ? e.actionAccuracy * 100 : null,
    ...(e.lossBreakdown ?? {}),
  }));

  const foldBoundaries = folds
    .filter((f) => f.status === "completed" || f.status === "running")
    .map((f) => `F${f.foldNum}E1`);

  return (
    <Card className="glass-card mb-6" data-testid="loss-curves">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            Loss Curves
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={() => setShowComponents(!showComponents)} data-testid="button-toggle-components">
            {showComponents ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            <span className="ml-1 text-xs">{showComponents ? "Hide" : "Show"} Components</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="label" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} interval={Math.max(1, Math.floor(chartData.length / 20))} />
            <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }} />
            <Line type="monotone" dataKey="trainLoss" stroke="#06b6d4" strokeWidth={1.5} dot={false} name="Train Loss" />
            <Line type="monotone" dataKey="valLoss" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Val Loss" />
            {showComponents && (
              <>
                <Line type="monotone" dataKey="L_ret" stroke="#8b5cf6" strokeWidth={1} dot={false} name="L_ret" strokeDasharray="4 2" />
                <Line type="monotone" dataKey="L_action" stroke="#ef4444" strokeWidth={1} dot={false} name="L_action" strokeDasharray="4 2" />
                <Line type="monotone" dataKey="L_mfe" stroke="#22c55e" strokeWidth={1} dot={false} name="L_mfe" strokeDasharray="4 2" />
                <Line type="monotone" dataKey="L_mae" stroke="#ec4899" strokeWidth={1} dot={false} name="L_mae" strokeDasharray="4 2" />
              </>
            )}
            {foldBoundaries.map((b, i) => (
              <ReferenceLine key={i} x={b} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" opacity={0.5} />
            ))}
            <Legend />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function AccuracyChart({ epochs }: { epochs: TrainingEpoch[] }) {
  if (!epochs.length) return null;

  const chartData = epochs
    .filter((e) => e.actionAccuracy !== null)
    .map((e) => ({
      label: `F${e.foldNum}E${e.epoch}`,
      accuracy: e.actionAccuracy ? +(e.actionAccuracy * 100).toFixed(1) : null,
    }));

  if (!chartData.length) return null;

  return (
    <Card className="glass-card" data-testid="accuracy-chart">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Target className="w-4 h-4 text-emerald-400" />Action Accuracy</CardTitle></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="label" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} interval={Math.max(1, Math.floor(chartData.length / 15))} />
            <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} domain={[0, 100]} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }} />
            <Area type="monotone" dataKey="accuracy" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} strokeWidth={1.5} name="Accuracy %" />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function ExpectancyChart({ epochs }: { epochs: TrainingEpoch[] }) {
  const sweepEpochs = epochs.filter((e) => e.expectancy !== null);
  if (!sweepEpochs.length) return null;

  const chartData = sweepEpochs.map((e) => ({
    label: `F${e.foldNum}E${e.epoch}`,
    expectancy: e.expectancy ? +e.expectancy.toFixed(4) : null,
    pf: e.profitFactor ? +e.profitFactor.toFixed(2) : null,
    threshold: e.threshold ? +e.threshold.toFixed(4) : null,
  }));

  return (
    <Card className="glass-card" data-testid="expectancy-chart">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><TrendingUp className="w-4 h-4 text-cyan-400" />Expectancy & Threshold</CardTitle></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="label" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} interval={Math.max(1, Math.floor(chartData.length / 15))} />
            <YAxis yAxisId="left" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }} />
            <Line yAxisId="left" type="monotone" dataKey="expectancy" stroke="#06b6d4" strokeWidth={1.5} dot={false} name="E[R]" />
            <Line yAxisId="right" type="monotone" dataKey="threshold" stroke="#f59e0b" strokeWidth={1} dot={false} name="Threshold" strokeDasharray="4 2" />
            <ReferenceLine yAxisId="left" y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" opacity={0.5} />
            <Legend />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function FoldResultsTable({ folds }: { folds: TrainingFold[] }) {
  if (!folds.length) return null;

  return (
    <Card className="glass-card mb-6" data-testid="fold-results-table">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Layers className="w-4 h-4 text-cyan-400" />Walk-Forward Fold Results</CardTitle></CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left py-2 px-2 text-muted-foreground">Fold</th>
                <th className="text-left py-2 px-2 text-muted-foreground">Window</th>
                <th className="text-left py-2 px-2 text-muted-foreground">Status</th>
                <th className="text-right py-2 px-2 text-muted-foreground">Trades</th>
                <th className="text-right py-2 px-2 text-muted-foreground">Win Rate</th>
                <th className="text-right py-2 px-2 text-muted-foreground">E[R]</th>
                <th className="text-right py-2 px-2 text-muted-foreground">PF</th>
                <th className="text-right py-2 px-2 text-muted-foreground">Sharpe</th>
                <th className="text-right py-2 px-2 text-muted-foreground">MaxDD</th>
                <th className="text-right py-2 px-2 text-muted-foreground">Total R</th>
                <th className="text-right py-2 px-2 text-muted-foreground">Threshold</th>
              </tr>
            </thead>
            <tbody>
              {folds.map((fold) => {
                const isRunning = fold.status === "running";
                const totalR = fold.totalR ?? 0;
                return (
                  <tr
                    key={fold.id}
                    className={cn(
                      "border-b border-border/30 hover:bg-muted/20",
                      isRunning && "bg-cyan-500/5 border-l-2 border-l-cyan-500",
                    )}
                    data-testid={`row-fold-${fold.foldNum}`}
                  >
                    <td className="py-2 px-2 font-mono font-medium">
                      {isRunning && <Loader2 className="w-3 h-3 inline mr-1 animate-spin text-cyan-400" />}
                      {fold.foldNum}
                    </td>
                    <td className="py-2 px-2 text-muted-foreground">
                      {fold.testStart && fold.testEnd ? `${fold.testStart} → ${fold.testEnd}` : "—"}
                    </td>
                    <td className="py-2 px-2">
                      <Badge
                        variant={isRunning ? "default" : fold.status === "completed" ? "secondary" : "outline"}
                        className={cn("text-[10px]", isRunning && "bg-cyan-500/20 text-cyan-400 border-cyan-500/50")}
                      >
                        {fold.status ?? "pending"}
                      </Badge>
                    </td>
                    <td className="py-2 px-2 text-right number-mono">{fold.trades ?? "—"}</td>
                    <td className="py-2 px-2 text-right number-mono">
                      {fold.winRate !== null ? `${(fold.winRate * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className={cn("py-2 px-2 text-right number-mono", (fold.expectancy ?? 0) >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {fold.expectancy !== null ? `${fold.expectancy >= 0 ? "+" : ""}${fold.expectancy.toFixed(4)}` : "—"}
                    </td>
                    <td className="py-2 px-2 text-right number-mono">{fold.profitFactor?.toFixed(2) ?? "—"}</td>
                    <td className="py-2 px-2 text-right number-mono">{fold.sharpe?.toFixed(2) ?? "—"}</td>
                    <td className="py-2 px-2 text-right number-mono text-red-400">{fold.maxDrawdown?.toFixed(2) ?? "—"}</td>
                    <td className={cn("py-2 px-2 text-right number-mono font-medium", totalR >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {fold.totalR !== null ? `${totalR >= 0 ? "+" : ""}${totalR.toFixed(2)}` : "—"}
                    </td>
                    <td className="py-2 px-2 text-right number-mono text-muted-foreground">{fold.finalThreshold?.toFixed(4) ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function FoldBarChart({ folds }: { folds: TrainingFold[] }) {
  const completed = folds.filter((f) => f.status === "completed" && f.totalR !== null);
  if (!completed.length) return null;

  const data = completed.map((f) => ({
    fold: `F${f.foldNum}`,
    totalR: +(f.totalR ?? 0).toFixed(2),
    trades: f.trades ?? 0,
  }));

  return (
    <Card className="glass-card" data-testid="fold-bar-chart">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><BarChart3 className="w-4 h-4 text-cyan-400" />R per Fold</CardTitle></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="fold" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
            <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }} />
            <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
            <Bar dataKey="totalR" name="Total R" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.totalR >= 0 ? "#22c55e" : "#ef4444"} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function PerSymbolEdge({ folds, symbols }: { folds: TrainingFold[]; symbols: string[] }) {
  const completedFolds = folds.filter((f) => f.status === "completed" && f.perSymbol);
  if (!completedFolds.length || !symbols.length) return null;

  return (
    <Card className="glass-card mb-6" data-testid="per-symbol-edge">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Zap className="w-4 h-4 text-cyan-400" />Per-Symbol Edge Heatmap</CardTitle></CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left py-2 px-2 text-muted-foreground">Symbol</th>
                {completedFolds.map((f) => (
                  <th key={f.foldNum} className="text-center py-2 px-1 text-muted-foreground">F{f.foldNum}</th>
                ))}
                <th className="text-right py-2 px-2 text-muted-foreground font-medium">Total</th>
              </tr>
            </thead>
            <tbody>
              {symbols.map((sym) => {
                let total = 0;
                return (
                  <tr key={sym} className="border-b border-border/30">
                    <td className="py-1.5 px-2 font-mono font-medium">{sym.replace("USDT", "")}</td>
                    {completedFolds.map((f) => {
                      const ps = f.perSymbol as Record<string, any> | null;
                      const symData = ps?.[sym];
                      const r = symData?.total_r ?? symData ?? null;
                      const rNum = typeof r === "number" ? r : (r?.total_r ?? 0);
                      total += rNum;
                      const intensity = Math.min(1, Math.abs(rNum) / 20);
                      return (
                        <td key={f.foldNum} className="text-center py-1.5 px-1">
                          <span
                            className={cn(
                              "inline-block w-full px-1 py-0.5 rounded text-[10px] number-mono",
                              rNum > 0 ? "text-emerald-400" : rNum < 0 ? "text-red-400" : "text-muted-foreground",
                            )}
                            style={{
                              backgroundColor: rNum > 0
                                ? `rgba(34, 197, 94, ${intensity * 0.2})`
                                : rNum < 0
                                  ? `rgba(239, 68, 68, ${intensity * 0.2})`
                                  : "transparent",
                            }}
                          >
                            {typeof rNum === "number" ? `${rNum >= 0 ? "+" : ""}${rNum.toFixed(1)}` : "—"}
                          </span>
                        </td>
                      );
                    })}
                    <td className={cn("text-right py-1.5 px-2 number-mono font-medium", total >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {total >= 0 ? "+" : ""}{total.toFixed(1)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function ConfigPanel({ session }: { session: TrainingSession }) {
  const [expanded, setExpanded] = useState(false);
  const config = session.config as Record<string, any> | null;

  return (
    <Card className="glass-card mb-6" data-testid="config-panel">
      <CardHeader>
        <div className="flex items-center justify-between cursor-pointer" onClick={() => setExpanded(!expanded)}>
          <CardTitle className="text-sm flex items-center gap-2">
            <Zap className="w-4 h-4 text-amber-400" />
            Training Configuration
          </CardTitle>
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </CardHeader>
      {expanded && (
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <span className="text-[10px] text-muted-foreground block">Symbols</span>
              <span className="text-xs font-mono">{session.symbols?.join(", ") ?? "—"}</span>
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground block">Window</span>
              <span className="text-xs font-mono">{session.trainMonths ?? "?"}m train / {session.testMonths ?? "?"}m test</span>
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground block">GPU</span>
              <span className="text-xs font-mono">{session.gpuName ?? "—"}</span>
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground block">Epochs/Fold</span>
              <span className="text-xs font-mono">{session.totalEpochs ?? "—"}</span>
            </div>
            {config && Object.entries(config).map(([key, val]) => (
              <div key={key}>
                <span className="text-[10px] text-muted-foreground block">{key}</span>
                <span className="text-xs font-mono">{typeof val === "object" ? JSON.stringify(val) : String(val)}</span>
              </div>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

function ModelKnowledge({ epochs, session }: { epochs: TrainingEpoch[]; session: TrainingSession }) {
  if (!epochs.length) return null;

  const sweepEpochs = epochs.filter((e) => e.expectancy !== null);
  const latestAccuracy = epochs[epochs.length - 1]?.actionAccuracy ?? 0;
  const firstAccuracy = epochs[0]?.actionAccuracy ?? 0;
  const accuracyImprovement = latestAccuracy - firstAccuracy;

  const latestLoss = epochs[epochs.length - 1]?.valLoss ?? 1;
  const bestLoss = Math.min(...epochs.filter((e) => e.valLoss !== null).map((e) => e.valLoss!));
  const lossReduction = epochs[0]?.valLoss ? (1 - bestLoss / epochs[0].valLoss) * 100 : 0;

  const latestExpect = sweepEpochs.length ? sweepEpochs[sweepEpochs.length - 1]?.expectancy ?? 0 : 0;
  const bestExpect = sweepEpochs.length ? Math.max(...sweepEpochs.map((e) => e.expectancy ?? -999)) : 0;

  const overallProgress = Math.min(100, Math.max(0,
    (latestAccuracy * 100 * 0.3) + (lossReduction * 0.3) + ((latestExpect > 0 ? 40 : 0))
  ));

  return (
    <Card className="glass-card mb-6" data-testid="model-knowledge">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Brain className="w-4 h-4 text-cyan-400" />Model Knowledge</CardTitle></CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-muted-foreground">Learning Progress</span>
              <span className="number-mono text-cyan-400">{overallProgress.toFixed(0)}%</span>
            </div>
            <div className="h-3 bg-muted/30 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-500"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="text-center">
              <span className="text-[10px] text-muted-foreground block">Action Accuracy</span>
              <span className="text-lg font-bold number-mono text-emerald-400">{(latestAccuracy * 100).toFixed(1)}%</span>
              {accuracyImprovement > 0 && (
                <span className="text-[10px] text-emerald-400 block">+{(accuracyImprovement * 100).toFixed(1)}%</span>
              )}
            </div>
            <div className="text-center">
              <span className="text-[10px] text-muted-foreground block">Best Val Loss</span>
              <span className="text-lg font-bold number-mono">{bestLoss.toFixed(4)}</span>
              <span className="text-[10px] text-emerald-400 block">-{lossReduction.toFixed(1)}%</span>
            </div>
            <div className="text-center">
              <span className="text-[10px] text-muted-foreground block">Best E[R]</span>
              <span className={cn("text-lg font-bold number-mono", bestExpect >= 0 ? "text-emerald-400" : "text-red-400")}>
                {bestExpect > -999 ? `${bestExpect >= 0 ? "+" : ""}${bestExpect.toFixed(4)}` : "—"}
              </span>
            </div>
            <div className="text-center">
              <span className="text-[10px] text-muted-foreground block">Sweep Epochs</span>
              <span className="text-lg font-bold number-mono">{sweepEpochs.length}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SessionHistory({ sessions, onSelect }: { sessions: TrainingSession[]; onSelect: (id: number) => void }) {
  if (!sessions.length) return null;

  return (
    <Card className="glass-card" data-testid="session-history">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><History className="w-4 h-4 text-muted-foreground" />Session History</CardTitle></CardHeader>
      <CardContent>
        <div className="space-y-2">
          {sessions.map((s) => {
            const agg = s.aggregateMetrics as Record<string, any> | null;
            const duration = s.completedAt ? s.completedAt - s.startedAt : Date.now() - s.startedAt;
            return (
              <div
                key={s.id}
                className="flex items-center justify-between p-2 rounded-md hover:bg-muted/20 cursor-pointer border border-border/30"
                onClick={() => onSelect(s.id)}
                data-testid={`card-session-${s.id}`}
              >
                <div className="flex items-center gap-2">
                  {s.status === "completed" ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : s.status === "running" ? (
                    <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-400" />
                  )}
                  <div>
                    <span className="text-xs font-medium">{s.sessionType}</span>
                    <span className="text-[10px] text-muted-foreground ml-2">
                      {format(new Date(s.startedAt), "MMM d, yyyy HH:mm")}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="number-mono text-muted-foreground">{s.completedFolds}/{s.totalFolds} folds</span>
                  <span className="number-mono text-muted-foreground">{formatDuration(duration)}</span>
                  {agg?.total_r !== undefined && (
                    <span className={cn("number-mono font-medium", agg.total_r >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {agg.total_r >= 0 ? "+" : ""}{agg.total_r.toFixed(1)}R
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

export default function TrainingMonitor() {
  const { data: activeData, isLoading } = useActiveTraining();
  const { data: sessionsData } = useSessions();
  const { subscribe } = useTradingWs();

  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null);

  const activeSession = activeData?.active ?? null;
  const isActive = !!activeSession;
  const displaySession = activeSession ?? activeData?.latest ?? null;
  const folds = activeData?.folds ?? [];
  const recentEpochs = activeData?.recentEpochs ?? [];

  const { data: detailData } = useSessionDetail(selectedSessionId && selectedSessionId !== displaySession?.id ? selectedSessionId : null);
  const detailSession = selectedSessionId && detailData ? detailData.session : null;
  const detailFolds = detailData?.folds ?? [];

  const viewingSession = detailSession ?? displaySession;
  const viewingFolds = detailSession ? detailFolds : folds;
  const viewingSessionId = viewingSession?.id ?? null;

  const { data: epochsData } = useSessionEpochs(viewingSessionId);
  const allEpochs = epochsData?.epochs ?? (viewingSessionId === displaySession?.id ? recentEpochs : []);

  useEffect(() => {
    const unsubs = [
      subscribe("TRAINING_EPOCH", () => {
        queryClient.invalidateQueries({ queryKey: ["/api/training/active"] });
        if (viewingSessionId) {
          queryClient.invalidateQueries({ queryKey: ["/api/training/sessions", viewingSessionId, "epochs"] });
        }
      }),
      subscribe("TRAINING_FOLD_END", () => {
        queryClient.invalidateQueries({ queryKey: ["/api/training/active"] });
        queryClient.invalidateQueries({ queryKey: ["/api/training/sessions"] });
      }),
      subscribe("TRAINING_SESSION_START", () => {
        queryClient.invalidateQueries({ queryKey: ["/api/training/active"] });
        queryClient.invalidateQueries({ queryKey: ["/api/training/sessions"] });
      }),
      subscribe("TRAINING_SESSION_END", () => {
        queryClient.invalidateQueries({ queryKey: ["/api/training/active"] });
        queryClient.invalidateQueries({ queryKey: ["/api/training/sessions"] });
      }),
    ];
    return () => unsubs.forEach((u) => u());
  }, [subscribe, viewingSessionId]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 space-y-0" data-testid="training-monitor-page">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <Brain className="w-5 h-5 text-cyan-400" />
          Training Monitor
        </h1>
        {selectedSessionId && selectedSessionId !== displaySession?.id && (
          <Button variant="ghost" size="sm" onClick={() => setSelectedSessionId(null)} data-testid="button-back-to-active">
            Back to {isActive ? "Active" : "Latest"}
          </Button>
        )}
      </div>

      <StatusBanner session={viewingSession} isActive={isActive && viewingSession?.id === activeSession?.id} />

      {viewingSession && (
        <>
          <OverviewCards session={viewingSession} />
          <ModelKnowledge epochs={allEpochs} session={viewingSession} />
          <LossCurves epochs={allEpochs} folds={viewingFolds} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <AccuracyChart epochs={allEpochs} />
            <ExpectancyChart epochs={allEpochs} />
          </div>

          <FoldResultsTable folds={viewingFolds} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <FoldBarChart folds={viewingFolds} />
            {viewingSession.symbols && viewingSession.symbols.length > 0 && (
              <PerSymbolEdge folds={viewingFolds} symbols={viewingSession.symbols} />
            )}
          </div>

          <ConfigPanel session={viewingSession} />
        </>
      )}

      {sessionsData?.sessions && sessionsData.sessions.length > 0 && (
        <SessionHistory
          sessions={sessionsData.sessions.filter((s) => s.id !== viewingSession?.id)}
          onSelect={setSelectedSessionId}
        />
      )}
    </div>
  );
}
