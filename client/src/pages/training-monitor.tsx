import { useQuery, useMutation } from "@tanstack/react-query";
import { useTradingWs } from "@/hooks/use-trading-ws";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useEffect, useState, useMemo, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  Brain, Clock, Layers, Zap, TrendingUp,
  BarChart3, Activity, ChevronDown, ChevronUp, Timer,
  CheckCircle2, XCircle, Loader2, Target, History,
  Trash2, ShieldCheck, AlertTriangle, Gauge, Trophy,
  ArrowUpRight, ArrowDownRight, Cpu, Hash,
  Database, RefreshCw, Download, CircleDot,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { formatDistanceToNow, format } from "date-fns";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine, Area, AreaChart,
  Legend, ComposedChart,
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

interface ReadyResponse {
  ready: boolean;
  unclearedSessions: number;
  runningSessions: number;
  message: string;
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

function useTrainingReady() {
  return useQuery<ReadyResponse>({
    queryKey: ["/api/training/ready"],
    refetchInterval: 15000,
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

function StatusBanner({ session, isActive, onClear, onClearAll, clearPending }: {
  session: TrainingSession | null;
  isActive: boolean;
  onClear?: (id: number) => void;
  onClearAll?: () => void;
  clearPending?: boolean;
}) {
  if (!session) {
    return (
      <div className="glass-card border border-border/50 p-6 mb-6" data-testid="status-banner">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-muted/20">
            <Brain className="w-8 h-8 text-muted-foreground" />
          </div>
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
  const isCompleted = session.status === "completed";
  const isFailed = session.status === "failed";
  const isDone = isCompleted || isFailed;
  const STALE_THRESHOLD_MS = 5 * 60 * 1000;
  const isStale = session.status === "running" && session.lastUpdateTs && (Date.now() - session.lastUpdateTs > STALE_THRESHOLD_MS);
  const isReallyActive = isActive && !isStale;
  const canClear = isDone || isStale || session.status === "running";

  return (
    <div className={cn(
      "glass-card border p-6 mb-6 relative overflow-hidden",
      isReallyActive ? "border-cyan-500/50 glow-cyan" : isStale ? "border-amber-500/30" : isCompleted ? "border-emerald-500/30" : isFailed ? "border-red-500/30" : "border-border/50",
    )} data-testid="status-banner">
      {isReallyActive && (
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 via-transparent to-cyan-500/5 animate-pulse pointer-events-none" />
      )}
      <div className="flex items-center justify-between mb-4 relative">
        <div className="flex items-center gap-3">
          {isReallyActive ? (
            <div className="relative p-3 rounded-xl bg-cyan-500/10">
              <Brain className="w-8 h-8 text-cyan-400" />
              <span className="absolute top-1 right-1 w-3 h-3 bg-cyan-400 rounded-full pulse-dot" />
            </div>
          ) : isStale ? (
            <div className="p-3 rounded-xl bg-amber-500/10">
              <AlertTriangle className="w-8 h-8 text-amber-400" />
            </div>
          ) : (
            <div className={cn("p-3 rounded-xl", isCompleted ? "bg-emerald-500/10" : "bg-red-500/10")}>
              {isCompleted ? (
                <CheckCircle2 className="w-8 h-8 text-emerald-400" />
              ) : (
                <XCircle className="w-8 h-8 text-red-400" />
              )}
            </div>
          )}
          <div>
            <h2 className="text-lg font-semibold">
              {isReallyActive ? "V5 TRAINING IN PROGRESS" : isStale ? "Training Stale — CLI Stopped?" : isCompleted ? "Training Complete" : "Training Failed"}
            </h2>
            <p className="text-xs text-muted-foreground">
              {session.sessionType === "walk_forward" ? "Walk-Forward Analysis" : "Single Training"} — Started {formatDistanceToNow(new Date(session.startedAt), { addSuffix: true })}
              {session.gpuName && ` on ${session.gpuName}`}
              {isDone && session.completedAt && ` — Finished ${formatDistanceToNow(new Date(session.completedAt), { addSuffix: true })}`}
            </p>
            {isStale && (
              <p className="text-xs text-amber-400 mt-1">No updates received for {formatDuration(Date.now() - (session.lastUpdateTs ?? session.startedAt))} — if the CLI was stopped, clear this session to start fresh</p>
            )}
            {isFailed && session.errorMessage && (
              <p className="text-xs text-red-400 mt-1">{session.errorMessage}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isReallyActive && eta && eta > 0 && (
            <Badge variant="outline" className="border-cyan-500/50 text-cyan-400" data-testid="badge-eta">
              <Timer className="w-3 h-3 mr-1" />
              ETA: {formatDuration(eta)}
            </Badge>
          )}
          {isStale && (
            <Badge variant="outline" className="border-amber-500/50 text-amber-400" data-testid="badge-stale">
              <AlertTriangle className="w-3 h-3 mr-1" />
              STALE
            </Badge>
          )}
          <Badge
            variant={isReallyActive ? "default" : isCompleted ? "secondary" : isStale ? "outline" : "destructive"}
            className={cn(
              isReallyActive && "bg-cyan-500/20 text-cyan-400 border-cyan-500/50",
              isStale && "border-amber-500/50 text-amber-400"
            )}
            data-testid="badge-status"
          >
            {isReallyActive && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
            {isStale ? "STALE" : session.status.toUpperCase()}
          </Badge>
          {canClear && onClear && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" className="border-red-500/30 text-red-400 hover:bg-red-500/10" data-testid="button-clear-session" disabled={clearPending}>
                  {clearPending ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Trash2 className="w-3 h-3 mr-1" />}
                  Clear
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent className="glass-card border-border/50">
                <AlertDialogHeader>
                  <AlertDialogTitle>{isStale ? "Clear Stale Session?" : session.status === "running" ? "Force Clear Running Session?" : "Clear Training Session?"}</AlertDialogTitle>
                  <AlertDialogDescription>
                    {isStale
                      ? "This session appears to be stale (no updates received). This will delete it and all its data so you can start fresh."
                      : session.status === "running"
                        ? "Warning: This session is marked as running. Only clear it if you've already stopped the CLI. This will permanently delete all training data."
                        : "This will permanently delete this training session and all its epoch/fold data. You need to clear previous sessions before starting new training."}
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel data-testid="button-cancel-clear">Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={() => onClear(session.id)} className="bg-red-500 hover:bg-red-600" data-testid="button-confirm-clear">
                    Clear Session
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>
      </div>

      {(isReallyActive || isStale) && (
        <div className="space-y-2 relative">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Fold {session.currentFold}/{session.totalFolds} — Epoch {session.currentEpoch}/{session.totalEpochs}</span>
            <span>{progressPct.toFixed(1)}%</span>
          </div>
          <Progress value={progressPct} className="h-2.5" data-testid="progress-bar" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Elapsed: {formatDuration(elapsed)}</span>
            {session.lastUpdateTs && (
              <span>Last update: {formatDistanceToNow(new Date(session.lastUpdateTs), { addSuffix: true })}</span>
            )}
          </div>
        </div>
      )}

      {isDone && (
        <CompletionSummary session={session} />
      )}
    </div>
  );
}

function CompletionSummary({ session }: { session: TrainingSession }) {
  const agg = session.aggregateMetrics as Record<string, any> | null;
  const duration = (session.completedAt ?? Date.now()) - session.startedAt;

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4 pt-4 border-t border-border/30">
      <div className="text-center">
        <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Duration</span>
        <span className="text-sm font-bold number-mono">{formatDuration(duration)}</span>
      </div>
      <div className="text-center">
        <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Folds Done</span>
        <span className="text-sm font-bold number-mono">{session.completedFolds}/{session.totalFolds}</span>
      </div>
      <div className="text-center">
        <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Total R</span>
        <span className={cn("text-sm font-bold number-mono", (agg?.total_r ?? 0) >= 0 ? "text-emerald-400" : "text-red-400")}>
          {agg?.total_r !== undefined ? `${agg.total_r >= 0 ? "+" : ""}${Number(agg.total_r).toFixed(1)}` : "—"}
        </span>
      </div>
      <div className="text-center">
        <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Avg E[R]</span>
        <span className={cn("text-sm font-bold number-mono", (agg?.avg_expectancy_r ?? 0) >= 0 ? "text-emerald-400" : "text-red-400")}>
          {agg?.avg_expectancy_r !== undefined ? `${agg.avg_expectancy_r >= 0 ? "+" : ""}${Number(agg.avg_expectancy_r).toFixed(4)}` : "—"}
        </span>
      </div>
      <div className="text-center">
        <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Profitable</span>
        <span className="text-sm font-bold number-mono">
          {agg?.profitable_folds !== undefined ? `${agg.profitable_folds}/${session.totalFolds}` : "—"}
        </span>
      </div>
    </div>
  );
}

function ReadinessGate({ ready }: { ready: ReadyResponse }) {
  if (ready.ready) return null;

  return (
    <Card className={cn(
      "glass-card mb-6 border",
      ready.runningSessions > 0 ? "border-cyan-500/30" : "border-amber-500/30",
    )} data-testid="readiness-gate">
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          {ready.runningSessions > 0 ? (
            <Activity className="w-5 h-5 text-cyan-400 animate-pulse" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-amber-400" />
          )}
          <div className="flex-1">
            <p className="text-sm font-medium">{ready.message}</p>
            {ready.unclearedSessions > 0 && ready.runningSessions === 0 && (
              <p className="text-xs text-muted-foreground mt-1">
                Clear or review previous session results to unlock new training
              </p>
            )}
          </div>
          {ready.runningSessions === 0 && (
            <ShieldCheck className="w-5 h-5 text-muted-foreground" />
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function OverviewCards({ session }: { session: TrainingSession }) {
  const elapsed = Date.now() - session.startedAt;
  const eta = session.estimatedCompletionTs ? Math.max(0, session.estimatedCompletionTs - Date.now()) : null;
  const agg = session.aggregateMetrics as Record<string, any> | null;
  const isActive = session.status === "running";

  const cards = [
    { icon: Layers, label: "Folds", value: `${session.completedFolds}`, sub: `/${session.totalFolds}`, color: "text-cyan-400" },
    { icon: Activity, label: "Epoch", value: `${session.currentEpoch}`, sub: `/${session.totalEpochs}`, color: "text-cyan-400" },
    { icon: Clock, label: "Elapsed", value: formatDuration(elapsed), sub: null, color: "text-amber-400" },
    { icon: Timer, label: "ETA", value: eta !== null && isActive ? formatDuration(eta) : "—", sub: null, color: "text-amber-400" },
    {
      icon: TrendingUp, label: "Total R",
      value: agg?.total_r !== undefined ? `${agg.total_r >= 0 ? "+" : ""}${Number(agg.total_r).toFixed(1)}` : "—",
      sub: null,
      color: (agg?.total_r ?? 0) >= 0 ? "text-emerald-400" : "text-red-400",
      valueColor: true,
    },
    {
      icon: Target, label: "Avg E[R]",
      value: agg?.avg_expectancy_r !== undefined ? `${agg.avg_expectancy_r >= 0 ? "+" : ""}${Number(agg.avg_expectancy_r).toFixed(4)}` : "—",
      sub: null,
      color: (agg?.avg_expectancy_r ?? 0) >= 0 ? "text-emerald-400" : "text-red-400",
      valueColor: true,
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6" data-testid="overview-cards">
      {cards.map((c) => (
        <Card key={c.label} className="glass-card hover:border-border/60 transition-colors">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <c.icon className={cn("w-4 h-4", c.color)} />
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{c.label}</span>
            </div>
            <div className={cn("text-xl font-bold number-mono", c.valueColor && c.color)} data-testid={`text-${c.label.toLowerCase().replace(/[^a-z]/g, "-")}`}>
              {c.value}
              {c.sub && <span className="text-sm text-muted-foreground">{c.sub}</span>}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function ModelKnowledge({ epochs, session }: { epochs: TrainingEpoch[]; session: TrainingSession }) {
  if (!epochs.length) return null;

  const sweepEpochs = epochs.filter((e) => e.expectancy !== null);
  const latestAccuracy = epochs[epochs.length - 1]?.actionAccuracy ?? 0;
  const firstAccuracy = epochs[0]?.actionAccuracy ?? 0;
  const accuracyImprovement = latestAccuracy - firstAccuracy;

  const bestLoss = Math.min(...epochs.filter((e) => e.valLoss !== null).map((e) => e.valLoss!));
  const lossReduction = epochs[0]?.valLoss ? (1 - bestLoss / epochs[0].valLoss) * 100 : 0;

  const bestExpect = sweepEpochs.length ? Math.max(...sweepEpochs.map((e) => e.expectancy ?? -999)) : 0;
  const bestWR = sweepEpochs.length ? Math.max(...sweepEpochs.filter((e) => e.winRate !== null).map((e) => e.winRate ?? 0)) : 0;
  const bestPF = sweepEpochs.length ? Math.max(...sweepEpochs.filter((e) => e.profitFactor !== null).map((e) => e.profitFactor ?? 0)) : 0;

  const overallProgress = Math.min(100, Math.max(0,
    (latestAccuracy * 100 * 0.3) + (lossReduction * 0.3) + ((bestExpect > 0 ? 40 : 0))
  ));

  const gaugeColor = overallProgress > 70 ? "from-emerald-500 to-cyan-400" : overallProgress > 40 ? "from-amber-500 to-cyan-400" : "from-red-500 to-amber-400";

  return (
    <Card className="glass-card mb-6" data-testid="model-knowledge">
      <CardHeader>
        <CardTitle className="text-sm flex items-center gap-2">
          <Gauge className="w-4 h-4 text-cyan-400" />
          Model Knowledge
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-5">
          <div>
            <div className="flex justify-between text-xs mb-2">
              <span className="text-muted-foreground">Learning Progress</span>
              <span className="number-mono text-cyan-400 font-medium">{overallProgress.toFixed(0)}%</span>
            </div>
            <div className="h-4 bg-muted/20 rounded-full overflow-hidden relative">
              <div
                className={cn("h-full rounded-full bg-gradient-to-r transition-all duration-700", gaugeColor)}
                style={{ width: `${overallProgress}%` }}
              />
              <div className="absolute inset-0 flex items-center px-2">
                {[25, 50, 75].map((mark) => (
                  <div key={mark} className="absolute h-full w-px bg-background/30" style={{ left: `${mark}%` }} />
                ))}
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center p-2 rounded-lg bg-muted/10">
              <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Accuracy</span>
              <span className="text-lg font-bold number-mono text-emerald-400">{(latestAccuracy * 100).toFixed(1)}%</span>
              {accuracyImprovement > 0 && (
                <span className="text-[10px] text-emerald-400 flex items-center justify-center gap-0.5">
                  <ArrowUpRight className="w-3 h-3" />{(accuracyImprovement * 100).toFixed(1)}%
                </span>
              )}
            </div>
            <div className="text-center p-2 rounded-lg bg-muted/10">
              <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Best Val Loss</span>
              <span className="text-lg font-bold number-mono">{bestLoss.toFixed(4)}</span>
              <span className="text-[10px] text-emerald-400 flex items-center justify-center gap-0.5">
                <ArrowDownRight className="w-3 h-3" />{lossReduction.toFixed(1)}%
              </span>
            </div>
            <div className="text-center p-2 rounded-lg bg-muted/10">
              <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Best E[R]</span>
              <span className={cn("text-lg font-bold number-mono", bestExpect >= 0 ? "text-emerald-400" : "text-red-400")}>
                {bestExpect > -999 ? `${bestExpect >= 0 ? "+" : ""}${bestExpect.toFixed(4)}` : "—"}
              </span>
            </div>
            <div className="text-center p-2 rounded-lg bg-muted/10">
              <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Best Win Rate</span>
              <span className="text-lg font-bold number-mono text-cyan-400">
                {bestWR > 0 ? `${(bestWR * 100).toFixed(1)}%` : "—"}
              </span>
            </div>
            <div className="text-center p-2 rounded-lg bg-muted/10">
              <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Best PF</span>
              <span className="text-lg font-bold number-mono text-cyan-400">
                {bestPF > 0 ? bestPF.toFixed(2) : "—"}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function LossCurves({ epochs, folds }: { epochs: TrainingEpoch[]; folds: TrainingFold[] }) {
  const [showComponents, setShowComponents] = useState(false);

  if (!epochs.length) {
    return (
      <Card className="glass-card mb-6">
        <CardHeader><CardTitle className="text-sm flex items-center gap-2"><BarChart3 className="w-4 h-4 text-cyan-400" />Loss Curves</CardTitle></CardHeader>
        <CardContent><p className="text-sm text-muted-foreground text-center py-12">Waiting for epoch data...</p></CardContent>
      </Card>
    );
  }

  const chartData = epochs.map((e) => ({
    label: `F${e.foldNum}E${e.epoch}`,
    epoch: e.epoch,
    fold: e.foldNum,
    trainLoss: e.trainLoss,
    valLoss: e.valLoss,
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
            <Badge variant="outline" className="ml-2 text-[10px]">{epochs.length} epochs</Badge>
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={() => setShowComponents(!showComponents)} data-testid="button-toggle-components">
            {showComponents ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            <span className="ml-1 text-xs">{showComponents ? "Hide" : "Show"} Components</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.2} />
            <XAxis dataKey="label" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} interval={Math.max(1, Math.floor(chartData.length / 20))} />
            <YAxis tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(225 40% 8%)", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11 }} />
            <Line type="monotone" dataKey="trainLoss" stroke="#06b6d4" strokeWidth={2} dot={false} name="Train Loss" />
            <Line type="monotone" dataKey="valLoss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Val Loss" />
            {showComponents && (
              <>
                <Line type="monotone" dataKey="L_ret" stroke="#8b5cf6" strokeWidth={1} dot={false} name="L_ret" strokeDasharray="4 2" />
                <Line type="monotone" dataKey="L_action" stroke="#ef4444" strokeWidth={1} dot={false} name="L_action" strokeDasharray="4 2" />
                <Line type="monotone" dataKey="L_mfe" stroke="#22c55e" strokeWidth={1} dot={false} name="L_mfe" strokeDasharray="4 2" />
                <Line type="monotone" dataKey="L_mae" stroke="#ec4899" strokeWidth={1} dot={false} name="L_mae" strokeDasharray="4 2" />
              </>
            )}
            {foldBoundaries.map((b, i) => (
              <ReferenceLine key={i} x={b} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" opacity={0.4} label={{ value: `F${i + 1}`, position: "top", fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            ))}
            <Legend wrapperStyle={{ fontSize: 10 }} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function CumulativeRCurve({ folds }: { folds: TrainingFold[] }) {
  const completed = folds.filter((f) => f.status === "completed" && f.totalR !== null);
  if (completed.length < 2) return null;

  let cumR = 0;
  const data = completed.map((f) => {
    cumR += f.totalR ?? 0;
    return {
      fold: `F${f.foldNum}`,
      totalR: +(f.totalR ?? 0).toFixed(2),
      cumulativeR: +cumR.toFixed(2),
      trades: f.trades ?? 0,
      winRate: f.winRate ? +(f.winRate * 100).toFixed(1) : null,
    };
  });

  return (
    <Card className="glass-card" data-testid="cumulative-r-curve">
      <CardHeader>
        <CardTitle className="text-sm flex items-center gap-2">
          <Trophy className="w-4 h-4 text-emerald-400" />
          Cumulative R
          <Badge variant="outline" className={cn("ml-2 text-[10px]", cumR >= 0 ? "text-emerald-400 border-emerald-500/30" : "text-red-400 border-red-500/30")}>
            {cumR >= 0 ? "+" : ""}{cumR.toFixed(1)}R
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.2} />
            <XAxis dataKey="fold" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            <YAxis yAxisId="left" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(225 40% 8%)", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11 }} />
            <Bar yAxisId="right" dataKey="totalR" name="Fold R" radius={[3, 3, 0, 0]} fillOpacity={0.6}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.totalR >= 0 ? "#22c55e" : "#ef4444"} />
              ))}
            </Bar>
            <Line yAxisId="left" type="monotone" dataKey="cumulativeR" stroke="#06b6d4" strokeWidth={2.5} dot={{ fill: "#06b6d4", r: 3 }} name="Cumulative R" />
            <ReferenceLine yAxisId="left" y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" opacity={0.4} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function AccuracyChart({ epochs }: { epochs: TrainingEpoch[] }) {
  const chartData = useMemo(() =>
    epochs.filter((e) => e.actionAccuracy !== null).map((e) => ({
      label: `F${e.foldNum}E${e.epoch}`,
      accuracy: e.actionAccuracy ? +(e.actionAccuracy * 100).toFixed(1) : null,
    })),
    [epochs]
  );

  if (!chartData.length) return null;

  return (
    <Card className="glass-card" data-testid="accuracy-chart">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Target className="w-4 h-4 text-emerald-400" />Action Accuracy</CardTitle></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.2} />
            <XAxis dataKey="label" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} interval={Math.max(1, Math.floor(chartData.length / 15))} />
            <YAxis tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} domain={[0, 100]} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(225 40% 8%)", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11 }} />
            <Area type="monotone" dataKey="accuracy" stroke="#22c55e" fill="#22c55e" fillOpacity={0.08} strokeWidth={2} name="Accuracy %" />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function ExpectancyChart({ epochs }: { epochs: TrainingEpoch[] }) {
  const sweepEpochs = useMemo(() => epochs.filter((e) => e.expectancy !== null), [epochs]);
  if (!sweepEpochs.length) return null;

  const chartData = sweepEpochs.map((e) => ({
    label: `F${e.foldNum}E${e.epoch}`,
    expectancy: e.expectancy ? +e.expectancy.toFixed(4) : null,
    threshold: e.threshold ? +e.threshold.toFixed(4) : null,
    winRate: e.winRate ? +(e.winRate * 100).toFixed(1) : null,
    pf: e.profitFactor ? +e.profitFactor.toFixed(2) : null,
  }));

  return (
    <Card className="glass-card" data-testid="expectancy-chart">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><TrendingUp className="w-4 h-4 text-cyan-400" />Expectancy & Threshold</CardTitle></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.2} />
            <XAxis dataKey="label" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} interval={Math.max(1, Math.floor(chartData.length / 15))} />
            <YAxis yAxisId="left" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(225 40% 8%)", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11 }} />
            <Line yAxisId="left" type="monotone" dataKey="expectancy" stroke="#06b6d4" strokeWidth={2} dot={false} name="E[R]" />
            <Line yAxisId="right" type="monotone" dataKey="threshold" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Threshold" strokeDasharray="4 2" />
            <ReferenceLine yAxisId="left" y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" opacity={0.4} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function WinRatePFChart({ epochs }: { epochs: TrainingEpoch[] }) {
  const sweepEpochs = useMemo(() => epochs.filter((e) => e.winRate !== null || e.profitFactor !== null), [epochs]);
  if (sweepEpochs.length < 2) return null;

  const chartData = sweepEpochs.map((e) => ({
    label: `F${e.foldNum}E${e.epoch}`,
    winRate: e.winRate ? +(e.winRate * 100).toFixed(1) : null,
    profitFactor: e.profitFactor ? +e.profitFactor.toFixed(2) : null,
  }));

  return (
    <Card className="glass-card" data-testid="winrate-pf-chart">
      <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Hash className="w-4 h-4 text-amber-400" />Win Rate & Profit Factor</CardTitle></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.2} />
            <XAxis dataKey="label" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} interval={Math.max(1, Math.floor(chartData.length / 15))} />
            <YAxis yAxisId="left" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} domain={[0, 100]} />
            <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(225 40% 8%)", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11 }} />
            <Line yAxisId="left" type="monotone" dataKey="winRate" stroke="#22c55e" strokeWidth={2} dot={false} name="Win Rate %" />
            <Line yAxisId="right" type="monotone" dataKey="profitFactor" stroke="#a78bfa" strokeWidth={2} dot={false} name="Profit Factor" />
            <ReferenceLine yAxisId="left" y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" opacity={0.3} />
            <ReferenceLine yAxisId="right" y={1} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" opacity={0.3} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function FoldResultsTable({ folds }: { folds: TrainingFold[] }) {
  if (!folds.length) return null;

  const completedFolds = folds.filter((f) => f.status === "completed");
  const totalR = completedFolds.reduce((sum, f) => sum + (f.totalR ?? 0), 0);
  const avgWR = completedFolds.length ? completedFolds.reduce((sum, f) => sum + (f.winRate ?? 0), 0) / completedFolds.length : 0;
  const profitableFolds = completedFolds.filter((f) => (f.totalR ?? 0) > 0).length;

  return (
    <Card className="glass-card mb-6" data-testid="fold-results-table">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            Walk-Forward Fold Results
          </CardTitle>
          <div className="flex items-center gap-3 text-xs">
            <span className="text-muted-foreground">{profitableFolds}/{completedFolds.length} profitable</span>
            <Badge variant="outline" className={cn("text-[10px]", totalR >= 0 ? "text-emerald-400 border-emerald-500/30" : "text-red-400 border-red-500/30")}>
              {totalR >= 0 ? "+" : ""}{totalR.toFixed(1)}R total
            </Badge>
            <Badge variant="outline" className="text-[10px] text-cyan-400 border-cyan-500/30">
              {(avgWR * 100).toFixed(1)}% avg WR
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left py-2.5 px-2 text-muted-foreground font-medium">Fold</th>
                <th className="text-left py-2.5 px-2 text-muted-foreground font-medium">Test Window</th>
                <th className="text-left py-2.5 px-2 text-muted-foreground font-medium">Status</th>
                <th className="text-right py-2.5 px-2 text-muted-foreground font-medium">Trades</th>
                <th className="text-right py-2.5 px-2 text-muted-foreground font-medium">Win Rate</th>
                <th className="text-right py-2.5 px-2 text-muted-foreground font-medium">E[R]</th>
                <th className="text-right py-2.5 px-2 text-muted-foreground font-medium">PF</th>
                <th className="text-right py-2.5 px-2 text-muted-foreground font-medium">Sharpe</th>
                <th className="text-right py-2.5 px-2 text-muted-foreground font-medium">MaxDD</th>
                <th className="text-right py-2.5 px-2 text-muted-foreground font-medium">Total R</th>
                <th className="text-right py-2.5 px-2 text-muted-foreground font-medium">Threshold</th>
              </tr>
            </thead>
            <tbody>
              {folds.map((fold) => {
                const isRunning = fold.status === "running";
                const foldR = fold.totalR ?? 0;
                return (
                  <tr
                    key={fold.id}
                    className={cn(
                      "border-b border-border/20 hover:bg-muted/10 transition-colors",
                      isRunning && "bg-cyan-500/5 border-l-2 border-l-cyan-500",
                    )}
                    data-testid={`row-fold-${fold.foldNum}`}
                  >
                    <td className="py-2.5 px-2 font-mono font-medium">
                      {isRunning && <Loader2 className="w-3 h-3 inline mr-1 animate-spin text-cyan-400" />}
                      {fold.foldNum}
                    </td>
                    <td className="py-2.5 px-2 text-muted-foreground text-[11px]">
                      {fold.testStart && fold.testEnd ? `${fold.testStart} → ${fold.testEnd}` : "—"}
                    </td>
                    <td className="py-2.5 px-2">
                      <Badge
                        variant={isRunning ? "default" : fold.status === "completed" ? "secondary" : "outline"}
                        className={cn("text-[10px]", isRunning && "bg-cyan-500/20 text-cyan-400 border-cyan-500/50")}
                      >
                        {fold.status ?? "pending"}
                      </Badge>
                    </td>
                    <td className="py-2.5 px-2 text-right number-mono">{fold.trades ?? "—"}</td>
                    <td className="py-2.5 px-2 text-right number-mono">
                      {fold.winRate !== null ? `${(fold.winRate * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className={cn("py-2.5 px-2 text-right number-mono", (fold.expectancy ?? 0) >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {fold.expectancy !== null ? `${fold.expectancy >= 0 ? "+" : ""}${fold.expectancy.toFixed(4)}` : "—"}
                    </td>
                    <td className="py-2.5 px-2 text-right number-mono">{fold.profitFactor?.toFixed(2) ?? "—"}</td>
                    <td className="py-2.5 px-2 text-right number-mono">{fold.sharpe?.toFixed(2) ?? "—"}</td>
                    <td className="py-2.5 px-2 text-right number-mono text-red-400">{fold.maxDrawdown?.toFixed(2) ?? "—"}</td>
                    <td className={cn("py-2.5 px-2 text-right number-mono font-medium", foldR >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {fold.totalR !== null ? `${foldR >= 0 ? "+" : ""}${foldR.toFixed(2)}` : "—"}
                    </td>
                    <td className="py-2.5 px-2 text-right number-mono text-muted-foreground">{fold.finalThreshold?.toFixed(4) ?? "—"}</td>
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
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.2} />
            <XAxis dataKey="fold" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            <YAxis tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
            <RechartsTooltip contentStyle={{ backgroundColor: "hsl(225 40% 8%)", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 11 }} />
            <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
            <Bar dataKey="totalR" name="Total R" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.totalR >= 0 ? "#22c55e" : "#ef4444"} fillOpacity={0.75} />
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
                <th className="text-left py-2 px-2 text-muted-foreground font-medium">Symbol</th>
                {completedFolds.map((f) => (
                  <th key={f.foldNum} className="text-center py-2 px-1 text-muted-foreground font-medium">F{f.foldNum}</th>
                ))}
                <th className="text-right py-2 px-2 text-muted-foreground font-medium">Total</th>
                <th className="text-center py-2 px-2 text-muted-foreground font-medium">Edge</th>
              </tr>
            </thead>
            <tbody>
              {symbols.map((sym) => {
                let total = 0;
                let positiveFolds = 0;
                return (
                  <tr key={sym} className="border-b border-border/20">
                    <td className="py-2 px-2 font-mono font-medium">{sym.replace("USDT", "")}</td>
                    {completedFolds.map((f) => {
                      const ps = f.perSymbol as Record<string, any> | null;
                      const symData = ps?.[sym];
                      const r = symData?.total_r ?? symData ?? null;
                      const rNum = typeof r === "number" ? r : (r?.total_r ?? 0);
                      total += rNum;
                      if (rNum > 0) positiveFolds++;
                      const intensity = Math.min(1, Math.abs(rNum) / 20);
                      return (
                        <td key={f.foldNum} className="text-center py-2 px-1">
                          <span
                            className={cn(
                              "inline-block w-full px-1 py-0.5 rounded text-[10px] number-mono font-medium",
                              rNum > 0 ? "text-emerald-400" : rNum < 0 ? "text-red-400" : "text-muted-foreground",
                            )}
                            style={{
                              backgroundColor: rNum > 0
                                ? `rgba(34, 197, 94, ${intensity * 0.25})`
                                : rNum < 0
                                  ? `rgba(239, 68, 68, ${intensity * 0.25})`
                                  : "transparent",
                            }}
                          >
                            {typeof rNum === "number" ? `${rNum >= 0 ? "+" : ""}${rNum.toFixed(1)}` : "—"}
                          </span>
                        </td>
                      );
                    })}
                    <td className={cn("text-right py-2 px-2 number-mono font-bold", total >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {total >= 0 ? "+" : ""}{total.toFixed(1)}
                    </td>
                    <td className="text-center py-2 px-2">
                      <Badge
                        variant="outline"
                        className={cn("text-[9px]",
                          positiveFolds > completedFolds.length * 0.6
                            ? "text-emerald-400 border-emerald-500/30 bg-emerald-500/10"
                            : positiveFolds > completedFolds.length * 0.4
                              ? "text-amber-400 border-amber-500/30 bg-amber-500/10"
                              : "text-red-400 border-red-500/30 bg-red-500/10"
                        )}
                      >
                        {positiveFolds > completedFolds.length * 0.6 ? "EDGE" : positiveFolds > completedFolds.length * 0.4 ? "WEAK" : "NONE"}
                      </Badge>
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
        <div className="flex items-center justify-between cursor-pointer select-none" onClick={() => setExpanded(!expanded)}>
          <CardTitle className="text-sm flex items-center gap-2">
            <Cpu className="w-4 h-4 text-amber-400" />
            Training Configuration
          </CardTitle>
          <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </Button>
        </div>
      </CardHeader>
      {expanded && (
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-2 rounded-lg bg-muted/10">
              <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Symbols</span>
              <span className="text-xs font-mono">{session.symbols?.join(", ") ?? "—"}</span>
            </div>
            <div className="p-2 rounded-lg bg-muted/10">
              <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Window</span>
              <span className="text-xs font-mono">{session.trainMonths ?? "?"}m train / {session.testMonths ?? "?"}m test</span>
            </div>
            <div className="p-2 rounded-lg bg-muted/10">
              <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">GPU</span>
              <span className="text-xs font-mono">{session.gpuName ?? "—"}</span>
            </div>
            <div className="p-2 rounded-lg bg-muted/10">
              <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">Epochs/Fold</span>
              <span className="text-xs font-mono">{session.totalEpochs ?? "—"}</span>
            </div>
            {config && Object.entries(config).slice(0, 12).map(([key, val]) => (
              <div key={key} className="p-2 rounded-lg bg-muted/10">
                <span className="text-[10px] text-muted-foreground block uppercase tracking-wider">{key.replace(/_/g, " ")}</span>
                <span className="text-xs font-mono break-all">{typeof val === "object" ? JSON.stringify(val) : String(val)}</span>
              </div>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

function SessionHistory({ sessions, onSelect, onDelete, deletePending }: {
  sessions: TrainingSession[];
  onSelect: (id: number) => void;
  onDelete?: (id: number) => void;
  deletePending?: boolean;
}) {
  if (!sessions.length) return null;

  return (
    <Card className="glass-card" data-testid="session-history">
      <CardHeader>
        <CardTitle className="text-sm flex items-center gap-2">
          <History className="w-4 h-4 text-muted-foreground" />
          Session History
          <Badge variant="outline" className="text-[10px] ml-2">{sessions.length}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {sessions.map((s) => {
            const agg = s.aggregateMetrics as Record<string, any> | null;
            const duration = s.completedAt ? s.completedAt - s.startedAt : Date.now() - s.startedAt;
            return (
              <div
                key={s.id}
                className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/10 cursor-pointer border border-border/20 transition-colors group"
                onClick={() => onSelect(s.id)}
                data-testid={`card-session-${s.id}`}
              >
                <div className="flex items-center gap-3">
                  {s.status === "completed" ? (
                    <div className="p-1.5 rounded-lg bg-emerald-500/10">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    </div>
                  ) : s.status === "running" ? (
                    <div className="p-1.5 rounded-lg bg-cyan-500/10">
                      <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
                    </div>
                  ) : (
                    <div className="p-1.5 rounded-lg bg-red-500/10">
                      <XCircle className="w-4 h-4 text-red-400" />
                    </div>
                  )}
                  <div>
                    <span className="text-xs font-medium">{s.sessionType === "walk_forward" ? "Walk-Forward" : s.sessionType}</span>
                    <span className="text-[10px] text-muted-foreground ml-2">
                      {format(new Date(s.startedAt), "MMM d, yyyy HH:mm")}
                    </span>
                    {s.symbols && <span className="text-[10px] text-muted-foreground ml-2">{s.symbols.length} symbols</span>}
                  </div>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="number-mono text-muted-foreground">{s.completedFolds}/{s.totalFolds} folds</span>
                  <span className="number-mono text-muted-foreground">{formatDuration(duration)}</span>
                  {agg?.total_r !== undefined && (
                    <span className={cn("number-mono font-medium", agg.total_r >= 0 ? "text-emerald-400" : "text-red-400")}>
                      {agg.total_r >= 0 ? "+" : ""}{Number(agg.total_r).toFixed(1)}R
                    </span>
                  )}
                  {onDelete && s.status !== "running" && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity text-red-400 hover:text-red-300 hover:bg-red-500/10"
                      onClick={(e) => { e.stopPropagation(); onDelete(s.id); }}
                      disabled={deletePending}
                      data-testid={`button-delete-session-${s.id}`}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
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

interface FreshnessSymbol {
  symbol: string;
  totalCandles: number;
  lastCandleTs: number | null;
  lastCandleDate: string | null;
  staleMinutes: number | null;
  status: "fresh" | "stale" | "critical" | "no_data";
  h1Status: string;
  h4Status: string;
  daysOfData: number;
}

interface FreshnessResponse {
  symbols: FreshnessSymbol[];
  totalSymbols: number;
  freshCount: number;
  staleCount: number;
  criticalCount: number;
  noDataCount: number;
  queriedAt: number;
}

interface ReadinessCheck {
  passed: boolean;
  label: string;
  details?: { symbol: string; count: number; needed: number }[];
}

interface RetrainReadinessResponse {
  ready: boolean;
  checks: {
    allSymbolsFresh: ReadinessCheck;
    minCandleCount: ReadinessCheck;
    gpuConnected: ReadinessCheck;
    noActiveSession: ReadinessCheck;
  };
  summary: string;
}

interface SyncResult {
  success: boolean;
  totalInserted: number;
  errorCount: number;
  results: { symbol: string; inserted: number; error?: string }[];
}

interface SyncProgress {
  running: boolean;
  current: string | null;
  completed: { symbol: string; inserted: number; error?: string }[];
  total: number;
}

function DataReadinessTab() {
  const { toast } = useToast();
  const [countdown, setCountdown] = useState(3600);
  const countdownRef = useRef(3600);

  const { data: freshness, isLoading: freshnessLoading, refetch: refetchFreshness } = useQuery<FreshnessResponse>({
    queryKey: ["/api/data/freshness"],
    refetchInterval: 3600000,
  });

  const { data: readiness, isLoading: readinessLoading, refetch: refetchReadiness } = useQuery<RetrainReadinessResponse>({
    queryKey: ["/api/data/retrain-readiness"],
    refetchInterval: 3600000,
  });

  const [isSyncing, setIsSyncing] = useState(false);

  const { data: syncProgress } = useQuery<SyncProgress>({
    queryKey: ["/api/data/sync-progress"],
    refetchInterval: isSyncing ? 1000 : false,
  });

  const syncMutation = useMutation({
    mutationFn: async () => {
      setIsSyncing(true);
      queryClient.invalidateQueries({ queryKey: ["/api/data/sync-progress"] });
      const res = await apiRequest("POST", "/api/data/sync-all");
      return res.json() as Promise<SyncResult>;
    },
    onSettled: () => {
      setIsSyncing(false);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["/api/data/freshness"] });
      queryClient.invalidateQueries({ queryKey: ["/api/data/retrain-readiness"] });
      queryClient.invalidateQueries({ queryKey: ["/api/data/sync-progress"] });
      toast({
        title: data.success ? "Sync complete" : "Sync completed with errors",
        description: data.success
          ? `Inserted ${data.totalInserted} new candles across ${data.results.filter((r) => r.inserted > 0).length} symbols`
          : `Inserted ${data.totalInserted} candles but ${data.errorCount} symbol(s) had errors`,
        variant: data.success ? "default" : "destructive",
      });
    },
    onError: (err: any) => {
      toast({ title: "Sync failed", description: err.message, variant: "destructive" });
    },
  });

  useEffect(() => {
    countdownRef.current = 3600;
    setCountdown(3600);
    const timer = setInterval(() => {
      countdownRef.current -= 1;
      if (countdownRef.current <= 0) {
        countdownRef.current = 3600;
      }
      setCountdown(countdownRef.current);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleManualRefresh = useCallback(() => {
    refetchFreshness();
    refetchReadiness();
    countdownRef.current = 3600;
    setCountdown(3600);
  }, [refetchFreshness, refetchReadiness]);

  const countdownMin = Math.floor(countdown / 60);
  const countdownSec = countdown % 60;

  const statusColor = (status: string) => {
    switch (status) {
      case "fresh": return "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
      case "stale": return "text-amber-400 bg-amber-500/10 border-amber-500/30";
      case "critical": return "text-red-400 bg-red-500/10 border-red-500/30";
      default: return "text-muted-foreground bg-muted/10 border-border/30";
    }
  };

  const statusLabel = (status: string) => {
    switch (status) {
      case "fresh": return "Fresh";
      case "stale": return "Stale";
      case "critical": return "Critical";
      default: return "No Data";
    }
  };

  const rowBg = (status: string) => {
    switch (status) {
      case "fresh": return "";
      case "stale": return "bg-amber-500/5";
      case "critical": return "bg-red-500/5";
      default: return "bg-muted/5";
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {freshness && (
            <div className="flex gap-2 text-xs">
              <Badge variant="outline" className="border-emerald-500/30 text-emerald-400" data-testid="badge-fresh-count">
                <CircleDot className="w-3 h-3 mr-1" />{freshness.freshCount} Fresh
              </Badge>
              {freshness.staleCount > 0 && (
                <Badge variant="outline" className="border-amber-500/30 text-amber-400" data-testid="badge-stale-count">
                  <AlertTriangle className="w-3 h-3 mr-1" />{freshness.staleCount} Stale
                </Badge>
              )}
              {freshness.criticalCount > 0 && (
                <Badge variant="outline" className="border-red-500/30 text-red-400" data-testid="badge-critical-count">
                  <XCircle className="w-3 h-3 mr-1" />{freshness.criticalCount} Critical
                </Badge>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground" data-testid="text-auto-refresh-countdown">
            Next auto-refresh in {countdownMin}m {countdownSec.toString().padStart(2, "0")}s
          </span>
          <Button variant="outline" size="sm" onClick={handleManualRefresh} className="border-border/50" data-testid="button-refresh-freshness">
            <RefreshCw className={cn("w-3 h-3 mr-1", freshnessLoading && "animate-spin")} />
            Refresh
          </Button>
          <Button
            variant="default"
            size="sm"
            onClick={() => syncMutation.mutate()}
            disabled={syncMutation.isPending}
            className="bg-cyan-600 hover:bg-cyan-700"
            data-testid="button-sync-all"
          >
            {syncMutation.isPending ? (
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            ) : (
              <Download className="w-3 h-3 mr-1" />
            )}
            Sync All Now
          </Button>
        </div>
      </div>

      {(isSyncing || syncProgress?.running) && (
        <Card className="glass-card border border-cyan-500/30" data-testid="card-sync-progress">
          <CardContent className="p-4">
            <div className="flex items-center gap-3 mb-3">
              <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-cyan-400">
                    Syncing {syncProgress?.current?.replace("USDT", "") || "..."} ({(syncProgress?.completed.length ?? 0)}/{syncProgress?.total ?? 20})
                  </p>
                  <span className="text-xs text-muted-foreground">
                    {syncProgress?.total ? Math.round(((syncProgress.completed.length) / syncProgress.total) * 100) : 0}%
                  </span>
                </div>
                <Progress value={syncProgress?.total ? ((syncProgress.completed.length) / syncProgress.total) * 100 : 0} className="h-1.5 mt-2" />
              </div>
            </div>
            {syncProgress && syncProgress.completed.length > 0 && (
              <div className="grid grid-cols-4 md:grid-cols-5 gap-1 text-xs">
                {syncProgress.completed.map((r) => (
                  <div key={r.symbol} className={cn("px-2 py-1 rounded flex items-center gap-1",
                    r.error ? "bg-red-500/10 text-red-400" : r.inserted > 0 ? "bg-emerald-500/10 text-emerald-400" : "bg-muted/10 text-muted-foreground"
                  )}>
                    {r.error ? <XCircle className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
                    {r.symbol.replace("USDT", "")}: {r.error ? "Err" : `+${r.inserted}`}
                  </div>
                ))}
                {syncProgress.current && (
                  <div className="px-2 py-1 rounded flex items-center gap-1 bg-cyan-500/10 text-cyan-400">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    {syncProgress.current.replace("USDT", "")}...
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {syncMutation.isSuccess && syncMutation.data && (
        <Card className="glass-card border border-emerald-500/30" data-testid="card-sync-results">
          <CardContent className="p-4">
            <div className="flex items-center gap-3 mb-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <p className="text-sm font-medium text-emerald-400">
                Sync complete — {syncMutation.data.totalInserted} new candles inserted
              </p>
            </div>
            {syncMutation.data.results.some((r) => r.inserted > 0 || r.error) && (
              <div className="grid grid-cols-4 md:grid-cols-5 gap-1 text-xs">
                {syncMutation.data.results
                  .filter((r) => r.inserted > 0 || r.error)
                  .map((r) => (
                    <div key={r.symbol} className={cn("px-2 py-1 rounded", r.error ? "bg-red-500/10 text-red-400" : "bg-emerald-500/10 text-emerald-400")}>
                      {r.symbol.replace("USDT", "")}: {r.error ? "Error" : `+${r.inserted}`}
                    </div>
                  ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <Card className="glass-card border border-border/50" data-testid="card-freshness-table">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Database className="w-4 h-4 text-cyan-400" />
            Per-Symbol Data Freshness (15m Candles)
          </CardTitle>
        </CardHeader>
        <CardContent>
          {freshnessLoading ? (
            <div className="flex items-center justify-center h-32">
              <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
            </div>
          ) : freshness ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs" data-testid="table-freshness">
                <thead>
                  <tr className="border-b border-border/30 text-muted-foreground">
                    <th className="text-left py-2 px-2 font-medium">Symbol</th>
                    <th className="text-left py-2 px-2 font-medium">Last 15m Candle</th>
                    <th className="text-right py-2 px-2 font-medium">Staleness</th>
                    <th className="text-center py-2 px-2 font-medium">Status</th>
                    <th className="text-center py-2 px-2 font-medium">1H (Resampled)</th>
                    <th className="text-center py-2 px-2 font-medium">4H (Resampled)</th>
                    <th className="text-right py-2 px-2 font-medium">Candles</th>
                    <th className="text-right py-2 px-2 font-medium">Days</th>
                  </tr>
                </thead>
                <tbody>
                  {freshness.symbols.map((s) => (
                    <tr key={s.symbol} className={cn("border-b border-border/10 hover:bg-muted/10 transition-colors", rowBg(s.status))} data-testid={`row-symbol-${s.symbol}`}>
                      <td className="py-2 px-2 font-medium text-foreground">{s.symbol.replace("USDT", "")}</td>
                      <td className="py-2 px-2 text-muted-foreground">
                        {s.lastCandleDate ? format(new Date(s.lastCandleDate), "MMM dd, HH:mm") : "—"}
                      </td>
                      <td className="py-2 px-2 text-right">
                        {s.staleMinutes !== null ? (
                          <span className={cn(
                            s.status === "fresh" ? "text-emerald-400" : s.status === "stale" ? "text-amber-400" : "text-red-400"
                          )}>
                            {s.staleMinutes < 60 ? `${s.staleMinutes}m` : `${Math.floor(s.staleMinutes / 60)}h ${s.staleMinutes % 60}m`}
                          </span>
                        ) : "—"}
                      </td>
                      <td className="py-2 px-2 text-center">
                        <Badge variant="outline" className={cn("text-[10px] py-0 px-1.5", statusColor(s.status))} data-testid={`badge-status-${s.symbol}`}>
                          {statusLabel(s.status)}
                        </Badge>
                      </td>
                      <td className="py-2 px-2 text-center">
                        <Badge variant="outline" className={cn("text-[10px] py-0 px-1.5",
                          s.h1Status === "synced" ? "text-emerald-400 border-emerald-500/30" : "text-amber-400 border-amber-500/30"
                        )}>
                          {s.h1Status === "synced" ? "In-Sync" : "Stale"}
                        </Badge>
                      </td>
                      <td className="py-2 px-2 text-center">
                        <Badge variant="outline" className={cn("text-[10px] py-0 px-1.5",
                          s.h4Status === "synced" ? "text-emerald-400 border-emerald-500/30" : "text-amber-400 border-amber-500/30"
                        )}>
                          {s.h4Status === "synced" ? "In-Sync" : "Stale"}
                        </Badge>
                      </td>
                      <td className="py-2 px-2 text-right number-mono text-muted-foreground">{s.totalCandles.toLocaleString()}</td>
                      <td className="py-2 px-2 text-right number-mono text-muted-foreground">{s.daysOfData}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Failed to load freshness data</p>
          )}
        </CardContent>
      </Card>

      <Card className="glass-card border border-border/50" data-testid="card-retrain-readiness">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            Retrain Readiness Checklist
          </CardTitle>
        </CardHeader>
        <CardContent>
          {readinessLoading ? (
            <div className="flex items-center justify-center h-20">
              <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
            </div>
          ) : readiness ? (
            <div className="space-y-3">
              {Object.entries(readiness.checks).map(([key, check]) => (
                <div key={key} className="flex items-start gap-3" data-testid={`check-${key}`}>
                  {check.passed ? (
                    <CheckCircle2 className="w-5 h-5 text-emerald-400 mt-0.5 shrink-0" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
                  )}
                  <div>
                    <p className={cn("text-sm", check.passed ? "text-emerald-400" : "text-red-400")}>{check.label}</p>
                    {check.details && check.details.length > 0 && (
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Missing: {check.details.map((d) => `${d.symbol.replace("USDT", "")} (${d.count.toLocaleString()}/${d.needed.toLocaleString()})`).join(", ")}
                      </p>
                    )}
                  </div>
                </div>
              ))}
              <div className={cn(
                "mt-4 p-3 rounded-lg border text-sm font-medium text-center",
                readiness.ready
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                  : "border-red-500/30 bg-red-500/10 text-red-400"
              )} data-testid="text-readiness-summary">
                {readiness.summary}
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Failed to load readiness data</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function TrainingMonitor() {
  const { data: activeData, isLoading } = useActiveTraining();
  const { data: sessionsData } = useSessions();
  const { data: readyData } = useTrainingReady();
  const { subscribe } = useTradingWs();
  const { toast } = useToast();

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

  const clearSessionMutation = useMutation({
    mutationFn: async (id: number) => {
      await apiRequest("DELETE", `/api/training/sessions/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/training/active"] });
      queryClient.invalidateQueries({ queryKey: ["/api/training/sessions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/training/ready"] });
      setSelectedSessionId(null);
      toast({ title: "Session cleared", description: "Training session data has been removed" });
    },
    onError: (err: any) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  const clearAllMutation = useMutation({
    mutationFn: async () => {
      await apiRequest("POST", "/api/training/sessions/clear-all");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/training/active"] });
      queryClient.invalidateQueries({ queryKey: ["/api/training/sessions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/training/ready"] });
      setSelectedSessionId(null);
      toast({ title: "All sessions cleared", description: "Training history has been cleared. Ready for new training." });
    },
    onError: (err: any) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

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
        queryClient.invalidateQueries({ queryKey: ["/api/training/ready"] });
      }),
      subscribe("TRAINING_SESSION_END", () => {
        queryClient.invalidateQueries({ queryKey: ["/api/training/active"] });
        queryClient.invalidateQueries({ queryKey: ["/api/training/sessions"] });
        queryClient.invalidateQueries({ queryKey: ["/api/training/ready"] });
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

  const allSessions = sessionsData?.sessions ?? [];
  const historySessions = allSessions.filter((s) => s.id !== viewingSession?.id);
  const hasUncleared = readyData && !readyData.ready && readyData.runningSessions === 0;

  return (
    <div className="p-4 md:p-6 space-y-0" data-testid="training-monitor-page">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <Brain className="w-5 h-5 text-cyan-400" />
          Training Monitor
        </h1>
        <div className="flex items-center gap-2">
          {selectedSessionId && selectedSessionId !== displaySession?.id && (
            <Button variant="ghost" size="sm" onClick={() => setSelectedSessionId(null)} data-testid="button-back-to-active">
              Back to {isActive ? "Active" : "Latest"}
            </Button>
          )}
          {hasUncleared && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" className="border-amber-500/30 text-amber-400 hover:bg-amber-500/10" data-testid="button-clear-all">
                  {clearAllMutation.isPending ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Trash2 className="w-3 h-3 mr-1" />}
                  Clear All Sessions
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent className="glass-card border-border/50">
                <AlertDialogHeader>
                  <AlertDialogTitle>Clear All Training Sessions?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will permanently delete all completed training sessions and their data.
                    This action is required before starting new training.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel data-testid="button-cancel-clear-all">Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={() => clearAllMutation.mutate()} className="bg-amber-500 hover:bg-amber-600 text-black" data-testid="button-confirm-clear-all">
                    Clear All
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>
      </div>

      <Tabs defaultValue="training" className="w-full">
        <TabsList className="mb-4 bg-muted/20 border border-border/30" data-testid="tabs-training-monitor">
          <TabsTrigger value="training" className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400" data-testid="tab-training">
            <Brain className="w-3.5 h-3.5 mr-1.5" />
            Training
          </TabsTrigger>
          <TabsTrigger value="data-readiness" className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400" data-testid="tab-data-readiness">
            <Database className="w-3.5 h-3.5 mr-1.5" />
            Data Readiness
          </TabsTrigger>
        </TabsList>

        <TabsContent value="training">
          {readyData && <ReadinessGate ready={readyData} />}

          <StatusBanner
            session={viewingSession}
            isActive={isActive && viewingSession?.id === activeSession?.id}
            onClear={(id) => clearSessionMutation.mutate(id)}
            onClearAll={() => clearAllMutation.mutate()}
            clearPending={clearSessionMutation.isPending}
          />

          {viewingSession && (
            <>
              <OverviewCards session={viewingSession} />
              <ModelKnowledge epochs={allEpochs} session={viewingSession} />
              <LossCurves epochs={allEpochs} folds={viewingFolds} />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <AccuracyChart epochs={allEpochs} />
                <ExpectancyChart epochs={allEpochs} />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <WinRatePFChart epochs={allEpochs} />
                <CumulativeRCurve folds={viewingFolds} />
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

          {historySessions.length > 0 && (
            <SessionHistory
              sessions={historySessions}
              onSelect={setSelectedSessionId}
              onDelete={(id) => clearSessionMutation.mutate(id)}
              deletePending={clearSessionMutation.isPending}
            />
          )}
        </TabsContent>

        <TabsContent value="data-readiness">
          <DataReadinessTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
