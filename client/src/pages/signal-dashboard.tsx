import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";
import { TrendingDown, TrendingUp, Target, Zap, AlertTriangle, RefreshCw } from "lucide-react";
import { useTradingWs } from "@/hooks/use-trading-ws";
import { queryClient } from "@/lib/queryClient";
import type { V5Signal } from "@shared/schema";

function fmt(n: number | null | undefined, decimals = 2) {
  if (n == null || !isFinite(n)) return "—";
  return n.toFixed(decimals);
}

function fmtPrice(n: number | null | undefined) {
  if (n == null || !isFinite(n)) return "—";
  if (n > 1000) return n.toLocaleString("en-US", { maximumFractionDigits: 2 });
  if (n > 10) return n.toFixed(3);
  return n.toFixed(4);
}

interface LeverageStats {
  leverage: number;
  riskPct: number;
  winRateNeeded: number;
  ev: number;
  maxLosingStreak: number;
  accountLossOnStreak: number;
  safe: boolean;
}

function calcLeverageStats(rr: number, winRate: number, accountEquity: number, riskPct: number, leverage: number): LeverageStats {
  const r = winRate / 100;
  const ev = r * rr - (1 - r);
  const winRateNeeded = 1 / (1 + rr);
  const p_loss = 1 - r;
  const streak10 = Math.pow(p_loss, 10);
  const streak13 = Math.pow(p_loss, 13);
  const lossPerTrade = (riskPct / 100) * leverage;
  const accountLossOnStreak = Math.min(1 - Math.pow(1 - lossPerTrade, 13), 1) * 100;
  const safe = accountLossOnStreak < 20 && ev > 0;
  const maxLosingStreak = Math.ceil(Math.log(0.01) / Math.log(p_loss));
  return { leverage, riskPct, winRateNeeded: winRateNeeded * 100, ev, maxLosingStreak, accountLossOnStreak, safe };
}

function LeverageCalculator({ signal }: { signal: V5Signal }) {
  const [winRate, setWinRate] = useState(50);
  const [riskPct, setRiskPct] = useState(1);
  const [leverage, setLeverage] = useState(5);
  const [equity] = useState(1500);

  const rr = signal.predictedRr ?? 1.5;
  const stats = calcLeverageStats(rr, winRate, equity, riskPct, leverage);
  const usdRisk = (equity * riskPct) / 100;
  const usdProfit = usdRisk * rr;
  const usdLoss = usdRisk;

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground font-medium uppercase tracking-wider">Leverage Calculator</div>
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="text-xs text-muted-foreground">Win Rate %</label>
          <input
            type="number"
            value={winRate}
            onChange={e => setWinRate(Number(e.target.value))}
            min={20} max={80} step={1}
            className="w-full mt-1 px-2 py-1.5 text-sm rounded bg-background border border-border text-foreground"
            data-testid="input-win-rate"
          />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Risk % / Trade</label>
          <input
            type="number"
            value={riskPct}
            onChange={e => setRiskPct(Number(e.target.value))}
            min={0.1} max={5} step={0.1}
            className="w-full mt-1 px-2 py-1.5 text-sm rounded bg-background border border-border text-foreground"
            data-testid="input-risk-pct"
          />
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Leverage</label>
          <input
            type="number"
            value={leverage}
            onChange={e => setLeverage(Number(e.target.value))}
            min={1} max={50} step={1}
            className="w-full mt-1 px-2 py-1.5 text-sm rounded bg-background border border-border text-foreground"
            data-testid="input-leverage"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-2 text-center">
          <div className="text-xs text-muted-foreground">Win</div>
          <div className="font-semibold text-emerald-400">+${usdProfit.toFixed(0)}</div>
        </div>
        <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-2 text-center">
          <div className="text-xs text-muted-foreground">Loss</div>
          <div className="font-semibold text-red-400">-${usdLoss.toFixed(0)}</div>
        </div>
        <div className={cn("rounded-lg border p-2 text-center", stats.ev > 0 ? "bg-emerald-500/10 border-emerald-500/20" : "bg-red-500/10 border-red-500/20")}>
          <div className="text-xs text-muted-foreground">EV per trade</div>
          <div className={cn("font-semibold", stats.ev > 0 ? "text-emerald-400" : "text-red-400")}>
            {stats.ev > 0 ? "+" : ""}{(stats.ev * usdRisk).toFixed(1)}R
          </div>
        </div>
        <div className={cn("rounded-lg border p-2 text-center", stats.accountLossOnStreak < 20 ? "bg-emerald-500/10 border-emerald-500/20" : "bg-red-500/10 border-red-500/20")}>
          <div className="text-xs text-muted-foreground">13-loss streak</div>
          <div className={cn("font-semibold", stats.accountLossOnStreak < 20 ? "text-emerald-400" : "text-red-400")}>
            -{stats.accountLossOnStreak.toFixed(0)}%
          </div>
        </div>
      </div>

      <div className={cn("rounded-lg border p-3 text-center text-sm font-medium", stats.safe ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" : "bg-red-500/10 border-red-500/20 text-red-400")}>
        {stats.safe
          ? `✓ Safe — max 13-loss streak = ${stats.accountLossOnStreak.toFixed(0)}% drawdown`
          : `⚠ DANGER — ${leverage}x leverage can wipe ${stats.accountLossOnStreak.toFixed(0)}% on 13 losses`}
      </div>

      <div className="text-xs text-muted-foreground space-y-0.5">
        <div>Min WR to be profitable: <span className="text-foreground">{stats.winRateNeeded.toFixed(1)}%</span></div>
        <div>Expected max losing streak (1%): <span className="text-foreground">{stats.maxLosingStreak} trades</span></div>
        <div>Predicted R:R from model: <span className="text-foreground">{fmt(rr)}:1</span></div>
      </div>
    </div>
  );
}

function SignalCard({ signal, isNew }: { signal: V5Signal; isNew?: boolean }) {
  const isLong = signal.direction === "LONG";
  const isShort = signal.direction === "SHORT";
  const age = formatDistanceToNow(new Date(signal.signalTs), { addSuffix: true });
  const rr = signal.predictedRr;
  const rrOk = rr != null && rr >= 1.3;

  return (
    <Card
      data-testid={`card-signal-${signal.id}`}
      className={cn(
        "border transition-all duration-300",
        isNew && "ring-1 ring-primary/50",
        isLong ? "border-emerald-500/30" : "border-red-500/30"
      )}
    >
      <CardHeader className="pb-2 pt-3 px-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isLong
              ? <TrendingUp className="w-4 h-4 text-emerald-400" />
              : <TrendingDown className="w-4 h-4 text-red-400" />}
            <span className="font-bold text-sm">{signal.symbol}</span>
            <Badge
              variant="outline"
              className={cn("text-xs px-1.5 py-0",
                isLong ? "border-emerald-500/50 text-emerald-400" : "border-red-500/50 text-red-400")}
            >
              {signal.direction}
            </Badge>
            {isNew && <Badge variant="outline" className="text-xs px-1.5 py-0 border-primary/50 text-primary animate-pulse">NEW</Badge>}
          </div>
          <span className="text-xs text-muted-foreground">{age}</span>
        </div>
      </CardHeader>

      <CardContent className="px-4 pb-4 space-y-3">
        <div className="grid grid-cols-3 gap-2 text-sm">
          <div className="text-center rounded bg-background/50 border border-border/50 p-2">
            <div className="text-xs text-muted-foreground mb-0.5">Entry</div>
            <div className="font-mono font-medium">{fmtPrice(signal.entryPrice)}</div>
          </div>
          <div className="text-center rounded bg-red-500/5 border border-red-500/20 p-2">
            <div className="text-xs text-red-400 mb-0.5">Stop Loss</div>
            <div className="font-mono font-medium text-red-300">{fmtPrice(signal.slPrice)}</div>
          </div>
          <div className="text-center rounded bg-emerald-500/5 border border-emerald-500/20 p-2">
            <div className="text-xs text-emerald-400 mb-0.5">Take Profit</div>
            <div className="font-mono font-medium text-emerald-300">{fmtPrice(signal.tpPrice)}</div>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-2 text-xs">
          <div>
            <span className="text-muted-foreground">Score</span>
            <div className="font-medium mt-0.5">{fmt(signal.score, 4)}</div>
          </div>
          <div>
            <span className="text-muted-foreground">μR</span>
            <div className="font-medium mt-0.5">{fmt(signal.muR, 4)}</div>
          </div>
          <div>
            <span className="text-muted-foreground">p_side</span>
            <div className="font-medium mt-0.5">{fmt(signal.pSide)}</div>
          </div>
          <div>
            <span className="text-muted-foreground">Conf</span>
            <div className="font-medium mt-0.5">{fmt(signal.confidence)}</div>
          </div>
        </div>

        {(signal.predictedMfeR != null || signal.predictedMaeR != null || signal.predictedRr != null) && (
          <>
            <Separator className="opacity-30" />
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">Pred MFE</span>
                <div className="font-medium mt-0.5 text-emerald-400">{fmt(signal.predictedMfeR)}R</div>
              </div>
              <div>
                <span className="text-muted-foreground">Pred MAE</span>
                <div className="font-medium mt-0.5 text-red-400">{fmt(signal.predictedMaeR)}R</div>
              </div>
              <div>
                <span className="text-muted-foreground">Pred R:R</span>
                <div className={cn("font-semibold mt-0.5", rrOk ? "text-emerald-400" : "text-amber-400")}>
                  {fmt(rr)}:1{!rrOk && " ⚠"}
                </div>
              </div>
            </div>
          </>
        )}

        {signal.ofGateReason && (
          <div className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded px-2 py-1">
            OF: {signal.ofGateReason}
          </div>
        )}

        <Separator className="opacity-30" />
        <LeverageCalculator signal={signal} />
      </CardContent>
    </Card>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <Target className="w-12 h-12 text-muted-foreground/40 mb-4" />
      <h3 className="text-lg font-medium text-muted-foreground mb-2">No signals yet</h3>
      <p className="text-sm text-muted-foreground/70 max-w-sm">
        Signals will appear here when the GPU trainer detects high-confidence SHORT opportunities.
        Connect the GPU trainer and ensure it's sending ENTER decisions.
      </p>
    </div>
  );
}

export default function SignalDashboard() {
  const [newSignalIds, setNewSignalIds] = useState<Set<number>>(new Set());
  const [lastKnownIds, setLastKnownIds] = useState<Set<number>>(new Set());

  const { data: signals = [] } = useQuery<V5Signal[]>({
    queryKey: ["/api/v5/signals"],
    refetchInterval: 30000,
  });

  const { data: settings } = useQuery<any>({
    queryKey: ["/api/settings"],
  });

  const accountEquity = settings?.account_equity_usd ?? 1500;

  const { subscribe } = useTradingWs();

  useEffect(() => {
    const unsub = subscribe("V5_SIGNAL", (payload) => {
      queryClient.invalidateQueries({ queryKey: ["/api/v5/signals"] });
      const id = payload.id as number;
      if (id) {
        setNewSignalIds(prev => new Set([...prev, id]));
        setTimeout(() => setNewSignalIds(prev => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        }), 10000);
      }
    });
    return unsub;
  }, [subscribe]);

  useEffect(() => {
    if (signals.length > 0) {
      const currentIds = new Set(signals.map(s => s.id));
      const newIds = new Set([...currentIds].filter(id => !lastKnownIds.has(id)));
      if (newIds.size > 0 && lastKnownIds.size > 0) {
        setNewSignalIds(prev => new Set([...prev, ...newIds]));
        setTimeout(() => setNewSignalIds(prev => {
          const next = new Set(prev);
          newIds.forEach(id => next.delete(id));
          return next;
        }), 10000);
      }
      setLastKnownIds(currentIds);
    }
  }, [signals]);

  const shortSignals = signals.filter(s => s.direction === "SHORT");
  const longSignals = signals.filter(s => s.direction === "LONG");
  const recentSignals = signals.slice(0, 20);
  const highConfidence = signals.filter(s => s.confidence >= 0.7).length;
  const withPredRr = signals.filter(s => s.predictedRr != null && s.predictedRr >= 1.3).length;

  return (
    <div className="p-4 md:p-6 space-y-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Target className="w-6 h-6 text-primary" />
            Signal Dashboard
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            High-confidence V5 signals for manual execution — SHORT specialist active
          </p>
        </div>
        <button
          onClick={() => queryClient.invalidateQueries({ queryKey: ["/api/v5/signals"] })}
          data-testid="button-refresh-signals"
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-border hover:bg-accent transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border border-border bg-card p-3" data-testid="stat-total-signals">
          <div className="text-xs text-muted-foreground">Total Signals</div>
          <div className="text-2xl font-bold mt-1">{signals.length}</div>
        </div>
        <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3" data-testid="stat-short-signals">
          <div className="text-xs text-red-400">SHORT Signals</div>
          <div className="text-2xl font-bold mt-1 text-red-400">{shortSignals.length}</div>
        </div>
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3" data-testid="stat-long-signals">
          <div className="text-xs text-emerald-400">LONG Signals</div>
          <div className="text-2xl font-bold mt-1 text-emerald-400">{longSignals.length}</div>
        </div>
        <div className="rounded-lg border border-primary/30 bg-primary/5 p-3" data-testid="stat-high-rr">
          <div className="text-xs text-primary">R:R ≥ 1.3</div>
          <div className="text-2xl font-bold mt-1 text-primary">{withPredRr}</div>
        </div>
      </div>

      <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 flex items-start gap-3">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-sm text-amber-200 space-y-1">
          <div className="font-medium text-amber-300">Risk Warning — Read Before Trading</div>
          <div className="text-xs text-amber-200/80 space-y-0.5">
            <div>• At <strong>15x leverage</strong>, a 13-loss streak costs <strong>74% of account</strong> (happens at 55% WR)</div>
            <div>• Safe max leverage: <strong>5x</strong> (13-streak = 16% drawdown at 50% WR)</div>
            <div>• Account equity: <strong>${accountEquity.toLocaleString()}</strong> • Risk 1% = ${(accountEquity * 0.01).toFixed(0)}/trade</div>
            <div>• Only trade signals with predicted R:R ≥ 1.3 and SHORT direction (LONG disabled)</div>
          </div>
        </div>
      </div>

      {recentSignals.length === 0 ? (
        <EmptyState />
      ) : (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-4 h-4 text-primary" />
            <h2 className="text-sm font-semibold">Recent Signals</h2>
            <span className="text-xs text-muted-foreground">({recentSignals.length} shown, newest first)</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recentSignals.map(signal => (
              <SignalCard
                key={signal.id}
                signal={signal}
                isNew={newSignalIds.has(signal.id)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
