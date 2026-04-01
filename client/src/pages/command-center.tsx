import { useQuery } from "@tanstack/react-query";
import { useState, useEffect, useCallback } from "react";
import { formatDistanceToNow } from "date-fns";
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Target,
  Wallet,
  BarChart3,
  Brain,
  Radio,
  Crosshair,
  Zap,
  Shield,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { CloseButton } from "@/components/position-actions";
import { useTradingWs } from "@/hooks/use-trading-ws";
import { queryClient } from "@/lib/queryClient";
import { TRADING_SYMBOLS } from "@shared/symbols";

const SYMBOLS = TRADING_SYMBOLS;

interface CycleLog {
  id: number;
  symbol: string;
  cycleTs: number;
  price: number | null;
  pEnter: number | null;
  direction: string | null;
  decision: string;
  reasons: string[] | null;
  laneSelected: string | null;
  htfScore: number | null;
  thresholdUsed: number | null;
  laneSizeMult: number | null;
  holdReason: string | null;
  v5Score: number | null;
  v5Threshold: number | null;
  v5Side: string | null;
  retMu: number | null;
  mfePred: number | null;
  maePred: number | null;
  pHold: number | null;
  pLong: number | null;
  pShort: number | null;
  createdAt: number;
  autoTradeResult?: { opened: boolean; positionId?: number; reason?: string };
}

function MetricCard({
  label,
  value,
  glow,
  icon: Icon,
  testId,
}: {
  label: string;
  value: string;
  glow?: "green" | "red" | "cyan";
  icon: typeof Activity;
  testId: string;
}) {
  const glowClass = glow === "green" ? "glow-green" : glow === "red" ? "glow-red" : glow === "cyan" ? "glow-cyan" : "";
  return (
    <div
      className={`glass-card rounded-lg p-4 animate-fade-in-up ${glowClass}`}
      data-testid={testId}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-muted-foreground" />
        <span className="text-xs text-muted-foreground uppercase tracking-wider">{label}</span>
      </div>
      <div className="number-mono text-2xl font-bold">{value}</div>
    </div>
  );
}

function Sparkline({ data }: { data: Array<{ close: number }> }) {
  if (!data || data.length === 0) return <div className="h-12 shimmer rounded" />;
  return (
    <ResponsiveContainer width="100%" height={48}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22c55e" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey="close"
          stroke="#22c55e"
          strokeWidth={1.5}
          fill="url(#sparkGrad)"
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function MarketCard({
  symbol,
  priceData,
}: {
  symbol: string;
  priceData?: { price: number; change24h: number; high24h: number; low24h: number };
}) {
  const { data: candles } = useQuery<Array<{ close: number }>>({
    queryKey: [`/api/market/candles?symbol=${symbol}&interval=15m&limit=96`],
    refetchInterval: 15000,
  });

  const change = priceData?.change24h ?? 0;
  const isPositive = change >= 0;

  return (
    <div
      className="glass-card rounded-lg p-4 animate-fade-in-up scanline"
      data-testid={`market-card-${symbol}`}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="font-bold text-sm">{symbol.replace("USDT", "")}</span>
        <Badge
          className={`no-default-hover-elevate no-default-active-elevate text-xs ${isPositive ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}`}
          data-testid={`badge-change-${symbol}`}
        >
          {isPositive ? "+" : ""}{change.toFixed(2)}%
        </Badge>
      </div>
      <div className="number-mono text-xl font-bold mb-2" data-testid={`price-${symbol}`}>
        {priceData ? `$${priceData.price.toLocaleString()}` : "—"}
      </div>
      <Sparkline data={candles ?? []} />
    </div>
  );
}

function ActionProbBar({ pHold, pLong, pShort }: { pHold: number | null; pLong: number | null; pShort: number | null }) {
  const hold = (pHold ?? 0) * 100;
  const long = (pLong ?? 0) * 100;
  const short = (pShort ?? 0) * 100;
  const total = hold + long + short;
  if (total === 0) return null;

  return (
    <div className="flex items-center gap-1.5 w-full">
      <div className="flex h-1.5 flex-1 rounded-full overflow-hidden bg-muted">
        <div className="bg-emerald-500" style={{ width: `${long}%` }} />
        <div className="bg-amber-500" style={{ width: `${hold}%` }} />
        <div className="bg-red-500" style={{ width: `${short}%` }} />
      </div>
      <span className="text-[10px] number-mono text-muted-foreground whitespace-nowrap">
        L{long.toFixed(0)} H{hold.toFixed(0)} S{short.toFixed(0)}
      </span>
    </div>
  );
}

interface RegimeSymbol {
  symbol: string;
  adx: number;
  chopIndex: number;
  bbw: number;
  tier: "TRENDING" | "SOFT_CHOP" | "HARD_CHOP";
}

interface GateStats {
  todayTotal: number;
  entered: number;
  blockedByV5Score: number;
  blockedByRegime: number;
  blockedByOBGate: number;
  blockedByCooldown: number;
  blockedByOther: number;
  longBias: number;
  shortBias: number;
  longEntered: number;
  shortEntered: number;
  hourlyBuckets: Array<{ hour: number; v5Score: number; regime: number; obGate: number; entered: number; other: number }>;
  regimeGrid: Array<{ symbol: string; regime: string; adx: number }>;
  trendingCount: number;
  softChopCount: number;
  hardChopCount: number;
  circuitBreakerActive: boolean;
  windowStart: number;
  windowEnd: number;
}

export default function CommandCenter() {
  const { subscribe } = useTradingWs();
  const [realtimeCycles, setRealtimeCycles] = useState<CycleLog[]>([]);
  const [chopExpanded, setChopExpanded] = useState(false);

  const { data: systemStatus, isLoading: statusLoading } = useQuery<{
    moneyConfig?: { account_equity_usd?: number };
    dailyPnl?: number;
    winRate?: number;
    profitFactor?: number;
    activePositions?: number;
    avgConfidence?: number;
    gpu?: { isAvailable?: boolean; lastActivity?: number | null };
    lastSignal?: { signalTs?: number; symbol?: string; direction?: string } | null;
  }>({
    queryKey: ["/api/system/status"],
  });

  const { data: prices } = useQuery<
    Record<string, { price: number; change24h: number; high24h: number; low24h: number }>
  >({
    queryKey: ["/api/market/prices"],
    refetchInterval: 15000,
  });

  const { data: cycleLogs, isLoading: cycleLogsLoading } = useQuery<CycleLog[]>({
    queryKey: ["/api/live/cycle-logs?limit=30"],
    refetchInterval: 30000,
  });

  const { data: positions } = useQuery<
    Array<{
      id?: number;
      symbol?: string;
      side?: string;
      entryPrice?: number;
      currentPrice?: number;
      pnlR?: number;
      pnl?: number;
      openedAt?: number;
      timestamp?: number;
    }>
  >({
    queryKey: ["/api/paper/positions?status=OPEN"],
  });

  const { data: equityCurve } = useQuery<Array<{ ts: number; r: number }>>({
    queryKey: ["/api/v5/equity-curve"],
  });

  const { data: regimeData } = useQuery<{ symbols: RegimeSymbol[]; blockedCount: number }>({
    queryKey: ["/api/market/regime"],
    refetchInterval: 60000,
  });

  const { data: gateStats } = useQuery<GateStats>({
    queryKey: ["/api/live/gate-stats"],
    refetchInterval: 15000,
  });

  const handleCycleUpdate = useCallback((payload: Record<string, unknown>) => {
    const newCycle: CycleLog = {
      id: (payload.id as number) ?? 0,
      symbol: (payload.symbol as string) ?? "BTCUSDT",
      cycleTs: (payload.cycleTs as number) ?? Date.now(),
      price: (payload.price as number) ?? null,
      pEnter: (payload.pEnter as number) ?? null,
      direction: (payload.direction as string) ?? null,
      decision: (payload.decision as string) ?? "UNKNOWN",
      reasons: (payload.reasons as string[]) ?? null,
      laneSelected: (payload.laneSelected as string) ?? null,
      htfScore: (payload.htfScore as number) ?? null,
      thresholdUsed: (payload.thresholdUsed as number) ?? null,
      laneSizeMult: (payload.laneSizeMult as number) ?? null,
      holdReason: (payload.holdReason as string) ?? null,
      v5Score: (payload.v5Score as number) ?? null,
      v5Threshold: (payload.v5Threshold as number) ?? null,
      v5Side: (payload.v5Side as string) ?? null,
      retMu: (payload.retMu as number) ?? null,
      mfePred: (payload.mfePred as number) ?? null,
      maePred: (payload.maePred as number) ?? null,
      pHold: (payload.pHold as number) ?? null,
      pLong: (payload.pLong as number) ?? null,
      pShort: (payload.pShort as number) ?? null,
      createdAt: Date.now(),
      autoTradeResult: payload.autoTradeResult as CycleLog["autoTradeResult"] ?? undefined,
    };
    setRealtimeCycles((prev) => [newCycle, ...prev].slice(0, 30));
    queryClient.invalidateQueries({ queryKey: ["/api/live/cycle-logs"] });
    if (newCycle.autoTradeResult?.opened) {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
    }
  }, []);

  useEffect(() => {
    const unsub = subscribe("CYCLE_UPDATE", handleCycleUpdate);
    return unsub;
  }, [subscribe, handleCycleUpdate]);

  const allCycles = realtimeCycles.length > 0
    ? [...realtimeCycles, ...(cycleLogs ?? [])].reduce((acc, c) => {
        if (!acc.find((x) => x.id === c.id)) acc.push(c);
        return acc;
      }, [] as CycleLog[]).sort((a, b) => b.cycleTs - a.cycleTs).slice(0, 30)
    : cycleLogs ?? [];

  const equity = systemStatus?.moneyConfig?.account_equity_usd ?? 0;
  const dailyPnl = systemStatus?.dailyPnl ?? 0;
  const winRate = systemStatus?.winRate ?? 0;
  const profitFactor = systemStatus?.profitFactor ?? 0;
  const activeCount = systemStatus?.activePositions ?? positions?.length ?? 0;
  const avgConfidence = systemStatus?.avgConfidence ?? 0;
  const gpuLive = systemStatus?.gpu?.isAvailable ?? false;
  const lastSignal = systemStatus?.lastSignal;

  const pEnterValues = allCycles.filter((c) => c.pEnter != null).map((c) => ({ value: c.pEnter! })).slice(0, 20);

  return (
    <div className="p-4 space-y-4" data-testid="command-center">
      <div
        className={`glass-card rounded-lg p-3 flex items-center gap-3 ${gpuLive ? "glow-green" : ""}`}
        data-testid="live-status-banner"
      >
        <div className={`w-3 h-3 rounded-full ${gpuLive ? "bg-emerald-400 pulse-dot" : "bg-muted-foreground"}`} />
        <span className="text-sm font-semibold">
          {gpuLive ? "V5 Neural Engine LIVE" : "V5 Neural Engine OFFLINE"}
        </span>
        {gpuLive && (
          <span className="text-xs text-muted-foreground">
            Scanning {SYMBOLS.length} assets every 15m
          </span>
        )}
        {lastSignal?.signalTs && (
          <span className="text-xs text-muted-foreground ml-auto flex items-center gap-1" data-testid="text-last-signal-time">
            <Zap className="w-3 h-3 text-amber-400" />
            Last Signal: {formatDistanceToNow(new Date(lastSignal.signalTs), { addSuffix: true })}
          </span>
        )}
        {pEnterValues.length > 0 && (
          <div className="ml-2 w-24" data-testid="heartbeat-sparkline">
            <ResponsiveContainer width="100%" height={20}>
              <AreaChart data={pEnterValues}>
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#22c55e"
                  strokeWidth={1}
                  fill="none"
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {regimeData && (() => {
        const hardChop = regimeData.symbols.filter((s) => s.tier === "HARD_CHOP");
        const softChop = regimeData.symbols.filter((s) => s.tier === "SOFT_CHOP");
        const affectedCount = hardChop.length + softChop.length;
        if (affectedCount < 3) return null;
        return (
          <div
            className="glass-card rounded-lg p-3 border border-amber-500/30 animate-fade-in-up"
            data-testid="chop-shield-strip"
          >
            <button
              className="flex items-center gap-2 w-full text-left"
              onClick={() => setChopExpanded(!chopExpanded)}
              data-testid="button-chop-toggle"
            >
              <Shield className="w-4 h-4 text-amber-400" />
              <span className="text-sm font-semibold text-amber-400">CHOP PROTECTION ACTIVE</span>
              <Badge className="no-default-hover-elevate no-default-active-elevate bg-amber-500/20 text-amber-400 text-xs" data-testid="badge-chop-count">
                {affectedCount} symbols affected
              </Badge>
              <span className="ml-auto text-muted-foreground">
                {chopExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </span>
            </button>
            {chopExpanded && (
              <div className="mt-3 grid grid-cols-2 gap-2" data-testid="chop-shield-details">
                {hardChop.length > 0 && (
                  <div>
                    <span className="text-xs text-red-400 uppercase tracking-wider font-semibold">Blocked (Hard Chop)</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {hardChop.map((s) => (
                        <Badge key={s.symbol} className="no-default-hover-elevate no-default-active-elevate text-[10px] bg-red-500/20 text-red-400" data-testid={`badge-hard-chop-${s.symbol}`}>
                          {s.symbol.replace("USDT", "")} ADX:{s.adx.toFixed(0)}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
                {softChop.length > 0 && (
                  <div>
                    <span className="text-xs text-amber-400 uppercase tracking-wider font-semibold">Throttled (Soft Chop)</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {softChop.map((s) => (
                        <Badge key={s.symbol} className="no-default-hover-elevate no-default-active-elevate text-[10px] bg-amber-500/20 text-amber-400" data-testid={`badge-soft-chop-${s.symbol}`}>
                          {s.symbol.replace("USDT", "")} ADX:{s.adx.toFixed(0)}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })()}

      {/* ── Signal Gate Scorecard ───────────────────────────────────────────── */}
      {gateStats && (
        <div className="glass-card rounded-lg p-4" data-testid="gate-scorecard">
          <div className="flex items-center gap-2 mb-3">
            <Shield className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold uppercase tracking-wider">Today's Signal Funnel</span>
            {gateStats.circuitBreakerActive && (
              <Badge className="no-default-hover-elevate no-default-active-elevate bg-red-500/20 text-red-400 text-xs animate-pulse" data-testid="badge-circuit-breaker">
                CIRCUIT BREAKER ACTIVE
              </Badge>
            )}
            <span className="ml-auto text-xs text-muted-foreground number-mono" data-testid="gate-total-count">
              {gateStats.todayTotal} signals today
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Funnel */}
            <div className="space-y-1.5">
              {[
                { label: "Signals received", count: gateStats.todayTotal, color: "text-foreground", bar: "bg-white/10", pct: 100 },
                { label: "Blocked by V5 score", count: gateStats.blockedByV5Score, color: "text-muted-foreground", bar: "bg-slate-500/40", pct: gateStats.todayTotal > 0 ? Math.round(gateStats.blockedByV5Score / gateStats.todayTotal * 100) : 0 },
                { label: "Blocked by regime gate", count: gateStats.blockedByRegime, color: "text-amber-400", bar: "bg-amber-500/40", pct: gateStats.todayTotal > 0 ? Math.round(gateStats.blockedByRegime / gateStats.todayTotal * 100) : 0 },
                { label: "Blocked by OB/flow gate", count: gateStats.blockedByOBGate, color: "text-orange-400", bar: "bg-orange-500/40", pct: gateStats.todayTotal > 0 ? Math.round(gateStats.blockedByOBGate / gateStats.todayTotal * 100) : 0 },
                { label: "Cooldown / other", count: gateStats.blockedByCooldown + gateStats.blockedByOther, color: "text-muted-foreground/70", bar: "bg-muted/30", pct: gateStats.todayTotal > 0 ? Math.round((gateStats.blockedByCooldown + gateStats.blockedByOther) / gateStats.todayTotal * 100) : 0 },
                { label: "ENTERED", count: gateStats.entered, color: "text-emerald-400", bar: "bg-emerald-500/50", pct: gateStats.todayTotal > 0 ? Math.round(gateStats.entered / gateStats.todayTotal * 100) : 0 },
              ].map(({ label, count, color, bar, pct }) => (
                <div key={label} className="flex items-center gap-2" data-testid={`funnel-row-${label.replace(/\s+/g, '-').toLowerCase()}`}>
                  <span className={`text-xs w-48 shrink-0 ${color}`}>{label}</span>
                  <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${bar} transition-all`} style={{ width: `${pct}%` }} />
                  </div>
                  <span className={`text-xs number-mono w-8 text-right ${color}`}>{count}</span>
                  <span className="text-[10px] text-muted-foreground/60 w-8 text-right">{pct}%</span>
                </div>
              ))}
            </div>

            {/* L/S bias */}
            <div className="space-y-3">
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1.5">LONG / SHORT Balance (Today's Entries)</p>
                <div className="flex h-5 rounded-full overflow-hidden" data-testid="ls-bias-bar">
                  <div
                    className="bg-emerald-500/70 flex items-center justify-center text-[10px] font-bold text-emerald-100 transition-all"
                    style={{ width: `${gateStats.longBias}%` }}
                  >
                    {gateStats.longBias > 15 && `${gateStats.longBias}%`}
                  </div>
                  <div
                    className="bg-red-500/70 flex items-center justify-center text-[10px] font-bold text-red-100 transition-all"
                    style={{ width: `${gateStats.shortBias}%` }}
                  >
                    {gateStats.shortBias > 15 && `${gateStats.shortBias}%`}
                  </div>
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-emerald-400">LONG {gateStats.longEntered}</span>
                  <span className="text-[10px] text-red-400">SHORT {gateStats.shortEntered}</span>
                </div>
                {gateStats.longBias >= 80 && (
                  <p className="text-[10px] text-amber-400/80 mt-1" data-testid="text-bias-warning">
                    Bullish tilt — {gateStats.trendingCount}/{gateStats.regimeGrid.length} symbols trending. Regime gate mostly blocking shorts.
                  </p>
                )}
                {gateStats.shortBias >= 80 && (
                  <p className="text-[10px] text-amber-400/80 mt-1" data-testid="text-bias-warning">
                    Bearish tilt — {gateStats.hardChopCount + gateStats.softChopCount}/{gateStats.regimeGrid.length} symbols in chop/bear. Regime gate mostly blocking longs.
                  </p>
                )}
              </div>

              {/* Mini regime grid */}
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1.5">Regime Grid (20 Symbols)</p>
                <div className="grid grid-cols-5 gap-1" data-testid="regime-grid">
                  {gateStats.regimeGrid.map((s) => {
                    const sym = s.symbol.replace("USDT", "");
                    const color = s.regime === "TRENDING" ? "bg-emerald-500/25 text-emerald-400 border-emerald-500/30"
                      : s.regime === "SOFT_CHOP" ? "bg-amber-500/25 text-amber-400 border-amber-500/30"
                      : "bg-red-500/25 text-red-400 border-red-500/30";
                    return (
                      <div key={s.symbol} className={`text-[9px] font-mono border rounded px-1 py-0.5 text-center ${color}`} data-testid={`regime-cell-${s.symbol}`} title={`${s.symbol}: ${s.regime} (ADX: ${s.adx.toFixed(0)})`}>
                        {sym}
                      </div>
                    );
                  })}
                </div>
                <div className="flex gap-2 mt-1.5">
                  <span className="text-[9px] text-emerald-400">● TRENDING</span>
                  <span className="text-[9px] text-amber-400">● SOFT CHOP</span>
                  <span className="text-[9px] text-red-400">● HARD CHOP</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-6 gap-3" data-testid="metrics-bar">
        {statusLoading ? (
          Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="glass-card rounded-lg p-4 h-24 shimmer" />
          ))
        ) : (
          <>
            <MetricCard
              label="Account Equity"
              value={`$${equity.toLocaleString()}`}
              glow="cyan"
              icon={Wallet}
              testId="metric-equity"
            />
            <MetricCard
              label="Daily P&L"
              value={`${dailyPnl >= 0 ? "+" : ""}$${dailyPnl.toLocaleString()}`}
              glow={dailyPnl >= 0 ? "green" : "red"}
              icon={dailyPnl >= 0 ? TrendingUp : TrendingDown}
              testId="metric-daily-pnl"
            />
            <MetricCard
              label="Win Rate"
              value={`${(winRate * 100).toFixed(1)}%`}
              glow={winRate >= 0.5 ? "green" : "red"}
              icon={Target}
              testId="metric-win-rate"
            />
            <MetricCard
              label="Profit Factor"
              value={profitFactor.toFixed(2)}
              glow={profitFactor >= 1 ? "green" : "red"}
              icon={BarChart3}
              testId="metric-profit-factor"
            />
            <MetricCard
              label="Active Positions"
              value={String(activeCount)}
              glow={activeCount > 0 ? "cyan" : undefined}
              icon={Crosshair}
              testId="metric-active-positions"
            />
            <MetricCard
              label="Model Confidence"
              value={`${(avgConfidence * 100).toFixed(0)}%`}
              glow={avgConfidence >= 0.7 ? "green" : avgConfidence >= 0.4 ? "cyan" : "red"}
              icon={Brain}
              testId="metric-confidence"
            />
          </>
        )}
      </div>

      <div className="grid grid-cols-4 gap-3" data-testid="market-grid">
        {SYMBOLS.map((sym) => (
          <MarketCard
            key={sym}
            symbol={sym}
            priceData={prices?.[sym]}
          />
        ))}
      </div>

      <div className="glass-card rounded-lg p-4" data-testid="signal-feed">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
          <span className="text-sm font-semibold uppercase tracking-wider">Live Signal Feed</span>
          <Radio className="w-4 h-4 text-emerald-400 ml-auto" />
        </div>

        {cycleLogsLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-10 shimmer rounded" />
            ))}
          </div>
        ) : allCycles.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <div className="shimmer rounded px-6 py-3 text-sm">Awaiting cycle data...</div>
          </div>
        ) : (
          <div className="space-y-1">
            <div className="grid grid-cols-8 gap-2 px-2 py-1 text-xs text-muted-foreground uppercase tracking-wider">
              <span>Time</span>
              <span>Symbol</span>
              <span>Direction</span>
              <span>Decision</span>
              <span>p_enter</span>
              <span>V5 Score</span>
              <span>Action Probs</span>
              <span>ret_mu / MFE / MAE</span>
            </div>
            {allCycles.map((cycle, idx) => {
              const dir = cycle.direction ?? "HOLD";
              const dirColor =
                dir === "LONG" ? "bg-emerald-500/20 text-emerald-400" :
                dir === "SHORT" ? "bg-red-500/20 text-red-400" :
                "bg-amber-500/20 text-amber-400";
              const decisionColor =
                cycle.decision === "ENTER" ? "bg-emerald-500/20 text-emerald-400" :
                cycle.decision === "HOLD" ? "bg-amber-500/20 text-amber-400" :
                "bg-muted text-muted-foreground";
              const scoreVal = cycle.v5Score;
              const scoreColor =
                scoreVal != null && scoreVal >= (cycle.v5Threshold ?? 0.02) ? "bg-emerald-500/20 text-emerald-400" :
                "bg-muted text-muted-foreground";

              return (
                <div
                  key={cycle.id ?? idx}
                  className="grid grid-cols-8 gap-2 px-2 py-2 rounded hover-elevate animate-signal-arrive items-center"
                  data-testid={`cycle-row-${idx}`}
                >
                  <span className="text-xs text-muted-foreground number-mono">
                    {cycle.cycleTs
                      ? formatDistanceToNow(new Date(cycle.cycleTs), { addSuffix: true })
                      : "—"}
                  </span>
                  <Badge className="no-default-hover-elevate no-default-active-elevate text-xs w-fit bg-muted text-foreground" data-testid={`cycle-symbol-${idx}`}>
                    {cycle.symbol?.replace("USDT", "") ?? "?"}
                  </Badge>
                  <Badge className={`no-default-hover-elevate no-default-active-elevate text-xs w-fit ${dirColor}`} data-testid={`cycle-dir-${idx}`}>
                    {dir}
                  </Badge>
                  <div className="flex items-center gap-1">
                    <Badge className={`no-default-hover-elevate no-default-active-elevate text-xs w-fit ${decisionColor}`} data-testid={`cycle-decision-${idx}`}>
                      {cycle.decision}
                    </Badge>
                    {cycle.autoTradeResult?.opened && (
                      <Badge className="no-default-hover-elevate no-default-active-elevate text-[10px] w-fit bg-cyan-500/20 text-cyan-400 animate-signal-arrive" data-testid={`cycle-executed-${idx}`}>
                        EXECUTED
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Progress value={(cycle.pEnter ?? 0) * 100} className="h-2 flex-1" />
                    <span className="number-mono text-xs">{cycle.pEnter != null ? (cycle.pEnter * 100).toFixed(0) + "%" : "—"}</span>
                  </div>
                  <Badge className={`no-default-hover-elevate no-default-active-elevate text-xs w-fit ${scoreColor}`} data-testid={`cycle-v5score-${idx}`}>
                    {scoreVal != null ? scoreVal.toFixed(4) : "—"}
                  </Badge>
                  <ActionProbBar pHold={cycle.pHold} pLong={cycle.pLong} pShort={cycle.pShort} />
                  <div className="text-[10px] number-mono text-muted-foreground space-x-1">
                    {cycle.retMu != null && <span className="text-foreground">{cycle.retMu.toFixed(4)}</span>}
                    {cycle.mfePred != null && <span className="text-emerald-400">{cycle.mfePred.toFixed(3)}</span>}
                    {cycle.maePred != null && <span className="text-red-400">{cycle.maePred.toFixed(3)}</span>}
                    {cycle.retMu == null && cycle.mfePred == null && "—"}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="grid grid-cols-5 gap-3" data-testid="bottom-section">
        <div className="col-span-3 glass-card rounded-lg p-4" data-testid="active-positions">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold uppercase tracking-wider">Active Positions</span>
          </div>

          {!positions || positions.length === 0 ? (
            <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
              No open positions
            </div>
          ) : (
            <div className="space-y-1">
              <div className="grid grid-cols-7 gap-2 px-2 py-1 text-xs text-muted-foreground uppercase tracking-wider">
                <span>Symbol</span>
                <span>Side</span>
                <span>Entry</span>
                <span>Current</span>
                <span>P&L (R)</span>
                <span>Duration</span>
                <span className="text-right">Action</span>
              </div>
              {positions.map((pos, idx) => {
                const pnl = pos.pnlR ?? pos.pnl ?? 0;
                const isProfit = pnl >= 0;
                const rowBg = isProfit ? "bg-emerald-500/5" : "bg-red-500/5";

                return (
                  <div
                    key={pos.id ?? idx}
                    className={`grid grid-cols-7 gap-2 px-2 py-2 rounded ${rowBg} items-center`}
                    data-testid={`position-row-${idx}`}
                  >
                    <span className="text-sm font-medium flex items-center gap-1">
                      {pos.symbol ?? "—"}
                      {pos.source === "v5_signal" && (
                        <Badge className="no-default-hover-elevate no-default-active-elevate text-[9px] px-1 py-0 bg-cyan-500/20 text-cyan-400" data-testid={`badge-v5-${idx}`}>V5</Badge>
                      )}
                    </span>
                    <Badge
                      className={`no-default-hover-elevate no-default-active-elevate text-xs w-fit ${
                        pos.side === "LONG" ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {pos.side ?? "—"}
                    </Badge>
                    <span className="number-mono text-sm">
                      ${pos.entryPrice?.toLocaleString() ?? "—"}
                    </span>
                    <span className="number-mono text-sm">
                      ${pos.currentPrice?.toLocaleString() ?? "—"}
                    </span>
                    <span
                      className={`number-mono text-sm font-medium ${isProfit ? "text-emerald-400" : "text-red-400"}`}
                    >
                      {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}R
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {pos.openedAt || pos.timestamp
                        ? formatDistanceToNow(new Date((pos.openedAt ?? pos.timestamp ?? 0) * 1000), { addSuffix: false })
                        : "—"}
                    </span>
                    <div className="flex justify-end">
                      {pos.id && (
                        <CloseButton positionId={pos.id as number} symbol={pos.symbol ?? "?"} side={pos.side ?? "?"} />
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="col-span-2 glass-card rounded-lg p-4" data-testid="equity-curve">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            <span className="text-sm font-semibold uppercase tracking-wider">Equity Curve</span>
          </div>

          {!equityCurve || equityCurve.length === 0 ? (
            <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
              No equity data
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={equityCurve}>
                <defs>
                  <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22c55e" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area
                  type="monotone"
                  dataKey="r"
                  stroke="#22c55e"
                  strokeWidth={2}
                  fill="url(#eqGrad)"
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}