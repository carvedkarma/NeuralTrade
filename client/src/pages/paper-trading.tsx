import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useTradingWs } from "@/hooks/use-trading-ws";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Scatter,
  ComposedChart,
} from "recharts";
import {
  RotateCcw,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  AlertTriangle,
  Brain,
  Shield,
  ShieldAlert,
  ShieldCheck,
  ArrowUpDown,
  Trash2,
} from "lucide-react";
import { CloseButton, PartialCloseButton, EditSLTPDialog } from "@/components/position-actions";
import { usePingMonitor } from "@/hooks/use-ping";
import { PingBadge } from "@/components/ping-badge";

interface Portfolio {
  startingEquity: number;
  currentEquity: number;
  equity: number;
  totalPnlR: number;
  totalPnlUsdt: number;
  totalPnl: number;
  maxDrawdownR: number;
  maxDrawdown: number;
  unrealizedPnl: number;
  unrealizedPnlR: number;
  realizedPnl: number;
  tradesCount: number;
  totalTrades: number;
  winRate: number;
  openPositions: number;
}

interface PaperConfig {
  paperTradingEnabled: boolean;
  isAutoTrading?: boolean;
  leverageEnabled?: boolean;
  leverageTiers?: { minScore: number; leverage: number }[];
  maxLeverage?: number;
}

interface Position {
  id?: string | number;
  symbol: string;
  side: string;
  entryPrice: number;
  currentPrice?: number;
  pnlR?: number;
  pnlUsdt?: number;
  duration?: string;
  stopLoss?: number;
  takeProfit?: number;
  entryTime?: number;
  exitTime?: number;
  exitPrice?: number;
  exitType?: string;
  status?: string;
  source?: string;
  leverage?: number;
  qty?: number;
  initialRiskUsdt?: number;
  v5Score?: number;
  peakProfit?: number;
  trailPrice?: number;
  trailMode?: string;
}

interface EquityPoint {
  ts: number;
  r: number;
  tradeR: number;
  symbol: string;
  side: string;
}

interface PositionHealth {
  score: number;
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  reason: string;
  factors: {
    pnlScore: number;
    slTpRatioScore: number;
    modelConfidenceScore: number;
    timeScore: number;
    mfeTrendScore: number;
  };
  currentPnlR: number;
  peakPnlR: number;
  giveback: number;
  latestV5Score: number | null;
  latestAdjustment: string | null;
}

function formatUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(value);
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

function formatDateTime(ts: number): string {
  return new Date(ts).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(ms: number): string {
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  if (hours > 24) {
    const days = Math.floor(hours / 24);
    return `${days}d ${hours % 24}h`;
  }
  return `${hours}h ${minutes}m`;
}

function formatPrice(price: number): string {
  if (price >= 1000) return price.toFixed(2);
  if (price >= 1) return price.toFixed(4);
  return price.toFixed(6);
}

const NEURAL_ADJUSTMENT_LABELS: Record<string, { label: string; color: string }> = {
  BREAKEVEN: { label: "BE Set", color: "text-amber-400" },
  TRAIL_TIGHTEN: { label: "Trail Tight", color: "text-cyan-400" },
  TRAIL_WIDEN: { label: "Trail Wide", color: "text-blue-400" },
  DIRECTION_FLIP_EXIT: { label: "Flip Exit", color: "text-red-400" },
  CONFIDENCE_DECAY_EXIT: { label: "Decay Exit", color: "text-orange-400" },
  CONFIDENCE_DECAY_TIGHTEN: { label: "Decay Tight", color: "text-orange-400" },
  MFE_PROTECTION_EXIT: { label: "MFE Lock", color: "text-emerald-400" },
  ADAPTIVE_TRAIL: { label: "Adapt Trail", color: "text-purple-400" },
};

function HealthGauge({ score, riskLevel }: { score: number; riskLevel: string }) {
  const color = score >= 70 ? "text-emerald-400" : score >= 45 ? "text-amber-400" : score >= 25 ? "text-orange-400" : "text-red-400";
  const bgColor = score >= 70 ? "bg-emerald-400" : score >= 45 ? "bg-amber-400" : score >= 25 ? "bg-orange-400" : "bg-red-400";
  const bgTrack = "bg-muted/40";

  return (
    <div className="flex items-center gap-2" data-testid="health-gauge">
      <div className={`relative w-16 h-1.5 rounded-full ${bgTrack}`}>
        <div
          className={`absolute top-0 left-0 h-full rounded-full transition-all duration-700 ${bgColor}`}
          style={{ width: `${Math.max(2, Math.min(100, score))}%` }}
        />
      </div>
      <span className={`text-[10px] font-bold number-mono ${color}`} data-testid="text-health-score">
        {score}
      </span>
      {riskLevel === "CRITICAL" && (
        <ShieldAlert className="w-3 h-3 text-red-400" />
      )}
      {riskLevel === "HIGH" && (
        <ShieldAlert className="w-3 h-3 text-orange-400" />
      )}
      {riskLevel === "LOW" && (
        <ShieldCheck className="w-3 h-3 text-emerald-400/60" />
      )}
    </div>
  );
}

function MfeTracker({ currentPnlR, peakPnlR, giveback }: { currentPnlR: number; peakPnlR: number; giveback: number }) {
  if (peakPnlR <= 0) return null;

  const rawGivebackPct = Math.round(giveback * 100);
  const givebackPct = Math.min(rawGivebackPct, 100);
  const givebackColor = givebackPct < 15 ? "text-emerald-400/70" : givebackPct < 40 ? "text-amber-400/70" : "text-red-400/70";

  return (
    <div className="flex items-center gap-1.5 text-[10px]" data-testid="mfe-tracker">
      <TrendingUp className="w-3 h-3 text-cyan-400/60" />
      <span className="text-muted-foreground">Peak</span>
      <span className="number-mono text-cyan-400">{peakPnlR.toFixed(2)}R</span>
      {givebackPct > 0 && (
        <>
          <span className="text-muted-foreground/50">|</span>
          <span className={`number-mono ${givebackColor}`} data-testid="text-giveback">
            -{givebackPct}% giveback
          </span>
        </>
      )}
    </div>
  );
}

function PositionPriceGauge({ pos, livePrice, health }: { pos: Position; livePrice?: number; health?: PositionHealth }) {
  const { entryPrice, stopLoss, takeProfit, side } = pos;
  const currentPrice = livePrice ?? pos.currentPrice;
  if (!currentPrice || !stopLoss || !takeProfit) return null;

  const isLong = side === "LONG";
  const lo = isLong ? stopLoss : takeProfit;
  const hi = isLong ? takeProfit : stopLoss;
  const range = hi - lo;
  if (range <= 0) return null;

  const entryPct = ((entryPrice - lo) / range) * 100;
  const pricePct = ((currentPrice - lo) / range) * 100;
  const clampedPricePct = Math.max(0, Math.min(100, pricePct));

  const priceDiff = isLong ? currentPrice - entryPrice : entryPrice - currentPrice;
  const livePnlUsdt = pos.qty ? priceDiff * pos.qty : (pos.pnlUsdt ?? 0);
  const livePnlR = pos.initialRiskUsdt ? livePnlUsdt / pos.initialRiskUsdt : (pos.pnlR ?? 0);
  const pnl = Math.round(livePnlR * 100) / 100;
  const pnlUsd = Math.round(livePnlUsdt * 100) / 100;
  const isProfit = pnl >= 0;

  const dur = pos.entryTime ? Date.now() - pos.entryTime : 0;

  const slDenom = isLong ? (entryPrice - stopLoss) : (stopLoss - entryPrice);
  const tpDenom = isLong ? (takeProfit - entryPrice) : (entryPrice - takeProfit);
  const slDist = slDenom > 0
    ? Math.max(0, Math.min(999, (isLong ? (currentPrice - stopLoss) : (stopLoss - currentPrice)) / slDenom * 100))
    : 0;
  const tpDist = tpDenom > 0
    ? Math.max(0, Math.min(999, (isLong ? (takeProfit - currentPrice) : (currentPrice - takeProfit)) / tpDenom * 100))
    : 0;

  const posId = typeof pos.id === "number" ? pos.id : parseInt(String(pos.id ?? "0"));

  const healthScore = health?.score ?? null;
  const isBreakeven = stopLoss != null && Math.abs(stopLoss - entryPrice) / entryPrice < 0.001;

  const trailPrice = pos.trailPrice;
  let trailPct: number | null = null;
  if (trailPrice && range > 0) {
    trailPct = Math.max(0, Math.min(100, ((trailPrice - lo) / range) * 100));
  }

  const pulseClass =
    healthScore !== null && healthScore < 15
      ? "animate-pulse border-red-500/60"
      : healthScore !== null && healthScore < 30
      ? "animate-pulse border-amber-500/50"
      : "border-border/50";

  const latestAdj = health?.latestAdjustment;
  const adjInfo = latestAdj ? NEURAL_ADJUSTMENT_LABELS[latestAdj] : null;

  return (
    <div className={`glass-card rounded-lg border p-3 space-y-3 ${pulseClass}`} data-testid={`position-gauge-${pos.symbol}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-semibold text-sm">{pos.symbol}</span>
          <Badge
            variant="outline"
            className={isLong ? "text-emerald-400 border-emerald-400/30 text-[10px] px-1.5" : "text-red-400 border-red-400/30 text-[10px] px-1.5"}
          >
            {side}
          </Badge>
          {pos.source === "v5_signal" && (
            <Badge className="no-default-hover-elevate no-default-active-elevate text-[10px] bg-cyan-500/20 text-cyan-400 px-1.5">V5</Badge>
          )}
          {pos.leverage != null && pos.leverage > 1 && (
            <Badge variant="outline" className="text-amber-400 border-amber-400/30 text-[10px] px-1.5" data-testid="badge-leverage">
              {pos.leverage}x
            </Badge>
          )}
          {isBreakeven && (
            <Badge variant="outline" className="text-amber-400 border-amber-400/30 text-[10px] px-1.5" data-testid="badge-breakeven">
              <Shield className="w-2.5 h-2.5 mr-0.5" />BE
            </Badge>
          )}
          {adjInfo && (
            <Badge variant="outline" className={`${adjInfo.color} border-current/30 text-[10px] px-1.5`} data-testid="badge-neural-status">
              <Brain className="w-2.5 h-2.5 mr-0.5" />{adjInfo.label}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-sm font-bold number-mono ${isProfit ? "text-emerald-400" : "text-red-400"}`} data-testid={`text-pnlr-${pos.symbol}`}>
            {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}R
          </span>
          <span className={`text-xs number-mono ${isProfit ? "text-emerald-400/70" : "text-red-400/70"}`} data-testid={`text-pnlusdt-${pos.symbol}`}>
            {pnlUsd >= 0 ? "+" : ""}${pnlUsd.toFixed(2)}
          </span>
        </div>
      </div>

      {health && (
        <div className="flex items-center justify-between">
          <HealthGauge score={health.score} riskLevel={health.riskLevel} />
          <MfeTracker currentPnlR={health.currentPnlR} peakPnlR={health.peakPnlR} giveback={health.giveback} />
        </div>
      )}

      <div className="space-y-1.5">
        <div className="relative h-8 rounded-md overflow-hidden bg-muted/30">
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-amber-400/80 z-10"
            style={{ left: `${Math.max(1, Math.min(99, entryPct))}%` }}
          >
            <div className="absolute -top-0.5 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[3px] border-r-[3px] border-t-[4px] border-l-transparent border-r-transparent border-t-amber-400" />
          </div>

          <div
            className={`absolute top-0 bottom-0 rounded-sm transition-all duration-500 ${
              isProfit ? "bg-emerald-500/20" : "bg-red-500/20"
            }`}
            style={{
              left: `${Math.min(clampedPricePct, Math.max(0, Math.min(100, entryPct)))}%`,
              width: `${Math.abs(clampedPricePct - Math.max(0, Math.min(100, entryPct)))}%`,
            }}
          />

          <div
            className={`absolute top-0 bottom-0 w-[3px] z-20 rounded-full transition-all duration-500 ${
              isProfit ? "bg-emerald-400 shadow-[0_0_8px_rgba(34,197,94,0.5)]" : "bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.5)]"
            }`}
            style={{ left: `${clampedPricePct}%` }}
          />

          {trailPct !== null && (
            <div
              className="absolute top-0 bottom-0 w-[2px] z-15 rounded-full bg-purple-400/70 transition-all duration-500"
              style={{ left: `${trailPct}%` }}
              data-testid="trail-level-indicator"
            >
              <div className="absolute -bottom-0.5 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[2px] border-r-[2px] border-b-[3px] border-l-transparent border-r-transparent border-b-purple-400" />
            </div>
          )}

          {isBreakeven && (
            <div
              className="absolute top-0 bottom-0 w-[2px] z-12 bg-amber-400/40"
              style={{ left: `${Math.max(1, Math.min(99, entryPct))}%` }}
            >
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-amber-400/60" />
            </div>
          )}

          <div
            className="absolute inset-y-0 left-0 flex items-center pl-1.5"
          >
            <span className="text-[9px] font-semibold text-red-400/80 number-mono">
              {isLong ? "SL" : "TP"}
            </span>
          </div>
          <div
            className="absolute inset-y-0 right-0 flex items-center pr-1.5"
          >
            <span className="text-[9px] font-semibold text-emerald-400/80 number-mono">
              {isLong ? "TP" : "SL"}
            </span>
          </div>
        </div>

        <div className="flex justify-between items-center text-[10px] number-mono text-muted-foreground">
          <span className="text-red-400/70">${formatPrice(isLong ? stopLoss : takeProfit)}</span>
          <div className="flex items-center gap-1 flex-wrap">
            <span className="text-amber-400/70">Entry ${formatPrice(entryPrice)}</span>
            <span className="text-foreground/50">→</span>
            <span className={isProfit ? "text-emerald-400" : "text-red-400"}>Now ${formatPrice(currentPrice)}</span>
            {trailPrice && (
              <>
                <span className="text-foreground/50">|</span>
                <span className="text-purple-400/70">Trail ${formatPrice(trailPrice)}</span>
              </>
            )}
          </div>
          <span className="text-emerald-400/70">${formatPrice(isLong ? takeProfit : stopLoss)}</span>
        </div>
      </div>

      <div className="flex items-center justify-between text-[10px]">
        <div className="flex gap-3 flex-wrap">
          <div>
            <span className="text-muted-foreground">SL Dist: </span>
            <span className={`number-mono ${slDist < 30 ? "text-red-400 font-semibold" : "text-muted-foreground"}`}>
              {slDist.toFixed(0)}%
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">TP Dist: </span>
            <span className={`number-mono ${tpDist < 30 ? "text-emerald-400 font-semibold" : "text-muted-foreground"}`}>
              {tpDist.toFixed(0)}%
            </span>
          </div>
          <div className="text-muted-foreground" data-testid={`text-duration-${pos.symbol}`}>
            {pos.entryTime ? (
              <span title={new Date(pos.entryTime).toLocaleString()}>
                {formatDateTime(pos.entryTime)} · {dur > 0 ? formatDuration(dur) : "-"}
              </span>
            ) : "-"}
          </div>
          {pos.leverage != null && (
            <div className="text-amber-400/70" data-testid={`text-leverage-${pos.symbol}`}>
              {pos.leverage}x
              {pos.v5Score != null && (
                <span className="ml-1 text-cyan-400/70" data-testid={`text-v5score-${pos.symbol}`}>
                  (V5: {pos.v5Score.toFixed(3)})
                </span>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-0.5">
          {posId > 0 && (
            <>
              <PartialCloseButton positionId={posId} symbol={pos.symbol} />
              <EditSLTPDialog
                positionId={posId}
                symbol={pos.symbol}
                side={pos.side}
                currentSL={pos.stopLoss ?? null}
                currentTP={pos.takeProfit ?? null}
                entryPrice={pos.entryPrice}
              />
              <CloseButton positionId={posId} symbol={pos.symbol} side={pos.side} livePrice={livePrice ?? pos.currentPrice} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function PaperTrading() {
  const [equityRange, setEquityRange] = useState<"7d" | "30d" | "all">("30d");
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [healthMap, setHealthMap] = useState<Record<number, PositionHealth>>({});
  const { subscribe } = useTradingWs();
  const ping = usePingMonitor();

  useEffect(() => {
    const unsub = subscribe("PRICE_TICK", (payload) => {
      setLivePrices(payload as Record<string, number>);
    });
    return unsub;
  }, [subscribe]);

  const { data: portfolio, isLoading: portfolioLoading } = useQuery<Portfolio>({
    queryKey: ["/api/paper/portfolio"],
    refetchInterval: 10000,
  });

  const { data: config } = useQuery<PaperConfig>({
    queryKey: ["/api/paper/config"],
  });

  const { data: status } = useQuery<{
    paperTradingEnabled: boolean;
    isAutoTrading: boolean;
  }>({
    queryKey: ["/api/paper/status"],
    refetchInterval: 5000,
  });

  const { data: openPositions } = useQuery<Position[]>({
    queryKey: ["/api/paper/positions", "?status=OPEN"],
    refetchInterval: 5000,
  });

  const { data: equityCurve } = useQuery<EquityPoint[]>({
    queryKey: ["/api/paper/equity-curve", `?range=${equityRange}`],
    refetchInterval: 30000,
  });

  useEffect(() => {
    if (!openPositions || openPositions.length === 0) {
      setHealthMap({});
      return;
    }
    const fetchHealth = async () => {
      const results: Record<number, PositionHealth> = {};
      await Promise.all(
        openPositions.map(async (pos) => {
          const posId = typeof pos.id === "number" ? pos.id : parseInt(String(pos.id ?? "0"));
          if (posId <= 0) return;
          try {
            const resp = await fetch(`/api/paper/positions/${posId}/health`);
            if (resp.ok) {
              results[posId] = await resp.json();
            }
          } catch {}
        })
      );
      setHealthMap(results);
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, [openPositions]);

  const enableMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/paper/enable"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/config"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/status"] });
    },
  });

  const disableMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/paper/disable"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/config"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/status"] });
    },
  });

  const startAutoMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/paper/start"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/status"] });
    },
  });

  const stopAutoMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/paper/stop"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/status"] });
    },
  });

  const resetMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/paper/reset"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/status"] });
    },
  });

  const clearHistoryMutation = useMutation({
    mutationFn: () => apiRequest("DELETE", "/api/paper/trade-history"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/trade-history"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/equity-curve"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/performance"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
    },
  });

  const clearAnalyticsMutation = useMutation({
    mutationFn: () => apiRequest("DELETE", "/api/paper/equity-curve"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/equity-curve"] });
    },
  });

  const paperEnabled = status?.paperTradingEnabled ?? config?.paperTradingEnabled ?? false;
  const autoTrading = status?.isAutoTrading ?? false;

  const handlePaperToggle = (checked: boolean) => {
    if (checked) {
      enableMutation.mutate();
    } else {
      disableMutation.mutate();
    }
  };

  const handleAutoToggle = (checked: boolean) => {
    if (checked) {
      startAutoMutation.mutate();
    } else {
      stopAutoMutation.mutate();
    }
  };

  const handleReset = () => {
    if (window.confirm("Are you sure you want to reset the paper trading portfolio? This will clear all positions and trade history.")) {
      resetMutation.mutate();
    }
  };

  const handleClearHistory = () => {
    if (window.confirm("Clear all trade history records? This cannot be undone.")) {
      clearHistoryMutation.mutate();
    }
  };

  const handleClearAnalytics = () => {
    if (window.confirm("Clear the equity curve data? This cannot be undone.")) {
      clearAnalyticsMutation.mutate();
    }
  };

  const pctChange = portfolio && portfolio.startingEquity > 0
    ? ((portfolio.currentEquity - portfolio.startingEquity) / portfolio.startingEquity) * 100
    : 0;

  const pnlPositive = (portfolio?.totalPnlR ?? 0) >= 0;

  return (
    <div className="p-4 space-y-4" data-testid="paper-trading">
      <div className="glass-card rounded-md p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Paper Trading</span>
            <Switch
              data-testid="switch-paper-trading"
              checked={paperEnabled}
              onCheckedChange={handlePaperToggle}
              disabled={enableMutation.isPending || disableMutation.isPending}
            />
            {paperEnabled && (
              <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Auto Trading</span>
            <Switch
              data-testid="switch-auto-trading"
              checked={autoTrading}
              onCheckedChange={handleAutoToggle}
              disabled={!paperEnabled || startAutoMutation.isPending || stopAutoMutation.isPending}
            />
            {autoTrading && (
              <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
            )}
          </div>

          <div className="ml-auto flex items-center gap-2">
            <PingBadge ping={ping} />
            <Button
              variant="outline"
              size="sm"
              data-testid="button-clear-history"
              onClick={handleClearHistory}
              disabled={clearHistoryMutation.isPending}
            >
              <Trash2 className="w-3.5 h-3.5 mr-1" />
              Clear History
            </Button>
            <Button
              variant="outline"
              size="sm"
              data-testid="button-clear-analytics"
              onClick={handleClearAnalytics}
              disabled={clearAnalyticsMutation.isPending}
            >
              <Trash2 className="w-3.5 h-3.5 mr-1" />
              Clear Analytics
            </Button>
            <Button
              variant="destructive"
              data-testid="button-reset-portfolio"
              onClick={handleReset}
              disabled={resetMutation.isPending}
            >
              <RotateCcw className="w-4 h-4 mr-1" />
              Reset Portfolio
            </Button>
          </div>
        </div>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <CardTitle className="text-sm font-medium">Open Positions</CardTitle>
              <Brain className="w-4 h-4 text-purple-400/60" />
            </div>
            {openPositions && openPositions.length > 0 && (
              <Badge variant="outline" className="text-[10px] text-cyan-400 border-cyan-400/30" data-testid="badge-open-count">
                {openPositions.length} active
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {(!openPositions || openPositions.length === 0) ? (
            <p className="text-sm text-muted-foreground text-center py-8" data-testid="text-no-open-positions">
              No open positions
            </p>
          ) : (
            <div className="grid gap-3 sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3" data-testid="positions-grid">
              {openPositions.map((pos, i) => {
                const posId = typeof pos.id === "number" ? pos.id : parseInt(String(pos.id ?? "0"));
                return (
                  <PositionPriceGauge
                    key={pos.id ?? i}
                    pos={pos}
                    livePrice={livePrices[pos.symbol]}
                    health={posId > 0 ? healthMap[posId] : undefined}
                  />
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Starting Equity</CardTitle>
            <DollarSign className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="number-mono text-2xl font-bold" data-testid="text-starting-equity">
              {portfolioLoading ? "..." : formatUsd(portfolio?.startingEquity ?? 0)}
            </div>
          </CardContent>
        </Card>

        <Card className={pctChange >= 0 ? "glow-green" : "glow-red"}>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Current Equity</CardTitle>
            {pctChange >= 0 ? (
              <TrendingUp className="w-4 h-4 text-emerald-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
          </CardHeader>
          <CardContent>
            <div className="number-mono text-2xl font-bold" data-testid="text-current-equity">
              {portfolioLoading ? "..." : formatUsd(portfolio?.currentEquity ?? 0)}
            </div>
            <p className={`text-xs mt-1 ${pctChange >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {pctChange >= 0 ? "+" : ""}{pctChange.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card className={pnlPositive ? "glow-green" : "glow-red"}>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total P&L</CardTitle>
            <Activity className="w-4 h-4 text-cyan-500" />
          </CardHeader>
          <CardContent>
            <div
              className={`number-mono text-2xl font-bold ${pnlPositive ? "text-emerald-400" : "text-red-400"}`}
              data-testid="text-total-pnl"
            >
              {portfolioLoading ? "..." : `${(portfolio?.totalPnlR ?? 0) >= 0 ? "+" : ""}${(portfolio?.totalPnlR ?? 0).toFixed(2)}R`}
            </div>
            <p className={`text-xs mt-1 ${pnlPositive ? "text-emerald-400" : "text-red-400"}`}>
              {portfolioLoading ? "" : formatUsd(portfolio?.totalPnlUsdt ?? 0)}
            </p>
          </CardContent>
        </Card>

        <Card className="glow-red">
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Max Drawdown</CardTitle>
            <AlertTriangle className="w-4 h-4 text-red-400" />
          </CardHeader>
          <CardContent>
            <div className="number-mono text-2xl font-bold text-red-400" data-testid="text-max-drawdown">
              {portfolioLoading ? "..." : `${(portfolio?.maxDrawdownR ?? 0).toFixed(2)}R`}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="glass-card rounded-md p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <h3 className="text-sm font-medium text-muted-foreground">Equity Curve</h3>
          <div className="flex gap-1">
            {(["7d", "30d", "all"] as const).map((range) => (
              <Button
                key={range}
                variant={equityRange === range ? "default" : "outline"}
                size="sm"
                data-testid={`button-range-${range}`}
                onClick={() => setEquityRange(range)}
              >
                {range === "all" ? "All" : range.toUpperCase()}
              </Button>
            ))}
          </div>
        </div>
        <div className="h-64" data-testid="chart-equity-curve">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={equityCurve ?? []}>
              <defs>
                <linearGradient id="greenGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="ts"
                tickFormatter={formatDate}
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickFormatter={(v: number) => `${v}R`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                  fontSize: 12,
                }}
                labelFormatter={(v: number) => formatDateTime(v)}
                formatter={(value: number, name: string) => {
                  if (name === "r") return [`${value}R`, "Cumulative"];
                  if (name === "tradeR") return [`${value}R`, "Trade"];
                  return [value, name];
                }}
              />
              <Area
                type="monotone"
                dataKey="r"
                stroke="#34d399"
                fill="url(#greenGradient)"
                strokeWidth={2}
              />
              <Scatter
                dataKey="tradeR"
                fill="#34d399"
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  if (!cx || !cy) return <circle r={0} />;
                  const color = (payload?.tradeR ?? 0) >= 0 ? "#34d399" : "#f87171";
                  return <circle cx={cx} cy={cy} r={4} fill={color} stroke="none" />;
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
