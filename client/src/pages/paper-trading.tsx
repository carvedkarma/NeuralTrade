import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useTradingWs } from "@/hooks/use-trading-ws";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Scatter,
  ComposedChart,
  ReferenceLine,
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
  Trash2,
  Zap,
  Eye,
  Star,
  Hourglass,
  BarChart2,
  Target,
  Award,
  History,
  Clock,
  Percent,
  Sigma,
  FlaskConical,
  CheckCircle2,
  XCircle,
  Lock,
} from "lucide-react";
import { CloseButton, PartialCloseButton, EditSLTPDialog } from "@/components/position-actions";
import { usePingMonitor } from "@/hooks/use-ping";
import { PingBadge } from "@/components/ping-badge";

interface MonteCarloStats {
  medianFinalEquity: number;
  p5FinalEquity: number;
  p95FinalEquity: number;
  medianMaxDrawdown: number;
  p95MaxDrawdown: number;
  confidenceLevel: string;
  isStatisticallySignificant: boolean;
}

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
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  sharpe: number;
  sharpeWarning?: string;
  expectancy: number;
  bestTrade: number;
  worstTrade: number;
  profitFactor: number;
  exposure: number;
  exposurePct: number;
  openPositions: number;
  monteCarloStats?: MonteCarloStats;
}

interface TradeHistory {
  id: number;
  positionId: number;
  symbol: string;
  side: string;
  entryTs: number;
  entryPrice: number;
  exitTs: number;
  exitPrice: number;
  grossR: number;
  netR: number;
  costR: number;
  pnlUsdt: number;
  riskUsdt: number;
  barsHeld: number;
  exitReason: string;
  maxFavorableR: number;
  signalConfidence: number;
  createdAt: number;
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

interface NeuralEvent {
  ts: number;
  symbol: string;
  side: string;
  adjustmentType: string;
  reason?: string;
  positionId?: number;
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

const NEURAL_ADJUSTMENT_LABELS: Record<string, { label: string; color: string; termColor: string }> = {
  BREAKEVEN: { label: "BE Set", color: "text-amber-400", termColor: "text-amber-400" },
  TRAIL_TIGHTEN: { label: "Trail Tight", color: "text-cyan-400", termColor: "text-cyan-400" },
  TRAIL_WIDEN: { label: "Trail Wide", color: "text-blue-400", termColor: "text-blue-400" },
  DIRECTION_FLIP_EXIT: { label: "Flip Exit", color: "text-red-400", termColor: "text-red-400" },
  CONFIDENCE_DECAY_EXIT: { label: "Decay Exit", color: "text-orange-400", termColor: "text-orange-400" },
  CONFIDENCE_DECAY_TIGHTEN: { label: "Decay Tight", color: "text-orange-400", termColor: "text-orange-400" },
  MFE_PROTECTION_EXIT: { label: "MFE Lock", color: "text-emerald-400", termColor: "text-emerald-400" },
  ADAPTIVE_TRAIL: { label: "Adapt Trail", color: "text-purple-400", termColor: "text-purple-400" },
};

const NEURAL_SWEEP_COLORS: Record<string, { gradient: string; animation: string; label: string; icon: string }> = {
  BREAKEVEN: { gradient: "from-amber-500/40", animation: "neural-sweep-amber", label: "BE LOCKED", icon: "lock" },
  TRAIL_TIGHTEN: { gradient: "from-cyan-500/30", animation: "neural-sweep-cyan", label: "TRAILING", icon: "trail" },
  TRAIL_WIDEN: { gradient: "from-cyan-500/30", animation: "neural-sweep-cyan", label: "TRAILING", icon: "trail" },
  DIRECTION_FLIP_EXIT: { gradient: "from-red-500/40", animation: "neural-sweep-red", label: "EXIT SIGNAL", icon: "exit" },
  CONFIDENCE_DECAY_EXIT: { gradient: "from-red-500/30", animation: "neural-sweep-red", label: "EXIT SIGNAL", icon: "exit" },
  CONFIDENCE_DECAY_TIGHTEN: { gradient: "from-orange-500/30", animation: "neural-sweep-amber", label: "DECAY TIGHT", icon: "decay" },
  MFE_PROTECTION_EXIT: { gradient: "from-emerald-500/30", animation: "neural-sweep-emerald", label: "MFE LOCK", icon: "lock" },
  ADAPTIVE_TRAIL: { gradient: "from-purple-500/30", animation: "neural-sweep-purple", label: "ADAPT", icon: "trail" },
};

function HealthRing({ score, size = 32 }: { score: number; size?: number }) {
  const r = (size - 4) / 2;
  const circ = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, score));
  const offset = circ - (pct / 100) * circ;
  const color = score >= 70 ? "#34d399" : score >= 45 ? "#fbbf24" : score >= 25 ? "#fb923c" : "#f87171";

  return (
    <svg width={size} height={size} className="animate-health-ring-pulse" style={{ color }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="currentColor" strokeOpacity={0.15} strokeWidth={2.5} />
      <circle
        cx={size / 2} cy={size / 2} r={r}
        fill="none" stroke="currentColor" strokeWidth={2.5}
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        className="transition-all duration-700"
      />
      <text x={size / 2} y={size / 2} textAnchor="middle" dominantBaseline="central" fill="currentColor" fontSize={8} fontFamily="var(--font-mono)" fontWeight="bold">
        {score}
      </text>
    </svg>
  );
}

function EegWave() {
  return (
    <svg className="absolute inset-0 w-full h-full opacity-[0.07] pointer-events-none" preserveAspectRatio="none" viewBox="0 0 300 40">
      <polyline
        points="0,20 20,20 30,10 40,30 50,15 60,25 70,20 90,20 100,8 110,32 120,20 140,20 150,12 160,28 170,20 190,20 200,5 210,35 220,20 240,20 260,20 270,10 280,30 290,20 300,20"
        fill="none" stroke="#34d399" strokeWidth="1.5"
        strokeDasharray="300" style={{ animation: "eeg-draw 4s linear infinite" }}
      />
    </svg>
  );
}

function NeuralWatchPanel({
  events,
  monitoredPositions,
  healthMap,
  flashingPositions,
}: {
  events: NeuralEvent[];
  monitoredPositions: Position[];
  healthMap: Record<number, PositionHealth>;
  flashingPositions: Set<number>;
}) {
  const isActive = events.length > 0 && Date.now() - events[0].ts < 60000;

  const latestAdjByPos: Record<number, NeuralEvent> = {};
  events.forEach((ev) => {
    if (ev.positionId && !latestAdjByPos[ev.positionId]) {
      latestAdjByPos[ev.positionId] = ev;
    }
  });

  return (
    <div className="relative overflow-hidden" data-testid="neural-watch-panel">
      <EegWave />
      <div className="flex items-center justify-between px-3 py-1.5 relative z-10">
        <div className="flex items-center gap-2">
          <Brain className={`w-3.5 h-3.5 text-emerald-400/70 ${isActive ? "animate-brain-pulse" : ""}`} />
          <span className="text-[10px] font-mono font-medium text-emerald-400/80 tracking-wider uppercase">
            Neural Monitor
          </span>
          <span className={`w-1.5 h-1.5 rounded-full ${isActive ? "bg-emerald-400 animate-pulse" : "bg-emerald-400/20"}`} />
        </div>
        {monitoredPositions.length > 0 && (
          <span className="text-[9px] font-mono text-emerald-400/50 flex items-center gap-1">
            <Eye className="w-2.5 h-2.5" />
            {monitoredPositions.length}
          </span>
        )}
      </div>

      <div className="px-3 pb-2.5 pt-1 relative z-10">
        {monitoredPositions.length === 0 ? (
          <div className="relative flex items-center justify-center py-3 gap-2 overflow-hidden">
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-1/2 -translate-y-1/2 w-1/4 h-full bg-gradient-to-r from-transparent via-emerald-400/5 to-transparent" style={{ animation: "empty-sweep 4s linear infinite" }} />
            </div>
            <Brain className="w-4 h-4 text-emerald-400/15 animate-brain-pulse" />
            <span className="text-[9px] font-mono text-muted-foreground/25">Monitoring...</span>
          </div>
        ) : (
          <div className="flex gap-1.5 overflow-x-auto scrollbar-hide" style={{ scrollbarWidth: "none" }}>
            {monitoredPositions.map((pos) => {
              const posId = typeof pos.id === "number" ? pos.id : parseInt(String(pos.id ?? "0"));
              const health = posId > 0 ? healthMap[posId] : undefined;
              const isFlash = posId > 0 && flashingPositions.has(posId);
              const latestAdj = latestAdjByPos[posId];
              const sweepInfo = latestAdj && isFlash ? NEURAL_SWEEP_COLORS[latestAdj.adjustmentType] : null;
              const adjLabel = latestAdj ? NEURAL_ADJUSTMENT_LABELS[latestAdj.adjustmentType] : null;
              const isMfeLock = latestAdj?.adjustmentType === "MFE_PROTECTION_EXIT" && isFlash;
              const healthScore = health?.score ?? 0;
              const healthColor = healthScore >= 70 ? "text-emerald-400" : healthScore >= 45 ? "text-amber-400" : healthScore >= 25 ? "text-orange-400" : "text-red-400";
              const borderColor = isFlash ? "border-cyan-400/50 bg-cyan-400/5" : "border-border/20 bg-black/20";

              return (
                <div
                  key={posId}
                  className={`relative flex items-center gap-1.5 px-2 py-1 rounded-full border transition-all duration-300 shrink-0 ${borderColor}`}
                  data-testid={`neural-watch-${pos.symbol}`}
                >
                  {sweepInfo && (
                    <div
                      className={`absolute inset-0 rounded-full bg-gradient-to-r ${sweepInfo.gradient} to-transparent pointer-events-none`}
                      style={{ animation: `${sweepInfo.animation} 0.8s ease-out forwards` }}
                    />
                  )}
                  <div className="relative z-10 flex items-center gap-1.5">
                    <span className="text-[10px] font-mono font-semibold text-foreground/80">{pos.symbol.replace("USDT", "")}</span>
                    <span className={`text-[7px] font-mono font-bold px-1 py-px rounded ${pos.side === "LONG" ? "text-emerald-400 bg-emerald-500/10" : "text-red-400 bg-red-500/10"}`}>
                      {pos.side === "LONG" ? "L" : "S"}
                    </span>
                    {health && (
                      <span className={`text-[9px] font-mono font-bold ${healthColor}`}>{healthScore}</span>
                    )}
                    {isMfeLock && <Shield className="w-2.5 h-2.5 text-emerald-400 animate-pulse" />}
                    {sweepInfo && adjLabel && (
                      <span className={`text-[7px] font-mono font-bold tracking-wider ${adjLabel.color} animate-pulse`}>
                        {sweepInfo.label}
                      </span>
                    )}
                    {isFlash && !sweepInfo && (
                      <Zap className="w-2.5 h-2.5 text-cyan-400/60 animate-pulse" />
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

interface CycleEvent {
  ts: number;
  symbol: string;
  direction: string;
  decision: string;
  price: number;
  v5Score: number | null;
  pEnter: number | null;
  opened: boolean;
  holdReason?: string;
}

const SCANNER_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "ADAUSDT"];

function RadarIcon({ isLive }: { isLive: boolean }) {
  return (
    <div className="relative w-4 h-4">
      <svg viewBox="0 0 20 20" className="w-4 h-4 text-cyan-400/60">
        <circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeWidth="1" strokeOpacity="0.3" />
        <circle cx="10" cy="10" r="4" fill="none" stroke="currentColor" strokeWidth="1" strokeOpacity="0.2" />
        <circle cx="10" cy="10" r="1.5" fill="currentColor" fillOpacity="0.5" />
      </svg>
      {isLive && (
        <svg viewBox="0 0 20 20" className="absolute inset-0 w-4 h-4 text-cyan-400 animate-radar-sweep" style={{ transformOrigin: "center" }}>
          <line x1="10" y1="10" x2="10" y2="2" stroke="currentColor" strokeWidth="1.5" strokeOpacity="0.6" strokeLinecap="round" />
        </svg>
      )}
    </div>
  );
}

function AiScannerGrid({
  events,
  lastCycleTs,
  cycleCount,
  webOpenCount,
  maxPositions,
}: {
  events: CycleEvent[];
  lastCycleTs: number;
  cycleCount: number;
  webOpenCount: number;
  maxPositions: number;
}) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 2000);
    return () => clearInterval(t);
  }, []);
  const isLive = lastCycleTs > 0 && now - lastCycleTs < 60000;

  const latestBySymbol: Record<string, CycleEvent> = {};
  events.forEach((ev) => {
    if (!latestBySymbol[ev.symbol]) latestBySymbol[ev.symbol] = ev;
  });

  const lastScanned = events.length > 0 ? events[0].symbol.replace("USDT", "") : null;

  const recentEvents = Object.values(latestBySymbol).filter(
    (ev) => now - ev.ts < 90000
  );
  const portfolioFullCount = recentEvents.filter(
    (ev) => ev.holdReason?.toLowerCase().includes("max total positions")
  ).length;
  const portfolioFull = portfolioFullCount >= 2;

  const hasSyncWarning = portfolioFull && webOpenCount === 0 && isLive;

  return (
    <div className="overflow-hidden" data-testid="ai-scanner-grid">
      <div className="flex items-center justify-between px-3 py-1.5">
        <div className="flex items-center gap-2">
          <RadarIcon isLive={isLive} />
          <span className="text-[10px] font-mono font-medium text-cyan-400/80 tracking-wider uppercase">
            Scanner
          </span>
          <span className={`w-1.5 h-1.5 rounded-full ${isLive ? "bg-emerald-400 animate-pulse" : "bg-muted-foreground/20"}`} data-testid="scanner-status-dot" />
          {isLive ? (
            <span className="text-[9px] font-mono text-emerald-400/50">LIVE</span>
          ) : (
            <span className="text-[9px] font-mono text-muted-foreground/30">OFF</span>
          )}
          {portfolioFull && (
            <span className="text-[8px] font-mono font-bold text-amber-400/80 flex items-center gap-1" data-testid="badge-portfolio-full">
              <Lock className="w-2.5 h-2.5" />
              {maxPositions}/{maxPositions}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {hasSyncWarning && (
            <span className="text-[8px] font-mono text-orange-400 flex items-center gap-1 animate-pulse" data-testid="badge-sync-warning">
              <ShieldAlert className="w-2.5 h-2.5" />SYNC
            </span>
          )}
          {lastScanned && isLive && !portfolioFull && (
            <span className="text-[9px] font-mono text-cyan-400/40 animate-pulse">
              {lastScanned}
            </span>
          )}
          {cycleCount > 0 && (
            <span className="text-[9px] font-mono text-muted-foreground/30">
              {cycleCount}
            </span>
          )}
        </div>
      </div>

      {hasSyncWarning && (
        <div className="px-3 py-1 flex items-center gap-2">
          <ShieldAlert className="w-2.5 h-2.5 text-orange-400 shrink-0" />
          <span className="text-[9px] font-mono text-orange-400/70">
            Trainer reports {maxPositions} open but web shows {webOpenCount}
          </span>
        </div>
      )}

      <div className="px-3 pb-2.5 pt-1">
        <div className="grid grid-cols-4 sm:grid-cols-8 gap-1">
          {SCANNER_SYMBOLS.map((sym) => {
            const ev = latestBySymbol[sym];
            const isRecent = ev && (now - ev.ts < 10000);
            const isEnter = ev?.decision === "ENTER";
            const isCooldown = ev?.decision === "COOLDOWN";
            const isHold = ev?.decision === "HOLD";
            const isOpened = ev?.opened === true;
            const isPortfolioBlock = isHold && (ev?.holdReason?.toLowerCase().includes("max total positions") ?? false);
            const shortName = sym.replace("USDT", "");
            const v5Pct = ev?.v5Score ? Math.min(100, (ev.v5Score / 10) * 100) : 0;

            const dirColor = ev?.direction === "LONG" ? "border-l-emerald-400/60" : ev?.direction === "SHORT" ? "border-l-red-400/60" : "border-l-transparent";
            let bgStyle = "bg-black/20";
            let animStyle = "";
            if (isRecent && isEnter) {
              bgStyle = "bg-emerald-500/8";
              animStyle = "tile-enter 1s ease-out";
            } else if (isRecent && isCooldown) {
              bgStyle = "bg-amber-500/5";
              animStyle = "tile-cooldown 1.2s ease-out";
            } else if (isRecent && isPortfolioBlock) {
              bgStyle = "bg-amber-500/3 opacity-60";
            } else if (isRecent && isHold) {
              bgStyle = "bg-black/10 opacity-50";
            } else if (isRecent) {
              bgStyle = "bg-cyan-500/3";
            }

            return (
              <div
                key={sym}
                className={`relative rounded border-l-2 ${dirColor} px-1.5 py-1 flex flex-col items-center gap-0.5 transition-all duration-500 ${bgStyle}`}
                style={animStyle ? { animation: animStyle } : undefined}
                data-testid={`scanner-tile-${sym}`}
              >
                <div className="flex items-center gap-1">
                  <span className={`text-[10px] font-mono font-bold leading-none ${(isHold || isPortfolioBlock) && isRecent ? "text-muted-foreground/30" : "text-foreground/70"}`}>
                    {shortName}
                  </span>
                  {isRecent && isPortfolioBlock && <Lock className="w-2 h-2 text-amber-500/40" />}
                  {isRecent && isCooldown && <Hourglass className="w-2 h-2 text-amber-400/50" />}
                </div>

                {isRecent && isEnter && (
                  <span
                    className="text-[7px] font-mono font-bold text-emerald-400 tracking-wider"
                    style={{ animation: "enter-fade 2s ease-out forwards" }}
                    data-testid={`badge-enter-${sym}`}
                  >
                    ENTER
                  </span>
                )}

                {isOpened && (
                  <span className="text-[7px] font-mono font-bold text-amber-300 flex items-center gap-0.5" data-testid={`badge-opened-${sym}`}>
                    <Star className="w-2 h-2" />OPEN
                  </span>
                )}

                <div className="w-full flex items-center gap-1">
                  <div className="flex-1 h-[2px] rounded-full bg-muted/20 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700 bg-cyan-400/50"
                      style={{ width: `${v5Pct}%` }}
                    />
                  </div>
                  {ev?.v5Score != null && (
                    <span className="text-[7px] font-mono text-cyan-400/40 leading-none">{ev.v5Score.toFixed(1)}</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

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

function PositionPriceGauge({ pos, livePrice, health, isFlashing, isNew, isGlowing }: { pos: Position; livePrice?: number; health?: PositionHealth; isFlashing?: boolean; isNew?: boolean; isGlowing?: boolean }) {
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
  const isBreakeven = stopLoss != null && Math.abs(stopLoss - entryPrice) / entryPrice < 0.0015;

  const trailPrice = pos.trailPrice;
  let trailPct: number | null = null;
  if (trailPrice && range > 0) {
    trailPct = Math.max(0, Math.min(100, ((trailPrice - lo) / range) * 100));
  }

  const borderAccent = isGlowing
    ? "border-l-emerald-400 shadow-[0_0_12px_1px_rgba(34,197,94,0.15)]"
    : isFlashing
    ? "border-l-cyan-400 shadow-[0_0_10px_1px_rgba(34,211,238,0.12)]"
    : healthScore !== null && healthScore < 15
    ? "border-l-red-500 animate-pulse"
    : healthScore !== null && healthScore < 30
    ? "border-l-amber-500"
    : isLong ? "border-l-emerald-400/40" : "border-l-red-400/40";

  const latestAdj = health?.latestAdjustment;
  const adjInfo = latestAdj ? NEURAL_ADJUSTMENT_LABELS[latestAdj] : null;

  const slDistDisplay = isBreakeven ? "BE" : `${slDist.toFixed(0)}%`;

  return (
    <div
      className={`rounded-lg border border-border/20 border-l-2 bg-black/30 p-2.5 space-y-2 transition-all duration-500 ${borderAccent}`}
      data-testid={`position-gauge-${pos.symbol}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="font-mono font-bold text-[13px] text-foreground/90">{pos.symbol.replace("USDT", "")}</span>
          <span className={`text-[8px] font-mono font-bold px-1 py-px rounded ${isLong ? "text-emerald-400 bg-emerald-500/10" : "text-red-400 bg-red-500/10"}`}>
            {side}
          </span>
          {pos.leverage != null && pos.leverage > 1 && (
            <span className="text-[8px] font-mono text-amber-400/70" data-testid="badge-leverage">{pos.leverage}x</span>
          )}
          {pos.source === "v5_signal" && (
            <span className="text-[7px] font-mono text-cyan-400/60 bg-cyan-500/8 px-1 rounded">V5</span>
          )}
          {isBreakeven && (
            <span className="text-[7px] font-mono text-amber-400 flex items-center gap-0.5" data-testid="badge-breakeven">
              <Shield className="w-2.5 h-2.5" />BE
            </span>
          )}
          {adjInfo && !isBreakeven && (
            <span className={`text-[7px] font-mono font-bold ${adjInfo.color} flex items-center gap-0.5`} data-testid="badge-neural-status">
              <Brain className="w-2.5 h-2.5" />{adjInfo.label}
            </span>
          )}
          {isNew && (
            <span className="text-[7px] font-mono text-emerald-300 animate-pulse" data-testid="badge-new-position">NEW</span>
          )}
          {isFlashing && (
            <span className="text-[7px] font-mono text-cyan-300 animate-pulse flex items-center gap-0.5" data-testid="badge-monitor-active">
              <Zap className="w-2 h-2" />ADJ
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`text-base font-bold number-mono ${isProfit ? "text-emerald-400" : "text-red-400"}`} data-testid={`text-pnlr-${pos.symbol}`}>
            {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}R
          </span>
          <span className={`text-[10px] number-mono ${isProfit ? "text-emerald-400/50" : "text-red-400/50"}`} data-testid={`text-pnlusdt-${pos.symbol}`}>
            {pnlUsd >= 0 ? "+" : ""}${pnlUsd.toFixed(0)}
          </span>
        </div>
      </div>

      {health && (
        <div className="flex items-center justify-between">
          <HealthGauge score={health.score} riskLevel={health.riskLevel} />
          <MfeTracker currentPnlR={health.currentPnlR} peakPnlR={health.peakPnlR} giveback={health.giveback} />
        </div>
      )}

      <div className="space-y-1">
        <div className="relative h-1 rounded-full overflow-hidden bg-muted/20">
          <div
            className="absolute top-0 bottom-0 w-px bg-amber-400/60 z-10"
            style={{ left: `${Math.max(1, Math.min(99, entryPct))}%` }}
          />

          <div
            className={`absolute top-0 bottom-0 rounded-full transition-all duration-500 ${
              isProfit ? "bg-emerald-500/25" : "bg-red-500/25"
            }`}
            style={{
              left: `${Math.min(clampedPricePct, Math.max(0, Math.min(100, entryPct)))}%`,
              width: `${Math.abs(clampedPricePct - Math.max(0, Math.min(100, entryPct)))}%`,
            }}
          />

          <div
            className={`absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full z-20 transition-all duration-500 ${
              isProfit ? "bg-emerald-400 shadow-[0_0_6px_rgba(34,197,94,0.6)]" : "bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.6)]"
            }`}
            style={{ left: `${clampedPricePct}%`, marginLeft: "-4px" }}
          />

          {trailPct !== null && (
            <div
              className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full z-15 bg-purple-400/70 transition-all duration-500"
              style={{ left: `${trailPct}%`, marginLeft: "-3px" }}
              data-testid="trail-level-indicator"
            />
          )}
        </div>

        <div className="flex justify-between items-center text-[9px] number-mono text-muted-foreground/50">
          <span className="text-red-400/50">{isLong ? "SL" : "TP"}</span>
          <div className="flex items-center gap-1">
            <span className="text-amber-400/50">{formatPrice(entryPrice)}</span>
            <span className="text-foreground/20">→</span>
            <span className={isProfit ? "text-emerald-400/70" : "text-red-400/70"}>{formatPrice(currentPrice)}</span>
            {trailPrice && (
              <>
                <span className="text-foreground/15">|</span>
                <span className="text-purple-400/50">T {formatPrice(trailPrice)}</span>
              </>
            )}
          </div>
          <span className="text-emerald-400/50">{isLong ? "TP" : "SL"}</span>
        </div>
      </div>

      <div className="flex items-center justify-between text-[9px]">
        <div className="flex items-center gap-2 text-muted-foreground/40 font-mono">
          <span className={isBreakeven ? "text-amber-400/70" : slDist < 30 ? "text-red-400/70" : ""}>SL {slDistDisplay}</span>
          <span className="text-foreground/10">|</span>
          <span className={tpDist < 30 ? "text-emerald-400/70" : ""}>TP {tpDist.toFixed(0)}%</span>
          <span className="text-foreground/10">|</span>
          <span data-testid={`text-duration-${pos.symbol}`}>
            {dur > 0 ? formatDuration(dur) : "-"}
          </span>
          {pos.v5Score != null && (
            <>
              <span className="text-foreground/10">|</span>
              <span className="text-cyan-400/40" data-testid={`text-v5score-${pos.symbol}`}>V5 {pos.v5Score.toFixed(2)}</span>
            </>
          )}
          {pos.leverage != null && (
            <span className="text-amber-400/30" data-testid={`text-leverage-${pos.symbol}`}>{pos.leverage}x</span>
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

function ScannerStatusBadge({ lastCycleTs, cycleCount }: { lastCycleTs: number; cycleCount: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 5000);
    return () => clearInterval(t);
  }, []);
  const isLive = lastCycleTs > 0 && now - lastCycleTs < 60000;

  return (
    <div className="flex items-center gap-1.5" data-testid="scanner-header-badge">
      <span className={`w-2 h-2 rounded-full ${isLive ? "bg-emerald-400 animate-pulse" : "bg-muted-foreground/30"}`} />
      <span className={`text-xs font-mono font-semibold ${isLive ? "text-emerald-400" : "text-muted-foreground/50"}`}>
        {isLive ? "LIVE SCANNER" : "SCANNER OFFLINE"}
      </span>
      {cycleCount > 0 && (
        <span className="text-[10px] font-mono text-muted-foreground/40 ml-1">({cycleCount})</span>
      )}
    </div>
  );
}

function MonteCarloStrip({ mc, totalTrades }: { mc: MonteCarloStats; totalTrades: number }) {
  const confColor = mc.confidenceLevel.toLowerCase().includes("very high")
    ? "text-emerald-400"
    : mc.confidenceLevel.toLowerCase().includes("high")
    ? "text-cyan-400"
    : mc.confidenceLevel.toLowerCase().includes("moderate")
    ? "text-amber-400"
    : "text-orange-400";

  const confBg = mc.confidenceLevel.toLowerCase().includes("very high")
    ? "border-emerald-400/20 bg-emerald-400/5"
    : mc.confidenceLevel.toLowerCase().includes("high")
    ? "border-cyan-400/20 bg-cyan-400/5"
    : mc.confidenceLevel.toLowerCase().includes("moderate")
    ? "border-amber-400/20 bg-amber-400/5"
    : "border-orange-400/20 bg-orange-400/5";

  return (
    <div className={`rounded-md border px-4 py-2.5 ${confBg}`} data-testid="monte-carlo-strip">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5">
        <div className="flex items-center gap-1.5">
          <FlaskConical className="w-3.5 h-3.5 text-purple-400/70" />
          <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">Monte Carlo</span>
          <Badge className={`no-default-hover-elevate no-default-active-elevate text-[9px] px-1.5 py-0 h-4 ${confColor} bg-current/10 border border-current/30`}>
            {mc.isStatisticallySignificant ? "✓ " : ""}{mc.confidenceLevel}
          </Badge>
        </div>

        <div className="flex items-center gap-1 text-[10px]">
          <span className="text-muted-foreground/60">Median Equity</span>
          <span className="font-mono font-semibold text-foreground">${mc.medianFinalEquity.toFixed(0)}</span>
        </div>

        <div className="flex items-center gap-1 text-[10px]">
          <span className="text-muted-foreground/60">5th–95th pct</span>
          <span className="font-mono text-red-400/80">${mc.p5FinalEquity.toFixed(0)}</span>
          <span className="text-muted-foreground/40">→</span>
          <span className="font-mono text-emerald-400/80">${mc.p95FinalEquity.toFixed(0)}</span>
        </div>

        <div className="flex items-center gap-1 text-[10px]">
          <span className="text-muted-foreground/60">DD p95</span>
          <span className="font-mono text-red-400/80">{mc.p95MaxDrawdown.toFixed(2)}R</span>
        </div>

        <div className="flex items-center gap-1 text-[10px] ml-auto">
          <span className="text-muted-foreground/60">n={totalTrades} trades simulated</span>
        </div>
      </div>
    </div>
  );
}

const EXIT_REASON_LABELS: Record<string, { label: string; color: string }> = {
  TAKE_PROFIT: { label: "TP", color: "text-emerald-400" },
  STOP_LOSS: { label: "SL", color: "text-red-400" },
  NEURAL_MFE: { label: "MFE", color: "text-cyan-400" },
  MFE_GIVEBACK: { label: "MFE", color: "text-cyan-400" },
  MFE_PROTECTION_EXIT: { label: "MFE Shield", color: "text-purple-400" },
  BREAKEVEN: { label: "BE", color: "text-amber-400" },
  DIRECTION_FLIP: { label: "Flip", color: "text-orange-400" },
  DIRECTION_FLIP_EXIT: { label: "Flip", color: "text-orange-400" },
  CONFIDENCE_DECAY: { label: "Decay", color: "text-orange-400" },
  NEURAL_FLIP: { label: "Flip", color: "text-orange-400" },
  NEURAL_DECAY: { label: "Decay", color: "text-orange-300" },
  FAILURE: { label: "Fail", color: "text-red-300" },
  TIME: { label: "Time", color: "text-muted-foreground" },
  FLIP: { label: "Flip", color: "text-orange-400" },
  MANUAL: { label: "Manual", color: "text-muted-foreground" },
  ADAPTIVE_TRAIL: { label: "Trail", color: "text-blue-400" },
  PARTIAL: { label: "Partial", color: "text-cyan-400/70" },
};

function TradeHistoryTable({ trades }: { trades: TradeHistory[] }) {
  if (trades.length === 0) {
    return (
      <div className="text-center py-6 text-sm text-muted-foreground" data-testid="text-no-trade-history">
        No closed trades yet
      </div>
    );
  }

  return (
    <div className="overflow-x-auto" data-testid="trade-history-table">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="border-b border-border/40">
            <th className="text-left py-2 px-2 text-muted-foreground/60 font-medium">Symbol</th>
            <th className="text-left py-2 px-2 text-muted-foreground/60 font-medium">Side</th>
            <th className="text-right py-2 px-2 text-muted-foreground/60 font-medium">Net R</th>
            <th className="text-right py-2 px-2 text-muted-foreground/60 font-medium">P&L $</th>
            <th className="text-right py-2 px-2 text-muted-foreground/60 font-medium">MFE</th>
            <th className="text-right py-2 px-2 text-muted-foreground/60 font-medium">Conf</th>
            <th className="text-center py-2 px-2 text-muted-foreground/60 font-medium">Exit</th>
            <th className="text-right py-2 px-2 text-muted-foreground/60 font-medium">Duration</th>
            <th className="text-right py-2 px-2 text-muted-foreground/60 font-medium">Closed</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t, i) => {
            const isWin = t.netR >= 0;
            const exitInfo = EXIT_REASON_LABELS[t.exitReason] ?? { label: t.exitReason, color: "text-muted-foreground" };
            const durMs = t.exitTs - t.entryTs;
            const durStr = durMs > 0 ? formatDuration(durMs) : "-";
            return (
              <tr
                key={t.id}
                className={`border-b border-border/20 transition-colors hover:bg-muted/20 ${i % 2 === 0 ? "" : "bg-muted/5"}`}
                data-testid={`trade-row-${t.id}`}
              >
                <td className="py-1.5 px-2 font-mono font-semibold text-foreground/90">{t.symbol.replace("USDT", "")}</td>
                <td className="py-1.5 px-2">
                  <span className={`font-semibold ${t.side === "LONG" ? "text-emerald-400" : "text-red-400"}`}>
                    {t.side === "LONG" ? "▲" : "▼"} {t.side}
                  </span>
                </td>
                <td className={`py-1.5 px-2 text-right font-mono font-bold ${isWin ? "text-emerald-400" : "text-red-400"}`}>
                  {isWin ? "+" : ""}{t.netR.toFixed(2)}R
                </td>
                <td className={`py-1.5 px-2 text-right font-mono ${isWin ? "text-emerald-400/70" : "text-red-400/70"}`}>
                  {t.pnlUsdt >= 0 ? "+" : ""}${t.pnlUsdt.toFixed(2)}
                </td>
                <td className="py-1.5 px-2 text-right font-mono text-cyan-400/70">
                  {t.maxFavorableR > 0 ? `${t.maxFavorableR.toFixed(2)}R` : "-"}
                </td>
                <td className="py-1.5 px-2 text-right font-mono text-muted-foreground/60">
                  {t.signalConfidence > 0 ? `${(t.signalConfidence * 100).toFixed(0)}%` : "-"}
                </td>
                <td className="py-1.5 px-2 text-center">
                  <span className={`font-mono font-semibold ${exitInfo.color}`}>{exitInfo.label}</span>
                </td>
                <td className="py-1.5 px-2 text-right font-mono text-muted-foreground/50">{durStr}</td>
                <td className="py-1.5 px-2 text-right font-mono text-muted-foreground/40">
                  {new Date(t.exitTs).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function PaperTrading() {
  const [equityRange, setEquityRange] = useState<"7d" | "30d" | "all">("30d");
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [healthMap, setHealthMap] = useState<Record<number, PositionHealth>>({});
  const [neuralEvents, setNeuralEvents] = useState<NeuralEvent[]>([]);
  const [flashingPositions, setFlashingPositions] = useState<Set<number>>(new Set());
  const [newPositions, setNewPositions] = useState<Set<number>>(new Set());
  const [glowPositions, setGlowPositions] = useState<Set<number>>(new Set());
  const [cycleEvents, setCycleEvents] = useState<CycleEvent[]>([]);
  const [lastCycleTs, setLastCycleTs] = useState(0);
  const [cycleCount, setCycleCount] = useState(0);
  const { subscribe } = useTradingWs();
  const ping = usePingMonitor();

  useEffect(() => {
    const unsub = subscribe("PRICE_TICK", (payload) => {
      setLivePrices(payload as Record<string, number>);
    });
    return unsub;
  }, [subscribe]);

  useEffect(() => {
    const unsub = subscribe("CYCLE_UPDATE", (payload) => {
      const p = payload as {
        symbol?: string;
        direction?: string;
        decision?: string;
        price?: number;
        v5Score?: number;
        pEnter?: number;
        cycleTs?: number;
        holdReason?: string;
        autoTradeResult?: { opened?: boolean; positionId?: number };
      };
      if (!p.symbol) return;

      const openedFromCycle = p.autoTradeResult?.opened === true;
      const ev: CycleEvent = {
        ts: p.cycleTs ?? Date.now(),
        symbol: p.symbol,
        direction: p.direction ?? "",
        decision: p.decision ?? "HOLD",
        price: p.price ?? 0,
        v5Score: p.v5Score ?? null,
        pEnter: p.pEnter ?? null,
        opened: openedFromCycle,
        holdReason: p.holdReason,
      };

      setCycleEvents((prev) => [ev, ...prev].slice(0, 40));
      setLastCycleTs(Date.now());
      setCycleCount((c) => c + 1);

      if (openedFromCycle && p.autoTradeResult?.positionId) {
        const posId = p.autoTradeResult.positionId;
        setGlowPositions((prev) => { const n = new Set(prev); n.add(posId); return n; });
        setTimeout(() => {
          setGlowPositions((prev) => { const n = new Set(prev); n.delete(posId); return n; });
        }, 5000);
        setNewPositions((prev) => { const n = new Set(prev); n.add(posId); return n; });
        setTimeout(() => {
          setNewPositions((prev) => { const n = new Set(prev); n.delete(posId); return n; });
        }, 10000);
        queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
        queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
      }
    });
    return unsub;
  }, [subscribe]);

  useEffect(() => {
    const unsub = subscribe("TRADE_OPENED", (payload) => {
      const p = payload as { positionId?: number; symbol?: string };
      if (!p.positionId) return;
      const posId = p.positionId;

      if (newPositions.has(posId)) return;

      setCycleEvents((prev) => {
        const updated = [...prev];
        const idx = updated.findIndex((e) => e.symbol === p.symbol && e.decision === "ENTER" && !e.opened);
        if (idx >= 0) updated[idx] = { ...updated[idx], opened: true };
        return updated;
      });

      setGlowPositions((prev) => { const n = new Set(prev); n.add(posId); return n; });
      setTimeout(() => {
        setGlowPositions((prev) => { const n = new Set(prev); n.delete(posId); return n; });
      }, 5000);

      setNewPositions((prev) => { const n = new Set(prev); n.add(posId); return n; });
      setTimeout(() => {
        setNewPositions((prev) => { const n = new Set(prev); n.delete(posId); return n; });
      }, 10000);

      queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
    });
    return unsub;
  }, [subscribe, newPositions]);

  useEffect(() => {
    const unsub = subscribe("TRADE_UPDATE", (payload) => {
      const p = payload as { positionId?: number; symbol?: string; side?: string; adjustmentType?: string; action?: string };
      if (p.action === "NEURAL_ADJUST" && p.adjustmentType && p.symbol) {
        const ev: NeuralEvent = {
          ts: Date.now(),
          symbol: p.symbol,
          side: (p.side as string) ?? "UNKNOWN",
          adjustmentType: p.adjustmentType,
          positionId: p.positionId,
        };
        setNeuralEvents((prev) => [ev, ...prev].slice(0, 20));

        if (p.positionId) {
          const posId = p.positionId;
          setFlashingPositions((prev) => { const n = new Set(prev); n.add(posId); return n; });
          setTimeout(() => {
            setFlashingPositions((prev) => {
              const next = new Set(prev);
              next.delete(posId);
              return next;
            });
          }, 3000);
        }

        queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
      }
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

  const { data: tradeHistory } = useQuery<TradeHistory[]>({
    queryKey: ["/api/paper/trade-history"],
    refetchInterval: 30000,
  });

  const { data: openPosSummary } = useQuery<{ count: number; symbols: string[] }>({
    queryKey: ["/api/paper/open-positions-summary"],
    refetchInterval: 15000,
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

  const winRate = portfolio?.winRate ?? 0;
  const sharpe = portfolio?.sharpe ?? 0;
  const profitFactor = portfolio?.profitFactor ?? 0;
  const expectancy = portfolio?.expectancy ?? 0;
  const totalTrades = portfolio?.totalTrades ?? 0;
  const winningTrades = portfolio?.winningTrades ?? 0;
  const losingTrades = portfolio?.losingTrades ?? 0;
  const mc = portfolio?.monteCarloStats;

  const recentTrades = (tradeHistory ?? []).slice(0, 15);

  return (
    <div className="p-4 space-y-4" data-testid="paper-trading">

      {/* ── Control Bar ─────────────────────────────────────────── */}
      <div className="glass-card rounded-md px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">

          <div className="flex items-center gap-4 divide-x divide-border/40">
            <div className="flex items-center gap-2 pr-4">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Paper</span>
              <Switch
                data-testid="switch-paper-trading"
                checked={paperEnabled}
                onCheckedChange={handlePaperToggle}
                disabled={enableMutation.isPending || disableMutation.isPending}
              />
              <span className={`w-1.5 h-1.5 rounded-full transition-colors ${paperEnabled ? "bg-emerald-400 animate-pulse" : "bg-muted-foreground/20"}`} />
            </div>

            <div className="flex items-center gap-2 pl-4">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Auto</span>
              <Switch
                data-testid="switch-auto-trading"
                checked={autoTrading}
                onCheckedChange={handleAutoToggle}
                disabled={!paperEnabled || startAutoMutation.isPending || stopAutoMutation.isPending}
              />
              <span className={`w-1.5 h-1.5 rounded-full transition-colors ${autoTrading ? "bg-emerald-400 animate-pulse" : "bg-muted-foreground/20"}`} />
            </div>
          </div>

          <div className="h-4 w-px bg-border/40" />
          <ScannerStatusBadge lastCycleTs={lastCycleTs} cycleCount={cycleCount} />

          <div className="ml-auto flex items-center gap-2">
            <PingBadge ping={ping} />
            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground h-8 px-2.5"
              data-testid="button-clear-history" onClick={handleClearHistory} disabled={clearHistoryMutation.isPending}>
              <Trash2 className="w-3.5 h-3.5 mr-1" />
              <span className="text-xs">History</span>
            </Button>
            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground h-8 px-2.5"
              data-testid="button-clear-analytics" onClick={handleClearAnalytics} disabled={clearAnalyticsMutation.isPending}>
              <Trash2 className="w-3.5 h-3.5 mr-1" />
              <span className="text-xs">Equity</span>
            </Button>
            <Button variant="destructive" size="sm" className="h-8 px-3"
              data-testid="button-reset-portfolio" onClick={handleReset} disabled={resetMutation.isPending}>
              <RotateCcw className="w-3.5 h-3.5 mr-1" />
              <span className="text-xs">Reset</span>
            </Button>
          </div>
        </div>
      </div>

      {/* ── Primary KPI Row ──────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-1 pt-3 px-4">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Starting Equity</CardTitle>
            <DollarSign className="w-3.5 h-3.5 text-muted-foreground/50" />
          </CardHeader>
          <CardContent className="px-4 pb-3">
            <div className="number-mono text-xl font-bold" data-testid="text-starting-equity">
              {portfolioLoading ? "..." : formatUsd(portfolio?.startingEquity ?? 0)}
            </div>
            <p className="text-[10px] text-muted-foreground/50 mt-0.5">paper capital</p>
          </CardContent>
        </Card>

        <Card className={pctChange >= 0 ? "glow-green" : "glow-red"}>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-1 pt-3 px-4">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Current Equity</CardTitle>
            {pctChange >= 0 ? <TrendingUp className="w-3.5 h-3.5 text-emerald-400" /> : <TrendingDown className="w-3.5 h-3.5 text-red-400" />}
          </CardHeader>
          <CardContent className="px-4 pb-3">
            <div className="number-mono text-xl font-bold" data-testid="text-current-equity">
              {portfolioLoading ? "..." : formatUsd(portfolio?.currentEquity ?? 0)}
            </div>
            <p className={`text-[10px] mt-0.5 font-mono ${pctChange >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {pctChange >= 0 ? "+" : ""}{pctChange.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card className={pnlPositive ? "glow-green" : "glow-red"}>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-1 pt-3 px-4">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Total P&amp;L</CardTitle>
            <Activity className="w-3.5 h-3.5 text-cyan-500/70" />
          </CardHeader>
          <CardContent className="px-4 pb-3">
            <div className={`number-mono text-xl font-bold ${pnlPositive ? "text-emerald-400" : "text-red-400"}`} data-testid="text-total-pnl">
              {portfolioLoading ? "..." : `${(portfolio?.totalPnlR ?? 0) >= 0 ? "+" : ""}${(portfolio?.totalPnlR ?? 0).toFixed(2)}R`}
            </div>
            <p className={`text-[10px] mt-0.5 font-mono ${pnlPositive ? "text-emerald-400/70" : "text-red-400/70"}`}>
              {portfolioLoading ? "" : formatUsd(portfolio?.totalPnlUsdt ?? 0)}
            </p>
          </CardContent>
        </Card>

        <Card className="glow-red">
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-1 pt-3 px-4">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Max Drawdown</CardTitle>
            <AlertTriangle className="w-3.5 h-3.5 text-red-400/70" />
          </CardHeader>
          <CardContent className="px-4 pb-3">
            <div className="number-mono text-xl font-bold text-red-400" data-testid="text-max-drawdown">
              {portfolioLoading ? "..." : `${(portfolio?.maxDrawdownR ?? 0).toFixed(2)}R`}
            </div>
            <p className="text-[10px] text-muted-foreground/50 mt-0.5">
              {portfolioLoading ? "" : `${(portfolio?.maxDrawdown ?? 0).toFixed(2)}%`}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* ── Secondary KPI Row ────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="glass-card rounded-md px-4 py-3 flex items-center justify-between gap-3" data-testid="kpi-winrate">
          <div>
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Win Rate</p>
            <p className="number-mono text-lg font-bold text-foreground mt-0.5" data-testid="text-win-rate">
              {portfolioLoading ? "..." : `${winRate.toFixed(1)}%`}
            </p>
            <p className="text-[10px] text-muted-foreground/50 font-mono">
              {portfolioLoading ? "" : `${winningTrades}W / ${losingTrades}L`}
            </p>
          </div>
          <div className="relative w-10 h-10 shrink-0">
            <svg viewBox="0 0 36 36" className="w-10 h-10 -rotate-90">
              <circle cx="18" cy="18" r="14" fill="none" stroke="hsl(var(--muted))" strokeWidth="3" />
              <circle cx="18" cy="18" r="14" fill="none"
                stroke={winRate >= 55 ? "#34d399" : winRate >= 45 ? "#fbbf24" : "#f87171"}
                strokeWidth="3"
                strokeDasharray={`${(winRate / 100) * 87.96} 87.96`}
                strokeLinecap="round"
                className="transition-all duration-700"
              />
            </svg>
          </div>
        </div>

        <div className="glass-card rounded-md px-4 py-3" data-testid="kpi-sharpe">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Sharpe Ratio</p>
          <p className={`number-mono text-lg font-bold mt-0.5 ${sharpe >= 2 ? "text-emerald-400" : sharpe >= 1 ? "text-cyan-400" : "text-amber-400"}`} data-testid="text-sharpe">
            {portfolioLoading ? "..." : sharpe.toFixed(2)}
          </p>
          {portfolio?.sharpeWarning ? (
            <p className="text-[10px] text-amber-400/70 mt-0.5 flex items-center gap-1">
              <AlertTriangle className="w-2.5 h-2.5 shrink-0" />
              may overfit
            </p>
          ) : (
            <p className="text-[10px] text-muted-foreground/50 mt-0.5">annualized</p>
          )}
        </div>

        <div className="glass-card rounded-md px-4 py-3" data-testid="kpi-pf">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Profit Factor</p>
          <p className={`number-mono text-lg font-bold mt-0.5 ${profitFactor >= 2 ? "text-emerald-400" : profitFactor >= 1.2 ? "text-cyan-400" : "text-red-400"}`} data-testid="text-profit-factor">
            {portfolioLoading ? "..." : profitFactor > 0 ? profitFactor.toFixed(2) : "—"}
          </p>
          <p className="text-[10px] text-muted-foreground/50 mt-0.5">gross wins / losses</p>
        </div>

        <div className="glass-card rounded-md px-4 py-3" data-testid="kpi-expectancy">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Expectancy</p>
          <p className={`number-mono text-lg font-bold mt-0.5 ${expectancy >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-expectancy">
            {portfolioLoading ? "..." : `${expectancy >= 0 ? "+" : ""}${expectancy.toFixed(3)}R`}
          </p>
          <p className="text-[10px] text-muted-foreground/50 mt-0.5">
            {totalTrades > 0 ? `per trade · n=${totalTrades}` : "no trades yet"}
          </p>
        </div>
      </div>

      {/* ── Monte Carlo Strip ────────────────────────────────────── */}
      {mc && <MonteCarloStrip mc={mc} totalTrades={totalTrades} />}

      {/* ── Open Positions + Scanner ─────────────────────────────── */}
      <div className="rounded-xl border border-border/20 bg-black/50 backdrop-blur-sm overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/10">
          <div className="flex items-center gap-2">
            <span className="text-[12px] font-mono font-semibold text-foreground/80 tracking-wide">Positions</span>
          </div>
          <div className="flex items-center gap-2">
            {openPositions && openPositions.length > 0 && (
              <span className="text-[9px] font-mono text-cyan-400/60" data-testid="badge-open-count">
                {openPositions.length} active
              </span>
            )}
            {portfolio?.unrealizedPnl != null && portfolio.unrealizedPnl !== 0 && (
              <span className={`text-[10px] font-mono font-bold number-mono ${portfolio.unrealizedPnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                {portfolio.unrealizedPnl >= 0 ? "+" : ""}{portfolio.unrealizedPnlR?.toFixed(2) ?? "0.00"}R
              </span>
            )}
          </div>
        </div>

        <div className="border-b border-border/10">
          <AiScannerGrid
            events={cycleEvents}
            lastCycleTs={lastCycleTs}
            cycleCount={cycleCount}
            webOpenCount={openPosSummary?.count ?? 0}
            maxPositions={4}
          />
        </div>

        <div className="border-b border-border/10">
          <NeuralWatchPanel
            events={neuralEvents}
            monitoredPositions={openPositions?.filter(p => p.source === "v5_signal") ?? []}
            healthMap={healthMap}
            flashingPositions={flashingPositions}
          />
        </div>

        <div className="p-3">
          {(!openPositions || openPositions.length === 0) ? (
            <p className="text-[11px] font-mono text-muted-foreground/30 text-center py-6" data-testid="text-no-open-positions">
              No open positions
            </p>
          ) : (
            <div className="grid gap-2 sm:grid-cols-1 md:grid-cols-2" data-testid="positions-grid">
              {openPositions.map((pos, i) => {
                const posId = typeof pos.id === "number" ? pos.id : parseInt(String(pos.id ?? "0"));
                return (
                  <PositionPriceGauge
                    key={pos.id ?? i}
                    pos={pos}
                    livePrice={livePrices[pos.symbol]}
                    health={posId > 0 ? healthMap[posId] : undefined}
                    isFlashing={posId > 0 && flashingPositions.has(posId)}
                    isNew={posId > 0 && newPositions.has(posId)}
                    isGlowing={posId > 0 && glowPositions.has(posId)}
                  />
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Equity Curve ─────────────────────────────────────────── */}
      <div className="glass-card rounded-md p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-cyan-400/60" />
            <h3 className="text-sm font-semibold">Equity Curve</h3>
            {equityCurve && equityCurve.length > 0 && (
              <span className="text-[10px] font-mono text-muted-foreground/40">{equityCurve.length} trades</span>
            )}
          </div>
          <div className="flex gap-1">
            {(["7d", "30d", "all"] as const).map((range) => (
              <Button
                key={range}
                variant={equityRange === range ? "default" : "outline"}
                size="sm"
                className="h-7 text-xs px-2.5"
                data-testid={`button-range-${range}`}
                onClick={() => setEquityRange(range)}
              >
                {range === "all" ? "All" : range.toUpperCase()}
              </Button>
            ))}
          </div>
        </div>
        <div className="h-80" data-testid="chart-equity-curve">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={equityCurve ?? []} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.25} />
                  <stop offset="80%" stopColor="#34d399" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
              <XAxis
                dataKey="ts"
                tickFormatter={formatDate}
                stroke="hsl(var(--muted-foreground))"
                fontSize={11}
                tick={{ fill: "hsl(var(--muted-foreground))" }}
              />
              <YAxis
                stroke="hsl(var(--muted-foreground))"
                fontSize={11}
                tick={{ fill: "hsl(var(--muted-foreground))" }}
                tickFormatter={(v: number) => `${v}R`}
                width={42}
              />
              <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeOpacity={0.4} strokeDasharray="4 4" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  fontSize: 11,
                  boxShadow: "0 4px 16px rgba(0,0,0,0.3)",
                }}
                labelFormatter={(v: number) => formatDateTime(v)}
                formatter={(value: number, name: string, props: any) => {
                  if (name === "r") return [`${(value as number).toFixed(2)}R`, "Cumulative"];
                  if (name === "tradeR") {
                    const sym = props?.payload?.symbol ?? "";
                    const side = props?.payload?.side ?? "";
                    return [`${(value as number) >= 0 ? "+" : ""}${(value as number).toFixed(2)}R`, `Trade${sym ? ` · ${sym}` : ""}${side ? ` ${side}` : ""}`];
                  }
                  return [value, name];
                }}
              />
              <Area
                type="monotone"
                dataKey="r"
                stroke="#34d399"
                fill="url(#equityGradient)"
                strokeWidth={2}
                dot={false}
              />
              <Scatter
                dataKey="tradeR"
                fill="#34d399"
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  if (!cx || !cy) return <circle r={0} />;
                  const color = (payload?.tradeR ?? 0) >= 0 ? "#34d399" : "#f87171";
                  const r = Math.abs(payload?.tradeR ?? 1) > 3 ? 6 : 4;
                  return <circle cx={cx} cy={cy} r={r} fill={color} stroke="rgba(0,0,0,0.3)" strokeWidth={1} />;
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Recent Trade History ──────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-muted-foreground/60" />
              <CardTitle className="text-sm font-semibold">Recent Trades</CardTitle>
              {recentTrades.length > 0 && (
                <span className="text-[10px] font-mono text-muted-foreground/40">last {recentTrades.length}</span>
              )}
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground/60">
              {portfolio && (
                <>
                  <span className="flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400/60" />
                    Avg W: <span className="text-emerald-400/80 ml-0.5">+{((portfolio.avgWin ?? 0) * 100).toFixed(1)}%</span>
                  </span>
                  <span className="flex items-center gap-1">
                    <XCircle className="w-3 h-3 text-red-400/60" />
                    Avg L: <span className="text-red-400/80 ml-0.5">-{((portfolio.avgLoss ?? 0) * 100).toFixed(1)}%</span>
                  </span>
                  <span>Best: <span className="text-emerald-400/80">+{((portfolio.bestTrade ?? 0) * 100).toFixed(1)}%</span></span>
                  <span>Worst: <span className="text-red-400/80">{((portfolio.worstTrade ?? 0) * 100).toFixed(1)}%</span></span>
                </>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="px-4 pb-4">
          <TradeHistoryTable trades={recentTrades} />
        </CardContent>
      </Card>

    </div>
  );
}
