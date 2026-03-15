import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";
import { formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Minus,
  Clock,
  Zap,
  Target,
  AlertTriangle,
  CheckCircle2,
  Eye,
  Activity,
  Shield,
} from "lucide-react";
import { TRADING_SYMBOLS } from "@shared/symbols";

interface SymbolIntelligence {
  symbol: string;
  price: number | null;
  direction: string | null;
  decision: string | null;
  holdReason: string | null;
  pLong: number | null;
  pShort: number | null;
  pHold: number | null;
  retMu: number | null;
  mfePred: number | null;
  maePred: number | null;
  v5Score: number | null;
  v5Threshold: number | null;
  htfH1Trend: number | null;
  htfH4Trend: number | null;
  htfScore: number | null;
  cycleTs: number | null;
  slopeOk: boolean | null;
  rangeOk: boolean | null;
  pEnter: number | null;
  retMuDirection: string | null;
}

interface NeuralIntelligenceResponse {
  symbols: Record<string, SymbolIntelligence>;
  updatedAt: number;
}

function pct(v: number | null): string {
  if (v === null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmt(v: number | null, decimals = 3): string {
  if (v === null) return "—";
  return v.toFixed(decimals);
}

function ageLabel(ts: number | null): string {
  if (!ts) return "Never";
  try {
    return formatDistanceToNow(new Date(ts), { addSuffix: true });
  } catch {
    return "—";
  }
}

function isStale(ts: number | null): boolean {
  if (!ts) return true;
  return Date.now() - ts > 20 * 60 * 1000;
}

function DirectionIcon({ dir }: { dir: string | null }) {
  if (dir === "LONG") return <TrendingUp className="w-4 h-4 text-emerald-400" />;
  if (dir === "SHORT") return <TrendingDown className="w-4 h-4 text-rose-400" />;
  return <Minus className="w-4 h-4 text-slate-500" />;
}

function ProbBar({
  pLong,
  pHold,
  pShort,
}: {
  pLong: number | null;
  pHold: number | null;
  pShort: number | null;
}) {
  const pl = pLong ?? 0;
  const ph = pHold ?? 0;
  const ps = pShort ?? 0;
  const total = pl + ph + ps || 1;

  return (
    <div className="flex flex-col gap-1 w-full" data-testid="prob-bar">
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-emerald-400 w-7 text-right font-mono">{pct(pLong)}</span>
        <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
            style={{ width: `${(pl / total) * 100}%` }}
          />
        </div>
        <span className="text-[10px] text-slate-400 w-7 font-mono">LONG</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-slate-400 w-7 text-right font-mono">{pct(pHold)}</span>
        <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-slate-600 transition-all duration-500"
            style={{ width: `${(ph / total) * 100}%` }}
          />
        </div>
        <span className="text-[10px] text-slate-500 w-7 font-mono">HOLD</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-rose-400 w-7 text-right font-mono">{pct(pShort)}</span>
        <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-rose-500 to-rose-400 transition-all duration-500"
            style={{ width: `${(ps / total) * 100}%` }}
          />
        </div>
        <span className="text-[10px] text-slate-400 w-7 font-mono">SHORT</span>
      </div>
    </div>
  );
}

function RetMuBar({ retMu }: { retMu: number | null }) {
  if (retMu === null) {
    return (
      <div className="flex items-center gap-1 text-slate-600">
        <span className="text-[10px] font-mono">—</span>
      </div>
    );
  }
  const isPos = retMu >= 0;
  const magnitude = Math.min(Math.abs(retMu) / 3, 1);
  return (
    <div className="flex flex-col gap-0.5 w-full">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-slate-400 font-mono">Expected Return</span>
        <span
          className={cn(
            "text-[11px] font-mono font-semibold",
            isPos ? "text-emerald-400" : "text-rose-400"
          )}
        >
          {isPos ? "+" : ""}{retMu.toFixed(3)}R
        </span>
      </div>
      <div className="relative w-full h-2 rounded-full bg-slate-800 overflow-hidden">
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-600" />
        {isPos ? (
          <div
            className="absolute left-1/2 top-0 h-full rounded-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
            style={{ width: `${magnitude * 50}%` }}
          />
        ) : (
          <div
            className="absolute top-0 h-full rounded-full bg-gradient-to-l from-rose-500 to-rose-400 transition-all duration-500"
            style={{ width: `${magnitude * 50}%`, right: "50%" }}
          />
        )}
      </div>
    </div>
  );
}

function MfeMaeRatio({
  mfe,
  mae,
}: {
  mfe: number | null;
  mae: number | null;
}) {
  if (mfe === null || mae === null || mae === 0) {
    return <span className="text-slate-600 text-[10px] font-mono">—</span>;
  }
  const ratio = mfe / Math.max(mae, 0.1);
  const isGood = ratio >= 2;
  return (
    <div className="flex items-center gap-1.5">
      <Target className="w-3 h-3 text-slate-500" />
      <span className={cn("text-[11px] font-mono font-semibold", isGood ? "text-cyan-400" : "text-amber-400")}>
        {mfe.toFixed(2)}R /{" "}
        <span className="text-slate-400">{mae.toFixed(2)}R</span>
      </span>
      <span
        className={cn(
          "text-[10px] px-1 py-0.5 rounded font-mono",
          isGood ? "bg-cyan-900/40 text-cyan-400" : "bg-amber-900/30 text-amber-400"
        )}
      >
        {ratio.toFixed(1)}:1
      </span>
    </div>
  );
}

function HtfDots({ h1, h4, slopeOk }: { h1: number | null; h4: number | null; slopeOk: boolean | null }) {
  const dot = (val: number | null, label: string) => {
    const color =
      val === 1 ? "bg-emerald-500" : val === -1 ? "bg-rose-500" : "bg-slate-700";
    return (
      <div className="flex items-center gap-0.5">
        <div className={cn("w-2 h-2 rounded-full", color)} />
        <span className="text-[9px] text-slate-500">{label}</span>
      </div>
    );
  };
  return (
    <div className="flex items-center gap-2">
      {dot(h1, "1H")}
      {dot(h4, "4H")}
      <div className="flex items-center gap-0.5">
        <div className={cn("w-2 h-2 rounded-full", slopeOk ? "bg-violet-500" : "bg-slate-700")} />
        <span className="text-[9px] text-slate-500">Slope</span>
      </div>
    </div>
  );
}

function ScoreBar({ score, threshold }: { score: number | null; threshold: number | null }) {
  if (score === null) return <span className="text-slate-600 text-[10px] font-mono">—</span>;
  const thr = threshold ?? 0.5;
  const isAbove = score >= thr;
  const pct = Math.min(Math.max(score / (thr * 4), 0), 1) * 100;
  const thrPct = Math.min((thr / (thr * 4)) * 100, 100);
  return (
    <div className="flex flex-col gap-0.5 w-full">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-slate-400 font-mono">V5 Score</span>
        <span className={cn("text-[11px] font-mono font-semibold", isAbove ? "text-violet-400" : "text-slate-500")}>
          {score.toFixed(3)}
        </span>
      </div>
      <div className="relative w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            isAbove ? "bg-violet-500" : "bg-slate-600"
          )}
          style={{ width: `${pct}%` }}
        />
        <div
          className="absolute top-0 bottom-0 w-px bg-amber-500/60"
          style={{ left: `${thrPct}%` }}
        />
      </div>
    </div>
  );
}

function DecisionBadge({ decision, holdReason }: { decision: string | null; holdReason: string | null }) {
  if (!decision) return null;

  const isEnter = decision === "ENTER";
  const isCooldown = decision === "COOLDOWN";

  return (
    <div
      data-testid={`decision-badge-${decision}`}
      className={cn(
        "flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold border",
        isEnter
          ? "bg-emerald-900/50 text-emerald-300 border-emerald-700/50"
          : isCooldown
          ? "bg-amber-900/30 text-amber-400 border-amber-700/30"
          : "bg-slate-800 text-slate-500 border-slate-700"
      )}
    >
      {isEnter ? <Zap className="w-2.5 h-2.5" /> : isCooldown ? <Clock className="w-2.5 h-2.5" /> : <Shield className="w-2.5 h-2.5" />}
      {decision}
    </div>
  );
}

function SymbolCard({ data }: { data: SymbolIntelligence }) {
  const stale = isStale(data.cycleTs);
  const shortSym = data.symbol.replace("USDT", "");
  const isEnter = data.decision === "ENTER";
  const isBullish = data.retMuDirection === "BULLISH";
  const isBearish = data.retMuDirection === "BEARISH";

  return (
    <div
      data-testid={`symbol-card-${data.symbol}`}
      className={cn(
        "relative rounded-xl border p-3 flex flex-col gap-2.5 transition-all duration-300",
        "bg-slate-900/80 backdrop-blur-sm",
        stale
          ? "border-slate-800 opacity-60"
          : isEnter
          ? "border-emerald-700/60 shadow-[0_0_20px_rgba(16,185,129,0.08)]"
          : isBullish
          ? "border-slate-700/80"
          : isBearish
          ? "border-slate-700/80"
          : "border-slate-800"
      )}
    >
      {isEnter && (
        <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-emerald-900/10 to-transparent pointer-events-none" />
      )}

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <DirectionIcon dir={data.direction} />
          <span className="font-bold text-sm text-white font-mono tracking-wide">{shortSym}</span>
          {data.price !== null && (
            <span className="text-[10px] text-slate-400 font-mono">
              ${data.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </span>
          )}
        </div>
        <DecisionBadge decision={data.decision} holdReason={data.holdReason} />
      </div>

      <ProbBar pLong={data.pLong} pHold={data.pHold} pShort={data.pShort} />

      <RetMuBar retMu={data.retMu} />

      <MfeMaeRatio mfe={data.mfePred} mae={data.maePred} />

      <ScoreBar score={data.v5Score} threshold={data.v5Threshold} />

      <div className="flex items-center justify-between">
        <HtfDots h1={data.htfH1Trend} h4={data.htfH4Trend} slopeOk={data.slopeOk} />
        <div className="flex items-center gap-1 text-[9px] text-slate-600">
          <Clock className="w-2.5 h-2.5" />
          <span className="font-mono">{ageLabel(data.cycleTs)}</span>
        </div>
      </div>

      {data.holdReason && data.decision !== "ENTER" && (
        <div className="text-[10px] text-slate-500 border-t border-slate-800 pt-1.5 leading-tight font-mono truncate">
          {data.holdReason}
        </div>
      )}

      {stale && (
        <div className="absolute top-2 right-2">
          <div className="w-1.5 h-1.5 rounded-full bg-slate-600" title="No recent data" />
        </div>
      )}
      {!stale && !isEnter && (
        <div className="absolute top-2 right-2">
          <div className={cn("w-1.5 h-1.5 rounded-full", isBullish ? "bg-emerald-500/50" : isBearish ? "bg-rose-500/50" : "bg-slate-600")} />
        </div>
      )}
    </div>
  );
}

function MarketBiasBar({ symbols }: { symbols: Record<string, SymbolIntelligence> }) {
  const entries = Object.values(symbols).filter(s => !isStale(s.cycleTs));
  const total = entries.length || 1;
  const bullish = entries.filter(s => s.retMuDirection === "BULLISH").length;
  const bearish = entries.filter(s => s.retMuDirection === "BEARISH").length;
  const neutral = total - bullish - bearish;
  const entering = entries.filter(s => s.decision === "ENTER").length;

  const bullPct = (bullish / total) * 100;
  const bearPct = (bearish / total) * 100;
  const neutPct = (neutral / total) * 100;

  return (
    <div className="flex items-center gap-4 flex-wrap">
      <div className="flex items-center gap-2 bg-slate-900/60 border border-slate-800 rounded-xl px-4 py-2.5 flex-wrap gap-y-1">
        <span className="text-[11px] text-slate-400 font-mono">Market Bias</span>
        <div className="flex items-center gap-1 h-3 rounded-full overflow-hidden w-32">
          <div className="h-full bg-emerald-500 transition-all duration-700" style={{ width: `${bullPct}%` }} />
          <div className="h-full bg-slate-600 transition-all duration-700" style={{ width: `${neutPct}%` }} />
          <div className="h-full bg-rose-500 transition-all duration-700" style={{ width: `${bearPct}%` }} />
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <span className="text-emerald-400">{bullish} Bullish</span>
          <span className="text-slate-500">{neutral} Neutral</span>
          <span className="text-rose-400">{bearish} Bearish</span>
        </div>
      </div>
      <div className="flex items-center gap-2 bg-emerald-900/20 border border-emerald-800/30 rounded-xl px-4 py-2.5">
        <Zap className="w-3.5 h-3.5 text-emerald-400" />
        <span className="text-[11px] text-slate-400 font-mono">Active Signals</span>
        <span className="text-lg font-bold text-emerald-400 font-mono leading-none">{entering}</span>
      </div>
      <div className="flex items-center gap-2 bg-slate-900/60 border border-slate-800 rounded-xl px-4 py-2.5">
        <Eye className="w-3.5 h-3.5 text-violet-400" />
        <span className="text-[11px] text-slate-400 font-mono">Tracked</span>
        <span className="text-lg font-bold text-violet-400 font-mono leading-none">{entries.length}</span>
        <span className="text-[11px] text-slate-500 font-mono">/ {total}</span>
      </div>
    </div>
  );
}

export default function NeuralMonitor() {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);

  const { data, isLoading, error } = useQuery<NeuralIntelligenceResponse>({
    queryKey: ["/api/neural-intelligence"],
    refetchInterval: 30000,
  });

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "CYCLE_UPDATE" && msg.payload?.symbol) {
          queryClient.invalidateQueries({ queryKey: ["/api/neural-intelligence"] });
        }
      } catch {}
    };

    return () => ws.close();
  }, [queryClient]);

  const sortedSymbols = useMemo(() => {
    if (!data?.symbols) return TRADING_SYMBOLS.map(s => ({ symbol: s } as SymbolIntelligence));
    return TRADING_SYMBOLS.map(s => data.symbols[s] ?? { symbol: s } as SymbolIntelligence).sort((a, b) => {
      const aEnter = a.decision === "ENTER" ? 1 : 0;
      const bEnter = b.decision === "ENTER" ? 1 : 0;
      if (aEnter !== bEnter) return bEnter - aEnter;
      const aScore = a.v5Score ?? -999;
      const bScore = b.v5Score ?? -999;
      return bScore - aScore;
    });
  }, [data]);

  return (
    <div className="flex flex-col gap-5 min-h-full p-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-violet-900/30 border border-violet-800/40">
            <Brain className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight">Neural Market Monitor</h1>
            <p className="text-xs text-slate-400 font-mono">
              Per-symbol V5 intelligence — updated every 15m cycle
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-slate-500 font-mono">
          <Activity className="w-3.5 h-3.5" />
          {data?.updatedAt ? (
            <span>Refreshed {ageLabel(data.updatedAt)}</span>
          ) : (
            <span>Waiting for data…</span>
          )}
        </div>
      </div>

      {data?.symbols && <MarketBiasBar symbols={data.symbols} />}

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {isLoading &&
          TRADING_SYMBOLS.map((sym) => (
            <div
              key={sym}
              data-testid={`symbol-card-skeleton-${sym}`}
              className="rounded-xl border border-slate-800 bg-slate-900/50 p-3 h-52 animate-pulse"
            />
          ))}

        {error && (
          <div className="col-span-full flex items-center gap-2 text-rose-400 text-sm">
            <AlertTriangle className="w-4 h-4" />
            Failed to load neural intelligence data. GPU trainer may not be running.
          </div>
        )}

        {!isLoading &&
          sortedSymbols.map((sym) => (
            <SymbolCard key={sym.symbol} data={sym} />
          ))}
      </div>

      <div className="mt-2 border-t border-slate-800/60 pt-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-[10px] text-slate-500 font-mono">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-emerald-500" />
          <span>Bullish — model expects positive return (ret_mu {">"} 0)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-rose-500" />
          <span>Bearish — model expects negative return (ret_mu {"<"} 0)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-violet-500" />
          <span>V5 Score above threshold → signal candidate</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Target className="w-2.5 h-2.5 text-cyan-400" />
          <span>MFE/MAE ratio — model's predicted risk-reward per trade</span>
        </div>
      </div>
    </div>
  );
}
