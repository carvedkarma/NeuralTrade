import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ClipboardList, ChevronDown, ChevronRight, TrendingUp, TrendingDown, Clock, Activity, Target, Shield, Zap } from "lucide-react";

interface TradeRecord {
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
  maxAdverseR?: number | null;
  initialSl?: number | null;
  takeProfit?: number | null;
  v5Score?: number | null;
  trailActive?: number | null;
  regime?: string | null;
  signalConfidence?: number | null;
  signalEdge?: number | null;
  fillType?: string | null;
}

function fmt(v: number, digits = 2): string {
  return v.toFixed(digits);
}

function fmtUsd(v: number): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2 }).format(v);
}

function fmtTs(ts: number): string {
  return new Date(ts).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function fmtDur(ms: number): string {
  if (ms <= 0) return "-";
  const m = Math.floor(ms / 60000);
  const h = Math.floor(m / 60);
  const d = Math.floor(h / 24);
  if (d > 0) return `${d}d ${h % 24}h`;
  if (h > 0) return `${h}h ${m % 60}m`;
  return `${m}m`;
}

function rColor(r: number): string {
  return r >= 0 ? "text-emerald-400" : "text-red-400";
}

function pnlColor(v: number): string {
  return v >= 0 ? "text-emerald-400" : "text-red-400";
}

function exitBadge(reason: string): { label: string; cls: string } {
  if (!reason) return { label: "—", cls: "text-muted-foreground border-muted-foreground/30" };
  const r = reason.toUpperCase();
  if (r === "TP1" || r === "TP2") return { label: reason, cls: "text-emerald-400 border-emerald-400/40" };
  if (r === "SL") return { label: "SL", cls: "text-red-400 border-red-400/40" };
  if (r.startsWith("TRAIL")) return { label: reason, cls: "text-cyan-400 border-cyan-400/40" };
  if (r.includes("MFE")) return { label: reason, cls: "text-violet-400 border-violet-400/40" };
  if (r.includes("NEURAL")) return { label: reason, cls: "text-blue-400 border-blue-400/40" };
  if (r === "TIME") return { label: "Time", cls: "text-amber-400 border-amber-400/40" };
  if (r === "FLIP") return { label: "Flip", cls: "text-orange-400 border-orange-400/40" };
  return { label: reason, cls: "text-muted-foreground border-muted-foreground/30" };
}

function DetailRow({ label, value, className = "" }: { label: string; value: React.ReactNode; className?: string }) {
  return (
    <div className="flex items-center justify-between py-1 border-b border-white/5 last:border-0">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className={`text-xs font-medium number-mono ${className}`}>{value}</span>
    </div>
  );
}

function TradeDetailPanel({ trade }: { trade: TradeRecord }) {
  const dur = trade.exitTs && trade.entryTs ? trade.exitTs - trade.entryTs : 0;
  const slDist = trade.initialSl != null ? Math.abs(trade.entryPrice - trade.initialSl) : null;
  const tpDist = trade.takeProfit != null ? Math.abs(trade.takeProfit - trade.entryPrice) : null;
  const rr = slDist && tpDist && slDist > 0 ? (tpDist / slDist) : null;
  const slippage = Math.abs(trade.exitPrice - (trade.exitReason === "SL" && trade.initialSl != null ? trade.initialSl : trade.takeProfit ?? trade.exitPrice));
  const fees = trade.riskUsdt != null && trade.costR != null ? trade.riskUsdt * (trade.costR ?? 0) : null;
  const pricePrecision = trade.symbol.includes("BTC") ? 1 : trade.symbol.includes("ETH") ? 2 : trade.entryPrice < 1 ? 6 : 4;

  const mfeUsd = trade.riskUsdt != null && trade.maxFavorableR != null ? trade.riskUsdt * trade.maxFavorableR : null;
  const maeUsd = trade.riskUsdt != null && trade.maxAdverseR != null ? trade.riskUsdt * trade.maxAdverseR : null;

  return (
    <div className="mt-3 pt-3 border-t border-white/10 grid grid-cols-1 sm:grid-cols-3 gap-x-6 gap-y-0">
      {/* Column 1: Price levels */}
      <div>
        <p className="text-[10px] uppercase tracking-widest text-muted-foreground/60 mb-1 flex items-center gap-1">
          <Target className="w-3 h-3" /> Price Levels
        </p>
        <DetailRow label="Entry" value={trade.entryPrice.toFixed(pricePrecision)} />
        <DetailRow label="Exit" value={trade.exitPrice.toFixed(pricePrecision)} className={pnlColor(trade.side === "LONG" ? trade.exitPrice - trade.entryPrice : trade.entryPrice - trade.exitPrice)} />
        <DetailRow
          label="Initial SL"
          value={trade.initialSl != null ? trade.initialSl.toFixed(pricePrecision) : "—"}
          className="text-red-400/80"
        />
        <DetailRow
          label="Take Profit"
          value={trade.takeProfit != null ? trade.takeProfit.toFixed(pricePrecision) : "—"}
          className="text-emerald-400/80"
        />
        {rr != null && (
          <DetailRow label="R:R Ratio" value={`1:${rr.toFixed(2)}`} className="text-cyan-400" />
        )}
        {slDist != null && (
          <DetailRow label="SL Distance" value={slDist.toFixed(pricePrecision)} />
        )}
      </div>

      {/* Column 2: P&L breakdown */}
      <div>
        <p className="text-[10px] uppercase tracking-widest text-muted-foreground/60 mb-1 flex items-center gap-1">
          <TrendingUp className="w-3 h-3" /> P&L Breakdown
        </p>
        <DetailRow label="Net P&L" value={fmtUsd(trade.pnlUsdt ?? 0)} className={pnlColor(trade.pnlUsdt ?? 0)} />
        <DetailRow label="Gross R" value={`${(trade.grossR ?? 0) >= 0 ? "+" : ""}${fmt(trade.grossR ?? 0)}R`} className={rColor(trade.grossR ?? 0)} />
        <DetailRow label="Net R" value={`${(trade.netR ?? 0) >= 0 ? "+" : ""}${fmt(trade.netR ?? 0)}R`} className={rColor(trade.netR ?? 0)} />
        <DetailRow label="Fees (R)" value={`-${fmt(trade.costR ?? 0)}R`} className="text-amber-400/70" />
        {fees != null && (
          <DetailRow label="Fees (USD)" value={`-${fmtUsd(fees)}`} className="text-amber-400/70" />
        )}
        <DetailRow label="Risk (USD)" value={fmtUsd(trade.riskUsdt ?? 0)} />
        <DetailRow label="MFE (R)" value={trade.maxFavorableR != null ? `+${fmt(trade.maxFavorableR)}R` : "—"} className="text-emerald-400/80" />
        {mfeUsd != null && (
          <DetailRow label="MFE (USD)" value={`+${fmtUsd(mfeUsd)}`} className="text-emerald-400/60" />
        )}
        {trade.maxAdverseR != null && (
          <DetailRow label="MAE (R)" value={`-${fmt(trade.maxAdverseR)}R`} className="text-red-400/70" />
        )}
        {maeUsd != null && (
          <DetailRow label="MAE (USD)" value={`-${fmtUsd(maeUsd)}`} className="text-red-400/60" />
        )}
      </div>

      {/* Column 3: Context & signal */}
      <div>
        <p className="text-[10px] uppercase tracking-widest text-muted-foreground/60 mb-1 flex items-center gap-1">
          <Zap className="w-3 h-3" /> Signal & Context
        </p>
        <DetailRow label="Entry Time" value={fmtTs(trade.entryTs)} />
        <DetailRow label="Exit Time" value={trade.exitTs ? fmtTs(trade.exitTs) : "—"} />
        <DetailRow label="Duration" value={fmtDur(dur)} />
        <DetailRow label="Bars Held" value={trade.barsHeld != null ? `${trade.barsHeld} bars (15m)` : "—"} />
        {trade.v5Score != null && (
          <DetailRow label="V5 Score" value={`${(trade.v5Score * 100).toFixed(1)}%`} className="text-violet-400" />
        )}
        {trade.signalConfidence != null && (
          <DetailRow label="Confidence" value={`${(trade.signalConfidence * 100).toFixed(1)}%`} className="text-blue-400" />
        )}
        {trade.signalEdge != null && (
          <DetailRow label="Edge" value={`${(trade.signalEdge * 100).toFixed(2)}%`} />
        )}
        {trade.regime != null && (
          <DetailRow label="Regime" value={trade.regime} className={trade.regime === "TRENDING" ? "text-emerald-400" : trade.regime?.includes("CHOP") ? "text-amber-400" : ""} />
        )}
        {trade.trailActive ? (
          <DetailRow label="Trailing SL" value="Active" className="text-cyan-400" />
        ) : null}
        <DetailRow label="Position ID" value={`#${trade.positionId}`} className="text-muted-foreground/60" />
      </div>
    </div>
  );
}

export default function TradeHistory() {
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const { data: tradeHistory, isLoading } = useQuery<TradeRecord[]>({
    queryKey: ["/api/paper/trade-history", "?limit=500"],
    refetchInterval: 30000,
  });

  const sorted = tradeHistory
    ? [...tradeHistory].sort((a, b) => (b.exitTs ?? 0) - (a.exitTs ?? 0))
    : [];

  const totalR = sorted.reduce((s, t) => s + (t.netR ?? 0), 0);
  const totalPnl = sorted.reduce((s, t) => s + (t.pnlUsdt ?? 0), 0);
  const wins = sorted.filter((t) => (t.netR ?? 0) > 0);
  const losses = sorted.filter((t) => (t.netR ?? 0) <= 0);
  const winRate = sorted.length > 0 ? ((wins.length / sorted.length) * 100).toFixed(1) : "0.0";
  const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + (t.netR ?? 0), 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((s, t) => s + (t.netR ?? 0), 0) / losses.length : 0;

  const toggle = (id: number) => setExpandedId(prev => prev === id ? null : id);

  return (
    <div className="p-4 space-y-4" data-testid="trade-history-page">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <ClipboardList className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold">Trade History</h2>
          {sorted.length > 0 && (
            <Badge variant="outline" className="text-[10px] text-muted-foreground" data-testid="badge-trade-count">
              {sorted.length} trades
            </Badge>
          )}
        </div>
        {sorted.length > 0 && (
          <div className="flex items-center gap-4 text-sm flex-wrap">
            <span className="text-muted-foreground">
              WR: <span className="font-semibold text-foreground" data-testid="text-win-rate">{winRate}%</span>
            </span>
            <span className="text-muted-foreground">
              Avg W: <span className={`font-semibold number-mono ${rColor(avgWin)}`}>+{fmt(avgWin)}R</span>
            </span>
            <span className="text-muted-foreground">
              Avg L: <span className={`font-semibold number-mono text-red-400`}>{fmt(avgLoss)}R</span>
            </span>
            <span className="text-muted-foreground">
              Total: <span className={`font-semibold number-mono ${rColor(totalR)}`} data-testid="text-total-r">
                {totalR >= 0 ? "+" : ""}{fmt(totalR)}R
              </span>
            </span>
            <span className={`font-semibold number-mono ${pnlColor(totalPnl)}`}>
              {totalPnl >= 0 ? "+" : ""}{fmtUsd(totalPnl)}
            </span>
          </div>
        )}
      </div>

      {/* Trade list */}
      {isLoading ? (
        <p className="text-sm text-muted-foreground text-center py-16">Loading trades...</p>
      ) : sorted.length === 0 ? (
        <p className="text-sm text-muted-foreground text-center py-16" data-testid="text-no-trades">
          No completed trades yet
        </p>
      ) : (
        <div className="space-y-1">
          {sorted.map((trade, i) => {
            const pnl = trade.netR ?? 0;
            const isExpanded = expandedId === trade.id;
            const badge = exitBadge(trade.exitReason);
            const dur = trade.exitTs && trade.entryTs ? trade.exitTs - trade.entryTs : 0;
            const pricePrecision = trade.symbol.includes("BTC") ? 1 : trade.symbol.includes("ETH") ? 2 : trade.entryPrice < 1 ? 6 : 4;

            return (
              <Card
                key={trade.id ?? i}
                className={`cursor-pointer transition-all ${isExpanded ? "border-cyan-500/30 bg-cyan-950/10" : "hover:border-white/20"}`}
                onClick={() => toggle(trade.id ?? i)}
                data-testid={`row-trade-history-${i}`}
              >
                <CardContent className="p-3">
                  {/* Summary row */}
                  <div className="flex items-center gap-2 flex-wrap">
                    {/* Expand icon */}
                    <span className="text-muted-foreground/40 shrink-0">
                      {isExpanded
                        ? <ChevronDown className="w-3.5 h-3.5" />
                        : <ChevronRight className="w-3.5 h-3.5" />}
                    </span>

                    {/* Side badge */}
                    <Badge
                      variant="outline"
                      className={`text-[10px] px-1.5 shrink-0 ${trade.side === "LONG" ? "text-emerald-400 border-emerald-400/30" : "text-red-400 border-red-400/30"}`}
                    >
                      {trade.side === "LONG"
                        ? <TrendingUp className="w-2.5 h-2.5 inline mr-0.5" />
                        : <TrendingDown className="w-2.5 h-2.5 inline mr-0.5" />}
                      {trade.side}
                    </Badge>

                    {/* Symbol */}
                    <span className="font-semibold text-sm min-w-[80px]">{trade.symbol}</span>

                    {/* Entry → Exit */}
                    <span className="number-mono text-sm text-muted-foreground">
                      {trade.entryPrice.toFixed(pricePrecision)}
                      <span className="text-muted-foreground/40 mx-1">→</span>
                      <span className={pnlColor(trade.side === "LONG" ? trade.exitPrice - trade.entryPrice : trade.entryPrice - trade.exitPrice)}>
                        {(trade.exitPrice ?? 0).toFixed(pricePrecision)}
                      </span>
                    </span>

                    {/* Net R */}
                    <span className={`number-mono font-bold text-sm min-w-[52px] ${rColor(pnl)}`}>
                      {pnl >= 0 ? "+" : ""}{fmt(pnl)}R
                    </span>

                    {/* P&L USD */}
                    <span className={`number-mono text-sm ${pnlColor(trade.pnlUsdt ?? 0)}`}>
                      {(trade.pnlUsdt ?? 0) >= 0 ? "+" : ""}{fmtUsd(trade.pnlUsdt ?? 0)}
                    </span>

                    <div className="flex-1" />

                    {/* Exit reason */}
                    <Badge variant="outline" className={`text-[10px] px-1.5 shrink-0 ${badge.cls}`}>
                      {badge.label}
                    </Badge>

                    {/* MFE */}
                    {trade.maxFavorableR != null && trade.maxFavorableR > 0 && (
                      <span className="text-[10px] text-emerald-400/60 number-mono shrink-0">
                        MFE +{fmt(trade.maxFavorableR)}R
                      </span>
                    )}

                    {/* V5 score */}
                    {trade.v5Score != null && (
                      <span className="text-[10px] text-violet-400/70 number-mono shrink-0">
                        V5 {(trade.v5Score * 100).toFixed(0)}%
                      </span>
                    )}

                    {/* Trail indicator */}
                    {trade.trailActive ? (
                      <span className="text-[10px] text-cyan-400/60 shrink-0">⬡ Trail</span>
                    ) : null}

                    {/* Duration */}
                    <span className="text-[10px] text-muted-foreground/50 flex items-center gap-0.5 shrink-0">
                      <Clock className="w-2.5 h-2.5" />
                      {dur > 0 ? fmtDur(dur) : "-"}
                    </span>

                    {/* Date */}
                    <span className="text-[10px] text-muted-foreground/50 whitespace-nowrap shrink-0">
                      {trade.exitTs ? fmtTs(trade.exitTs) : "-"}
                    </span>
                  </div>

                  {/* Expanded detail panel */}
                  {isExpanded && <TradeDetailPanel trade={trade} />}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
