import { useState, useEffect, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { format } from "date-fns";
import {
  ArrowUpRight,
  ArrowDownRight,
  Save,
  Clock,
  DollarSign,
  TrendingUp,
  TrendingDown,
  FileText,
  Activity,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { apiRequest } from "@/lib/queryClient";
import { queryClient } from "@/lib/queryClient";
import TradeReplayChart from "@/components/trade-replay-chart";
import type { ReplayData } from "@/components/trade-replay-chart";

interface Trade {
  id: number;
  symbol: string;
  side: string;
  entryTime: number;
  entryPrice: number;
  exitTime: number | null;
  exitPrice: number | null;
  stopLoss: number | null;
  takeProfit: number | null;
  initialSl: number | null;
  sizePct: number | null;
  pEnter: number | null;
  costsBps: number | null;
  outcome: string | null;
  grossR: number | null;
  costR: number | null;
  netR: number | null;
  sizedR: number | null;
  status: string;
  reasons: string[] | null;
  pnlUsd: number | null;
  pnlUsdGross: number | null;
  pnlUsdCost: number | null;
  riskUsdUsed: number | null;
  equitySnapshotUsd: number | null;
  barsHeld: number | null;
  leverage: number | null;
  modelVersion: string | null;
  notes: string | null;
  createdAt: number;
  lane: string | null;
  htfScore: number | null;
  laneThresholdUsed: number | null;
  laneSizeMult: number | null;
  exitReason: string | null;
  laneHorizon: number | null;
  maxFavorableR: number | null;
  maxAdverseR: number | null;
  timeExit: boolean | null;
  breakevenMoved: boolean | null;
  trailUpdates: number | null;
  tmActions: Array<{ ts: number; action: string; reason: string; price?: number; sl?: number; ur?: number }> | null;
  policy: string | null;
}

interface TradeEvent {
  id: number;
  ts: string;
  type: string;
  message: string;
  data?: Record<string, unknown>;
}


interface TradeDetailModalProps {
  tradeId: number | null;
  open: boolean;
  onClose: () => void;
}

function fmtR(val: number | null | undefined): string {
  if (val === null || val === undefined) return "N/A";
  return `${val >= 0 ? "+" : ""}${val.toFixed(2)}R`;
}

function fmtR0(val: number | null | undefined): string {
  const v = val ?? 0;
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}R`;
}

function fmtUsd(val: number | null | undefined): string {
  if (val === null || val === undefined) return "N/A";
  return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtUsd0(val: number | null | undefined): string {
  const v = val ?? 0;
  return `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtPrice(val: number | null | undefined): string {
  if (val === null || val === undefined) return "N/A";
  return val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 });
}

function fmtTs(ts: string | number | null | undefined): string {
  if (!ts) return "N/A";
  try {
    return format(new Date(typeof ts === "string" ? ts : ts), "MMM dd, yyyy HH:mm:ss");
  } catch {
    return "N/A";
  }
}

function outcomeBadgeVariant(outcome: string | null): "default" | "secondary" | "destructive" | "outline" {
  if (!outcome) return "outline";
  const upper = outcome.toUpperCase();
  if (upper === "TP" || upper === "WIN") return "default";
  if (upper === "SL" || upper === "LOSS") return "destructive";
  return "secondary";
}

function outcomeLabel(outcome: string | null): string {
  if (!outcome) return "OPEN";
  const upper = outcome.toUpperCase();
  if (upper === "TP") return "Take Profit";
  if (upper === "SL") return "Stop Loss";
  if (upper === "EXPIRE") return "Expired";
  return outcome;
}

export default function TradeDetailModal({ tradeId, open, onClose }: TradeDetailModalProps) {
  const [notesValue, setNotesValue] = useState("");
  const [notesSaved, setNotesSaved] = useState(false);

  const { data: tradeData, isLoading: tradeLoading } = useQuery<{ trade: Trade; events: TradeEvent[] }>({
    queryKey: ["/api/pro/trades", String(tradeId)],
    enabled: tradeId !== null && open,
  });

  const { data: replayData, isLoading: replayLoading } = useQuery<ReplayData>({
    queryKey: ["/api/pro/trades", String(tradeId), "replay"],
    queryFn: async () => {
      const res = await fetch(`/api/pro/trades/${tradeId}/replay?preBars=50&postBars=10`);
      if (!res.ok) throw new Error("Failed to fetch replay data");
      return res.json();
    },
    enabled: tradeId !== null && open,
  });

  const trade = tradeData?.trade;
  const events = tradeData?.events ?? [];

  useEffect(() => {
    if (trade?.notes !== undefined) {
      setNotesValue(trade.notes ?? "");
      setNotesSaved(false);
    }
  }, [trade?.notes]);

  const notesMutation = useMutation({
    mutationFn: async (notes: string) => {
      await apiRequest("PATCH", `/api/pro/trades/${tradeId}/notes`, { notes });
    },
    onSuccess: () => {
      setNotesSaved(true);
      queryClient.invalidateQueries({ queryKey: ["/api/pro/trades", String(tradeId)] });
      setTimeout(() => setNotesSaved(false), 2000);
    },
  });


  const htfFlags = useMemo(() => {
    if (!trade?.reasons) return [];
    return trade.reasons.filter(
      (r) =>
        r.toLowerCase().includes("htf") ||
        r.toLowerCase().includes("h1") ||
        r.toLowerCase().includes("h4") ||
        r.toLowerCase().includes("alignment") ||
        r.toLowerCase().includes("slope") ||
        r.toLowerCase().includes("range")
    );
  }, [trade?.reasons]);

  const gateDecisions = useMemo(() => {
    if (!trade?.reasons) return [];
    return trade.reasons.filter(
      (r) =>
        r.toLowerCase().includes("gate") ||
        r.toLowerCase().includes("threshold") ||
        r.toLowerCase().includes("cooldown") ||
        r.toLowerCase().includes("hold")
    );
  }, [trade?.reasons]);

  const isLong = trade?.side?.toUpperCase() === "LONG" || trade?.side?.toUpperCase() === "BUY";

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent
        className="max-w-3xl max-h-[90vh] overflow-y-auto rounded-md"
        data-testid="trade-detail-modal"
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 flex-wrap" data-testid="trade-detail-title">
            {tradeLoading ? (
              <Skeleton className="h-6 w-48" />
            ) : (
              <>
                <span className="text-lg font-semibold" data-testid="text-trade-symbol">
                  {trade?.symbol ?? "—"}
                </span>
                <Badge
                  variant={isLong ? "default" : "destructive"}
                  className={`text-xs ${isLong ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}`}
                  data-testid="badge-trade-side"
                >
                  {isLong ? (
                    <ArrowUpRight className="h-3 w-3 mr-1" />
                  ) : (
                    <ArrowDownRight className="h-3 w-3 mr-1" />
                  )}
                  {trade?.side?.toUpperCase() ?? "—"}
                </Badge>
                <Badge
                  variant={outcomeBadgeVariant(trade?.outcome ?? null)}
                  className="text-xs"
                  data-testid="badge-trade-outcome"
                >
                  {outcomeLabel(trade?.outcome ?? null)}
                </Badge>
                {trade?.lane && (
                  <Badge
                    variant="outline"
                    className={`text-xs ${trade.lane === "CORE" ? "text-blue-400 border-blue-400/30" : trade.lane === "FLOW" ? "text-amber-400 border-amber-400/30" : trade.lane === "SCALP" ? "text-purple-400 border-purple-400/30" : ""}`}
                    data-testid="badge-trade-lane"
                  >
                    {trade.lane}
                  </Badge>
                )}
                {trade?.htfScore != null && (
                  <Badge variant="secondary" className="text-xs" data-testid="badge-htf-score">
                    HTF {trade.htfScore}/3
                  </Badge>
                )}
                {trade?.modelVersion && (
                  <span className="text-xs text-muted-foreground" data-testid="text-model-version">
                    v{trade.modelVersion}
                  </span>
                )}
              </>
            )}
          </DialogTitle>
          <DialogDescription className="sr-only">
            Trade detail view for trade {tradeId}
          </DialogDescription>
        </DialogHeader>

        {tradeLoading ? (
          <div className="space-y-4">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-48 w-full" />
          </div>
        ) : trade ? (
          <div className="space-y-4">
            <Card className="rounded-md" data-testid="card-pnl-block">
              <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-1.5">
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                  PnL Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Net R</p>
                    <p
                      className={`text-sm font-semibold ${trade.netR !== null ? (trade.netR >= 0 ? "text-emerald-400" : "text-red-400") : ""}`}
                      data-testid="text-net-r"
                    >
                      {fmtR(trade.netR)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Gross R</p>
                    <p className="text-sm font-semibold" data-testid="text-gross-r">
                      {fmtR(trade.grossR)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Cost R</p>
                    <p className="text-sm font-semibold" data-testid="text-cost-r">
                      {fmtR0(trade.costR)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Net PnL ($)</p>
                    <p
                      className={`text-sm font-semibold ${(trade.pnlUsd ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}
                      data-testid="text-net-pnl-usd"
                    >
                      {fmtUsd0(trade.pnlUsd)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Cost ($)</p>
                    <p className="text-sm font-semibold" data-testid="text-cost-usd">
                      {fmtUsd0(trade.pnlUsdCost)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Risk ($) Used</p>
                    <p className="text-sm font-semibold" data-testid="text-risk-usd">
                      {fmtUsd0(trade.riskUsdUsed)}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="rounded-md" data-testid="card-entry-explanation">
              <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-1.5">
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  Entry Explanation
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">p_enter</p>
                    <p className="text-sm font-semibold" data-testid="text-p-enter">
                      {trade.pEnter !== null ? `${(trade.pEnter * 100).toFixed(1)}%` : "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Direction</p>
                    <p className="text-sm font-semibold" data-testid="text-direction">
                      {trade.side?.toUpperCase() ?? "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Lane</p>
                    <p className="text-sm font-semibold" data-testid="text-lane">
                      {trade.lane ?? trade.policy ?? "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">HTF Score</p>
                    <p className="text-sm font-semibold" data-testid="text-htf-score">
                      {trade.htfScore != null ? `${trade.htfScore}/3` : "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Lane Threshold</p>
                    <p className="text-sm font-semibold" data-testid="text-lane-threshold">
                      {trade.laneThresholdUsed != null ? `${(trade.laneThresholdUsed * 100).toFixed(1)}%` : "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Size Mult</p>
                    <p className="text-sm font-semibold" data-testid="text-size-mult">
                      {trade.laneSizeMult != null ? `${trade.laneSizeMult.toFixed(2)}x` : "N/A"}
                    </p>
                  </div>
                </div>

                {htfFlags.length > 0 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1.5">HTF Alignment Flags</p>
                    <div className="flex flex-wrap gap-1.5">
                      {htfFlags.map((flag, idx) => (
                        <Badge key={idx} variant="secondary" className="text-xs">
                          {flag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {gateDecisions.length > 0 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1.5">Gate Decisions</p>
                    <div className="flex flex-wrap gap-1.5">
                      {gateDecisions.map((g, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {g}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {trade.reasons && trade.reasons.length > 0 && htfFlags.length === 0 && gateDecisions.length === 0 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1.5">Reasons</p>
                    <div className="flex flex-wrap gap-1.5">
                      {trade.reasons.map((r, idx) => (
                        <Badge key={idx} variant="secondary" className="text-xs">
                          {r}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="rounded-md" data-testid="card-exit-explanation">
              <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-1.5">
                  <TrendingDown className="h-4 w-4 text-muted-foreground" />
                  Exit Explanation
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Exit Reason</p>
                    <p className="text-sm font-semibold" data-testid="text-exit-reason">
                      {trade.exitReason || outcomeLabel(trade.outcome)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Bars Held</p>
                    <p className="text-sm font-semibold" data-testid="text-bars-held">
                      {`${trade.barsHeld ?? 0} bars`}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Horizon</p>
                    <p className="text-sm font-semibold" data-testid="text-horizon">
                      {trade.laneHorizon != null ? `${trade.laneHorizon} bars` : "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Exit Price</p>
                    <p className="text-sm font-semibold" data-testid="text-exit-price">
                      {fmtPrice(trade.exitPrice)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Entry Price</p>
                    <p className="text-sm font-semibold" data-testid="text-entry-price">
                      {fmtPrice(trade.entryPrice)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Stop Loss</p>
                    <p className="text-sm font-semibold text-red-400" data-testid="text-stop-loss">
                      {fmtPrice(trade.stopLoss)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Initial SL</p>
                    <p className="text-sm font-semibold text-red-400/70" data-testid="text-initial-sl">
                      {fmtPrice(trade.initialSl ?? trade.stopLoss)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Take Profit</p>
                    <p className="text-sm font-semibold text-emerald-400" data-testid="text-take-profit">
                      {fmtPrice(trade.takeProfit)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">MFE (Best R)</p>
                    <p className={`text-sm font-semibold ${(trade.maxFavorableR ?? 0) > 0 ? "text-emerald-400" : ""}`} data-testid="text-mfe">
                      {trade.maxFavorableR != null ? fmtR(trade.maxFavorableR) : "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">MAE (Worst R)</p>
                    <p className={`text-sm font-semibold ${(trade.maxAdverseR ?? 0) < 0 ? "text-red-400" : ""}`} data-testid="text-mae">
                      {trade.maxAdverseR != null ? fmtR(trade.maxAdverseR) : "N/A"}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {trade.breakevenMoved && (
                    <Badge variant="secondary" className="text-xs" data-testid="badge-breakeven">
                      BE Moved
                    </Badge>
                  )}
                  {trade.timeExit && (
                    <Badge variant="secondary" className="text-xs" data-testid="badge-time-exit">
                      Time Exit
                    </Badge>
                  )}
                  {(trade.trailUpdates ?? 0) > 0 && (
                    <Badge variant="secondary" className="text-xs" data-testid="badge-trail-updates">
                      Trail x{trade.trailUpdates}
                    </Badge>
                  )}
                </div>
                {trade.exitPrice != null && trade.entryPrice != null && (
                  (() => {
                    const isl = trade.initialSl ?? trade.stopLoss;
                    const origRisk = isl != null ? Math.abs(trade.entryPrice - isl) : 0;
                    const isLongDir = trade.side?.toUpperCase() === "LONG" || trade.side?.toUpperCase() === "BUY";
                    const calcR = origRisk > 0
                      ? (isLongDir
                          ? (trade.exitPrice - trade.entryPrice) / origRisk
                          : (trade.entryPrice - trade.exitPrice) / origRisk)
                      : 0;
                    const match = trade.grossR != null ? Math.abs(calcR - trade.grossR) <= 0.01 : true;
                    return (
                      <p className={`text-xs mt-1 ${match ? "text-muted-foreground" : "text-amber-400"}`} data-testid="text-pnl-check">
                        PnL check: (exit-entry)/orig_risk = {calcR >= 0 ? "+" : ""}{calcR.toFixed(2)}R
                        {trade.grossR != null && !match && ` (stored: ${trade.grossR >= 0 ? "+" : ""}${trade.grossR.toFixed(2)}R)`}
                      </p>
                    );
                  })()
                )}
              </CardContent>
            </Card>

            <TradeReplayChart
              data={replayData}
              isLoading={replayLoading}
              tmActions={trade.tmActions ?? undefined}
            />

            <Card className="rounded-md" data-testid="card-notes">
              <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-1.5">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  Notes
                </CardTitle>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => notesMutation.mutate(notesValue)}
                  disabled={notesMutation.isPending}
                  data-testid="button-save-notes"
                >
                  <Save className="h-3.5 w-3.5 mr-1.5" />
                  {notesMutation.isPending ? "Saving..." : notesSaved ? "Saved" : "Save"}
                </Button>
              </CardHeader>
              <CardContent>
                <Textarea
                  value={notesValue}
                  onChange={(e) => setNotesValue(e.target.value)}
                  placeholder="Add trade notes..."
                  className="resize-none text-sm min-h-[80px]"
                  data-testid="textarea-notes"
                />
              </CardContent>
            </Card>

            {trade.tmActions && trade.tmActions.length > 0 && (
              <Card className="rounded-md" data-testid="card-tm-timeline">
                <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium flex items-center gap-1.5">
                    <Activity className="h-4 w-4 text-muted-foreground" />
                    Trade Manager Timeline
                  </CardTitle>
                  <Badge variant="secondary" className="text-xs" data-testid="badge-tm-action-count">
                    {trade.tmActions.length} action{trade.tmActions.length !== 1 ? "s" : ""}
                  </Badge>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {trade.tmActions.map((act, idx) => {
                      const actionColor =
                        act.action === "CLOSE_FULL" ? "text-red-400 border-red-400/30" :
                        act.action === "TRAIL_SL" ? "text-blue-400 border-blue-400/30" :
                        act.action === "MOVE_SL" ? "text-emerald-400 border-emerald-400/30" :
                        "";
                      return (
                        <div
                          key={idx}
                          className="flex items-start gap-3 p-2 rounded-md bg-muted/30"
                          data-testid={`tm-action-row-${idx}`}
                        >
                          <div className="flex items-center gap-1.5 text-xs text-muted-foreground shrink-0">
                            <Clock className="h-3 w-3" />
                            {fmtTs(act.ts)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <Badge variant="outline" className={`text-xs ${actionColor}`}>
                              {act.action}
                            </Badge>
                            <p className="text-xs text-muted-foreground mt-1 break-words">
                              {act.reason}
                            </p>
                            <div className="flex flex-wrap gap-3 mt-1">
                              {act.price != null && (
                                <span className="text-xs text-muted-foreground">
                                  Price: <span className="font-mono">{act.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                                </span>
                              )}
                              {act.sl != null && (
                                <span className="text-xs text-muted-foreground">
                                  SL: <span className="font-mono text-red-400">{act.sl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                                </span>
                              )}
                              {act.ur != null && (
                                <span className="text-xs text-muted-foreground">
                                  uR: <span className={`font-mono ${act.ur >= 0 ? "text-emerald-400" : "text-red-400"}`}>{act.ur >= 0 ? "+" : ""}{act.ur.toFixed(3)}</span>
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            )}

            <Card className="rounded-md" data-testid="card-events-timeline">
              <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-1.5">
                  <Activity className="h-4 w-4 text-muted-foreground" />
                  Trade Events
                </CardTitle>
                <Badge variant="secondary" className="text-xs">
                  {events.length} event{events.length !== 1 ? "s" : ""}
                </Badge>
              </CardHeader>
              <CardContent>
                {events.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    No events recorded
                  </p>
                ) : (
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {events.map((evt) => (
                      <div
                        key={evt.id}
                        className="flex items-start gap-3 p-2 rounded-md bg-muted/30"
                        data-testid={`event-row-${evt.id}`}
                      >
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground shrink-0">
                          <Clock className="h-3 w-3" />
                          {fmtTs(evt.ts)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <Badge variant="outline" className="text-xs mb-1">
                            {evt.type}
                          </Badge>
                          <p className="text-xs text-muted-foreground break-words">
                            {evt.message}
                          </p>
                          {evt.data && Object.keys(evt.data).length > 0 && (
                            <pre className="text-xs text-muted-foreground mt-1 bg-muted/50 p-1.5 rounded-md overflow-x-auto">
                              {JSON.stringify(evt.data, null, 2)}
                            </pre>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-8">
            Trade not found
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
