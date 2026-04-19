import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Activity, AlertTriangle, RotateCcw, Zap } from "lucide-react";

interface V7State {
  enabled: boolean;
  notionalUsd: number;
  universe: { tradeable: string[]; probationary: string[]; disabled: string[] };
  config: {
    topPct: number; holdMinutes: number; rollingDays: number;
    minBufferSamples: number; killThresholdBps: number; killCostBps: number;
  };
  bufferSizes: Record<string, number>;
  thresholds: Record<string, number | null>;
  cumGrossBps: Record<string, number>;
  killed: Record<string, { ts: number; cumGrossBps: number; cumNetBps: number; reason: string } | null>;
  recentSnapshots: Array<{ symbol: string; ts: number; threshold: number; n: number }>;
}

interface V7Performance {
  tradeable: { live: any; ref: any; divergence: { drift_bps: number; band: string; ref_net_bps_6: number } | null };
  probationary: { live: any; ref: any; divergence: { drift_bps: number; band: string; ref_net_bps_6: number } | null };
  note: string;
}

function bpsColor(v: number): string {
  if (v > 0) return "text-emerald-400";
  if (v < 0) return "text-red-400";
  return "text-muted-foreground";
}

function bandBadge(band: string) {
  const cls =
    band === "WITHIN_BAND" ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
    : band === "ABOVE_BAND" ? "bg-sky-500/15 text-sky-400 border-sky-500/30"
    : "bg-amber-500/15 text-amber-400 border-amber-500/30";
  return <Badge variant="outline" className={cls}>{band.replace(/_/g, " ")}</Badge>;
}

export default function V7PathAPanel() {
  const { toast } = useToast();
  const [notionalInput, setNotionalInput] = useState("");

  const stateQ = useQuery<V7State>({
    queryKey: ["/api/v7/state"],
    refetchInterval: 5000,
  });
  const perfQ = useQuery<V7Performance>({
    queryKey: ["/api/v7/performance"],
    refetchInterval: 15000,
  });

  const enableMut = useMutation({
    mutationFn: async (v: boolean) => apiRequest(v ? "POST" : "POST", v ? "/api/v7/enable" : "/api/v7/disable"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v7/state"] });
      toast({ title: "V7 Path A updated" });
    },
    onError: (e: any) => toast({ title: "Toggle failed", description: e?.message, variant: "destructive" }),
  });

  const notionalMut = useMutation({
    mutationFn: async (usd: number) => apiRequest("POST", "/api/v7/notional", { usd }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v7/state"] });
      setNotionalInput("");
      toast({ title: "Notional updated" });
    },
    onError: (e: any) => toast({ title: "Update failed", description: e?.message, variant: "destructive" }),
  });

  const resumeMut = useMutation({
    mutationFn: async (sym: string) => apiRequest("POST", `/api/v7/resume/${sym}`),
    onSuccess: (_d, sym) => {
      queryClient.invalidateQueries({ queryKey: ["/api/v7/state"] });
      toast({ title: `${sym} resumed`, description: "Kill switch cleared" });
    },
    onError: (e: any) => toast({ title: "Resume failed", description: e?.message, variant: "destructive" }),
  });

  const s = stateQ.data;
  const p = perfQ.data;

  return (
    <Card data-testid="card-v7-path-a">
      <CardHeader className="pb-2 pt-3 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <Zap className="w-3.5 h-3.5 text-violet-400" />
            V7 Path A — Top {s?.config.topPct.toFixed(2) ?? "0.50"}% Selectivity Paper Engine
          </CardTitle>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground uppercase tracking-wider">Engine</span>
            <Switch
              data-testid="switch-v7-enabled"
              checked={!!s?.enabled}
              onCheckedChange={(v) => enableMut.mutate(v)}
              disabled={enableMut.isPending || stateQ.isLoading}
            />
            <span className={`w-1.5 h-1.5 rounded-full ${s?.enabled ? "bg-emerald-400 animate-pulse" : "bg-muted-foreground/30"}`} />
          </div>
        </div>
      </CardHeader>

      <CardContent className="px-4 pb-3 space-y-3">
        {/* Config row */}
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
          <span>Hold: <span className="text-foreground font-mono">{s?.config.holdMinutes}m</span></span>
          <span>Rolling: <span className="text-foreground font-mono">{s?.config.rollingDays}d</span></span>
          <span>Min samples: <span className="text-foreground font-mono">{s?.config.minBufferSamples}</span></span>
          <span>Kill@: <span className="text-foreground font-mono">{s?.config.killThresholdBps} bps net</span></span>
          <span>Notional: <span className="text-foreground font-mono">${s?.notionalUsd.toFixed(2)}</span></span>
          <span className="ml-auto inline-flex items-center gap-1">
            <Input
              data-testid="input-v7-notional"
              className="h-6 w-20 text-xs"
              placeholder="usd"
              value={notionalInput}
              onChange={(e) => setNotionalInput(e.target.value)}
            />
            <Button
              data-testid="button-v7-notional-set"
              size="sm" variant="outline" className="h-6 text-xs"
              onClick={() => {
                const v = parseFloat(notionalInput);
                if (isFinite(v) && v > 0) notionalMut.mutate(v);
              }}
              disabled={notionalMut.isPending || !notionalInput}
            >Set</Button>
          </span>
        </div>

        {/* Universe / per-symbol grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {[...(s?.universe.tradeable ?? []), ...(s?.universe.probationary ?? [])].map(sym => {
            const isProb = s?.universe.probationary.includes(sym);
            const k = s?.killed[sym];
            const buf = s?.bufferSizes[sym] ?? 0;
            const thr = s?.thresholds[sym];
            const cum = s?.cumGrossBps[sym] ?? 0;
            const warmup = !!s && buf < (s.config.minBufferSamples ?? 50);
            return (
              <div key={sym}
                className={`rounded-md border p-2 ${k ? "border-red-500/40 bg-red-500/5" : isProb ? "border-amber-500/30 bg-amber-500/5" : "border-border bg-card"}`}
                data-testid={`tile-v7-${sym}`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold">{sym}</span>
                  {isProb && <Badge variant="outline" className="text-[9px] h-4 px-1 bg-amber-500/15 text-amber-400 border-amber-500/30">PROB</Badge>}
                  {k && (
                    <Button
                      data-testid={`button-v7-resume-${sym}`}
                      size="sm" variant="outline" className="h-5 text-[10px] px-1.5 border-red-500/40 text-red-400"
                      onClick={() => resumeMut.mutate(sym)}
                      disabled={resumeMut.isPending}
                    ><RotateCcw className="w-2.5 h-2.5 mr-0.5" />Resume</Button>
                  )}
                </div>
                <div className="mt-1 text-[10px] text-muted-foreground space-y-0.5 font-mono">
                  <div>buf: <span className={warmup ? "text-amber-400" : "text-foreground"}>{buf}</span>{warmup && <span className="text-amber-400"> (warm-up)</span>}</div>
                  <div>thr: <span className="text-foreground" data-testid={`text-v7-threshold-${sym}`}>{thr != null ? thr.toExponential(3) : "—"}</span></div>
                  <div>cum gross: <span className={bpsColor(cum)} data-testid={`text-v7-cumbps-${sym}`}>{cum.toFixed(1)} bps</span></div>
                  {k && (
                    <div className="text-red-400 flex items-start gap-0.5 leading-tight pt-0.5">
                      <AlertTriangle className="w-2.5 h-2.5 mt-0.5 shrink-0" />
                      <span>killed: {k.cumNetBps.toFixed(0)} net bps</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Divergence vs back-test */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {[
            { name: "Tradeable Book (ADA/XRP/AVAX)", d: p?.tradeable, testId: "tradeable" },
            { name: "Probationary (SOL)", d: p?.probationary, testId: "probationary" },
          ].map(b => (
            <div key={b.testId} className="rounded-md border border-border p-2" data-testid={`v7-divergence-${b.testId}`}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold flex items-center gap-1">
                  <Activity className="w-3 h-3 text-muted-foreground" />{b.name}
                </span>
                {b.d?.divergence && bandBadge(b.d.divergence.band)}
              </div>
              {b.d?.live?.n ? (
                <div className="text-[11px] grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono">
                  <span className="text-muted-foreground">trades</span><span>{b.d.live.n}</span>
                  <span className="text-muted-foreground">gross/trade</span><span className={bpsColor(b.d.live.gross_mean_bps)}>{b.d.live.gross_mean_bps.toFixed(2)} bps</span>
                  <span className="text-muted-foreground">net@4 / @6 / @8</span>
                  <span>
                    <span className={bpsColor(b.d.live.net_mean_bps_4)}>{b.d.live.net_mean_bps_4.toFixed(1)}</span>
                    <span className="text-muted-foreground"> / </span>
                    <span className={bpsColor(b.d.live.net_mean_bps_6)}>{b.d.live.net_mean_bps_6.toFixed(1)}</span>
                    <span className="text-muted-foreground"> / </span>
                    <span className={bpsColor(b.d.live.net_mean_bps_8)}>{b.d.live.net_mean_bps_8.toFixed(1)}</span>
                  </span>
                  <span className="text-muted-foreground">winrate</span><span>{b.d.live.wr_pct.toFixed(1)}%</span>
                  <span className="text-muted-foreground">ref net@6</span><span className="text-foreground">{b.d.divergence?.ref_net_bps_6.toFixed(1)} bps</span>
                  <span className="text-muted-foreground">drift</span><span className={bpsColor(b.d.divergence?.drift_bps ?? 0)}>{(b.d.divergence?.drift_bps ?? 0).toFixed(1)} bps</span>
                </div>
              ) : (
                <div className="text-[11px] text-muted-foreground">No closed trades yet.</div>
              )}
            </div>
          ))}
        </div>

        <div className="text-[10px] text-muted-foreground italic">
          {p?.note ?? "Net@6bps assumes 6 bps round-trip cost; live cost should be verified separately."}
          {" "}Threshold uses live |returnH2| rolling 30d 99.5th percentile per symbol. ETH: disabled.
        </div>
      </CardContent>
    </Card>
  );
}
