import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Brain,
  Target,
  Shield,
  Clock,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  CheckCircle2,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { format } from "date-fns";

interface LiveTradeRecord {
  id: number;
  symbol: string;
  side: string;
  entryTime: number;
  entryPrice: number;
  exitTime: number | null;
  exitPrice: number | null;
  stopLoss: number | null;
  takeProfit: number | null;
  sizePct: number | null;
  pEnter: number | null;
  costsBps: number | null;
  outcome: string | null;
  grossR: number | null;
  netR: number | null;
  sizedR: number | null;
  status: string;
  reasons: string[] | null;
  createdAt: number;
}

interface ModelLearningStatsEntry {
  id: number;
  symbol: string;
  modelVersion: string;
  trainedUntilTs: number | null;
  trainingSamples: number | null;
  valPrAuc: number | null;
  valPrecision: number | null;
  valRecall: number | null;
  valF1: number | null;
  bestPolicyThreshold: number | null;
  bestPolicyCooldown: number | null;
  bestPolicyTpMult: number | null;
  bestPolicySlMult: number | null;
  pfNet: number | null;
  eNet: number | null;
  tradesPerDay: number | null;
  profitableRegimes: number | null;
  totalRegimes: number | null;
  promoted: boolean | null;
  promotionReason: string | null;
  trend7d: string | null;
  prevPfNet: number | null;
  prevENet: number | null;
  prevTradesPerDay: number | null;
  createdAt: number;
}

interface LiveCycleLog {
  id: number;
  symbol: string;
  cycleTs: number;
  price: number | null;
  pEnter: number | null;
  htfH1Trend: number | null;
  htfH4Trend: number | null;
  slopeOk: boolean | null;
  rangeOk: boolean | null;
  direction: string | null;
  thresholdUsed: number | null;
  decision: string;
  reasons: string[] | null;
  laneSelected: string | null;
  htfScore: number | null;
  holdReason: string | null;
  eNetPred: number | null;
  enterLogit: number | null;
  temperatureUsed: number | null;
  scalpAtrRatio: number | null;
  scalpTrZ: number | null;
  scalpBbZ: number | null;
  scalpEma20Slope: number | null;
  scalpMacdHist: number | null;
  scalpVolRatio: number | null;
  scalpVolExpansionOk: boolean | null;
  scalpMomentumOk: boolean | null;
  createdAt: number;
}

interface LiveSummary {
  openPositions: number;
  openTrades: Array<{ symbol: string; side: string; entryPrice: number; pEnter: number }>;
  closedTradesCount: number;
  winRate: string;
  totalNetR: string;
  avgNetR: string;
  learningStats: Record<string, ModelLearningStatsEntry | null>;
  symbols: string[];
}

function TrendBadge({ trend }: { trend: string | null }) {
  if (!trend) return <Badge variant="outline">N/A</Badge>;
  if (trend === "Improving") return <Badge className="bg-green-600 text-white"><TrendingUp className="w-3 h-3 mr-1" />Improving</Badge>;
  if (trend === "Worse") return <Badge variant="destructive"><TrendingDown className="w-3 h-3 mr-1" />Worse</Badge>;
  return <Badge variant="secondary"><Minus className="w-3 h-3 mr-1" />Flat</Badge>;
}

function SymbolBadge({ symbol }: { symbol: string }) {
  const colors: Record<string, string> = {
    BTCUSDT: "bg-orange-600 text-white",
    ETHUSDT: "bg-blue-600 text-white",
    SOLUSDT: "bg-purple-600 text-white",
    BNBUSDT: "bg-yellow-600 text-white",
    AVAXUSDT: "bg-red-600 text-white",
    XRPUSDT: "bg-slate-600 text-white",
    ADAUSDT: "bg-sky-600 text-white",
  };
  return <Badge className={colors[symbol] || "bg-gray-600 text-white"}>{symbol.replace("USDT", "")}</Badge>;
}

function formatTs(ts: number | null) {
  if (!ts) return "—";
  return format(new Date(ts), "MMM dd HH:mm");
}

function formatPrice(p: number | null) {
  if (p === null || p === undefined) return "—";
  if (p > 1000) return `$${p.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  return `$${p.toFixed(4)}`;
}

export function LiveSystemOverview() {
  const { data: summary } = useQuery<LiveSummary>({
    queryKey: ["/api/live/summary"],
    refetchInterval: 15000,
  });

  if (!summary) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Activity className="w-5 h-5" />Live System</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">No live data yet. Start the live runner on your GPU machine.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Live System Summary
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Open Positions</p>
            <p className="text-2xl font-bold" data-testid="text-open-positions">{summary.openPositions}</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Closed Trades</p>
            <p className="text-2xl font-bold" data-testid="text-closed-trades">{summary.closedTradesCount}</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Win Rate</p>
            <p className="text-2xl font-bold" data-testid="text-win-rate">{summary.winRate}%</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Total Net R</p>
            <p className={`text-2xl font-bold ${parseFloat(summary.totalNetR) >= 0 ? 'text-green-500' : 'text-red-500'}`} data-testid="text-total-net-r">
              {summary.totalNetR}R
            </p>
          </div>
        </div>

        {summary.openTrades.length > 0 && (
          <div className="mt-4">
            <p className="text-xs text-muted-foreground mb-2">Open Positions</p>
            <div className="flex flex-wrap gap-2">
              {summary.openTrades.map((t, i) => (
                <Badge key={i} variant="outline" className="flex items-center gap-1">
                  <SymbolBadge symbol={t.symbol} />
                  {t.side === "LONG" ? <ArrowUpRight className="w-3 h-3 text-green-500" /> : <ArrowDownRight className="w-3 h-3 text-red-500" />}
                  {formatPrice(t.entryPrice)}
                  <span className="text-muted-foreground text-xs">p={t.pEnter?.toFixed(2)}</span>
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function LearningStatsCards() {
  const { data: statsMap } = useQuery<Record<string, ModelLearningStatsEntry | null>>({
    queryKey: ["/api/live/learning-stats/latest"],
    refetchInterval: 60000,
  });

  const symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "XRPUSDT", "ADAUSDT"];
  const hasAny = statsMap && Object.values(statsMap).some(v => v !== null);

  if (!hasAny) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Brain className="w-5 h-5" />Model Learning Stats</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">No learning stats yet. Stats appear after the first training cycle completes.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {symbols.map(sym => {
        const stats = statsMap?.[sym];
        if (!stats) return (
          <Card key={sym}>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center justify-between gap-2 text-base">
                <span className="flex items-center gap-2"><SymbolBadge symbol={sym} /> Model Stats</span>
                <Badge variant="outline">No Data</Badge>
              </CardTitle>
            </CardHeader>
          </Card>
        );

        return (
          <Card key={sym} data-testid={`card-learning-stats-${sym}`}>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center justify-between gap-2 text-base">
                <span className="flex items-center gap-2">
                  <SymbolBadge symbol={sym} />
                  v{stats.modelVersion}
                </span>
                <div className="flex items-center gap-2">
                  <TrendBadge trend={stats.trend7d} />
                  {stats.promoted && <Badge className="bg-green-600 text-white"><CheckCircle2 className="w-3 h-3 mr-1" />Promoted</Badge>}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div>
                  <p className="text-xs text-muted-foreground">PR-AUC</p>
                  <p className="font-mono font-semibold">{stats.valPrAuc?.toFixed(3) ?? "—"}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">PF Net</p>
                  <p className={`font-mono font-semibold ${(stats.pfNet ?? 0) >= 1 ? 'text-green-500' : 'text-red-500'}`}>
                    {stats.pfNet?.toFixed(2) ?? "—"}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">E[Net R]</p>
                  <p className={`font-mono font-semibold ${(stats.eNet ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {stats.eNet?.toFixed(3) ?? "—"}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Trades/Day</p>
                  <p className="font-mono">{stats.tradesPerDay?.toFixed(1) ?? "—"}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Profitable Regimes</p>
                  <p className="font-mono">{stats.profitableRegimes ?? "—"}/{stats.totalRegimes ?? "—"}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Training Samples</p>
                  <p className="font-mono">{stats.trainingSamples?.toLocaleString() ?? "—"}</p>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
                <div>
                  <p className="text-xs text-muted-foreground">Threshold</p>
                  <p className="font-mono">{stats.bestPolicyThreshold != null ? `${(stats.bestPolicyThreshold * 100).toFixed(0)}%` : "—"}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Cooldown</p>
                  <p className="font-mono">{stats.bestPolicyCooldown ?? "—"} bars</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Last Trained</p>
                  <p className="font-mono text-xs">{formatTs(stats.trainedUntilTs)}</p>
                </div>
              </div>

              {(stats.prevPfNet != null || stats.prevENet != null) && (
                <div className="mt-3 pt-3 border-t text-xs text-muted-foreground">
                  <span>Prev: PF={stats.prevPfNet?.toFixed(2)} E={stats.prevENet?.toFixed(3)} TPD={stats.prevTradesPerDay?.toFixed(1)}</span>
                </div>
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

export function LiveTradeHistory() {
  const { data: trades } = useQuery<LiveTradeRecord[]>({
    queryKey: ["/api/live/trades"],
    refetchInterval: 15000,
  });

  if (!trades || trades.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Target className="w-5 h-5" />Live Trade History</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">No trades recorded yet. Trades appear when the live runner opens or closes positions.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Target className="w-5 h-5" />Live Trade History</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="table-live-trades">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-2">Symbol</th>
                <th className="py-2 pr-2">Side</th>
                <th className="py-2 pr-2">Entry</th>
                <th className="py-2 pr-2">Exit</th>
                <th className="py-2 pr-2">SL/TP</th>
                <th className="py-2 pr-2">p_enter</th>
                <th className="py-2 pr-2">Outcome</th>
                <th className="py-2 pr-2">Net R</th>
                <th className="py-2 pr-2">Time</th>
              </tr>
            </thead>
            <tbody>
              {trades.map(t => (
                <tr key={t.id} className="border-b border-border/50" data-testid={`row-trade-${t.id}`}>
                  <td className="py-2 pr-2"><SymbolBadge symbol={t.symbol} /></td>
                  <td className="py-2 pr-2">
                    <Badge variant={t.side === "LONG" ? "default" : "destructive"} className="text-xs">
                      {t.side === "LONG" ? <ArrowUpRight className="w-3 h-3 mr-0.5" /> : <ArrowDownRight className="w-3 h-3 mr-0.5" />}
                      {t.side}
                    </Badge>
                  </td>
                  <td className="py-2 pr-2 font-mono text-xs">{formatPrice(t.entryPrice)}</td>
                  <td className="py-2 pr-2 font-mono text-xs">{formatPrice(t.exitPrice)}</td>
                  <td className="py-2 pr-2 font-mono text-xs">
                    {t.stopLoss ? formatPrice(t.stopLoss) : "—"} / {t.takeProfit ? formatPrice(t.takeProfit) : "—"}
                  </td>
                  <td className="py-2 pr-2 font-mono text-xs">{t.pEnter?.toFixed(2) ?? "—"}</td>
                  <td className="py-2 pr-2">
                    {t.status === "open" ? (
                      <Badge variant="outline"><Clock className="w-3 h-3 mr-1" />Open</Badge>
                    ) : t.outcome === "TP" ? (
                      <Badge className="bg-green-600 text-white"><CheckCircle2 className="w-3 h-3 mr-1" />TP</Badge>
                    ) : t.outcome === "SL" ? (
                      <Badge variant="destructive"><XCircle className="w-3 h-3 mr-1" />SL</Badge>
                    ) : (
                      <Badge variant="secondary"><AlertCircle className="w-3 h-3 mr-1" />{t.outcome || "?"}</Badge>
                    )}
                  </td>
                  <td className={`py-2 pr-2 font-mono text-xs font-semibold ${(t.netR ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {t.netR != null ? `${t.netR >= 0 ? '+' : ''}${t.netR.toFixed(2)}R` : "—"}
                  </td>
                  <td className="py-2 pr-2 text-xs text-muted-foreground">{formatTs(t.entryTime)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

export function LiveCycleLogTable() {
  const { data: logs } = useQuery<LiveCycleLog[]>({
    queryKey: ["/api/live/cycle-logs"],
    refetchInterval: 15000,
  });

  if (!logs || logs.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><BarChart3 className="w-5 h-5" />Inference Cycle Log</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">No cycle logs yet. Logs appear every 15m when the live runner processes each symbol.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><BarChart3 className="w-5 h-5" />Inference Cycle Log</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" data-testid="table-cycle-logs">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="py-2 pr-2">Time</th>
                <th className="py-2 pr-2">Symbol</th>
                <th className="py-2 pr-2">Price</th>
                <th className="py-2 pr-2">p_enter</th>
                <th className="py-2 pr-2">E[net]</th>
                <th className="py-2 pr-2">Lane</th>
                <th className="py-2 pr-2">HTF</th>
                <th className="py-2 pr-2">Direction</th>
                <th className="py-2 pr-2">Decision</th>
                <th className="py-2 pr-2">Hold Reason</th>
              </tr>
            </thead>
            <tbody>
              {logs.slice(0, 50).map(l => (
                <tr key={l.id} className="border-b border-border/50" data-testid={`row-cycle-${l.id}`}>
                  <td className="py-2 pr-2 text-xs text-muted-foreground font-mono">{formatTs(l.cycleTs)}</td>
                  <td className="py-2 pr-2"><SymbolBadge symbol={l.symbol} /></td>
                  <td className="py-2 pr-2 font-mono text-xs">{formatPrice(l.price)}</td>
                  <td className="py-2 pr-2 font-mono text-xs font-semibold">
                    {l.pEnter != null ? `${(l.pEnter * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="py-2 pr-2 font-mono text-xs">
                    {l.eNetPred != null ? l.eNetPred.toFixed(3) : "—"}
                  </td>
                  <td className="py-2 pr-2">
                    {l.laneSelected ? (
                      <Badge variant={l.laneSelected === "CORE" ? "default" : l.laneSelected === "HOLD" ? "outline" : "secondary"} className="text-xs">
                        {l.laneSelected}
                      </Badge>
                    ) : "—"}
                  </td>
                  <td className="py-2 pr-2 text-xs">
                    <span className="font-mono">
                      {l.htfScore != null ? `S:${l.htfScore}` : "?"}{" "}
                      H1:{l.htfH1Trend ?? "?"} H4:{l.htfH4Trend ?? "?"}
                    </span>
                    {l.slopeOk && <CheckCircle2 className="w-3 h-3 text-green-500 inline ml-1" />}
                  </td>
                  <td className="py-2 pr-2">
                    {l.direction ? (
                      <Badge variant={l.direction === "LONG" ? "default" : l.direction === "SHORT" ? "destructive" : "secondary"} className="text-xs">
                        {l.direction}
                      </Badge>
                    ) : "—"}
                  </td>
                  <td className="py-2 pr-2">
                    <Badge variant={l.decision.startsWith("ENTER") ? "default" : "outline"} className="text-xs">
                      {l.decision}
                    </Badge>
                  </td>
                  <td className="py-2 pr-2 text-xs text-muted-foreground max-w-[200px] truncate">
                    {l.holdReason || (l.reasons?.join(", ")) || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

export function ScalpGatesPanel() {
  const { data: logs } = useQuery<LiveCycleLog[]>({
    queryKey: ["/api/live/cycle-logs"],
    refetchInterval: 15000,
  });

  const withGateData = (logs || []).filter(l => l.scalpAtrRatio != null);
  const total = withGateData.length;

  if (total === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Shield className="w-4 h-4" />SCALP Gate Diagnostics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">No SCALP gate data yet. Appears when the v4.5 runner sends cycle logs.</p>
        </CardContent>
      </Card>
    );
  }

  const volExpOkCount = withGateData.filter(l => l.scalpVolExpansionOk).length;
  const momOkCount = withGateData.filter(l => l.scalpMomentumOk).length;
  const avgAtrRatio = withGateData.reduce((s, l) => s + (l.scalpAtrRatio || 0), 0) / total;
  const avgVolRatio = withGateData.reduce((s, l) => s + (l.scalpVolRatio || 0), 0) / total;
  const avgTrZ = withGateData.reduce((s, l) => s + (l.scalpTrZ || 0), 0) / total;
  const avgBbZ = withGateData.reduce((s, l) => s + (l.scalpBbZ || 0), 0) / total;
  const bothOk = withGateData.filter(l => l.scalpVolExpansionOk && l.scalpMomentumOk).length;

  return (
    <Card data-testid="card-scalp-gates">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Shield className="w-4 h-4" />SCALP Gate Diagnostics
          <Badge variant="outline" className="ml-auto text-xs">{total} cycles</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Vol Expansion OK</p>
            <p className="text-xl font-bold" data-testid="text-vol-expansion-pct">
              {((volExpOkCount / total) * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground">{volExpOkCount}/{total}</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Momentum OK</p>
            <p className="text-xl font-bold" data-testid="text-momentum-pct">
              {((momOkCount / total) * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground">{momOkCount}/{total}</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Avg ATR Ratio</p>
            <p className="text-xl font-bold font-mono" data-testid="text-avg-atr-ratio">
              {avgAtrRatio.toFixed(3)}
            </p>
            <p className="text-xs text-muted-foreground">min: 1.20</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Avg Vol Ratio</p>
            <p className="text-xl font-bold font-mono" data-testid="text-avg-vol-ratio">
              {avgVolRatio.toFixed(3)}
            </p>
            <p className="text-xs text-muted-foreground">min: 1.20</p>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-border/50">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Both Gates Pass</p>
            <p className="text-lg font-bold">{((bothOk / total) * 100).toFixed(1)}%</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Avg TR Z-Score</p>
            <p className="text-lg font-bold font-mono">{avgTrZ.toFixed(2)}</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Avg BB Width Z</p>
            <p className="text-lg font-bold font-mono">{avgBbZ.toFixed(2)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function LiveSystemTab() {
  return (
    <div className="space-y-4">
      <LiveSystemOverview />
      <ScalpGatesPanel />
      
      <Tabs defaultValue="learning" className="w-full">
        <TabsList>
          <TabsTrigger value="learning" data-testid="tab-live-learning">
            <Brain className="w-4 h-4 mr-1" />Model Stats
          </TabsTrigger>
          <TabsTrigger value="trades" data-testid="tab-live-trades">
            <Target className="w-4 h-4 mr-1" />Trades
          </TabsTrigger>
          <TabsTrigger value="cycles" data-testid="tab-live-cycles">
            <BarChart3 className="w-4 h-4 mr-1" />Cycle Logs
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="learning" className="mt-4">
          <LearningStatsCards />
        </TabsContent>
        
        <TabsContent value="trades" className="mt-4">
          <LiveTradeHistory />
        </TabsContent>
        
        <TabsContent value="cycles" className="mt-4">
          <LiveCycleLogTable />
        </TabsContent>
      </Tabs>
    </div>
  );
}
