import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import {
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Clock,
  TrendingUp,
  Coffee,
  Activity,
  Filter,
} from "lucide-react";
import {
  ComposedChart,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";

const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"] as const;
const TIMEFRAMES = ["15m", "1h", "4h"] as const;

type PriceData = Record<string, { price: number; change24h: number; high24h: number; low24h: number }>;

interface CandleRow {
  id: number;
  symbol: string;
  timestamp: number;
  timeframe: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface V5Signal {
  id: number;
  symbol: string;
  direction: string;
  confidence: number;
  score: number | null;
  muR: number | null;
  pSide: number | null;
  lane: string | null;
  regime: string | null;
  entryPrice: number | null;
  slPrice: number | null;
  tpPrice: number | null;
  thresholdUsed: number | null;
  htfScore: number | null;
  sizeMultiplier: number | null;
  signalTs: number;
  createdAt: number;
}

interface PaperPosition {
  id: number;
  symbol: string;
  side: string;
  status: string;
  entryTs: number;
  entryPrice: number;
  qty: number;
  notionalUsdt: number;
  stopLoss: number | null;
  tp1: number | null;
  realizedPnlUsdt: number | null;
  barsOpen: number | null;
  initialStopDistance: number | null;
  peakProfit: number | null;
}

function formatPrice(price: number | null | undefined): string {
  if (price == null) return "-";
  if (price >= 1000) return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (price >= 1) return price.toFixed(4);
  return price.toFixed(6);
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function DirectionBadge({ direction }: { direction: string }) {
  if (direction === "LONG") {
    return (
      <Badge data-testid="badge-direction-long" className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
        <ArrowUpRight className="w-3 h-3 mr-1" />
        LONG
      </Badge>
    );
  }
  if (direction === "SHORT") {
    return (
      <Badge data-testid="badge-direction-short" className="bg-red-500/20 text-red-400 border-red-500/30">
        <ArrowDownRight className="w-3 h-3 mr-1" />
        SHORT
      </Badge>
    );
  }
  return (
    <Badge data-testid="badge-direction-hold" className="bg-amber-500/20 text-amber-400 border-amber-500/30">
      <Minus className="w-3 h-3 mr-1" />
      HOLD
    </Badge>
  );
}

function LaneBadge({ lane }: { lane: string | null }) {
  if (!lane) return <span className="text-muted-foreground">-</span>;
  const colors: Record<string, string> = {
    CORE: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
    FLOW: "bg-violet-500/20 text-violet-400 border-violet-500/30",
    SCALP: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  };
  return (
    <Badge data-testid={`badge-lane-${lane}`} className={colors[lane.toUpperCase()] || "bg-muted text-muted-foreground"}>
      {lane.toUpperCase()}
    </Badge>
  );
}

export default function LiveTrading() {
  const [selectedSymbol, setSelectedSymbol] = useState<string>("BTCUSDT");
  const [timeframe, setTimeframe] = useState<string>("15m");
  const [historyFilter, setHistoryFilter] = useState<string>("ALL");

  const { data: prices, isLoading: pricesLoading } = useQuery<PriceData>({
    queryKey: ["/api/market/prices"],
    refetchInterval: 15000,
  });

  const { data: candles, isLoading: candlesLoading } = useQuery<CandleRow[]>({
    queryKey: ["/api/market/candles", `?symbol=${selectedSymbol}&interval=${timeframe}&limit=200`],
    refetchInterval: 30000,
  });

  const { data: latestSignal, isLoading: signalLoading } = useQuery<V5Signal[]>({
    queryKey: ["/api/v5/signals", `?symbol=${selectedSymbol}&limit=1`],
    refetchInterval: 30000,
  });

  const { data: positions, isLoading: positionsLoading } = useQuery<PaperPosition[]>({
    queryKey: ["/api/paper/positions", "?status=OPEN"],
    refetchInterval: 15000,
  });

  const { data: signalHistory } = useQuery<V5Signal[]>({
    queryKey: ["/api/v5/signals", "?limit=50"],
    refetchInterval: 60000,
  });

  const signal = latestSignal?.[0] ?? null;

  const chartData = useMemo(() => {
    if (!candles?.length) return [];
    return candles.map((c) => ({
      time: formatTime(c.timestamp),
      ts: c.timestamp,
      close: c.close,
      open: c.open,
      high: c.high,
      low: c.low,
      volume: c.volume,
      bullish: c.close >= c.open,
    }));
  }, [candles]);

  const filteredHistory = useMemo(() => {
    if (!signalHistory) return [];
    if (historyFilter === "ALL") return signalHistory;
    return signalHistory.filter((s) => s.symbol === historyFilter);
  }, [signalHistory, historyFilter]);

  return (
    <div className="p-4 space-y-4" data-testid="live-trading">
      <div className="flex items-center gap-2 flex-wrap" data-testid="symbol-tabs">
        {SYMBOLS.map((sym) => {
          const p = prices?.[sym];
          const isActive = selectedSymbol === sym;
          return (
            <Button
              key={sym}
              variant={isActive ? "default" : "ghost"}
              data-testid={`tab-${sym}`}
              onClick={() => setSelectedSymbol(sym)}
              className={isActive ? "" : ""}
            >
              <div className="flex flex-col items-start gap-0.5">
                <span className="text-xs font-semibold">{sym.replace("USDT", "")}</span>
                {pricesLoading ? (
                  <Skeleton className="h-3 w-12" />
                ) : p ? (
                  <div className="flex items-center gap-1">
                    <span className="text-xs number-mono">${formatPrice(p.price)}</span>
                    <span
                      className={`text-[10px] number-mono ${p.change24h >= 0 ? "text-emerald-400" : "text-red-400"}`}
                      data-testid={`change-${sym}`}
                    >
                      {p.change24h >= 0 ? "+" : ""}{p.change24h}%
                    </span>
                  </div>
                ) : (
                  <span className="text-[10px] text-muted-foreground">-</span>
                )}
              </div>
            </Button>
          );
        })}
      </div>

      <Card className="glass-card" data-testid="price-chart-card">
        <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-500" />
            {selectedSymbol} Price Chart
          </CardTitle>
          <div className="flex items-center gap-1" data-testid="timeframe-selector">
            {TIMEFRAMES.map((tf) => (
              <Button
                key={tf}
                size="sm"
                variant={timeframe === tf ? "default" : "ghost"}
                data-testid={`btn-tf-${tf}`}
                onClick={() => setTimeframe(tf)}
              >
                {tf}
              </Button>
            ))}
          </div>
        </CardHeader>
        <CardContent>
          {candlesLoading ? (
            <Skeleton className="h-[300px] w-full" data-testid="chart-skeleton" />
          ) : chartData.length === 0 ? (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">
              No candle data available
            </div>
          ) : (
            <div className="h-[300px]" data-testid="price-chart">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(142 70% 45%)" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="hsl(142 70% 45%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 15% 20%)" opacity={0.3} />
                  <XAxis
                    dataKey="time"
                    tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }}
                    tickLine={false}
                    axisLine={false}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    yAxisId="price"
                    domain={["auto", "auto"]}
                    tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v: number) => formatPrice(v)}
                    width={70}
                  />
                  <YAxis
                    yAxisId="volume"
                    orientation="right"
                    domain={[0, (max: number) => max * 4]}
                    tick={false}
                    axisLine={false}
                    width={0}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(225 38% 9%)",
                      border: "1px solid hsl(220 30% 14%)",
                      borderRadius: "6px",
                      fontSize: "12px",
                    }}
                    labelStyle={{ color: "hsl(210 40% 95%)" }}
                    formatter={(value: number, name: string) => [
                      formatPrice(value),
                      name === "close" ? "Price" : name === "volume" ? "Volume" : name,
                    ]}
                  />
                  <Area
                    yAxisId="price"
                    type="monotone"
                    dataKey="close"
                    stroke="hsl(142 70% 45%)"
                    strokeWidth={1.5}
                    fill="url(#priceGradient)"
                    dot={false}
                  />
                  <Bar
                    yAxisId="volume"
                    dataKey="volume"
                    fill="hsl(199 89% 48%)"
                    opacity={0.2}
                    radius={[1, 1, 0, 0]}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card className="glass-card" data-testid="signal-detail-panel">
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-cyan-500" />
              Latest Signal
            </CardTitle>
            {signal && (
              <span className="text-xs text-muted-foreground flex items-center gap-1" data-testid="signal-time-ago">
                <Clock className="w-3 h-3" />
                {formatDistanceToNow(new Date(signal.signalTs), { addSuffix: true })}
              </span>
            )}
          </CardHeader>
          <CardContent>
            {signalLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : !signal ? (
              <div className="flex flex-col items-center justify-center h-40 text-muted-foreground gap-2">
                <Coffee className="w-8 h-8" />
                <span>No signal for {selectedSymbol}</span>
              </div>
            ) : (
              <div className="space-y-4 animate-signal-arrive">
                <div className="flex items-center gap-4">
                  <div
                    className={`flex items-center gap-2 px-4 py-2 rounded-md text-lg font-bold ${
                      signal.direction === "LONG"
                        ? "bg-emerald-500/15 text-emerald-400 glow-green"
                        : signal.direction === "SHORT"
                          ? "bg-red-500/15 text-red-400 glow-red"
                          : "bg-amber-500/15 text-amber-400 glow-amber"
                    }`}
                    data-testid="signal-direction-badge"
                  >
                    {signal.direction === "LONG" && <ArrowUpRight className="w-5 h-5" />}
                    {signal.direction === "SHORT" && <ArrowDownRight className="w-5 h-5" />}
                    {signal.direction === "HOLD" && <Minus className="w-5 h-5" />}
                    {signal.direction}
                  </div>

                  <div className="flex flex-col items-center" data-testid="signal-confidence">
                    <span className="text-2xl font-bold number-mono text-cyan-400">
                      {(signal.confidence * 100).toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Confidence</span>
                    <div className="w-24 h-1.5 bg-muted rounded-full mt-1">
                      <div
                        className="h-full rounded-full bg-cyan-500"
                        style={{ width: `${Math.min(signal.confidence * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Score</span>
                    <span className="number-mono" data-testid="signal-score">{signal.score?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">mu_R</span>
                    <span className="number-mono" data-testid="signal-mu-r">{signal.muR?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">p_side</span>
                    <span className="number-mono" data-testid="signal-p-side">{signal.pSide?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Threshold</span>
                    <span className="number-mono" data-testid="signal-threshold">{signal.thresholdUsed?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Entry</span>
                    <span className="number-mono" data-testid="signal-entry">${formatPrice(signal.entryPrice)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">SL</span>
                    <span className="number-mono text-red-400" data-testid="signal-sl">${formatPrice(signal.slPrice)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">TP</span>
                    <span className="number-mono text-emerald-400" data-testid="signal-tp">${formatPrice(signal.tpPrice)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Lane</span>
                    <LaneBadge lane={signal.lane} />
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Regime</span>
                    <Badge variant="secondary" data-testid="signal-regime">{signal.regime ?? "-"}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">HTF Score</span>
                    <span className="number-mono" data-testid="signal-htf">{signal.htfScore?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Size Mult</span>
                    <span className="number-mono" data-testid="signal-size-mult">{signal.sizeMultiplier?.toFixed(2) ?? "-"}</span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="glass-card" data-testid="active-positions-panel">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-500" />
              Active Positions
              {positions && positions.length > 0 && (
                <span className="inline-flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
                  <span className="text-xs text-muted-foreground">{positions.length}</span>
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {positionsLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : !positions || positions.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-40 text-muted-foreground gap-2">
                <Coffee className="w-8 h-8" />
                <span>No active positions</span>
              </div>
            ) : (
              <div className="max-h-64 overflow-y-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs">Symbol</TableHead>
                      <TableHead className="text-xs">Side</TableHead>
                      <TableHead className="text-xs">Entry</TableHead>
                      <TableHead className="text-xs">Current</TableHead>
                      <TableHead className="text-xs">P&L</TableHead>
                      <TableHead className="text-xs">Duration</TableHead>
                      <TableHead className="text-xs">SL</TableHead>
                      <TableHead className="text-xs">TP</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {positions.map((pos) => {
                      const curPrice = prices?.[pos.symbol]?.price;
                      const pnlR = curPrice && pos.initialStopDistance
                        ? ((pos.side === "LONG" ? curPrice - pos.entryPrice : pos.entryPrice - curPrice) / pos.initialStopDistance)
                        : null;
                      const isProfit = pnlR != null && pnlR > 0;

                      return (
                        <TableRow
                          key={pos.id}
                          data-testid={`position-row-${pos.id}`}
                          className={isProfit ? "bg-emerald-500/5" : pnlR != null && pnlR < 0 ? "bg-red-500/5" : ""}
                        >
                          <TableCell className="text-xs font-medium">{pos.symbol}</TableCell>
                          <TableCell>
                            <Badge
                              className={
                                pos.side === "LONG"
                                  ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                                  : "bg-red-500/20 text-red-400 border-red-500/30"
                              }
                            >
                              {pos.side}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-xs number-mono">${formatPrice(pos.entryPrice)}</TableCell>
                          <TableCell className="text-xs number-mono">
                            {curPrice ? `$${formatPrice(curPrice)}` : "-"}
                          </TableCell>
                          <TableCell className={`text-xs number-mono font-medium ${isProfit ? "text-emerald-400" : "text-red-400"}`}>
                            {pnlR != null ? `${pnlR >= 0 ? "+" : ""}${pnlR.toFixed(2)}R` : "-"}
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {formatDistanceToNow(new Date(pos.entryTs), { addSuffix: false })}
                          </TableCell>
                          <TableCell className="text-xs number-mono">{pos.stopLoss ? `$${formatPrice(pos.stopLoss)}` : "-"}</TableCell>
                          <TableCell className="text-xs number-mono">{pos.tp1 ? `$${formatPrice(pos.tp1)}` : "-"}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="glass-card" data-testid="signal-history-panel">
        <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-500" />
            Signal History
          </CardTitle>
          <div className="flex items-center gap-2" data-testid="history-filter">
            <Filter className="w-3 h-3 text-muted-foreground" />
            <Select value={historyFilter} onValueChange={setHistoryFilter}>
              <SelectTrigger className="w-32" data-testid="select-history-filter">
                <SelectValue placeholder="All Symbols" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">All Symbols</SelectItem>
                {SYMBOLS.map((sym) => (
                  <SelectItem key={sym} value={sym}>{sym}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          <div className="max-h-64 overflow-y-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Time</TableHead>
                  <TableHead className="text-xs">Symbol</TableHead>
                  <TableHead className="text-xs">Direction</TableHead>
                  <TableHead className="text-xs">Confidence</TableHead>
                  <TableHead className="text-xs">Score</TableHead>
                  <TableHead className="text-xs">Lane</TableHead>
                  <TableHead className="text-xs">Regime</TableHead>
                  <TableHead className="text-xs">Entry</TableHead>
                  <TableHead className="text-xs">SL</TableHead>
                  <TableHead className="text-xs">TP</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredHistory.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={10} className="text-center text-muted-foreground py-8">
                      No signals found
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredHistory.map((sig) => (
                    <TableRow key={sig.id} data-testid={`signal-row-${sig.id}`}>
                      <TableCell className="text-xs text-muted-foreground number-mono">
                        {new Date(sig.signalTs).toLocaleString([], {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </TableCell>
                      <TableCell className="text-xs font-medium">{sig.symbol}</TableCell>
                      <TableCell><DirectionBadge direction={sig.direction} /></TableCell>
                      <TableCell className="text-xs number-mono">{(sig.confidence * 100).toFixed(1)}%</TableCell>
                      <TableCell className="text-xs number-mono">{sig.score?.toFixed(4) ?? "-"}</TableCell>
                      <TableCell><LaneBadge lane={sig.lane} /></TableCell>
                      <TableCell>
                        <Badge variant="secondary" className="text-[10px]">{sig.regime ?? "-"}</Badge>
                      </TableCell>
                      <TableCell className="text-xs number-mono">${formatPrice(sig.entryPrice)}</TableCell>
                      <TableCell className="text-xs number-mono text-red-400">${formatPrice(sig.slPrice)}</TableCell>
                      <TableCell className="text-xs number-mono text-emerald-400">${formatPrice(sig.tpPrice)}</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
