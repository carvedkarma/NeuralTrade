import { useState, useMemo, useEffect, useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useTradingWs } from "@/hooks/use-trading-ws";
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
  Plus,
  Loader2,
  Radio,
  ScanLine,
  Layers,
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
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { CloseButton, PartialCloseButton, EditSLTPDialog, SLTPProgressBar } from "@/components/position-actions";

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


function NewTradePanel({ prices }: { prices: PriceData | undefined }) {
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [side, setSide] = useState<"LONG" | "SHORT">("LONG");
  const [entryPrice, setEntryPrice] = useState("");
  const [stopLoss, setStopLoss] = useState("");
  const [takeProfit, setTakeProfit] = useState("");
  const [riskPercent, setRiskPercent] = useState(1);

  const { data: portfolio } = useQuery<{ currentEquity: number }>({
    queryKey: ["/api/paper/portfolio"],
  });

  const mutation = useMutation({
    mutationFn: () =>
      apiRequest("POST", "/api/paper/manual-open", {
        symbol,
        side,
        entryPrice: parseFloat(entryPrice),
        stopLoss: parseFloat(stopLoss),
        takeProfit: parseFloat(takeProfit),
        riskPercent,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
      toast({ title: `Opened ${side} ${symbol}` });
      setOpen(false);
    },
    onError: (err: Error) => {
      toast({ title: "Failed to open trade", description: err.message, variant: "destructive" });
    },
  });

  const equity = portfolio?.currentEquity ?? 10000;
  const riskUsd = equity * (riskPercent / 100);
  const ep = parseFloat(entryPrice) || 0;
  const sl = parseFloat(stopLoss) || 0;
  const tp = parseFloat(takeProfit) || 0;
  const stopDist = Math.abs(ep - sl);
  const posSize = stopDist > 0 ? riskUsd / stopDist : 0;
  const rrRatio = stopDist > 0 ? Math.abs(tp - ep) / stopDist : 0;

  const setRRRatio = (ratio: number) => {
    if (!stopDist || !ep) return;
    const tpCalc = side === "LONG" ? ep + stopDist * ratio : ep - stopDist * ratio;
    setTakeProfit(tpCalc.toFixed(2));
  };

  const handleSymbolSelect = (sym: string) => {
    setSymbol(sym);
    const p = prices?.[sym]?.price;
    if (p) setEntryPrice(p.toString());
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button className="bg-emerald-600 hover:bg-emerald-700" data-testid="button-new-trade">
          <Plus className="w-4 h-4 mr-1" />
          New Trade
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[420px] sm:max-w-[420px] overflow-y-auto" data-testid="new-trade-panel">
        <SheetHeader>
          <SheetTitle>Open Manual Trade</SheetTitle>
        </SheetHeader>
        <div className="space-y-5 mt-4">
          <div>
            <Label className="text-xs text-muted-foreground mb-2 block">Symbol</Label>
            <div className="grid grid-cols-3 gap-1.5" data-testid="trade-symbol-selector">
              {SYMBOLS.map((sym) => (
                <Button
                  key={sym}
                  variant={symbol === sym ? "default" : "outline"}
                  size="sm"
                  onClick={() => handleSymbolSelect(sym)}
                  data-testid={`trade-symbol-${sym}`}
                >
                  {sym.replace("USDT", "")}
                </Button>
              ))}
            </div>
          </div>

          <div>
            <Label className="text-xs text-muted-foreground mb-2 block">Side</Label>
            <div className="grid grid-cols-2 gap-2" data-testid="trade-side-selector">
              <Button
                variant={side === "LONG" ? "default" : "outline"}
                className={side === "LONG" ? "bg-emerald-600 hover:bg-emerald-700" : ""}
                onClick={() => setSide("LONG")}
                data-testid="trade-side-long"
              >
                <ArrowUpRight className="w-4 h-4 mr-1" /> LONG
              </Button>
              <Button
                variant={side === "SHORT" ? "default" : "outline"}
                className={side === "SHORT" ? "bg-red-600 hover:bg-red-700" : ""}
                onClick={() => setSide("SHORT")}
                data-testid="trade-side-short"
              >
                <ArrowDownRight className="w-4 h-4 mr-1" /> SHORT
              </Button>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="trade-entry">Entry Price</Label>
            <Input
              id="trade-entry"
              type="number"
              step="any"
              value={entryPrice}
              onChange={(e) => setEntryPrice(e.target.value)}
              data-testid="input-trade-entry"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="trade-sl">Stop Loss</Label>
            <Input
              id="trade-sl"
              type="number"
              step="any"
              value={stopLoss}
              onChange={(e) => setStopLoss(e.target.value)}
              data-testid="input-trade-sl"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="trade-tp">Take Profit</Label>
            <Input
              id="trade-tp"
              type="number"
              step="any"
              value={takeProfit}
              onChange={(e) => setTakeProfit(e.target.value)}
              data-testid="input-trade-tp"
            />
            <div className="flex gap-1">
              {[1, 2, 3].map((r) => (
                <Button
                  key={r}
                  variant="outline"
                  size="sm"
                  className="text-xs flex-1"
                  onClick={() => setRRRatio(r)}
                  data-testid={`button-rr-${r}`}
                >
                  {r}:1 R:R
                </Button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <Label>Risk: {riskPercent}% (${riskUsd.toFixed(2)})</Label>
            <Slider
              value={[riskPercent]}
              onValueChange={([v]) => setRiskPercent(v)}
              min={0.5}
              max={5}
              step={0.5}
              data-testid="slider-risk-percent"
            />
          </div>

          <div className="glass-card rounded-md p-3 space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Position Size</span>
              <span className="number-mono" data-testid="text-position-size">{posSize.toFixed(6)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">R:R Ratio</span>
              <span className="number-mono" data-testid="text-rr-ratio">{rrRatio.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Risk USD</span>
              <span className="number-mono" data-testid="text-risk-usd">${riskUsd.toFixed(2)}</span>
            </div>
          </div>

          <Button
            className={`w-full ${side === "LONG" ? "bg-emerald-600 hover:bg-emerald-700" : "bg-red-600 hover:bg-red-700"}`}
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || !ep || !sl || !tp}
            data-testid="button-submit-trade"
          >
            {mutation.isPending && <Loader2 className="w-4 h-4 animate-spin mr-1" />}
            Open {side} Position
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}

export default function LiveTrading() {
  const { toast } = useToast();
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

  const { data: bybitStatus } = useQuery<any>({
    queryKey: ["/api/bybit/status"],
    refetchInterval: 30000,
  });

  const { data: bybitPositions } = useQuery<any>({
    queryKey: ["/api/bybit/positions"],
    refetchInterval: 10000,
    enabled: !!bybitStatus?.connected,
  });

  const { data: bybitBalance } = useQuery<any>({
    queryKey: ["/api/bybit/balance"],
    refetchInterval: 15000,
    enabled: !!bybitStatus?.connected,
  });

  const closeBybitMutation = useMutation({
    mutationFn: (symbol: string) => apiRequest("POST", `/api/bybit/close/${symbol}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bybit/positions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/bybit/balance"] });
      toast({ title: "Position closed on Bybit" });
    },
    onError: (err: any) => {
      toast({ title: "Failed to close", description: err.message, variant: "destructive" });
    },
  });

  const { data: signalHistory } = useQuery<V5Signal[]>({
    queryKey: ["/api/v5/signals", "?limit=50"],
    refetchInterval: 60000,
  });

  const { data: cycleLog } = useQuery<
    Array<{
      id: number;
      symbol: string;
      cycleTs: number;
      price: number | null;
      pEnter: number | null;
      direction: string | null;
      decision: string;
      laneSelected: string | null;
      htfScore: number | null;
      reasons: string[];
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
    }>
  >({
    queryKey: [`/api/live/cycle-logs?symbol=${selectedSymbol}&limit=10`],
    refetchInterval: 30000,
  });

  const { data: allRecentCycles } = useQuery<
    Array<{ symbol: string; cycleTs: number }>
  >({
    queryKey: ["/api/live/cycle-logs?limit=50"],
    refetchInterval: 30000,
  });

  const { subscribe } = useTradingWs();

  useEffect(() => {
    const unsub = subscribe("CYCLE_UPDATE", (payload: Record<string, unknown>) => {
      queryClient.invalidateQueries({ queryKey: ["/api/live/cycle-logs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v5/signals"] });
      const autoTrade = payload.autoTradeResult as { opened?: boolean; positionId?: number } | undefined;
      if (autoTrade?.opened) {
        queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
        toast({
          title: `V5 Auto-Trade: ${(payload.direction as string)?.toUpperCase()} ${payload.symbol}`,
          description: `Position opened @ $${Number(payload.price).toFixed(2)} | p_enter: ${((payload.pEnter as number) * 100).toFixed(0)}%`,
        });
      }
    });
    return unsub;
  }, [subscribe, toast]);

  const signal = latestSignal?.[0] ?? null;
  const latestCycle = cycleLog?.[0] ?? null;

  const lastScanPerSymbol = useMemo(() => {
    const map: Record<string, number> = {};
    if (!allRecentCycles) return map;
    for (const c of allRecentCycles) {
      if (!map[c.symbol]) map[c.symbol] = c.cycleTs;
    }
    return map;
  }, [allRecentCycles]);

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

  const hasAnyScan = Object.keys(lastScanPerSymbol).length > 0;

  return (
    <div className="p-4 space-y-4" data-testid="live-trading">
      {bybitStatus?.liveTradingEnabled && (
        <div className="glass-card rounded-lg px-4 py-3 border-red-500/40 bg-red-500/5" data-testid="live-trading-banner">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-red-400 animate-pulse" />
              <span className="text-sm font-bold text-red-400">LIVE TRADING ACTIVE</span>
              <Badge className="bg-red-500/20 text-red-400 border-red-500/30">Bybit Mainnet</Badge>
            </div>
            {bybitBalance && (
              <div className="flex items-center gap-4 text-sm">
                <span className="text-muted-foreground">Balance: <span className="number-mono text-emerald-400" data-testid="text-live-balance">${parseFloat(bybitBalance.walletBalance || "0").toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span></span>
                <span className="text-muted-foreground">Equity: <span className="number-mono text-foreground" data-testid="text-live-equity">${parseFloat(bybitBalance.equity || "0").toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span></span>
                {parseFloat(bybitBalance.unrealisedPnl || "0") !== 0 && (
                  <span className="text-muted-foreground">UPL: <span className={`number-mono ${parseFloat(bybitBalance.unrealisedPnl) >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-live-upl">${parseFloat(bybitBalance.unrealisedPnl || "0").toFixed(2)}</span></span>
                )}
              </div>
            )}
          </div>
          {bybitPositions?.positions?.length > 0 && (
            <div className="mt-3 border-t border-red-500/20 pt-3">
              <p className="text-xs text-muted-foreground mb-2">Live Positions ({bybitPositions.positions.length})</p>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {bybitPositions.positions.map((pos: any) => (
                  <div key={pos.symbol} className="glass-card rounded-md p-2.5 border-red-500/20" data-testid={`live-position-${pos.symbol}`}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold">{pos.symbol}</span>
                        <Badge className={pos.side === "LONG" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "bg-red-500/20 text-red-400 border-red-500/30"}>
                          {pos.side}
                        </Badge>
                        <span className="text-[10px] text-muted-foreground">{pos.leverage}x</span>
                      </div>
                      <span className={`number-mono text-sm font-bold ${parseFloat(pos.unrealisedPnl) >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid={`text-live-pnl-${pos.symbol}`}>
                        ${parseFloat(pos.unrealisedPnl).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                      <span>Qty: {pos.size} | Avg: ${formatPrice(parseFloat(pos.avgPrice))}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-5 px-2 text-[10px] text-red-400 hover:text-red-300"
                        onClick={() => {
                          if (window.confirm(`Close ${pos.symbol} position on Bybit?`)) {
                            closeBybitMutation.mutate(pos.symbol);
                          }
                        }}
                        disabled={closeBybitMutation.isPending}
                        data-testid={`button-close-live-${pos.symbol}`}
                      >
                        Close
                      </Button>
                    </div>
                    {(pos.stopLoss !== "0" || pos.takeProfit !== "0") && (
                      <div className="text-[10px] text-muted-foreground/60 mt-0.5">
                        {pos.stopLoss !== "0" && <span>SL: ${formatPrice(parseFloat(pos.stopLoss))}</span>}
                        {pos.stopLoss !== "0" && pos.takeProfit !== "0" && <span> | </span>}
                        {pos.takeProfit !== "0" && <span>TP: ${formatPrice(parseFloat(pos.takeProfit))}</span>}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="glass-card rounded-lg px-4 py-2.5 flex items-center gap-3 flex-wrap" data-testid="market-scanner-indicator">
        <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${hasAnyScan ? "bg-emerald-400 pulse-dot" : "bg-muted-foreground"}`} />
        <ScanLine className={`w-4 h-4 ${hasAnyScan ? "text-emerald-400" : "text-muted-foreground"}`} />
        <span className={`text-sm font-semibold ${hasAnyScan ? "text-foreground" : "text-muted-foreground"}`} data-testid="text-scanner-status">
          {hasAnyScan ? "Market Scanner Active" : "Market Scanner Idle"}
        </span>
        {lastScanPerSymbol[selectedSymbol] && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground ml-2" data-testid="text-symbol-last-scan">
            <Clock className="w-3 h-3" />
            <span>{selectedSymbol}: {formatDistanceToNow(new Date(lastScanPerSymbol[selectedSymbol]), { addSuffix: true })}</span>
          </div>
        )}
        <div className="ml-auto flex items-center gap-3">
          {SYMBOLS.map((sym) => {
            const ts = lastScanPerSymbol[sym];
            if (!ts) return null;
            return (
              <span key={sym} className="text-[10px] text-muted-foreground number-mono" data-testid={`text-scan-${sym}`}>
                {sym.replace("USDT", "")}: {formatDistanceToNow(new Date(ts), { addSuffix: false })}
              </span>
            );
          })}
        </div>
      </div>

      <div className="flex items-center gap-2 flex-wrap" data-testid="symbol-tabs">
        <NewTradePanel prices={prices} />
        <div className="w-px h-8 bg-border" />
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
                    <span className="text-muted-foreground">Engine</span>
                    <Badge data-testid="signal-engine" className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30">
                      {signal.lane === "V5" ? "V5" : signal.lane ?? "V5"}
                    </Badge>
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

            {latestCycle && (
              <div className="mt-4 pt-3 border-t border-border/50 space-y-2 animate-signal-arrive" data-testid="cycle-log-panel">
                <div className="flex items-center gap-2 mb-2">
                  <Radio className="w-3.5 h-3.5 text-cyan-400" />
                  <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Latest Cycle Log</span>
                  <span className="text-[10px] text-muted-foreground ml-auto number-mono" data-testid="text-cycle-time">
                    {formatDistanceToNow(new Date(latestCycle.cycleTs), { addSuffix: true })}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">p_enter</span>
                    <span className="number-mono" data-testid="cycle-p-enter">{latestCycle.pEnter?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Direction</span>
                    <span className={`text-xs font-medium ${latestCycle.direction === "LONG" ? "text-emerald-400" : latestCycle.direction === "SHORT" ? "text-red-400" : "text-amber-400"}`} data-testid="cycle-direction">
                      {latestCycle.direction ?? "-"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Decision</span>
                    <span className="number-mono" data-testid="cycle-decision">{latestCycle.decision}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">V5 Score</span>
                    <span className={`number-mono font-medium ${latestCycle.v5Score != null && latestCycle.v5Score >= (latestCycle.v5Threshold ?? 0.02) ? "text-emerald-400" : "text-muted-foreground"}`} data-testid="cycle-v5-score">
                      {latestCycle.v5Score?.toFixed(4) ?? "-"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">V5 Threshold</span>
                    <span className="number-mono" data-testid="cycle-v5-threshold">{latestCycle.v5Threshold?.toFixed(4) ?? latestCycle.thresholdUsed?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">ret_mu</span>
                    <span className="number-mono" data-testid="cycle-ret-mu">{latestCycle.retMu?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">MFE pred</span>
                    <span className="number-mono text-emerald-400" data-testid="cycle-mfe">{latestCycle.mfePred?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">MAE pred</span>
                    <span className="number-mono text-red-400" data-testid="cycle-mae">{latestCycle.maePred?.toFixed(4) ?? "-"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">HTF Score</span>
                    <span className="number-mono" data-testid="cycle-htf">{latestCycle.htfScore ?? "-"}</span>
                  </div>
                </div>
                {(latestCycle.pHold != null || latestCycle.pLong != null || latestCycle.pShort != null) && (
                  <div className="mt-2" data-testid="cycle-action-probs">
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Action Probabilities</span>
                    <div className="flex items-center gap-2 mt-1">
                      <div className="flex h-2 flex-1 rounded-full overflow-hidden bg-muted">
                        <div className="bg-emerald-500" style={{ width: `${(latestCycle.pLong ?? 0) * 100}%` }} />
                        <div className="bg-amber-500" style={{ width: `${(latestCycle.pHold ?? 0) * 100}%` }} />
                        <div className="bg-red-500" style={{ width: `${(latestCycle.pShort ?? 0) * 100}%` }} />
                      </div>
                    </div>
                    <div className="flex justify-between mt-1 text-[10px] number-mono">
                      <span className="text-emerald-400">LONG {((latestCycle.pLong ?? 0) * 100).toFixed(1)}%</span>
                      <span className="text-amber-400">HOLD {((latestCycle.pHold ?? 0) * 100).toFixed(1)}%</span>
                      <span className="text-red-400">SHORT {((latestCycle.pShort ?? 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                )}
                {latestCycle.holdReason && (
                  <div className="mt-1.5" data-testid="cycle-hold-reason">
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Hold Reason</span>
                    <div className="mt-0.5">
                      <Badge variant="secondary" className="text-[10px]">{latestCycle.holdReason}</Badge>
                    </div>
                  </div>
                )}
                {latestCycle.reasons && latestCycle.reasons.length > 0 && (
                  <div className="mt-1.5" data-testid="cycle-reasons">
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Reasons</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {latestCycle.reasons.map((r, i) => (
                        <Badge key={i} variant="secondary" className="text-[10px]">{r}</Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {cycleLog && cycleLog.length > 1 && (
              <div className="mt-4 pt-3 border-t border-border/50" data-testid="cycle-history-mini">
                <div className="flex items-center gap-2 mb-2">
                  <Layers className="w-3.5 h-3.5 text-muted-foreground" />
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Recent Cycles</span>
                </div>
                <div className="space-y-1">
                  {cycleLog.slice(1, 6).map((c, idx) => (
                    <div key={c.id ?? idx} className="flex items-center gap-2 text-[10px] number-mono py-0.5" data-testid={`mini-cycle-${idx}`}>
                      <span className="text-muted-foreground w-16 shrink-0">
                        {formatDistanceToNow(new Date(c.cycleTs), { addSuffix: false })}
                      </span>
                      <Badge
                        className={`no-default-hover-elevate no-default-active-elevate text-[10px] ${
                          c.direction === "LONG" ? "bg-emerald-500/20 text-emerald-400" :
                          c.direction === "SHORT" ? "bg-red-500/20 text-red-400" :
                          "bg-amber-500/20 text-amber-400"
                        }`}
                      >
                        {c.direction ?? "HOLD"}
                      </Badge>
                      <Badge
                        className={`no-default-hover-elevate no-default-active-elevate text-[10px] ${
                          c.decision === "ENTER" ? "bg-emerald-500/20 text-emerald-400" :
                          "bg-muted text-muted-foreground"
                        }`}
                      >
                        {c.decision}
                      </Badge>
                      <span>{c.pEnter != null ? `p=${(c.pEnter * 100).toFixed(0)}%` : ""}</span>
                      {c.retMu != null && <span className="text-foreground">mu={c.retMu.toFixed(3)}</span>}
                      {c.v5Score != null && (
                        <span className={`text-[10px] number-mono ${c.v5Score >= (c.v5Threshold ?? 0.02) ? "text-emerald-400" : "text-muted-foreground"}`}>
                          s={c.v5Score.toFixed(3)}
                        </span>
                      )}
                    </div>
                  ))}
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
                      <TableHead className="text-xs">SL/TP</TableHead>
                      <TableHead className="text-xs">Duration</TableHead>
                      <TableHead className="text-xs text-right">Actions</TableHead>
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
                          <TableCell className="text-xs font-medium">
                            <span className="flex items-center gap-1">
                              {pos.symbol}
                              {pos.source === "v5_signal" && (
                                <Badge className="no-default-hover-elevate no-default-active-elevate text-[9px] px-1 py-0 bg-cyan-500/20 text-cyan-400">V5</Badge>
                              )}
                            </span>
                          </TableCell>
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
                          <TableCell className="text-xs">
                            <div className="space-y-0.5">
                              <div className="flex justify-between gap-2 text-[10px]">
                                <span className="text-red-400 number-mono">{pos.stopLoss ? `$${formatPrice(pos.stopLoss)}` : "-"}</span>
                                <span className="text-emerald-400 number-mono">{pos.tp1 ? `$${formatPrice(pos.tp1)}` : "-"}</span>
                              </div>
                              <SLTPProgressBar
                                entryPrice={pos.entryPrice}
                                currentPrice={curPrice}
                                stopLoss={pos.stopLoss}
                                takeProfit={pos.tp1}
                                side={pos.side}
                              />
                            </div>
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {formatDistanceToNow(new Date(pos.entryTs), { addSuffix: false })}
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex items-center justify-end gap-0.5">
                              <PartialCloseButton positionId={pos.id} symbol={pos.symbol} />
                              <EditSLTPDialog
                                positionId={pos.id}
                                symbol={pos.symbol}
                                side={pos.side}
                                currentSL={pos.stopLoss}
                                currentTP={pos.tp1}
                                entryPrice={pos.entryPrice}
                              />
                              <CloseButton positionId={pos.id} symbol={pos.symbol} side={pos.side} />
                            </div>
                          </TableCell>
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
                      <TableCell>
                        <Badge className="text-[10px] bg-cyan-500/20 text-cyan-400">
                          {sig.lane === "V5" ? "V5" : sig.lane ?? "V5"}
                        </Badge>
                      </TableCell>
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
