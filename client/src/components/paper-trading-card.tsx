import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3,
  Activity,
  AlertTriangle,
  Shield,
  Play,
  Square,
  RotateCcw,
  Clock,
  Target
} from "lucide-react";
import { format } from "date-fns";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
} from "recharts";

interface PortfolioSummary {
  equity: number;
  startingEquity: number;
  availableBalance: number;
  realizedPnl: number;
  unrealizedPnl: number;
  totalPnl: number;
  maxDrawdown: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  exposure: number;
  avgWin: number;
  avgLoss: number;
  sharpe: number;
  expectancy: number;
  currentDrawdown: number;
  bestTrade: number;
  worstTrade: number;
  profitFactor: number;
  isAutoTrading: boolean;
  openPosition: {
    id: number;
    side: "LONG" | "SHORT";
    entryPrice: number;
    qty: number;
    notional: number;
    stopLoss: number | null;
    tp1: number | null;
    tp2: number | null;
    barsOpen: number | null;
    unrealizedPnl: number;
    entryTs: number;
  } | null;
}

interface EquityPoint {
  ts: number;
  equityUsdt: number;
  drawdownPct: number;
}

interface Position {
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
  tp2: number | null;
  exitTs: number | null;
  exitPrice: number | null;
  realizedPnlUsdt: number | null;
  exitReason: string | null;
}

export function PerformanceCard() {
  const { data: portfolio } = useQuery<PortfolioSummary>({
    queryKey: ["/api/paper/portfolio"],
    refetchInterval: 5000,
  });

  const startMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/paper/start"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper"] });
    },
  });

  const stopMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/paper/stop"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper"] });
    },
  });

  const resetMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/paper/reset"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper"] });
    },
  });

  if (!portfolio) return null;

  const pnlPct = ((portfolio.equity - portfolio.startingEquity) / portfolio.startingEquity) * 100;
  const isAutoTrading = portfolio.isAutoTrading ?? false;

  return (
    <Card data-testid="card-performance">
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-primary" />
          Performance
        </CardTitle>
        <Badge 
          variant="secondary"
          className={pnlPct >= 0 ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}
          data-testid="badge-pnl-pct"
        >
          {pnlPct >= 0 ? "+" : ""}{pnlPct.toFixed(2)}%
        </Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center">
          <p className="text-xs text-muted-foreground">Account Equity</p>
          <p className="text-3xl font-bold text-emerald-400" data-testid="text-equity">
            ${portfolio.equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        </div>

        <div className="flex justify-center gap-2">
          {isAutoTrading ? (
            <Button
              variant="destructive"
              size="sm"
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
              className="gap-2"
              data-testid="button-stop-trading"
            >
              <Square className="h-4 w-4" />
              Stop Trading
            </Button>
          ) : (
            <Button
              variant="default"
              size="sm"
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
              className="gap-2 bg-emerald-600"
              data-testid="button-start-trading"
            >
              <Play className="h-4 w-4" />
              Start Trading
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => resetMutation.mutate()}
            disabled={resetMutation.isPending}
            data-testid="button-reset"
          >
            <RotateCcw className={`h-4 w-4 ${resetMutation.isPending ? "animate-spin" : ""}`} />
          </Button>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="p-2 rounded bg-muted/30">
            <p className="text-xs text-muted-foreground">Total</p>
            <p className="text-lg font-semibold" data-testid="text-total">{portfolio.totalTrades}</p>
          </div>
          <div className="p-2 rounded bg-emerald-500/10">
            <p className="text-xs text-muted-foreground">Wins</p>
            <p className="text-lg font-semibold text-emerald-400" data-testid="text-wins">{portfolio.winningTrades}</p>
          </div>
          <div className="p-2 rounded bg-red-500/10">
            <p className="text-xs text-muted-foreground">Losses</p>
            <p className="text-lg font-semibold text-red-400" data-testid="text-losses">{portfolio.losingTrades}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Win Rate</span>
            <span className={(portfolio.winRate ?? 0) >= 50 ? "text-emerald-400" : "text-red-400"}>
              {(portfolio.winRate ?? 0).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Profit Factor</span>
            <span className={(portfolio.profitFactor ?? 0) >= 1 ? "text-emerald-400" : "text-red-400"}>
              {(portfolio.profitFactor ?? 0).toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Avg Win</span>
            <span className="text-emerald-400">+{(portfolio.avgWin ?? 0).toFixed(2)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Avg Loss</span>
            <span className="text-red-400">-{Math.abs(portfolio.avgLoss ?? 0).toFixed(2)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Sharpe</span>
            <span>{(portfolio.sharpe ?? 0).toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Expectancy</span>
            <span className={(portfolio.expectancy ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}>
              {(portfolio.expectancy ?? 0) >= 0 ? "+" : ""}{(portfolio.expectancy ?? 0).toFixed(2)}%
            </span>
          </div>
        </div>

        <Separator />

        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-400" />
            <span className="text-muted-foreground">Max Drawdown</span>
            <span className="ml-auto text-red-400">-{(portfolio.maxDrawdown ?? 0).toFixed(2)}%</span>
          </div>
          <div className="flex justify-between pl-6">
            <span className="text-muted-foreground">Current DD</span>
            <span className="text-red-400">-{(portfolio.currentDrawdown ?? 0).toFixed(2)}%</span>
          </div>
        </div>

        <Separator />

        <div className="flex justify-between text-sm">
          <div className="flex items-center gap-1">
            <TrendingUp className="h-3 w-3 text-emerald-400" />
            <span className="text-muted-foreground">Best:</span>
            <span className="text-emerald-400">+{(portfolio.bestTrade ?? 0).toFixed(2)}%</span>
          </div>
          <div className="flex items-center gap-1">
            <TrendingDown className="h-3 w-3 text-red-400" />
            <span className="text-muted-foreground">Worst:</span>
            <span className="text-red-400">{(portfolio.worstTrade ?? 0).toFixed(2)}%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function EquityPerformanceCard() {
  const { data: portfolio } = useQuery<PortfolioSummary>({
    queryKey: ["/api/paper/portfolio"],
    refetchInterval: 5000,
  });

  const { data: equityCurve } = useQuery<EquityPoint[]>({
    queryKey: ["/api/paper/equity"],
    refetchInterval: 10000,
  });

  if (!portfolio) return null;

  const pnlPct = ((portfolio.equity - portfolio.startingEquity) / portfolio.startingEquity) * 100;
  const chartData = equityCurve?.map((p) => ({
    time: format(new Date(p.ts), "MM/dd HH:mm"),
    equity: p.equityUsdt,
  })) ?? [];

  return (
    <Card data-testid="card-equity-performance">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          Performance
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <p className="text-xs text-muted-foreground flex items-center gap-1">
            <BarChart3 className="h-3 w-3" />
            Equity
          </p>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold" data-testid="text-equity-value">
              ${portfolio.equity.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
            </span>
            <Badge 
              variant="secondary"
              className={pnlPct >= 0 ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}
            >
              {pnlPct >= 0 ? "+" : ""}{pnlPct.toFixed(2)}%
            </Badge>
          </div>
        </div>

        {chartData.length > 1 ? (
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={chartData}>
              <XAxis dataKey="time" tick={false} axisLine={false} />
              <YAxis domain={['auto', 'auto']} hide />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))', 
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                }}
              />
              <ReferenceLine y={portfolio.startingEquity} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
              <Line 
                type="monotone" 
                dataKey="equity" 
                stroke="hsl(var(--primary))" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[120px] flex items-center justify-center text-muted-foreground text-xs">
            Start trading to see equity curve
          </div>
        )}

        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <Target className="h-3 w-3" /> Win Rate
            </p>
            <p className={`font-medium ${(portfolio.winRate ?? 0) >= 50 ? "text-emerald-400" : "text-red-400"}`}>
              {(portfolio.winRate ?? 0).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <TrendingUp className="h-3 w-3" /> Profit Factor
            </p>
            <p className="font-medium">{(portfolio.profitFactor ?? 0).toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <Activity className="h-3 w-3" /> Trades
            </p>
            <p className="font-medium">{portfolio.totalTrades ?? 0}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function RiskStatusCard() {
  const { data: portfolio } = useQuery<PortfolioSummary>({
    queryKey: ["/api/paper/portfolio"],
    refetchInterval: 5000,
  });

  if (!portfolio) return null;

  const currentDD = portfolio.currentDrawdown ?? 0;
  const maxDD = portfolio.maxDrawdown ?? 0;
  const exposurePct = portfolio.startingEquity > 0 
    ? ((portfolio.exposure ?? 0) / portfolio.startingEquity) * 100 
    : 0;

  let riskLevel: "normal" | "warning" | "danger" = "normal";
  let riskMessage = "Trading conditions are favorable";

  if (currentDD > 10 || exposurePct > 50) {
    riskLevel = "danger";
    riskMessage = "High risk - consider reducing exposure";
  } else if (currentDD > 5 || exposurePct > 30) {
    riskLevel = "warning";
    riskMessage = "Elevated risk - monitor closely";
  }

  const riskColors = {
    normal: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
    warning: "bg-amber-500/10 border-amber-500/30 text-amber-400",
    danger: "bg-red-500/10 border-red-500/30 text-red-400",
  };

  return (
    <Card data-testid="card-risk-status">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Shield className="h-4 w-4 text-primary" />
          Risk Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className={`p-3 rounded border ${riskColors[riskLevel]}`}>
          <p className="font-medium capitalize">{riskLevel}</p>
          <p className="text-xs opacity-80">{riskMessage}</p>
        </div>

        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Current Drawdown</span>
            <span className="text-red-400">-{currentDD.toFixed(2)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Max Drawdown</span>
            <span className="text-red-400">-{maxDD.toFixed(2)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Market Exposure</span>
            <span>{exposurePct.toFixed(0)}%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function OpenPositionCard() {
  const { data: portfolio } = useQuery<PortfolioSummary>({
    queryKey: ["/api/paper/portfolio"],
    refetchInterval: 5000,
  });

  if (!portfolio?.openPosition) {
    return (
      <Card data-testid="card-no-position">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Target className="h-4 w-4 text-muted-foreground" />
            Open Position
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground text-center py-4">No open position</p>
        </CardContent>
      </Card>
    );
  }

  const pos = portfolio.openPosition;
  const isLong = pos.side === "LONG";
  const pnlPct = pos.notional > 0 ? (pos.unrealizedPnl / pos.notional) * 100 : 0;

  return (
    <Card data-testid="card-open-position">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Open Position
          </span>
          <Badge 
            className={isLong ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}
            data-testid="badge-position-side"
          >
            {isLong ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
            {pos.side}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-muted-foreground text-xs">Entry Price</p>
            <p className="font-medium">${pos.entryPrice.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Size</p>
            <p className="font-medium">{pos.qty.toFixed(6)} BTC</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Notional</p>
            <p className="font-medium">${pos.notional.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Unrealized PnL</p>
            <p className={`font-medium ${pos.unrealizedPnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {pos.unrealizedPnl >= 0 ? "+" : ""}${pos.unrealizedPnl.toFixed(2)} ({pnlPct.toFixed(2)}%)
            </p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="p-2 rounded bg-red-500/10 border border-red-500/20">
            <p className="text-muted-foreground">SL</p>
            <p className="font-medium text-red-400">${pos.stopLoss?.toFixed(0) ?? "N/A"}</p>
          </div>
          <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20">
            <p className="text-muted-foreground">TP1</p>
            <p className="font-medium text-amber-400">${pos.tp1?.toFixed(0) ?? "N/A"}</p>
          </div>
          <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
            <p className="text-muted-foreground">TP2</p>
            <p className="font-medium text-emerald-400">${pos.tp2?.toFixed(0) ?? "N/A"}</p>
          </div>
        </div>

        <div className="flex justify-between text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {pos.barsOpen ?? 0} bars
          </span>
          <span>Opened {format(new Date(pos.entryTs), "HH:mm MMM d")}</span>
        </div>
      </CardContent>
    </Card>
  );
}

export function PositionHistoryCard() {
  const { data: positions } = useQuery<Position[]>({
    queryKey: ["/api/paper/positions", "CLOSED"],
    refetchInterval: 5000,
  });

  const closedPositions = positions?.filter(p => p.status === "CLOSED").slice(0, 10) ?? [];

  return (
    <Card data-testid="card-position-history">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          Position History
        </CardTitle>
      </CardHeader>
      <CardContent>
        {closedPositions.length > 0 ? (
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {closedPositions.map((pos) => {
              const pnlPct = pos.notionalUsdt > 0 && pos.realizedPnlUsdt 
                ? (pos.realizedPnlUsdt / pos.notionalUsdt) * 100 
                : 0;
              const isWin = (pos.realizedPnlUsdt ?? 0) > 0;

              return (
                <div 
                  key={pos.id} 
                  className="p-3 rounded bg-muted/30 space-y-2"
                  data-testid={`position-row-${pos.id}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge 
                        variant="secondary"
                        className={pos.side === "LONG" ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}
                      >
                        {pos.side}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {format(new Date(pos.entryTs), "MMM d HH:mm")}
                      </span>
                    </div>
                    <Badge 
                      variant="secondary"
                      className={isWin ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}
                    >
                      {isWin ? "+" : ""}{pnlPct.toFixed(2)}%
                    </Badge>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <span className="text-muted-foreground">Entry: </span>
                      <span>${pos.entryPrice.toFixed(0)}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Exit: </span>
                      <span>${pos.exitPrice?.toFixed(0) ?? "-"}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">PnL: </span>
                      <span className={isWin ? "text-emerald-400" : "text-red-400"}>
                        ${pos.realizedPnlUsdt?.toFixed(2) ?? "0"}
                      </span>
                    </div>
                  </div>
                  {pos.exitReason && (
                    <div className="text-xs">
                      <Badge variant="outline" className="text-xs">
                        {pos.exitReason}
                      </Badge>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-8">
            No closed positions yet. Start trading to build history.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export { PerformanceCard as PortfolioCard };
export { EquityPerformanceCard as EquityCurveCard };
export { RiskStatusCard as PaperTradingStatsCard };
export { PositionHistoryCard as RecentTradesCard };
