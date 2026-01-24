import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { 
  TrendingUp, 
  TrendingDown, 
  Wallet, 
  Target, 
  AlertTriangle, 
  RotateCcw,
  DollarSign,
  BarChart3,
  Activity,
  Clock,
  Percent,
  ArrowUpRight,
  ArrowDownRight
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

interface Trade {
  id: number;
  positionId: number;
  ts: number;
  action: string;
  price: number;
  qty: number;
  feeUsdt: number;
  pnlUsdt: number;
  reason: string | null;
}

export function PortfolioCard() {
  const { data: portfolio } = useQuery<PortfolioSummary>({
    queryKey: ["/api/paper/portfolio"],
    refetchInterval: 5000,
  });

  const resetMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/paper/reset"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper"] });
    },
  });

  if (!portfolio) return null;

  const pnlPct = ((portfolio.equity - portfolio.startingEquity) / portfolio.startingEquity) * 100;
  const isProfitable = portfolio.totalPnl >= 0;

  return (
    <Card data-testid="card-portfolio">
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Wallet className="h-4 w-4 text-primary" />
          Paper Portfolio
        </CardTitle>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => resetMutation.mutate()}
          disabled={resetMutation.isPending}
          data-testid="button-reset-portfolio"
        >
          <RotateCcw className={`h-4 w-4 ${resetMutation.isPending ? "animate-spin" : ""}`} />
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-baseline justify-between">
          <span className="text-2xl font-bold" data-testid="text-equity">
            ${portfolio.equity.toFixed(2)}
          </span>
          <Badge 
            variant={isProfitable ? "secondary" : "destructive"}
            className={isProfitable ? "bg-emerald-500/20 text-emerald-400" : ""}
            data-testid="badge-pnl-pct"
          >
            {isProfitable ? <ArrowUpRight className="h-3 w-3 mr-1" /> : <ArrowDownRight className="h-3 w-3 mr-1" />}
            {pnlPct >= 0 ? "+" : ""}{pnlPct.toFixed(2)}%
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-muted-foreground text-xs">Realized PnL</p>
            <p className={`font-medium ${portfolio.realizedPnl >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-realized-pnl">
              {portfolio.realizedPnl >= 0 ? "+" : ""}${portfolio.realizedPnl.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Unrealized PnL</p>
            <p className={`font-medium ${portfolio.unrealizedPnl >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-unrealized-pnl">
              {portfolio.unrealizedPnl >= 0 ? "+" : ""}${portfolio.unrealizedPnl.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Win Rate</p>
            <p className="font-medium" data-testid="text-win-rate">{portfolio.winRate.toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Max Drawdown</p>
            <p className="font-medium text-amber-400" data-testid="text-max-dd">{portfolio.maxDrawdown.toFixed(2)}%</p>
          </div>
        </div>

        <Separator />

        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Trades: {portfolio.totalTrades}</span>
          <span className="text-emerald-400">{portfolio.winningTrades}W</span>
          <span className="text-red-400">{portfolio.losingTrades}L</span>
          <span>Exposure: ${portfolio.exposure.toFixed(0)}</span>
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
            <p className="font-medium" data-testid="text-entry-price">${pos.entryPrice.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Size</p>
            <p className="font-medium" data-testid="text-size">{pos.qty.toFixed(6)} BTC</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Notional</p>
            <p className="font-medium" data-testid="text-notional">${pos.notional.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Unrealized PnL</p>
            <p className={`font-medium ${pos.unrealizedPnl >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-pos-pnl">
              {pos.unrealizedPnl >= 0 ? "+" : ""}${pos.unrealizedPnl.toFixed(2)} ({pnlPct.toFixed(2)}%)
            </p>
          </div>
        </div>

        <Separator />

        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="p-2 rounded bg-red-500/10 border border-red-500/20">
            <p className="text-muted-foreground">Stop Loss</p>
            <p className="font-medium text-red-400" data-testid="text-sl">${pos.stopLoss?.toFixed(2) ?? "N/A"}</p>
          </div>
          <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20">
            <p className="text-muted-foreground">TP1</p>
            <p className="font-medium text-amber-400" data-testid="text-tp1">${pos.tp1?.toFixed(2) ?? "N/A"}</p>
          </div>
          <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
            <p className="text-muted-foreground">TP2</p>
            <p className="font-medium text-emerald-400" data-testid="text-tp2">${pos.tp2?.toFixed(2) ?? "N/A"}</p>
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

export function EquityCurveCard() {
  const { data: equityCurve } = useQuery<EquityPoint[]>({
    queryKey: ["/api/paper/equity"],
    refetchInterval: 10000,
  });

  const { data: portfolio } = useQuery<PortfolioSummary>({
    queryKey: ["/api/paper/portfolio"],
    refetchInterval: 5000,
  });

  const chartData = equityCurve?.map((p) => ({
    time: format(new Date(p.ts), "MM/dd HH:mm"),
    equity: p.equityUsdt,
    drawdown: p.drawdownPct,
  })) ?? [];

  const startEquity = portfolio?.startingEquity ?? 10000;

  return (
    <Card data-testid="card-equity-curve">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-primary" />
          Equity Curve
        </CardTitle>
      </CardHeader>
      <CardContent>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <XAxis 
                dataKey="time" 
                tick={{ fontSize: 10 }} 
                tickLine={false}
                axisLine={false}
              />
              <YAxis 
                domain={['auto', 'auto']}
                tick={{ fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `$${v.toFixed(0)}`}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))', 
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
              />
              <ReferenceLine y={startEquity} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
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
          <div className="h-[200px] flex items-center justify-center text-muted-foreground text-sm">
            No equity data yet. Start trading to see your curve.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function RecentTradesCard() {
  const { data: trades } = useQuery<Trade[]>({
    queryKey: ["/api/paper/trades"],
    refetchInterval: 5000,
  });

  const recentTrades = trades?.slice(0, 10) ?? [];

  return (
    <Card data-testid="card-recent-trades">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          Recent Paper Trades
        </CardTitle>
      </CardHeader>
      <CardContent>
        {recentTrades.length > 0 ? (
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {recentTrades.map((trade) => (
              <div 
                key={trade.id} 
                className="flex items-center justify-between p-2 rounded bg-muted/30 text-xs"
                data-testid={`trade-row-${trade.id}`}
              >
                <div className="flex items-center gap-2">
                  <Badge 
                    variant="secondary"
                    className={
                      trade.action === "OPEN" ? "bg-blue-500/20 text-blue-400" :
                      trade.action === "CLOSE" ? "bg-slate-500/20 text-slate-400" :
                      "bg-amber-500/20 text-amber-400"
                    }
                  >
                    {trade.action}
                  </Badge>
                  <span className="text-muted-foreground">{trade.qty.toFixed(4)} BTC</span>
                  <span>@ ${trade.price.toFixed(2)}</span>
                </div>
                <div className="flex items-center gap-2">
                  {trade.pnlUsdt !== 0 && (
                    <span className={trade.pnlUsdt >= 0 ? "text-emerald-400" : "text-red-400"}>
                      {trade.pnlUsdt >= 0 ? "+" : ""}${trade.pnlUsdt.toFixed(2)}
                    </span>
                  )}
                  <span className="text-muted-foreground">
                    {format(new Date(trade.ts), "HH:mm")}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-4">No trades yet</p>
        )}
      </CardContent>
    </Card>
  );
}

export function PaperTradingStatsCard() {
  const { data: portfolio } = useQuery<PortfolioSummary>({
    queryKey: ["/api/paper/portfolio"],
    refetchInterval: 5000,
  });

  if (!portfolio) return null;

  const avgWin = portfolio.winningTrades > 0 
    ? (portfolio.realizedPnl > 0 ? portfolio.realizedPnl / portfolio.winningTrades : 0)
    : 0;
  const avgLoss = portfolio.losingTrades > 0 
    ? (portfolio.realizedPnl < 0 ? Math.abs(portfolio.realizedPnl) / portfolio.losingTrades : 0)
    : 0;
  const profitFactor = avgLoss > 0 ? avgWin / avgLoss : avgWin > 0 ? Infinity : 0;
  const expectancy = portfolio.totalTrades > 0 
    ? portfolio.realizedPnl / portfolio.totalTrades 
    : 0;

  return (
    <Card data-testid="card-paper-stats">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Percent className="h-4 w-4 text-primary" />
          Paper Trading Statistics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-muted-foreground text-xs">Total Trades</p>
            <p className="font-medium" data-testid="text-total-trades">{portfolio.totalTrades}</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Win Rate</p>
            <p className="font-medium" data-testid="text-stats-winrate">{portfolio.winRate.toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Profit Factor</p>
            <p className={`font-medium ${profitFactor >= 1 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-pf">
              {profitFactor === Infinity ? "∞" : profitFactor.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Expectancy</p>
            <p className={`font-medium ${expectancy >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-expectancy">
              ${expectancy.toFixed(2)}/trade
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Max Drawdown</p>
            <p className="font-medium text-amber-400" data-testid="text-stats-dd">{portfolio.maxDrawdown.toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Total PnL</p>
            <p className={`font-medium ${portfolio.totalPnl >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-total-pnl">
              {portfolio.totalPnl >= 0 ? "+" : ""}${portfolio.totalPnl.toFixed(2)}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
