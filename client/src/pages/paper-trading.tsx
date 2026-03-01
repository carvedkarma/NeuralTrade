import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Scatter,
  ComposedChart,
} from "recharts";
import {
  RotateCcw,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  AlertTriangle,
} from "lucide-react";
import { CloseButton, PartialCloseButton, EditSLTPDialog } from "@/components/position-actions";

interface Portfolio {
  startingEquity: number;
  currentEquity: number;
  totalPnlR: number;
  totalPnlUsdt: number;
  maxDrawdownR: number;
  dailyPnlR: number;
  weeklyPnlR: number;
  tradesCount: number;
  winRate: number;
}

interface PaperConfig {
  paperTradingEnabled: boolean;
  isAutoTrading?: boolean;
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
}

interface EquityPoint {
  ts: number;
  r: number;
  tradeR: number;
  symbol: string;
  side: string;
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

export default function PaperTrading() {
  const [equityRange, setEquityRange] = useState<"7d" | "30d" | "all">("30d");

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
    refetchInterval: 10000,
  });

  const { data: closedPositions } = useQuery<Position[]>({
    queryKey: ["/api/paper/positions", "?status=CLOSED&limit=100"],
  });

  const { data: equityCurve } = useQuery<EquityPoint[]>({
    queryKey: ["/api/v5/equity-curve", `?range=${equityRange}`],
  });

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

  const pctChange = portfolio && portfolio.startingEquity > 0
    ? ((portfolio.currentEquity - portfolio.startingEquity) / portfolio.startingEquity) * 100
    : 0;

  const pnlPositive = (portfolio?.totalPnlR ?? 0) >= 0;

  return (
    <div className="p-4 space-y-4" data-testid="paper-trading">
      <div className="glass-card rounded-md p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Paper Trading</span>
            <Switch
              data-testid="switch-paper-trading"
              checked={paperEnabled}
              onCheckedChange={handlePaperToggle}
              disabled={enableMutation.isPending || disableMutation.isPending}
            />
            {paperEnabled && (
              <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Auto Trading</span>
            <Switch
              data-testid="switch-auto-trading"
              checked={autoTrading}
              onCheckedChange={handleAutoToggle}
              disabled={!paperEnabled || startAutoMutation.isPending || stopAutoMutation.isPending}
            />
            {autoTrading && (
              <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
            )}
          </div>

          <div className="ml-auto">
            <Button
              variant="destructive"
              data-testid="button-reset-portfolio"
              onClick={handleReset}
              disabled={resetMutation.isPending}
            >
              <RotateCcw className="w-4 h-4 mr-1" />
              Reset Portfolio
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Starting Equity</CardTitle>
            <DollarSign className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="number-mono text-2xl font-bold" data-testid="text-starting-equity">
              {portfolioLoading ? "..." : formatUsd(portfolio?.startingEquity ?? 0)}
            </div>
          </CardContent>
        </Card>

        <Card className={pctChange >= 0 ? "glow-green" : "glow-red"}>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Current Equity</CardTitle>
            {pctChange >= 0 ? (
              <TrendingUp className="w-4 h-4 text-emerald-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
          </CardHeader>
          <CardContent>
            <div className="number-mono text-2xl font-bold" data-testid="text-current-equity">
              {portfolioLoading ? "..." : formatUsd(portfolio?.currentEquity ?? 0)}
            </div>
            <p className={`text-xs mt-1 ${pctChange >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {pctChange >= 0 ? "+" : ""}{pctChange.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card className={pnlPositive ? "glow-green" : "glow-red"}>
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total P&L</CardTitle>
            <Activity className="w-4 h-4 text-cyan-500" />
          </CardHeader>
          <CardContent>
            <div
              className={`number-mono text-2xl font-bold ${pnlPositive ? "text-emerald-400" : "text-red-400"}`}
              data-testid="text-total-pnl"
            >
              {portfolioLoading ? "..." : `${(portfolio?.totalPnlR ?? 0) >= 0 ? "+" : ""}${(portfolio?.totalPnlR ?? 0).toFixed(2)}R`}
            </div>
            <p className={`text-xs mt-1 ${pnlPositive ? "text-emerald-400" : "text-red-400"}`}>
              {portfolioLoading ? "" : formatUsd(portfolio?.totalPnlUsdt ?? 0)}
            </p>
          </CardContent>
        </Card>

        <Card className="glow-red">
          <CardHeader className="flex flex-row items-center justify-between gap-1 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Max Drawdown</CardTitle>
            <AlertTriangle className="w-4 h-4 text-red-400" />
          </CardHeader>
          <CardContent>
            <div className="number-mono text-2xl font-bold text-red-400" data-testid="text-max-drawdown">
              {portfolioLoading ? "..." : `${(portfolio?.maxDrawdownR ?? 0).toFixed(2)}R`}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="glass-card rounded-md p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <h3 className="text-sm font-medium text-muted-foreground">Equity Curve</h3>
          <div className="flex gap-1">
            {(["7d", "30d", "all"] as const).map((range) => (
              <Button
                key={range}
                variant={equityRange === range ? "default" : "outline"}
                size="sm"
                data-testid={`button-range-${range}`}
                onClick={() => setEquityRange(range)}
              >
                {range === "all" ? "All" : range.toUpperCase()}
              </Button>
            ))}
          </div>
        </div>
        <div className="h-64" data-testid="chart-equity-curve">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={equityCurve ?? []}>
              <defs>
                <linearGradient id="greenGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="ts"
                tickFormatter={formatDate}
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickFormatter={(v: number) => `${v}R`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                  fontSize: 12,
                }}
                labelFormatter={(v: number) => formatDateTime(v)}
                formatter={(value: number, name: string) => {
                  if (name === "r") return [`${value}R`, "Cumulative"];
                  if (name === "tradeR") return [`${value}R`, "Trade"];
                  return [value, name];
                }}
              />
              <Area
                type="monotone"
                dataKey="r"
                stroke="#34d399"
                fill="url(#greenGradient)"
                strokeWidth={2}
              />
              <Scatter
                dataKey="tradeR"
                fill="#34d399"
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  if (!cx || !cy) return <circle r={0} />;
                  const color = (payload?.tradeR ?? 0) >= 0 ? "#34d399" : "#f87171";
                  return <circle cx={cx} cy={cy} r={4} fill={color} stroke="none" />;
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Open Positions</CardTitle>
        </CardHeader>
        <CardContent>
          {(!openPositions || openPositions.length === 0) ? (
            <p className="text-sm text-muted-foreground text-center py-8" data-testid="text-no-open-positions">
              No open positions
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Side</TableHead>
                  <TableHead>Entry Price</TableHead>
                  <TableHead>Current Price</TableHead>
                  <TableHead>P&L (R)</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>SL</TableHead>
                  <TableHead>TP</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {openPositions.map((pos, i) => {
                  const pnl = pos.pnlR ?? 0;
                  const dur = pos.entryTime ? Date.now() - pos.entryTime : 0;
                  const posId = typeof pos.id === "number" ? pos.id : parseInt(String(pos.id ?? "0"));
                  return (
                    <TableRow key={pos.id ?? i} data-testid={`row-open-position-${i}`}>
                      <TableCell className="font-medium">
                        <div className="flex items-center gap-1.5">
                          {pos.symbol}
                          {pos.source === "v5_signal" && (
                            <Badge className="no-default-hover-elevate no-default-active-elevate text-[10px] bg-cyan-500/20 text-cyan-400" data-testid={`badge-v5-signal-${i}`}>
                              V5
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={pos.side === "LONG" ? "text-emerald-400 border-emerald-400/30" : "text-red-400 border-red-400/30"}
                        >
                          {pos.side}
                        </Badge>
                      </TableCell>
                      <TableCell className="number-mono">{pos.entryPrice?.toFixed(2)}</TableCell>
                      <TableCell className="number-mono">{pos.currentPrice?.toFixed(2) ?? "-"}</TableCell>
                      <TableCell className={`number-mono ${pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}R
                      </TableCell>
                      <TableCell className="text-muted-foreground">{dur > 0 ? formatDuration(dur) : "-"}</TableCell>
                      <TableCell className="number-mono">{pos.stopLoss?.toFixed(2) ?? "-"}</TableCell>
                      <TableCell className="number-mono">{pos.takeProfit?.toFixed(2) ?? "-"}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-0.5">
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
                              <CloseButton positionId={posId} symbol={pos.symbol} side={pos.side} />
                            </>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Trade History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-h-96 overflow-auto">
            {(!closedPositions || closedPositions.length === 0) ? (
              <p className="text-sm text-muted-foreground text-center py-8" data-testid="text-no-trades">
                No completed trades yet
              </p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Date</TableHead>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Side</TableHead>
                    <TableHead>Entry</TableHead>
                    <TableHead>Exit</TableHead>
                    <TableHead>P&L R</TableHead>
                    <TableHead>P&L USD</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Exit Type</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...closedPositions]
                    .sort((a, b) => (b.exitTime ?? 0) - (a.exitTime ?? 0))
                    .map((trade, i) => {
                      const pnl = trade.pnlR ?? 0;
                      const dur = trade.entryTime && trade.exitTime ? trade.exitTime - trade.entryTime : 0;
                      return (
                        <TableRow key={trade.id ?? i} data-testid={`row-trade-history-${i}`}>
                          <TableCell className="text-muted-foreground">
                            {trade.exitTime ? formatDateTime(trade.exitTime) : "-"}
                          </TableCell>
                          <TableCell className="font-medium">{trade.symbol}</TableCell>
                          <TableCell>
                            <Badge
                              variant="outline"
                              className={trade.side === "LONG" ? "text-emerald-400 border-emerald-400/30" : "text-red-400 border-red-400/30"}
                            >
                              {trade.side}
                            </Badge>
                          </TableCell>
                          <TableCell className="number-mono">{trade.entryPrice?.toFixed(2)}</TableCell>
                          <TableCell className="number-mono">{trade.exitPrice?.toFixed(2) ?? "-"}</TableCell>
                          <TableCell className={`number-mono ${pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                            {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}R
                          </TableCell>
                          <TableCell className="number-mono">
                            {trade.pnlUsdt != null ? formatUsd(trade.pnlUsdt) : "-"}
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {dur > 0 ? formatDuration(dur) : "-"}
                          </TableCell>
                          <TableCell>
                            <span className="text-xs text-muted-foreground">{trade.exitType ?? "-"}</span>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                </TableBody>
              </Table>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}