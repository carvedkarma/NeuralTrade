import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useTradingWs } from "@/hooks/use-trading-ws";
import {
  ArrowUpRight,
  ArrowDownRight,
  Wallet,
  TrendingUp,
  DollarSign,
  ShieldAlert,
  Power,
  PowerOff,
  Loader2,
  RefreshCw,
  XCircle,
  AlertTriangle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function BitgetTradeHistory() {
  const { data: trades, isLoading } = useQuery<any[]>({
    queryKey: ["/api/live/trades"],
    refetchInterval: 30000,
  });

  const bitgetTrades = (trades || []).filter(
    (t: any) => t.exchange === "bitget" || t.reason?.includes("bitget")
  );

  const displayTrades = bitgetTrades.length > 0 ? bitgetTrades : (trades || []).slice(0, 20);
  const showingAll = bitgetTrades.length === 0 && (trades || []).length > 0;

  return (
    <div className="glass-card rounded-md p-4" data-testid="bitget-trade-history">
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold text-lg">
          Trade History {showingAll && <span className="text-xs text-muted-foreground ml-1">(all exchanges)</span>}
        </h2>
        <Badge variant="outline" className="text-xs">{displayTrades.length} trades</Badge>
      </div>
      {isLoading ? (
        <div className="flex items-center justify-center py-6 text-muted-foreground">
          <Loader2 className="w-4 h-4 animate-spin mr-2" /> Loading...
        </div>
      ) : displayTrades.length === 0 ? (
        <p className="text-sm text-muted-foreground text-center py-4" data-testid="no-trade-history">No trade history yet</p>
      ) : (
        <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Side</TableHead>
                <TableHead className="text-right">Entry</TableHead>
                <TableHead className="text-right">Exit</TableHead>
                <TableHead className="text-right">PnL</TableHead>
                <TableHead className="text-right">Time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {displayTrades.map((t: any, idx: number) => {
                const pnl = parseFloat(t.realizedPnl || t.pnl || "0");
                return (
                  <TableRow key={t.id || idx} data-testid={`trade-history-row-${idx}`}>
                    <TableCell>
                      <Badge className="bg-primary/10 text-primary border-primary/20 font-mono text-xs">
                        {(t.symbol || "").replace("USDT", "")}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {t.side === "LONG" || t.side === "Buy" ? (
                        <span className="text-emerald-400 text-xs">LONG</span>
                      ) : (
                        <span className="text-red-400 text-xs">SHORT</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right number-mono text-xs">{formatPrice(t.entryPrice || t.avgEntryPrice || 0)}</TableCell>
                    <TableCell className="text-right number-mono text-xs">{formatPrice(t.exitPrice || t.avgExitPrice || 0)}</TableCell>
                    <TableCell className={`text-right number-mono text-xs font-medium ${pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {pnl >= 0 ? "+" : ""}{formatUsd(pnl)}
                    </TableCell>
                    <TableCell className="text-right text-xs text-muted-foreground">
                      {t.closedAt || t.updatedAt ? new Date(t.closedAt || t.updatedAt).toLocaleDateString([], { month: "short", day: "numeric" }) : "-"}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}

function formatUsd(val: string | number): string {
  const num = typeof val === "string" ? parseFloat(val) : val;
  if (!Number.isFinite(num)) return "$0.00";
  return "$" + num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatPrice(val: string | number): string {
  const num = typeof val === "string" ? parseFloat(val) : val;
  if (!Number.isFinite(num) || num === 0) return "-";
  if (num >= 1000) return num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (num >= 1) return num.toFixed(4);
  return num.toFixed(6);
}

export default function BitgetTrading() {
  const { toast } = useToast();
  const { subscribe } = useTradingWs();

  const { data: bitgetStatus, isLoading: statusLoading } = useQuery<any>({
    queryKey: ["/api/bitget/status"],
    refetchInterval: 15000,
  });

  const { data: positions, isLoading: positionsLoading, refetch: refetchPositions } = useQuery<any>({
    queryKey: ["/api/bitget/positions"],
    refetchInterval: 10000,
    enabled: !!bitgetStatus?.connected,
  });

  const { data: balance, isLoading: balanceLoading } = useQuery<any>({
    queryKey: ["/api/bitget/balance"],
    refetchInterval: 10000,
    enabled: !!bitgetStatus?.connected,
  });

  const { data: recentSignals } = useQuery<any[]>({
    queryKey: ["/api/v5/signals", "?limit=20"],
    refetchInterval: 30000,
  });

  const toggleMutation = useMutation({
    mutationFn: (enabled: boolean) => apiRequest("POST", "/api/bitget/toggle", { enabled }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bitget/status"] });
      toast({ title: "Trading status updated" });
    },
    onError: (err: Error) => {
      toast({ title: "Failed to toggle trading", description: err.message, variant: "destructive" });
    },
  });

  const closeMutation = useMutation({
    mutationFn: (symbol: string) => apiRequest("POST", `/api/bitget/close/${symbol}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bitget/positions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/bitget/balance"] });
      toast({ title: "Position closed" });
    },
    onError: (err: Error) => {
      toast({ title: "Failed to close position", description: err.message, variant: "destructive" });
    },
  });

  useEffect(() => {
    const unsubs = [
      subscribe("LIVE_TRADE_OPEN", () => {
        queryClient.invalidateQueries({ queryKey: ["/api/bitget/positions"] });
        queryClient.invalidateQueries({ queryKey: ["/api/bitget/balance"] });
      }),
      subscribe("LIVE_TRADE_CLOSE", () => {
        queryClient.invalidateQueries({ queryKey: ["/api/bitget/positions"] });
        queryClient.invalidateQueries({ queryKey: ["/api/bitget/balance"] });
      }),
    ];
    return () => unsubs.forEach((u) => u());
  }, [subscribe]);

  const isConnected = bitgetStatus?.connected ?? false;
  const isLiveEnabled = bitgetStatus?.liveTradingEnabled ?? false;
  const openPositions = positions?.positions || [];
  const equity = parseFloat(balance?.equity || "0");
  const walletBal = parseFloat(balance?.walletBalance || "0");
  const availableBal = parseFloat(balance?.availableBalance || "0");
  const unrealizedPnl = parseFloat(balance?.unrealisedPnl || "0");

  if (!isConnected) {
    return (
      <div className="p-6 flex flex-col items-center justify-center min-h-[60vh]" data-testid="bitget-not-connected">
        <div className="glass-card rounded-lg p-8 max-w-md text-center space-y-4">
          <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto" />
          <h2 className="text-xl font-semibold">Bitget Not Connected</h2>
          <p className="text-sm text-muted-foreground">
            {statusLoading
              ? "Checking connection..."
              : "Go to Settings to enter your Bitget API credentials and connect your account."}
          </p>
          {bitgetStatus?.error && (
            <p className="text-xs text-red-400" data-testid="bitget-connection-error">{bitgetStatus.error}</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4" data-testid="bitget-trading-page">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold tracking-tight">Bitget Trading</h1>
          <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30" data-testid="bitget-connected-badge">
            Connected
          </Badge>
          {isLiveEnabled && (
            <Badge className="bg-red-500/20 text-red-400 border-red-500/50 animate-pulse" data-testid="bitget-live-badge">
              LIVE TRADING ACTIVE
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              refetchPositions();
              queryClient.invalidateQueries({ queryKey: ["/api/bitget/balance"] });
            }}
            data-testid="button-refresh-bitget"
          >
            <RefreshCw className="w-3.5 h-3.5 mr-1" />
            Refresh
          </Button>
          <Button
            size="sm"
            className={isLiveEnabled
              ? "bg-red-600 hover:bg-red-700 text-white"
              : "bg-emerald-600 hover:bg-emerald-700 text-white"
            }
            onClick={() => {
              if (!isLiveEnabled) {
                if (window.confirm("⚠️ START LIVE TRADING?\n\nThis will execute REAL trades with REAL money on Bitget when V5 signals fire.\n\nAre you absolutely sure?")) {
                  toggleMutation.mutate(true);
                }
              } else {
                toggleMutation.mutate(false);
              }
            }}
            disabled={toggleMutation.isPending}
            data-testid="button-toggle-bitget-trading"
          >
            {toggleMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin mr-1" />
            ) : isLiveEnabled ? (
              <PowerOff className="w-4 h-4 mr-1" />
            ) : (
              <Power className="w-4 h-4 mr-1" />
            )}
            {isLiveEnabled ? "STOP TRADING" : "START TRADING"}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card className="glass-card" data-testid="card-equity">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <DollarSign className="w-4 h-4 text-cyan-400" />
              <span className="text-xs text-muted-foreground">Total Equity</span>
            </div>
            <p className="text-xl font-bold number-mono text-cyan-400">{formatUsd(equity)}</p>
          </CardContent>
        </Card>
        <Card className="glass-card" data-testid="card-wallet">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <Wallet className="w-4 h-4 text-amber-400" />
              <span className="text-xs text-muted-foreground">Wallet Balance</span>
            </div>
            <p className="text-xl font-bold number-mono">{formatUsd(walletBal)}</p>
          </CardContent>
        </Card>
        <Card className="glass-card" data-testid="card-upnl">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              <span className="text-xs text-muted-foreground">Unrealized PnL</span>
            </div>
            <p className={`text-xl font-bold number-mono ${unrealizedPnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {unrealizedPnl >= 0 ? "+" : ""}{formatUsd(unrealizedPnl)}
            </p>
          </CardContent>
        </Card>
        <Card className="glass-card" data-testid="card-available">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <ShieldAlert className="w-4 h-4 text-violet-400" />
              <span className="text-xs text-muted-foreground">Available Margin</span>
            </div>
            <p className="text-xl font-bold number-mono">{formatUsd(availableBal)}</p>
          </CardContent>
        </Card>
      </div>

      <div className="glass-card rounded-md p-4" data-testid="bitget-positions-section">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-lg">Open Positions ({openPositions.length})</h2>
        </div>
        {positionsLoading ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading positions...
          </div>
        ) : openPositions.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground" data-testid="no-positions">
            No open positions
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Side</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                  <TableHead className="text-right">Entry</TableHead>
                  <TableHead className="text-right">Mark</TableHead>
                  <TableHead className="text-right">PnL</TableHead>
                  <TableHead className="text-right">Lev</TableHead>
                  <TableHead className="text-right">SL</TableHead>
                  <TableHead className="text-right">TP</TableHead>
                  <TableHead className="text-right">Liq</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {openPositions.map((pos: any, idx: number) => {
                  const pnl = parseFloat(pos.unrealisedPnl || "0");
                  return (
                    <TableRow key={`${pos.symbol}-${idx}`} data-testid={`position-row-${pos.symbol}`}>
                      <TableCell>
                        <Badge className="bg-primary/10 text-primary border-primary/20 font-mono">
                          {(pos.symbol || "").replace("USDT", "")}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {pos.side === "LONG" ? (
                          <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                            <ArrowUpRight className="w-3 h-3 mr-0.5" /> LONG
                          </Badge>
                        ) : (
                          <Badge className="bg-red-500/20 text-red-400 border-red-500/30">
                            <ArrowDownRight className="w-3 h-3 mr-0.5" /> SHORT
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-right number-mono text-sm">{pos.size}</TableCell>
                      <TableCell className="text-right number-mono text-sm">{formatPrice(pos.avgPrice)}</TableCell>
                      <TableCell className="text-right number-mono text-sm">{formatPrice(pos.markPrice)}</TableCell>
                      <TableCell className={`text-right number-mono text-sm font-medium ${pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {pnl >= 0 ? "+" : ""}{formatUsd(pnl)}
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="outline" className="number-mono text-xs">{pos.leverage}x</Badge>
                      </TableCell>
                      <TableCell className="text-right number-mono text-sm text-red-400/70">{formatPrice(pos.stopLoss)}</TableCell>
                      <TableCell className="text-right number-mono text-sm text-emerald-400/70">{formatPrice(pos.takeProfit)}</TableCell>
                      <TableCell className="text-right number-mono text-sm text-amber-400/70">{formatPrice(pos.liqPrice)}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="destructive"
                          size="sm"
                          className="h-7 text-xs"
                          onClick={() => {
                            if (window.confirm(`Close ${pos.side} position on ${pos.symbol}?`)) {
                              closeMutation.mutate(pos.symbol);
                            }
                          }}
                          disabled={closeMutation.isPending}
                          data-testid={`button-close-${pos.symbol}`}
                        >
                          <XCircle className="w-3 h-3 mr-0.5" /> Close
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      <div className="glass-card rounded-md p-4" data-testid="recent-signals-section">
        <h2 className="font-semibold text-lg mb-3">Recent V5 Signals</h2>
        {!recentSignals?.length ? (
          <p className="text-sm text-muted-foreground text-center py-4">No recent signals</p>
        ) : (
          <div className="space-y-1.5 max-h-[300px] overflow-y-auto">
            {recentSignals.slice(0, 15).map((sig: any) => (
              <div
                key={sig.id}
                className="flex items-center justify-between py-1.5 px-2 rounded border border-border/20 hover:bg-muted/20 transition-colors"
                data-testid={`signal-row-${sig.id}`}
              >
                <div className="flex items-center gap-2">
                  <Badge className="bg-primary/10 text-primary border-primary/20 font-mono text-xs">
                    {(sig.symbol || "").replace("USDT", "")}
                  </Badge>
                  {sig.direction === "LONG" ? (
                    <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30 text-xs">
                      <ArrowUpRight className="w-2.5 h-2.5 mr-0.5" /> LONG
                    </Badge>
                  ) : sig.direction === "SHORT" ? (
                    <Badge className="bg-red-500/20 text-red-400 border-red-500/30 text-xs">
                      <ArrowDownRight className="w-2.5 h-2.5 mr-0.5" /> SHORT
                    </Badge>
                  ) : (
                    <Badge className="bg-gray-500/20 text-gray-400 border-gray-500/30 text-xs">HOLD</Badge>
                  )}
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span className="number-mono">Score: {(sig.score ?? 0).toFixed(3)}</span>
                  <span className="number-mono">Conf: {((sig.confidence ?? 0) * 100).toFixed(0)}%</span>
                  <span>{new Date(sig.signalTs).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <BitgetTradeHistory />

      {bitgetStatus?.config && (
        <div className="glass-card rounded-md p-4" data-testid="bitget-config-info">
          <h2 className="font-semibold text-lg mb-3">Trading Configuration</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground text-xs">Risk Per Trade</span>
              <p className="number-mono font-medium">{bitgetStatus.config.riskPerTradePct}%</p>
            </div>
            <div>
              <span className="text-muted-foreground text-xs">Max Daily Loss</span>
              <p className="number-mono font-medium">${bitgetStatus.config.maxDailyLossUsdt}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-xs">Daily Loss Used</span>
              <p className="number-mono font-medium text-red-400">${(bitgetStatus.config.dailyLossUsdt ?? 0).toFixed(2)}</p>
            </div>
            <div>
              <span className="text-muted-foreground text-xs">Max Positions</span>
              <p className="number-mono font-medium">6</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
