import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Trade } from "@shared/schema";
import { TrendingUp, TrendingDown, Clock, ArrowRight, Target, AlertTriangle } from "lucide-react";
import { format } from "date-fns";

interface TradeHistoryProps {
  trades: Trade[];
}

export function TradeHistory({ trades }: TradeHistoryProps) {
  return (
    <Card className="overflow-visible h-full" data-testid="card-trade-history">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-muted-foreground" data-testid="text-trades-title">Trade History</CardTitle>
          <Badge variant="secondary" className="text-xs">{trades.filter(t => t.status === 'closed').length} closed</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[280px]">
          <div className="px-4 pb-4 space-y-2">
            {trades.length === 0 ? (
              <div className="flex items-center justify-center py-8 text-muted-foreground text-sm" data-testid="text-no-trades">
                No trades yet - Start the strategy to begin trading
              </div>
            ) : (
              trades.map((trade, index) => {
                const isLong = trade.side === "LONG";
                const isProfitable = trade.pnlPercent !== null && trade.pnlPercent > 0;
                const isOpen = trade.status === "open";

                return (
                  <div
                    key={trade.id}
                    className={`p-3 rounded-md border ${
                      isOpen 
                        ? "border-primary/50 bg-primary/5 animate-pulse" 
                        : isProfitable
                        ? "border-emerald-500/30 bg-emerald-500/5"
                        : "border-red-500/30 bg-red-500/5"
                    }`}
                    data-testid={`trade-item-${index}`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        {isLong ? (
                          <div className="w-6 h-6 rounded-full bg-emerald-500/20 flex items-center justify-center">
                            <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
                          </div>
                        ) : (
                          <div className="w-6 h-6 rounded-full bg-red-500/20 flex items-center justify-center">
                            <TrendingDown className="h-3.5 w-3.5 text-red-400" />
                          </div>
                        )}
                        <div>
                          <span className={`text-sm font-medium ${isLong ? "text-emerald-400" : "text-red-400"}`} data-testid={`text-trade-side-${index}`}>
                            {trade.side}
                          </span>
                          {trade.signalType && (
                            <Badge variant="secondary" className="ml-2 text-[10px] py-0">
                              {trade.signalType.toUpperCase()}
                            </Badge>
                          )}
                        </div>
                        {isOpen && (
                          <Badge variant="outline" className="text-xs border-yellow-500/50 text-yellow-400 animate-pulse" data-testid={`badge-trade-open-${index}`}>
                            LIVE
                          </Badge>
                        )}
                      </div>
                      {trade.pnlPercent !== null && (
                        <div className="text-right">
                          <span className={`text-sm font-mono font-bold ${isProfitable ? "text-emerald-400" : "text-red-400"}`} data-testid={`text-trade-pnl-${index}`}>
                            {isProfitable ? "+" : ""}{trade.pnlPercent.toFixed(2)}%
                          </span>
                          {trade.pnl !== null && (
                            <p className={`text-xs font-mono ${isProfitable ? "text-emerald-400/70" : "text-red-400/70"}`}>
                              {isProfitable ? "+" : ""}${trade.pnl.toFixed(2)}
                            </p>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="mt-3 flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <div className="text-center">
                          <p className="text-muted-foreground text-[10px]">ENTRY</p>
                          <p className="font-mono font-medium text-yellow-400" data-testid={`text-trade-entry-${index}`}>
                            ${trade.entryPrice.toLocaleString()}
                          </p>
                        </div>
                        <ArrowRight className="w-4 h-4 text-muted-foreground" />
                        {trade.exitPrice !== null ? (
                          <div className="text-center">
                            <p className="text-muted-foreground text-[10px]">EXIT</p>
                            <p className={`font-mono font-medium ${isProfitable ? "text-emerald-400" : "text-red-400"}`} data-testid={`text-trade-exit-${index}`}>
                              ${trade.exitPrice.toLocaleString()}
                            </p>
                          </div>
                        ) : (
                          <div className="flex items-center gap-3">
                            <div className="text-center">
                              <p className="text-muted-foreground text-[10px] flex items-center gap-0.5">
                                <AlertTriangle className="w-2.5 h-2.5" /> STOP
                              </p>
                              <p className="font-mono font-medium text-red-400">
                                ${trade.stopLoss.toLocaleString()}
                              </p>
                            </div>
                            <div className="text-center">
                              <p className="text-muted-foreground text-[10px] flex items-center gap-0.5">
                                <Target className="w-2.5 h-2.5" /> TARGET
                              </p>
                              <p className="font-mono font-medium text-emerald-400">
                                ${trade.takeProfit.toLocaleString()}
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        <span data-testid={`text-trade-time-${index}`}>{format(new Date(trade.timestamp), "MMM d, HH:mm")}</span>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
