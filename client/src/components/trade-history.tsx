import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Trade } from "@shared/schema";
import { TrendingUp, TrendingDown, Clock } from "lucide-react";
import { format } from "date-fns";

interface TradeHistoryProps {
  trades: Trade[];
}

export function TradeHistory({ trades }: TradeHistoryProps) {
  return (
    <Card className="overflow-visible h-full" data-testid="card-trade-history">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground" data-testid="text-trades-title">Recent Trades</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[280px]">
          <div className="px-4 pb-4 space-y-2">
            {trades.length === 0 ? (
              <div className="flex items-center justify-center py-8 text-muted-foreground text-sm" data-testid="text-no-trades">
                No trades yet
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
                        ? "border-primary/30 bg-primary/5" 
                        : "border-border bg-card"
                    }`}
                    data-testid={`trade-item-${index}`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        {isLong ? (
                          <TrendingUp className="h-4 w-4 text-emerald-400" />
                        ) : (
                          <TrendingDown className="h-4 w-4 text-red-400" />
                        )}
                        <span className={`text-sm font-medium ${isLong ? "text-emerald-400" : "text-red-400"}`} data-testid={`text-trade-side-${index}`}>
                          {trade.side}
                        </span>
                        {isOpen && (
                          <Badge variant="outline" className="text-xs border-primary/50 text-primary" data-testid={`badge-trade-open-${index}`}>
                            Open
                          </Badge>
                        )}
                      </div>
                      {trade.pnlPercent !== null && (
                        <span className={`text-sm font-mono font-medium ${isProfitable ? "text-emerald-400" : "text-red-400"}`} data-testid={`text-trade-pnl-${index}`}>
                          {isProfitable ? "+" : ""}{trade.pnlPercent.toFixed(2)}%
                        </span>
                      )}
                    </div>

                    <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                      <div className="space-y-0.5">
                        <span className="text-muted-foreground">Entry</span>
                        <p className="font-mono" data-testid={`text-trade-entry-${index}`}>${trade.entryPrice.toLocaleString()}</p>
                      </div>
                      {trade.exitPrice !== null ? (
                        <div className="space-y-0.5">
                          <span className="text-muted-foreground">Exit</span>
                          <p className="font-mono" data-testid={`text-trade-exit-${index}`}>${trade.exitPrice.toLocaleString()}</p>
                        </div>
                      ) : (
                        <div className="space-y-0.5">
                          <span className="text-muted-foreground">Target / Stop</span>
                          <p className="font-mono" data-testid={`text-trade-targets-${index}`}>
                            <span className="text-emerald-400">${trade.takeProfit.toLocaleString()}</span>
                            {" / "}
                            <span className="text-red-400">${trade.stopLoss.toLocaleString()}</span>
                          </p>
                        </div>
                      )}
                    </div>

                    <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span data-testid={`text-trade-time-${index}`}>{format(new Date(trade.timestamp), "MMM d, HH:mm")}</span>
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
