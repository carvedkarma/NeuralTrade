import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { Trade } from "@shared/schema";
import { TrendingUp, TrendingDown, Target, AlertTriangle, Clock, DollarSign } from "lucide-react";

interface ActiveTradePanelProps {
  trade: Trade;
  currentPrice: number;
}

export function ActiveTradePanel({ trade, currentPrice }: ActiveTradePanelProps) {
  const isLong = trade.side === "LONG";
  const priceDiff = isLong 
    ? currentPrice - trade.entryPrice 
    : trade.entryPrice - currentPrice;
  const unrealizedPnl = priceDiff * trade.size;
  const unrealizedPnlPercent = (priceDiff / trade.entryPrice) * 100;
  const isProfit = unrealizedPnl >= 0;

  const stopDistance = Math.abs(trade.entryPrice - trade.stopLoss);
  const tpDistance = Math.abs(trade.takeProfit - trade.entryPrice);
  const currentDistance = isLong 
    ? currentPrice - trade.entryPrice 
    : trade.entryPrice - currentPrice;
  
  const rMultiple = currentDistance / stopDistance;
  
  const progressToTP = Math.min(100, Math.max(0, (currentDistance / tpDistance) * 100));
  const progressToSL = Math.min(100, Math.max(0, (-currentDistance / stopDistance) * 100));

  const timeOpen = Date.now() - trade.timestamp;
  const candlesOpen = Math.floor(timeOpen / (15 * 60 * 1000));

  return (
    <Card className={`border-2 ${isLong ? "border-emerald-500/50" : "border-red-500/50"}`} data-testid="card-active-trade-panel">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            {isLong ? (
              <TrendingUp className="w-5 h-5 text-emerald-400" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-400" />
            )}
            <CardTitle className="text-base font-semibold">Active Trade</CardTitle>
          </div>
          <Badge 
            className={`animate-pulse ${isLong ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "bg-red-500/20 text-red-400 border-red-500/30"}`}
            data-testid="badge-trade-side"
          >
            {trade.side}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Entry</p>
            <p className="font-mono font-medium text-yellow-400" data-testid="text-trade-entry">
              ${trade.entryPrice.toLocaleString()}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Current</p>
            <p className="font-mono font-medium" data-testid="text-trade-current">
              ${currentPrice.toLocaleString()}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">P&L</p>
            <p className={`font-mono font-bold ${isProfit ? "text-emerald-400" : "text-red-400"}`} data-testid="text-trade-pnl">
              {isProfit ? "+" : ""}{unrealizedPnlPercent.toFixed(2)}%
            </p>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1 text-red-400">
              <AlertTriangle className="w-3 h-3" />
              <span>Stop: ${trade.stopLoss.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-1 text-emerald-400">
              <Target className="w-3 h-3" />
              <span>TP: ${trade.takeProfit.toLocaleString()}</span>
            </div>
          </div>
          
          <div className="relative h-3 bg-muted rounded-full overflow-hidden">
            <div 
              className="absolute left-0 h-full bg-red-500/50 transition-all"
              style={{ width: `${progressToSL}%` }}
            />
            <div 
              className="absolute right-0 h-full bg-emerald-500/50 transition-all"
              style={{ width: `${progressToTP}%` }}
            />
            <div 
              className="absolute top-0 bottom-0 w-0.5 bg-yellow-400"
              style={{ left: '50%' }}
            />
          </div>
          <p className="text-center text-xs text-muted-foreground">
            {rMultiple >= 0 ? `+${rMultiple.toFixed(2)}R` : `${rMultiple.toFixed(2)}R`} of target
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-muted-foreground" />
            <div className="text-xs">
              <p className="text-muted-foreground">Time Open</p>
              <p className="font-medium" data-testid="text-trade-time">{candlesOpen} candles ({Math.floor(timeOpen / 60000)}m)</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <DollarSign className="w-4 h-4 text-muted-foreground" />
            <div className="text-xs">
              <p className="text-muted-foreground">Size</p>
              <p className="font-medium font-mono" data-testid="text-trade-size">{trade.size.toFixed(4)} BTC</p>
            </div>
          </div>
        </div>

        <div className="text-xs text-center text-muted-foreground">
          Signal: <Badge variant="secondary" className="text-xs">{trade.signalType?.toUpperCase() ?? 'MANUAL'}</Badge>
        </div>
      </CardContent>
    </Card>
  );
}
