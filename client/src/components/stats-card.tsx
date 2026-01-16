import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, BarChart3, Target, Activity } from "lucide-react";

interface StatsCardProps {
  equity: number;
  dailyPnl: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
}

export function StatsCard({ equity, dailyPnl, winRate, profitFactor, totalTrades }: StatsCardProps) {
  const isDailyPositive = dailyPnl >= 0;

  return (
    <Card className="overflow-visible" data-testid="card-stats">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground" data-testid="text-stats-title">Performance</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-1.5">
            <BarChart3 className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Equity</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-mono font-bold" data-testid="text-equity">
              ${equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
            <Badge 
              variant="secondary" 
              className={`font-mono ${isDailyPositive ? "text-emerald-400" : "text-red-400"}`}
              data-testid="badge-daily-pnl"
            >
              {isDailyPositive ? "+" : ""}{dailyPnl.toFixed(2)}%
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 pt-3 border-t border-border">
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Target className="h-3 w-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Win Rate</span>
            </div>
            <span className={`text-sm font-mono font-medium ${winRate >= 50 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-win-rate">
              {winRate.toFixed(1)}%
            </span>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <TrendingUp className="h-3 w-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Profit Factor</span>
            </div>
            <span className={`text-sm font-mono font-medium ${profitFactor >= 1 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-profit-factor">
              {profitFactor.toFixed(2)}
            </span>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Activity className="h-3 w-3 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Trades</span>
            </div>
            <span className="text-sm font-mono font-medium" data-testid="text-total-trades">
              {totalTrades}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
