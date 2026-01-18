import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BarChart3, TrendingUp, TrendingDown, Award, AlertTriangle } from "lucide-react";
import type { PerformanceStats } from "@shared/schema";

interface PerformanceStatsCardProps {
  stats?: PerformanceStats;
  equity: number;
}

export function PerformanceStatsCard({ stats, equity }: PerformanceStatsCardProps) {
  if (!stats) {
    return (
      <Card className="overflow-visible" data-testid="card-performance-stats">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-amber-400" />
            <CardTitle className="text-sm font-medium">Performance Stats</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4 text-muted-foreground text-sm">
            Start trading to see stats
          </div>
        </CardContent>
      </Card>
    );
  }

  const equityChange = equity - 10000;
  const equityChangePercent = (equityChange / 10000) * 100;

  return (
    <Card className="overflow-visible" data-testid="card-performance-stats">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-amber-400" />
            <CardTitle className="text-sm font-medium">Performance</CardTitle>
          </div>
          <Badge 
            variant="secondary"
            className={equityChange >= 0 ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}
            data-testid="badge-equity-change"
          >
            {equityChange >= 0 ? "+" : ""}{equityChangePercent.toFixed(2)}%
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center p-3 rounded-md bg-muted/50">
          <p className="text-xs text-muted-foreground">Account Equity</p>
          <p className={`text-2xl font-bold font-mono ${equityChange >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-equity">
            ${equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="p-2 rounded-md bg-muted/50">
            <p className="text-xs text-muted-foreground">Total</p>
            <p className="font-mono font-bold" data-testid="text-total-trades">{stats.totalTrades}</p>
          </div>
          <div className="p-2 rounded-md bg-emerald-500/10">
            <p className="text-xs text-muted-foreground">Wins</p>
            <p className="font-mono font-bold text-emerald-400" data-testid="text-wins">{stats.winningTrades}</p>
          </div>
          <div className="p-2 rounded-md bg-red-500/10">
            <p className="text-xs text-muted-foreground">Losses</p>
            <p className="font-mono font-bold text-red-400" data-testid="text-losses">{stats.losingTrades}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Win Rate</span>
              <span className={`font-mono ${stats.winRate >= 50 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-win-rate">
                {stats.winRate.toFixed(1)}%
              </span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div 
                className={`h-full ${stats.winRate >= 50 ? "bg-emerald-500" : "bg-red-500"}`}
                style={{ width: `${Math.min(100, stats.winRate)}%` }}
              />
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Profit Factor</span>
              <span className={`font-mono ${stats.profitFactor >= 1 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-pf">
                {stats.profitFactor.toFixed(2)}
              </span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div 
                className={`h-full ${stats.profitFactor >= 1 ? "bg-emerald-500" : "bg-red-500"}`}
                style={{ width: `${Math.min(100, (stats.profitFactor / 3) * 100)}%` }}
              />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
            <span className="text-muted-foreground">Avg Win</span>
            <span className="font-mono text-emerald-400" data-testid="text-avg-win">+{stats.avgWin.toFixed(2)}%</span>
          </div>
          <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
            <span className="text-muted-foreground">Avg Loss</span>
            <span className="font-mono text-red-400" data-testid="text-avg-loss">-{stats.avgLoss.toFixed(2)}%</span>
          </div>
          <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
            <span className="text-muted-foreground">Sharpe</span>
            <span className="font-mono" data-testid="text-sharpe">{stats.sharpeRatio.toFixed(2)}</span>
          </div>
          <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
            <span className="text-muted-foreground">Expectancy</span>
            <span className={`font-mono ${stats.expectancy >= 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-expectancy">
              {stats.expectancy >= 0 ? "+" : ""}{stats.expectancy.toFixed(2)}%
            </span>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1 text-muted-foreground">
              <AlertTriangle className="w-3 h-3 text-yellow-400" />
              Max Drawdown
            </div>
            <span className="font-mono text-red-400" data-testid="text-max-dd">-{stats.maxDrawdown.toFixed(2)}%</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Current DD</span>
            <span className="font-mono text-red-400" data-testid="text-current-dd">-{stats.currentDrawdown.toFixed(2)}%</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-2 border-t border-border text-xs">
          <div className="flex items-center gap-1">
            <Award className="w-3 h-3 text-emerald-400" />
            <span className="text-muted-foreground">Best:</span>
            <span className="font-mono text-emerald-400" data-testid="text-best">+{stats.bestTrade.toFixed(2)}%</span>
          </div>
          <div className="flex items-center gap-1">
            <TrendingDown className="w-3 h-3 text-red-400" />
            <span className="text-muted-foreground">Worst:</span>
            <span className="font-mono text-red-400" data-testid="text-worst">{stats.worstTrade.toFixed(2)}%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
