import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, Activity, Zap, Target, BarChart3, ArrowUpDown } from "lucide-react";
import type { StrategySignal, MultiTimeframeScore, DashboardData } from "@shared/schema";

interface Strategy {
  id: string;
  name: string;
  description: string;
  icon: typeof TrendingUp;
  conditions: string[];
  riskLevel: "low" | "medium" | "high";
  timeframe: string;
  winRate: string;
  status: "active" | "waiting" | "inactive";
  signal?: "LONG" | "SHORT" | "HOLD";
}

interface StrategySelectorCardProps {
  indicators?: DashboardData["indicators"];
  mtfScore?: MultiTimeframeScore;
  currentPrice?: number;
  strategySignal?: StrategySignal;
}

export function StrategySelectorCard({ indicators, mtfScore, strategySignal }: StrategySelectorCardProps) {
  const rsi = indicators?.rsi?.value ?? 50;
  const macdSignal = indicators?.macd?.signal ?? "neutral";
  const adx = indicators?.adx?.value ?? 20;
  const stochK = indicators?.stochastic?.value ?? 50;
  const bbPercentB = indicators?.bollingerBands?.value ?? 50;
  const mtfDirection = mtfScore?.direction ?? "neutral";
  const mtfAlignment = mtfScore?.alignment ?? 0;

  const strategies: Strategy[] = [
    {
      id: "trend_follow",
      name: "Trend Following",
      description: "Rides strong trends using Kalman crossovers with MTF confirmation",
      icon: TrendingUp,
      conditions: ["ADX > 25", "MTF aligned", "Kalman crossover"],
      riskLevel: "medium",
      timeframe: "15m-4h",
      winRate: "55-65%",
      status: adx > 25 && mtfAlignment > 0.5 ? "active" : adx > 20 ? "waiting" : "inactive",
      signal: adx > 25 && mtfDirection === "bullish" ? "LONG" : adx > 25 && mtfDirection === "bearish" ? "SHORT" : "HOLD",
    },
    {
      id: "mean_reversion",
      name: "Mean Reversion",
      description: "Fades overextended moves at Bollinger Band extremes",
      icon: ArrowUpDown,
      conditions: ["RSI extreme", "BB %B < 5 or > 95", "Low ADX"],
      riskLevel: "high",
      timeframe: "5m-15m",
      winRate: "45-55%",
      status: (rsi < 25 || rsi > 75) && adx < 25 ? "active" : (rsi < 35 || rsi > 65) ? "waiting" : "inactive",
      signal: rsi < 25 && bbPercentB < 10 ? "LONG" : rsi > 75 && bbPercentB > 90 ? "SHORT" : "HOLD",
    },
    {
      id: "breakout",
      name: "Breakout Hunter",
      description: "Catches momentum breakouts with volume confirmation",
      icon: Zap,
      conditions: ["BB squeeze", "Volume spike", "MACD crossover"],
      riskLevel: "high",
      timeframe: "15m-1h",
      winRate: "40-50%",
      status: macdSignal !== "neutral" && adx > 20 ? "active" : "waiting",
      signal: macdSignal === "bullish" && adx > 25 ? "LONG" : macdSignal === "bearish" && adx > 25 ? "SHORT" : "HOLD",
    },
    {
      id: "scalping",
      name: "Stochastic Scalper",
      description: "Quick trades on stochastic oversold/overbought bounces",
      icon: Activity,
      conditions: ["Stoch K/D cross", "RSI divergence", "Tight stops"],
      riskLevel: "high",
      timeframe: "5m",
      winRate: "50-60%",
      status: stochK < 20 || stochK > 80 ? "active" : stochK < 30 || stochK > 70 ? "waiting" : "inactive",
      signal: stochK < 20 ? "LONG" : stochK > 80 ? "SHORT" : "HOLD",
    },
    {
      id: "confluence",
      name: "Confluence Master",
      description: "Only trades when multiple indicators align perfectly",
      icon: Target,
      conditions: ["3+ indicators agree", "MTF > 70%", "Whale support"],
      riskLevel: "low",
      timeframe: "15m-1h",
      winRate: "60-70%",
      status: mtfAlignment > 0.7 ? "active" : mtfAlignment > 0.5 ? "waiting" : "inactive",
      signal: mtfAlignment > 0.7 && mtfDirection === "bullish" ? "LONG" : 
              mtfAlignment > 0.7 && mtfDirection === "bearish" ? "SHORT" : "HOLD",
    },
    {
      id: "kalman_retest",
      name: "Kalman Retest",
      description: "Current strategy - enters on Kalman line retests after crossover",
      icon: BarChart3,
      conditions: ["Kalman crossover", "Price retest", "ATR-based stops"],
      riskLevel: "medium",
      timeframe: "15m",
      winRate: "55-60%",
      status: strategySignal?.type === "retest" ? "active" : "waiting",
      signal: strategySignal?.direction ?? "HOLD",
    },
  ];

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "low": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "medium": return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "high": return "bg-red-500/20 text-red-400 border-red-500/30";
      default: return "bg-muted text-muted-foreground";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "waiting": return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "inactive": return "bg-muted text-muted-foreground";
      default: return "bg-muted text-muted-foreground";
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case "LONG": return "text-emerald-400";
      case "SHORT": return "text-red-400";
      default: return "text-muted-foreground";
    }
  };

  return (
    <Card className="overflow-visible" data-testid="card-strategies">
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-400" />
          <CardTitle className="text-sm font-medium">AI Trading Strategies</CardTitle>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Multiple strategies analyzing market conditions in real-time
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {strategies.map((strategy) => {
          const Icon = strategy.icon;
          return (
            <div 
              key={strategy.id}
              className={`p-3 rounded-lg border ${strategy.status === "active" ? "border-emerald-500/30 bg-emerald-500/5" : "border-border bg-card"}`}
              data-testid={`strategy-${strategy.id}`}
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <Icon className={`w-4 h-4 ${strategy.status === "active" ? "text-emerald-400" : "text-muted-foreground"}`} />
                  <span className="text-sm font-medium">{strategy.name}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Badge className={`text-[10px] ${getStatusColor(strategy.status)}`}>
                    {strategy.status.toUpperCase()}
                  </Badge>
                  {strategy.signal !== "HOLD" && (
                    <Badge className={`text-[10px] ${getSignalColor(strategy.signal ?? "HOLD")} bg-transparent border`}>
                      {strategy.signal}
                    </Badge>
                  )}
                </div>
              </div>
              <p className="text-xs text-muted-foreground mb-2">{strategy.description}</p>
              <div className="flex flex-wrap items-center gap-2 text-[10px]">
                <Badge variant="outline" className={getRiskColor(strategy.riskLevel)}>
                  {strategy.riskLevel.toUpperCase()} RISK
                </Badge>
                <span className="text-muted-foreground">{strategy.timeframe}</span>
                <span className="text-muted-foreground">WR: {strategy.winRate}</span>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
