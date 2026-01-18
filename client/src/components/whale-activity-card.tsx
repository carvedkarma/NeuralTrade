import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Fish, TrendingUp, TrendingDown, Minus, ArrowUpRight, ArrowDownRight } from "lucide-react";
import type { WhaleActivity } from "@shared/schema";

interface WhaleActivityCardProps {
  whaleActivity?: WhaleActivity;
}

export function WhaleActivityCard({ whaleActivity }: WhaleActivityCardProps) {
  if (!whaleActivity) {
    return (
      <Card className="overflow-visible" data-testid="card-whale-activity">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <Fish className="w-5 h-5 text-blue-400" />
            <CardTitle className="text-sm font-medium">Whale Activity</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4 text-muted-foreground text-sm">
            Detecting large orders...
          </div>
        </CardContent>
      </Card>
    );
  }

  const formatMoney = (value: number) => {
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
    return `$${value.toFixed(0)}`;
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case "bullish": return "text-emerald-400";
      case "bearish": return "text-red-400";
      default: return "text-yellow-400";
    }
  };

  const getSentimentBg = (sentiment: string) => {
    switch (sentiment) {
      case "bullish": return "bg-emerald-500/20 border-emerald-500/30";
      case "bearish": return "bg-red-500/20 border-red-500/30";
      default: return "bg-yellow-500/20 border-yellow-500/30";
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case "bullish": return <TrendingUp className="w-4 h-4" />;
      case "bearish": return <TrendingDown className="w-4 h-4" />;
      default: return <Minus className="w-4 h-4" />;
    }
  };

  const netFlowPositive = whaleActivity.netFlow >= 0;

  return (
    <Card className="overflow-visible" data-testid="card-whale-activity">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Fish className="w-5 h-5 text-blue-400" />
            <CardTitle className="text-sm font-medium">Whale Activity</CardTitle>
          </div>
          <Badge 
            className={`${getSentimentBg(whaleActivity.whaleActivity)} ${getSentimentColor(whaleActivity.whaleActivity)} flex items-center gap-1`}
            data-testid="badge-whale-sentiment"
          >
            {getSentimentIcon(whaleActivity.whaleActivity)}
            {whaleActivity.whaleActivity.toUpperCase()}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 rounded-md bg-emerald-500/10 border border-emerald-500/20">
            <div className="flex items-center gap-1 text-xs text-muted-foreground mb-1">
              <ArrowUpRight className="w-3 h-3 text-emerald-400" />
              Large Buys
            </div>
            <div className="text-lg font-bold font-mono text-emerald-400" data-testid="text-large-buys">
              {formatMoney(whaleActivity.largeBuys)}
            </div>
          </div>
          <div className="p-3 rounded-md bg-red-500/10 border border-red-500/20">
            <div className="flex items-center gap-1 text-xs text-muted-foreground mb-1">
              <ArrowDownRight className="w-3 h-3 text-red-400" />
              Large Sells
            </div>
            <div className="text-lg font-bold font-mono text-red-400" data-testid="text-large-sells">
              {formatMoney(whaleActivity.largeSells)}
            </div>
          </div>
        </div>

        <div className="p-3 rounded-md bg-muted/50 border border-border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground">Net Flow</span>
            <span className={`text-sm font-mono font-bold ${netFlowPositive ? "text-emerald-400" : "text-red-400"}`} data-testid="text-net-flow">
              {netFlowPositive ? "+" : ""}{formatMoney(whaleActivity.netFlow)}
            </span>
          </div>
          <div className="relative h-2 bg-muted rounded-full overflow-hidden">
            <div 
              className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-border z-10"
            />
            {netFlowPositive ? (
              <div 
                className="absolute top-0 bottom-0 left-1/2 bg-emerald-500 rounded-r-full transition-all"
                style={{ 
                  width: `${Math.min(50, (whaleActivity.netFlow / (whaleActivity.largeBuys + whaleActivity.largeSells)) * 100)}%` 
                }}
              />
            ) : (
              <div 
                className="absolute top-0 bottom-0 right-1/2 bg-red-500 rounded-l-full transition-all"
                style={{ 
                  width: `${Math.min(50, (Math.abs(whaleActivity.netFlow) / (whaleActivity.largeBuys + whaleActivity.largeSells)) * 100)}%` 
                }}
              />
            )}
          </div>
          <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
            <span>Selling</span>
            <span>Buying</span>
          </div>
        </div>

        <p className="text-xs text-center text-muted-foreground">
          Tracking orders &gt; $50K in the last 500 trades
        </p>
      </CardContent>
    </Card>
  );
}
