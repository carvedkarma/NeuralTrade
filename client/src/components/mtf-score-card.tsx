import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Layers, TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { MultiTimeframeScore } from "@shared/schema";

interface MTFScoreCardProps {
  mtfScore?: MultiTimeframeScore;
}

export function MTFScoreCard({ mtfScore }: MTFScoreCardProps) {
  if (!mtfScore) {
    return (
      <Card className="overflow-visible" data-testid="card-mtf-score">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <CardTitle className="text-sm font-medium">Multi-Timeframe Score</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4 text-muted-foreground text-sm">
            Loading MTF data...
          </div>
        </CardContent>
      </Card>
    );
  }

  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case "bullish": return "text-emerald-400";
      case "bearish": return "text-red-400";
      default: return "text-yellow-400";
    }
  };

  const getDirectionBg = (direction: string) => {
    switch (direction) {
      case "bullish": return "bg-emerald-500/20 border-emerald-500/30";
      case "bearish": return "bg-red-500/20 border-red-500/30";
      default: return "bg-yellow-500/20 border-yellow-500/30";
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "up": return <TrendingUp className="w-3 h-3 text-emerald-400" />;
      case "down": return <TrendingDown className="w-3 h-3 text-red-400" />;
      default: return <Minus className="w-3 h-3 text-yellow-400" />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case "up": return "bg-emerald-500";
      case "down": return "bg-red-500";
      default: return "bg-yellow-500";
    }
  };

  return (
    <Card className="overflow-visible" data-testid="card-mtf-score">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <CardTitle className="text-sm font-medium">Multi-Timeframe</CardTitle>
          </div>
          <Badge className={`${getDirectionBg(mtfScore.direction)} ${getDirectionColor(mtfScore.direction)}`} data-testid="badge-mtf-direction">
            {mtfScore.direction.toUpperCase()}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center">
          <div className={`text-3xl font-bold font-mono ${getDirectionColor(mtfScore.direction)}`} data-testid="text-mtf-score">
            {mtfScore.score > 0 ? "+" : ""}{mtfScore.score.toFixed(2)}
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {(mtfScore.alignment * 100).toFixed(0)}% alignment
          </p>
        </div>

        <div className="space-y-2">
          {mtfScore.details.map((tf) => (
            <div 
              key={tf.timeframe} 
              className="flex items-center justify-between p-2 rounded-md bg-muted/50"
              data-testid={`tf-${tf.timeframe}`}
            >
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium w-8">{tf.timeframe.toUpperCase()}</span>
                {getTrendIcon(tf.trend)}
              </div>
              <div className="flex items-center gap-2">
                <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${getTrendColor(tf.trend)}`}
                    style={{ width: `${tf.weight * 100}%` }}
                  />
                </div>
                <span className="text-[10px] text-muted-foreground w-8">
                  {(tf.weight * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="text-center text-xs text-muted-foreground">
          Higher alignment = stronger signal confirmation
        </div>
      </CardContent>
    </Card>
  );
}
