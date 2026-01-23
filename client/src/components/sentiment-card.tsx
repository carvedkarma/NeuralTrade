import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { Sentiment } from "@shared/schema";
import { TrendingUp, TrendingDown, Minus, Newspaper, MessageCircle } from "lucide-react";

interface SentimentCardProps {
  sentiment?: Sentiment;
}

export function SentimentCard({ sentiment }: SentimentCardProps) {
  if (!sentiment) {
    return (
      <Card data-testid="card-sentiment-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <MessageCircle className="h-4 w-4 text-muted-foreground" />
            Market Sentiment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground" data-testid="text-no-sentiment">
            Loading sentiment data...
          </p>
        </CardContent>
      </Card>
    );
  }

  const fearGreedValue = sentiment.fearGreed?.value ?? 50;
  const fearGreedColor = 
    fearGreedValue <= 25 ? "text-red-400" :
    fearGreedValue <= 45 ? "text-orange-400" :
    fearGreedValue <= 55 ? "text-amber-400" :
    fearGreedValue <= 75 ? "text-lime-400" : "text-emerald-400";

  const fearGreedBg = 
    fearGreedValue <= 25 ? "bg-red-500" :
    fearGreedValue <= 45 ? "bg-orange-500" :
    fearGreedValue <= 55 ? "bg-amber-500" :
    fearGreedValue <= 75 ? "bg-lime-500" : "bg-emerald-500";

  const SignalIcon = 
    sentiment.fearGreed?.signal === "bullish" ? TrendingUp :
    sentiment.fearGreed?.signal === "bearish" ? TrendingDown : Minus;

  const signalColor = 
    sentiment.fearGreed?.signal === "bullish" ? "text-emerald-400" :
    sentiment.fearGreed?.signal === "bearish" ? "text-red-400" : "text-amber-400";

  return (
    <Card data-testid="card-sentiment">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <MessageCircle className="h-4 w-4 text-primary" />
          Market Sentiment
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {sentiment.fearGreed && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Fear & Greed Index</span>
              <div className="flex items-center gap-2">
                <span className={`text-lg font-bold ${fearGreedColor}`} data-testid="text-fear-greed-value">
                  {sentiment.fearGreed.value}
                </span>
                <Badge 
                  variant="outline" 
                  className={`text-xs capitalize ${signalColor}`}
                  data-testid="badge-fear-greed-classification"
                >
                  {sentiment.fearGreed.classification}
                </Badge>
              </div>
            </div>
            <Progress 
              value={sentiment.fearGreed.value} 
              className="h-2" 
              data-testid="progress-fear-greed"
            />
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Extreme Fear</span>
              <span>Extreme Greed</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <SignalIcon className={`h-3 w-3 ${signalColor}`} />
              <span className="text-muted-foreground" data-testid="text-fear-greed-description">
                {sentiment.fearGreed.description}
              </span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <div className="bg-muted/50 rounded p-2 space-y-1">
            <div className="text-xs text-muted-foreground">Social Score</div>
            <div className="flex items-center gap-2">
              <Progress 
                value={sentiment.socialScore * 100} 
                className="h-1.5 flex-1" 
                data-testid="progress-social-score"
              />
              <span className="text-xs font-medium" data-testid="text-social-score">
                {(sentiment.socialScore * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          <div className="bg-muted/50 rounded p-2 space-y-1">
            <div className="text-xs text-muted-foreground">News Score</div>
            <div className="flex items-center gap-2">
              <Progress 
                value={Math.abs(sentiment.newsScore) * 50 + 50} 
                className="h-1.5 flex-1" 
                data-testid="progress-news-score"
              />
              <span className={`text-xs font-medium ${sentiment.newsScore > 0 ? "text-emerald-400" : sentiment.newsScore < 0 ? "text-red-400" : ""}`} data-testid="text-news-score">
                {sentiment.newsScore > 0 ? "+" : ""}{(sentiment.newsScore * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {sentiment.topNews.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground flex items-center gap-1">
              <Newspaper className="h-3 w-3" />
              Top News
            </div>
            <div className="space-y-1.5">
              {sentiment.topNews.slice(0, 3).map((news, i) => (
                <div 
                  key={i} 
                  className="text-xs bg-muted/30 rounded p-2"
                  data-testid={`news-item-${i}`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-muted-foreground line-clamp-2">{news.title}</p>
                    <Badge 
                      variant="outline" 
                      className={`text-[10px] flex-shrink-0 ${
                        news.sentiment === "bullish" ? "text-emerald-400" :
                        news.sentiment === "bearish" ? "text-red-400" : "text-muted-foreground"
                      }`}
                    >
                      {news.source}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
