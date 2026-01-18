import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Brain, TrendingUp, TrendingDown, AlertTriangle, Lightbulb, Loader2 } from "lucide-react";
import type { AIAnalysis } from "@shared/schema";
import { useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface AIAnalysisCardProps {
  analysis?: AIAnalysis;
}

export function AIAnalysisCard({ analysis }: AIAnalysisCardProps) {
  const analyzeMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/ai/analyze"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] });
    },
  });

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case "STRONG_BUY": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "BUY": return "bg-green-500/20 text-green-400 border-green-500/30";
      case "HOLD": return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "SELL": return "bg-orange-500/20 text-orange-400 border-orange-500/30";
      case "STRONG_SELL": return "bg-red-500/20 text-red-400 border-red-500/30";
      default: return "bg-muted text-muted-foreground";
    }
  };

  const getRecommendationIcon = (rec: string) => {
    if (rec.includes("BUY")) return <TrendingUp className="w-4 h-4" />;
    if (rec.includes("SELL")) return <TrendingDown className="w-4 h-4" />;
    return null;
  };

  return (
    <Card className="overflow-visible" data-testid="card-ai-analysis">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            <CardTitle className="text-sm font-medium">AI Market Analysis</CardTitle>
          </div>
          <Button 
            size="sm" 
            variant="outline"
            onClick={() => analyzeMutation.mutate()}
            disabled={analyzeMutation.isPending}
            data-testid="button-analyze"
          >
            {analyzeMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              "Analyze"
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {analysis ? (
          <>
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">AI Recommendation</span>
              <Badge className={`${getRecommendationColor(analysis.recommendation)} flex items-center gap-1`} data-testid="badge-recommendation">
                {getRecommendationIcon(analysis.recommendation)}
                {analysis.recommendation.replace("_", " ")}
              </Badge>
            </div>

            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">Confidence</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-purple-500 transition-all"
                    style={{ width: `${analysis.confidence * 100}%` }}
                  />
                </div>
                <span className="text-sm font-mono" data-testid="text-confidence">
                  {(analysis.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">Market Summary</p>
              <p className="text-sm" data-testid="text-summary">{analysis.marketSummary}</p>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">Trend Analysis</p>
              <p className="text-sm text-muted-foreground">{analysis.trendExplanation}</p>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">Signal Reasoning</p>
              <p className="text-sm">{analysis.signalReasoning}</p>
            </div>

            {analysis.keyInsights.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Lightbulb className="w-3 h-3" />
                  Key Insights
                </div>
                <ul className="space-y-1">
                  {analysis.keyInsights.map((insight, i) => (
                    <li key={i} className="text-xs text-emerald-400 flex items-start gap-1">
                      <span className="mt-1">•</span>
                      {insight}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {analysis.warnings.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <AlertTriangle className="w-3 h-3 text-yellow-400" />
                  Warnings
                </div>
                <ul className="space-y-1">
                  {analysis.warnings.map((warning, i) => (
                    <li key={i} className="text-xs text-yellow-400 flex items-start gap-1">
                      <span className="mt-1">•</span>
                      {warning}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-6 text-muted-foreground">
            <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Click "Analyze" to get AI market insights</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
