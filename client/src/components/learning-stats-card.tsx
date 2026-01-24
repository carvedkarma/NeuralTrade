import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import type { LearningStats } from "@shared/schema";
import { 
  Database, Cpu, Brain, Activity, Layers, Clock, 
  CheckCircle2, AlertCircle, Zap, TrendingUp, BarChart3,
  Globe, MessageCircle, Newspaper, Gauge, History, BookOpen, Target
} from "lucide-react";
import { SiReddit, SiX } from "react-icons/si";

interface LearningStatsCardProps {
  learningStats?: LearningStats;
}

export function DataSourcesCard({ learningStats }: LearningStatsCardProps) {
  if (!learningStats) {
    return (
      <Card data-testid="card-data-sources-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Database className="h-4 w-4 text-muted-foreground" />
            Data Sources
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading data source stats...</p>
        </CardContent>
      </Card>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "text-emerald-400 bg-emerald-500/20";
      case "fallback": return "text-amber-400 bg-amber-500/20";
      case "error": return "text-red-400 bg-red-500/20";
      default: return "text-muted-foreground bg-muted";
    }
  };

  const formatTime = (ts: number | null) => {
    if (!ts) return "Never";
    const seconds = Math.floor((Date.now() - ts) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  return (
    <Card data-testid="card-data-sources">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Database className="h-4 w-4 text-primary" />
          Data Sources
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {learningStats.dataSources.map((source, i) => (
          <div key={i} className="flex items-center justify-between p-2 bg-muted/30 rounded" data-testid={`source-${source.name.toLowerCase().replace(/\s/g, '-')}`}>
            <div className="flex items-center gap-2">
              <Badge className={getStatusColor(source.status)} variant="outline">
                {source.status === "active" ? <CheckCircle2 className="h-3 w-3 mr-1" /> : 
                 source.status === "error" ? <AlertCircle className="h-3 w-3 mr-1" /> : null}
                {source.status}
              </Badge>
              <span className="text-sm font-medium">{source.name}</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>{source.candlesCollected} candles</span>
              <span>{formatTime(source.lastFetch)}</span>
            </div>
          </div>
        ))}
        
        <div className="pt-2 border-t">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center gap-1">
              <BarChart3 className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Total Candles:</span>
              <span className="font-medium" data-testid="text-total-candles">
                {learningStats.dataIngestion.candlesTotal}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Time Range:</span>
              <span className="font-medium" data-testid="text-time-range">
                {learningStats.dataIngestion.timeRangeDays.toFixed(1)} days
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function PatternLearningCard({ learningStats }: LearningStatsCardProps) {
  if (!learningStats) {
    return (
      <Card data-testid="card-pattern-learning-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Layers className="h-4 w-4 text-muted-foreground" />
            Pattern Learning
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading pattern stats...</p>
        </CardContent>
      </Card>
    );
  }

  const { patternLearning } = learningStats;
  const pl = patternLearning as any;
  const activePatterns = pl.activePatterns || 0;
  const immaturePatterns = pl.immaturePatterns || 0;
  const maxPatterns = pl.maxPatterns || 30;
  const canCreate = pl.canCreatePatterns ?? false;
  const minSupport = pl.minSupportRequired || 50;
  const simDist = pl.similarityDistribution;
  const similarityHealthy = pl.similarityHealthy ?? true;

  return (
    <Card data-testid="card-pattern-learning">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Layers className="h-4 w-4 text-primary" />
          Pattern Memory
          {!canCreate && (
            <Badge variant="outline" className="ml-auto text-xs text-amber-400">
              Learning Frozen
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-muted/30 rounded p-2 text-center">
            <div className="text-xs text-muted-foreground">Active</div>
            <div className="text-lg font-bold text-emerald-400" data-testid="text-active-patterns">
              {activePatterns}
            </div>
            <div className="text-[10px] text-muted-foreground">/{maxPatterns} max</div>
          </div>
          <div className="bg-muted/30 rounded p-2 text-center">
            <div className="text-xs text-muted-foreground">Immature</div>
            <div className="text-lg font-bold text-amber-400" data-testid="text-immature-patterns">
              {immaturePatterns}
            </div>
            <div className="text-[10px] text-muted-foreground">&lt;{minSupport} samples</div>
          </div>
          <div className="bg-muted/30 rounded p-2 text-center">
            <div className="text-xs text-muted-foreground">Similarity</div>
            <div className={`text-lg font-bold ${similarityHealthy ? 'text-emerald-400' : 'text-red-400'}`} data-testid="text-avg-similarity">
              {simDist?.mean ? (simDist.mean * 100).toFixed(0) : (patternLearning.avgSimilarity * 100).toFixed(0)}%
            </div>
            <div className="text-[10px] text-muted-foreground">
              {similarityHealthy ? 'healthy' : 'too high'}
            </div>
          </div>
        </div>

        {simDist && simDist.count > 0 && (
          <div className="bg-muted/20 rounded p-2 text-xs">
            <div className="text-muted-foreground mb-1">Similarity Distribution</div>
            <div className="flex justify-between">
              <span>Min: {(simDist.min * 100).toFixed(0)}%</span>
              <span>Median: {(simDist.median * 100).toFixed(0)}%</span>
              <span>Max: {(simDist.max * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}

        {!canCreate && pl.requiredData && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded p-2 text-xs">
            <div className="font-medium text-amber-400 mb-1">Pattern creation frozen</div>
            <div className="text-muted-foreground">
              Need {pl.requiredData.trades} trades (have {pl.currentData?.trades || 0}) and{' '}
              {pl.requiredData.candles} candles (have {pl.currentData?.candles || 0})
            </div>
          </div>
        )}

        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground">Clusters by Regime</div>
          {pl.clustersByRegime && Object.entries(pl.clustersByRegime).map(([regime, stats]: [string, any]) => (
            <div key={regime} className="flex items-center justify-between text-xs">
              <span className="capitalize">{regime.replace("_", " ")}</span>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className={stats.mature > 0 ? "text-emerald-400" : "text-muted-foreground"}>
                  {stats.mature} mature
                </Badge>
                <span className="text-muted-foreground">/ {stats.total} total</span>
              </div>
            </div>
          ))}
        </div>

        {patternLearning.topPatternOutcomes && patternLearning.topPatternOutcomes.length > 0 && activePatterns > 0 && (
          <div className="space-y-2 pt-2 border-t">
            <div className="text-xs font-medium text-muted-foreground">Cluster Summary</div>
            {patternLearning.topPatternOutcomes.map((pattern, i) => (
              <div key={i} className="flex items-center justify-between text-xs bg-muted/20 rounded p-1.5">
                <span>{pattern.pattern}</span>
                <span className="text-muted-foreground">({pattern.count} total)</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function FeatureComputationCard({ learningStats }: LearningStatsCardProps) {
  if (!learningStats) {
    return (
      <Card data-testid="card-feature-computation-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Cpu className="h-4 w-4 text-muted-foreground" />
            Feature Engine
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading feature stats...</p>
        </CardContent>
      </Card>
    );
  }

  const { featureComputation } = learningStats;

  return (
    <Card data-testid="card-feature-computation">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Cpu className="h-4 w-4 text-primary" />
          Feature Engine
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-muted/30 rounded p-2">
            <div className="text-xs text-muted-foreground">Total</div>
            <div className="text-lg font-bold text-primary" data-testid="text-total-features">
              {featureComputation.totalFeatures}
            </div>
          </div>
          <div className="bg-muted/30 rounded p-2">
            <div className="text-xs text-muted-foreground">Computed</div>
            <div className="text-lg font-bold text-emerald-400" data-testid="text-features-computed">
              {featureComputation.featuresComputed}
            </div>
          </div>
          <div className="bg-muted/30 rounded p-2">
            <div className="text-xs text-muted-foreground">Time</div>
            <div className="text-lg font-bold" data-testid="text-computation-time">
              {featureComputation.computationTime}ms
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground">Feature Categories</div>
          <div className="grid grid-cols-2 gap-1.5">
            {Object.entries(featureComputation.featureCategories).map(([cat, count]) => (
              <div key={cat} className="flex items-center justify-between text-xs bg-muted/20 rounded px-2 py-1">
                <span className="capitalize">{cat}</span>
                <Badge variant="secondary" className="text-[10px]">{count}</Badge>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-2 pt-2 border-t">
          <div className="text-xs font-medium text-muted-foreground">Top Features by Importance</div>
          {featureComputation.topFeatures.slice(0, 5).map((feature, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-mono">{feature.name}</span>
                  <span className="text-muted-foreground">{(feature.importance * 100).toFixed(0)}%</span>
                </div>
                <Progress value={feature.importance * 100} className="h-1 mt-1" />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function ModelPerformanceCard({ learningStats }: LearningStatsCardProps) {
  if (!learningStats) {
    return (
      <Card data-testid="card-model-performance-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Brain className="h-4 w-4 text-muted-foreground" />
            ML Ensemble
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading model stats...</p>
        </CardContent>
      </Card>
    );
  }

  const { modelPerformance, ensembleStats } = learningStats;

  // Calculate overall signal frequency from all models
  const totalSignals = modelPerformance.reduce((acc, m) => ({
    long: acc.long + m.signalDistribution.long,
    short: acc.short + m.signalDistribution.short,
    hold: acc.hold + m.signalDistribution.hold,
  }), { long: 0, short: 0, hold: 0 });
  const totalCount = totalSignals.long + totalSignals.short + totalSignals.hold;
  const signalPcts = totalCount > 0 ? {
    long: ((totalSignals.long / totalCount) * 100).toFixed(0),
    short: ((totalSignals.short / totalCount) * 100).toFixed(0),
    hold: ((totalSignals.hold / totalCount) * 100).toFixed(0),
  } : { long: "0", short: "0", hold: "100" };

  return (
    <Card data-testid="card-model-performance">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Brain className="h-4 w-4 text-primary" />
          ML Ensemble Performance
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-muted/30 rounded p-2">
            <div className="text-xs text-muted-foreground">Predictions</div>
            <div className="text-lg font-bold text-primary" data-testid="text-total-predictions">
              {ensembleStats.totalPredictions}
            </div>
          </div>
          <div className="bg-muted/30 rounded p-2">
            <div className="text-xs text-muted-foreground">Consensus</div>
            <div className="text-lg font-bold text-emerald-400" data-testid="text-consensus-rate">
              {(ensembleStats.consensusRate * 100).toFixed(0)}%
            </div>
          </div>
          <div className="bg-muted/30 rounded p-2">
            <div className="text-xs text-muted-foreground">Confidence</div>
            <div className="text-lg font-bold text-amber-400" data-testid="text-avg-confidence">
              {(ensembleStats.avgConfidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Signal Frequency Breakdown */}
        <div className="bg-muted/20 rounded p-2">
          <div className="text-xs font-medium text-muted-foreground mb-2">Signal Frequency (All Models)</div>
          <div className="flex items-center gap-2 text-xs">
            <Badge variant="outline" className="text-emerald-400">
              <TrendingUp className="h-2.5 w-2.5 mr-1" />
              {signalPcts.long}% Long
            </Badge>
            <Badge variant="outline" className="text-red-400">
              {signalPcts.short}% Short
            </Badge>
            <Badge variant="outline" className="text-muted-foreground">
              {signalPcts.hold}% Hold
            </Badge>
          </div>
          {Number(signalPcts.hold) > 80 && (
            <div className="text-[10px] text-amber-400 mt-1">
              System is conservative - mostly HOLD (normal in choppy markets)
            </div>
          )}
        </div>

        <div className="space-y-3">
          {modelPerformance.map((model, i) => {
            const m = model as any; // Access optional fields
            const hasDirectionalData = m.directionalStats && m.directionalStats.total > 0;
            
            return (
              <div key={i} className="bg-muted/20 rounded p-2 space-y-2" data-testid={`model-${model.modelName.toLowerCase().replace(/\s/g, '-')}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Zap className="h-3 w-3 text-primary" />
                    <span className="text-sm font-medium">{model.modelName}</span>
                    <Badge variant="outline" className="text-[10px]">{model.weight}%</Badge>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {model.predictionsToday} predictions
                  </span>
                </div>
                
                {/* Directional Accuracy (excludes HOLD) */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex flex-col">
                    <div className="flex justify-between text-xs text-muted-foreground mb-1">
                      <span>Dir. Accuracy (ex-HOLD)</span>
                      <span className={hasDirectionalData ? (m.directionalAccuracy >= 50 ? "text-emerald-400" : "text-amber-400") : "text-muted-foreground"}>
                        {hasDirectionalData ? `${m.directionalAccuracy}%` : "N/A"}
                      </span>
                    </div>
                    <Progress 
                      value={hasDirectionalData ? m.directionalAccuracy : 0} 
                      className={`h-1.5 ${!hasDirectionalData ? "opacity-30" : ""}`} 
                    />
                    {hasDirectionalData && (
                      <div className="text-[10px] text-muted-foreground mt-0.5">
                        {m.directionalStats.correct}/{m.directionalStats.total} correct
                      </div>
                    )}
                  </div>
                  <div className="flex flex-col">
                    <div className="flex justify-between text-xs text-muted-foreground mb-1">
                      <span>HOLD Rate</span>
                      <span className={(m.holdRate || 0) > 80 ? "text-amber-400" : "text-muted-foreground"}>
                        {m.holdRate || 0}%
                      </span>
                    </div>
                    <Progress value={m.holdRate || 0} className="h-1.5" />
                  </div>
                </div>

                <div className="flex items-center gap-2 text-xs">
                  <Badge variant="outline" className="text-emerald-400 text-[10px]">
                    <TrendingUp className="h-2 w-2 mr-1" />
                    {model.signalDistribution.long} Long
                  </Badge>
                  <Badge variant="outline" className="text-red-400 text-[10px]">
                    {model.signalDistribution.short} Short
                  </Badge>
                  <Badge variant="outline" className="text-muted-foreground text-[10px]">
                    {model.signalDistribution.hold} Hold
                  </Badge>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

export function LearningOverviewCard({ learningStats }: LearningStatsCardProps) {
  if (!learningStats) {
    return (
      <Card data-testid="card-learning-overview-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Activity className="h-4 w-4 text-muted-foreground" />
            Learning Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading learning stats...</p>
        </CardContent>
      </Card>
    );
  }

  const totalSignals = learningStats.modelPerformance.reduce((acc, m) => acc + m.predictionsToday, 0) / 3;

  return (
    <Card data-testid="card-learning-overview">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          System Learning Overview
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 rounded-lg p-3 text-center">
            <Database className="h-5 w-5 mx-auto mb-1 text-blue-400" />
            <div className="text-xs text-muted-foreground">Data Points</div>
            <div className="text-xl font-bold text-blue-400" data-testid="text-data-points">
              {learningStats.dataIngestion.candlesTotal}
            </div>
          </div>
          <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 rounded-lg p-3 text-center">
            <Layers className="h-5 w-5 mx-auto mb-1 text-purple-400" />
            <div className="text-xs text-muted-foreground">Patterns</div>
            <div className="text-xl font-bold text-purple-400" data-testid="text-patterns-count">
              {learningStats.patternLearning.totalPatterns}
            </div>
          </div>
          <div className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 rounded-lg p-3 text-center">
            <Cpu className="h-5 w-5 mx-auto mb-1 text-emerald-400" />
            <div className="text-xs text-muted-foreground">Features</div>
            <div className="text-xl font-bold text-emerald-400" data-testid="text-features-count">
              {learningStats.featureComputation.totalFeatures}
            </div>
          </div>
          <div className="bg-gradient-to-br from-amber-500/10 to-amber-600/5 rounded-lg p-3 text-center">
            <Brain className="h-5 w-5 mx-auto mb-1 text-amber-400" />
            <div className="text-xs text-muted-foreground">Predictions</div>
            <div className="text-xl font-bold text-amber-400" data-testid="text-predictions-count">
              {learningStats.ensembleStats.totalPredictions}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function SocialAwarenessCard({ learningStats }: LearningStatsCardProps) {
  if (!learningStats?.socialAwareness) {
    return (
      <Card data-testid="card-social-awareness-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Globe className="h-4 w-4 text-muted-foreground" />
            Live Feeling (Current)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading live sentiment data...</p>
        </CardContent>
      </Card>
    );
  }

  const { socialAwareness } = learningStats;

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "text-emerald-400 bg-emerald-500/20";
      case "idle": return "text-muted-foreground bg-muted";
      case "error": return "text-red-400 bg-red-500/20";
      default: return "text-muted-foreground bg-muted";
    }
  };

  const formatTime = (ts: number | null) => {
    if (!ts) return "Never";
    const seconds = Math.floor((Date.now() - ts) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  const getPlatformIcon = (icon: string) => {
    switch (icon) {
      case "gauge": return <Gauge className="h-4 w-4" />;
      case "newspaper": return <Newspaper className="h-4 w-4" />;
      case "twitter": return <SiX className="h-4 w-4" />;
      case "reddit": return <SiReddit className="h-4 w-4" />;
      default: return <MessageCircle className="h-4 w-4" />;
    }
  };

  const getSentimentColor = (sentiment: number) => {
    if (sentiment >= 0.6) return "text-emerald-400";
    if (sentiment <= 0.4) return "text-red-400";
    return "text-amber-400";
  };

  const getSentimentLabel = (sentiment: number) => {
    if (sentiment >= 0.7) return "Bullish";
    if (sentiment >= 0.55) return "Slightly Bullish";
    if (sentiment >= 0.45) return "Neutral";
    if (sentiment >= 0.3) return "Slightly Bearish";
    return "Bearish";
  };

  return (
    <Card data-testid="card-social-awareness">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Globe className="h-4 w-4 text-cyan-400" />
          Live Feeling (Current)
          <Badge variant="outline" className="ml-auto text-xs">
            {socialAwareness.totalItemsRead} items read
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-[10px] text-amber-500/80 bg-amber-500/10 rounded px-2 py-1 mb-2">
          Affects live signals only, not backtests
        </div>
        {/* Global Sentiment Summary */}
        <div className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Global Market Sentiment</span>
            <span className={`text-lg font-bold ${getSentimentColor(socialAwareness.globalSentiment)}`}>
              {(socialAwareness.globalSentiment * 100).toFixed(0)}%
            </span>
          </div>
          <Progress 
            value={socialAwareness.globalSentiment * 100} 
            className="h-2"
          />
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>Bearish</span>
            <span className={getSentimentColor(socialAwareness.globalSentiment)}>
              {getSentimentLabel(socialAwareness.globalSentiment)}
            </span>
            <span>Bullish</span>
          </div>
        </div>

        {/* Platform List */}
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground font-medium">Data Sources by Platform</div>
          {socialAwareness.platforms.map((platform, i) => (
            <div 
              key={i} 
              className="flex items-center justify-between p-2 bg-muted/30 rounded"
              data-testid={`platform-${platform.platform.toLowerCase().replace(/[\s\/]/g, '-')}`}
            >
              <div className="flex items-center gap-2">
                <div className="text-muted-foreground">
                  {getPlatformIcon(platform.icon)}
                </div>
                <span className="text-sm font-medium">{platform.platform}</span>
                <Badge className={getStatusColor(platform.status)} variant="outline">
                  {platform.status}
                </Badge>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-muted-foreground">
                  <span className="font-medium text-foreground">{platform.itemsRead}</span> reads
                </span>
                <span className={getSentimentColor(platform.sentiment)}>
                  {(platform.sentiment * 100).toFixed(0)}%
                </span>
                <span className="text-muted-foreground">{formatTime(platform.lastFetch)}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Sentiment Confidence Modifier */}
        <div className="pt-2 border-t">
          <div className="text-xs text-muted-foreground font-medium mb-2">Signal Confidence Impact</div>
          <div className="bg-muted/30 rounded-lg p-3">
            {(() => {
              const sentiment = socialAwareness.globalSentiment;
              const modifier = sentiment >= 0.6 ? +5 : sentiment <= 0.4 ? -5 : 0;
              const direction = sentiment >= 0.6 ? "bullish" : sentiment <= 0.4 ? "bearish" : "neutral";
              
              return (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Gauge className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm">
                      {direction === "bullish" && "Bullish sentiment boosts LONG confidence"}
                      {direction === "bearish" && "Bearish sentiment boosts SHORT confidence"}
                      {direction === "neutral" && "Neutral sentiment - no modifier"}
                    </span>
                  </div>
                  <Badge 
                    variant="outline" 
                    className={
                      modifier > 0 ? "text-emerald-400 border-emerald-500/30" :
                      modifier < 0 ? "text-red-400 border-red-500/30" :
                      "text-muted-foreground"
                    }
                  >
                    {modifier > 0 ? `+${modifier}%` : modifier < 0 ? `${modifier}%` : "0%"}
                  </Badge>
                </div>
              );
            })()}
          </div>
        </div>

        {/* Last Update */}
        <div className="pt-2 border-t text-xs text-muted-foreground flex items-center gap-1">
          <Clock className="h-3 w-3" />
          Last social update: {formatTime(socialAwareness.lastGlobalUpdate)}
        </div>
      </CardContent>
    </Card>
  );
}

interface HistoricalDataStatus {
  totalCandles: number;
  daysOfData: number;
  startDate: string | null;
  endDate: string | null;
  backfillComplete: boolean;
  completionPct: number;
  expectedFor365Days: number;
}

interface IntegrityReport {
  totalCandles: number;
  daysOfData: number;
  completionPct: number;
  missingRanges: Array<{ start: string; end: string; gapCandles: number }>;
  duplicateCount: number;
  lastCandleTs: number | null;
  alignmentHealthy: boolean;
  overallHealth: "complete" | "missing_ranges" | "out_of_sync" | "no_data";
}

interface HistoricalLearningCardProps extends LearningStatsCardProps {
  historicalStatus?: HistoricalDataStatus | null;
  integrityReport?: IntegrityReport | null;
  onBackfill?: () => void;
  backfillInProgress?: boolean;
  backfillProgress?: number;
}

export function HistoricalLearningCard({ 
  learningStats, 
  historicalStatus,
  integrityReport,
  onBackfill,
  backfillInProgress,
  backfillProgress
}: HistoricalLearningCardProps) {
  if (!learningStats?.historicalLearning) {
    return (
      <Card data-testid="card-historical-learning-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <History className="h-4 w-4 text-muted-foreground" />
            Historical Data Learning
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading historical learning stats...</p>
        </CardContent>
      </Card>
    );
  }

  const { historicalLearning } = learningStats;

  const formatTime = (ts: number | null) => {
    if (!ts) return "Never";
    const seconds = Math.floor((Date.now() - ts) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  const displayDays = historicalStatus?.daysOfData || 
    Math.round((new Date(historicalLearning.dataRangeEnd).getTime() - 
                new Date(historicalLearning.dataRangeStart).getTime()) / (24 * 60 * 60 * 1000));
  
  const displayCandles = historicalStatus?.totalCandles || historicalLearning.totalHistoricalCandles;

  return (
    <Card data-testid="card-historical-learning">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <History className="h-4 w-4 text-indigo-400" />
          Historical Data Learning
          <Badge variant="outline" className="ml-auto text-xs bg-blue-500/10 text-blue-400 border-blue-500/30">
            Price-only
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-[10px] text-blue-500/80 bg-blue-500/10 rounded px-2 py-1 mb-2">
          Backtests use historical price/volume only. No sentiment applied retroactively.
        </div>
        
        {/* Backfill Status */}
        {backfillInProgress && (
          <div className="bg-gradient-to-br from-amber-500/10 to-orange-500/10 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Fetching Historical Data...</span>
              <span className="text-lg font-bold text-amber-400">
                {backfillProgress?.toFixed(0) || 0}%
              </span>
            </div>
            <Progress value={backfillProgress || 0} className="h-2" />
          </div>
        )}
        
        {/* Data Completion Progress */}
        <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Data Completion (1 Year Target)</span>
            <span className="text-lg font-bold text-blue-400">
              {(historicalStatus?.completionPct ?? 0).toFixed(1)}%
            </span>
          </div>
          <Progress 
            value={historicalStatus?.completionPct ?? 0} 
            className="h-2"
          />
          <div className="text-xs text-muted-foreground mt-1">
            {displayCandles.toLocaleString()} / {(historicalStatus?.expectedFor365Days ?? 35040).toLocaleString()} candles
          </div>
        </div>

        {/* Data Health Status */}
        {integrityReport && (
          <div className="flex items-center gap-2 p-2 bg-muted/30 rounded">
            {integrityReport.overallHealth === "complete" && (
              <Badge variant="outline" className="text-emerald-400 bg-emerald-500/10 border-emerald-500/30">
                <CheckCircle2 className="h-3 w-3 mr-1" /> Healthy
              </Badge>
            )}
            {integrityReport.overallHealth === "missing_ranges" && (
              <Badge variant="outline" className="text-amber-400 bg-amber-500/10 border-amber-500/30">
                <AlertCircle className="h-3 w-3 mr-1" /> {integrityReport.missingRanges.length} Gaps
              </Badge>
            )}
            {integrityReport.overallHealth === "out_of_sync" && (
              <Badge variant="outline" className="text-red-400 bg-red-500/10 border-red-500/30">
                <AlertCircle className="h-3 w-3 mr-1" /> Out of Sync
              </Badge>
            )}
            {integrityReport.overallHealth === "no_data" && (
              <Badge variant="outline" className="text-muted-foreground">
                No Data
              </Badge>
            )}
            <span className="text-xs text-muted-foreground ml-auto">
              {integrityReport.duplicateCount > 0 && `${integrityReport.duplicateCount} dupes`}
              {integrityReport.duplicateCount > 0 && !integrityReport.alignmentHealthy && " | "}
              {!integrityReport.alignmentHealthy && "alignment issues"}
            </span>
          </div>
        )}

        {/* Learning Progress */}
        <div className="bg-gradient-to-br from-indigo-500/10 to-purple-500/10 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Training Progress</span>
            <span className="text-lg font-bold text-indigo-400">
              {historicalLearning.learningProgress.toFixed(0)}%
            </span>
          </div>
          <Progress 
            value={historicalLearning.learningProgress} 
            className="h-2"
          />
          <div className="text-xs text-muted-foreground mt-1">
            {historicalLearning.epochsCompleted} training epochs completed
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <Database className="h-4 w-4 mx-auto mb-1 text-blue-400" />
            <div className="text-xs text-muted-foreground">Historical Candles (15m)</div>
            <div className="text-lg font-bold" data-testid="text-historical-candles">
              {displayCandles.toLocaleString()}
            </div>
            <div className="text-[10px] text-muted-foreground">{displayDays} days</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <BookOpen className="h-4 w-4 mx-auto mb-1 text-purple-400" />
            <div className="text-xs text-muted-foreground">Pattern Clusters</div>
            <div className="text-lg font-bold" data-testid="text-patterns-learned">
              {learningStats.patternLearning?.totalPatterns ?? 0}
            </div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <Target className="h-4 w-4 mx-auto mb-1 text-amber-400" />
            <div className="text-xs text-muted-foreground">Backtest Trades</div>
            <div className="text-lg font-bold" data-testid="text-backtest-trades">
              {historicalLearning.backtestTrades}
            </div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <TrendingUp className="h-4 w-4 mx-auto mb-1 text-emerald-400" />
            <div className="text-xs text-muted-foreground">Backtest Win Rate</div>
            <div className="text-lg font-bold text-emerald-400" data-testid="text-historical-winrate">
              {historicalLearning.historicalWinRate.toFixed(1)}%
            </div>
            <div className="text-[10px] text-muted-foreground">(executed trades only)</div>
          </div>
        </div>

        {/* Training Range */}
        <div className="pt-2 border-t">
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs text-muted-foreground">Training Coverage</div>
            <Badge variant="outline" className={
              (historicalLearning.trainingCoverage ?? 100) >= 80 
                ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30" 
                : "text-amber-400 bg-amber-500/10 border-amber-500/30"
            }>
              {historicalLearning.trainingCoverage ?? 100}% Used
            </Badge>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs mb-3">
            <div className="bg-muted/20 rounded p-2">
              <div className="text-muted-foreground">In Training</div>
              <div className="font-medium">{(historicalLearning.candlesUsedForTraining ?? 0).toLocaleString()}</div>
            </div>
            <div className="bg-muted/20 rounded p-2">
              <div className="text-muted-foreground">Available</div>
              <div className="font-medium">{(historicalLearning.candlesAvailable ?? 0).toLocaleString()}</div>
            </div>
          </div>
          
          <div className="text-xs text-muted-foreground mb-2">Data Range</div>
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">{historicalStatus?.startDate || historicalLearning.dataRangeStart}</span>
            <span className="text-muted-foreground">→</span>
            <span className="font-medium">{historicalStatus?.endDate || historicalLearning.dataRangeEnd}</span>
          </div>
          <div className="text-xs text-muted-foreground mt-2 flex items-center gap-1">
            <Clock className="h-3 w-3" />
            Last training: {formatTime(historicalLearning.lastTrainingTime)}
          </div>
          
          {/* Backfill Button */}
          {onBackfill && !backfillInProgress && displayDays < 300 && (
            <Button
              onClick={onBackfill}
              variant="outline"
              size="sm"
              className="mt-3 w-full"
              data-testid="button-backfill"
            >
              <Database className="h-3 w-3 mr-2" />
              Fetch 1 Year Historical Data
            </Button>
          )}
          {historicalStatus?.backfillComplete && (
            <div className="mt-2 text-[10px] text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Historical data complete
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
