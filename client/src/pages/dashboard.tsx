import { useQuery, useMutation } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { SignalCard } from "@/components/signal-card";
import { RegimeCard } from "@/components/regime-card";
import { FeaturesCard } from "@/components/features-card";
import { FuturesMetricsCard } from "@/components/futures-metrics-card";
import { RiskModeCard } from "@/components/risk-mode-card";
import { StatsCard } from "@/components/stats-card";
import { TradeHistory } from "@/components/trade-history";
import { PriceChart } from "@/components/price-chart";
import { StrategyControl } from "@/components/strategy-control";
import { ActiveTradePanel } from "@/components/active-trade-panel";
import { ThemeToggle } from "@/components/theme-toggle";
import { AIAnalysisCard } from "@/components/ai-analysis-card";
import { IndicatorsCard } from "@/components/indicators-card";
import { MTFScoreCard } from "@/components/mtf-score-card";
import { WhaleActivityCard } from "@/components/whale-activity-card";
import { PerformanceStatsCard } from "@/components/performance-stats-card";
import { StrategySelectorCard } from "@/components/strategy-selector-card";
import { ShotPlanCard } from "@/components/shot-plan-card";
import { SentimentCard } from "@/components/sentiment-card";
import { 
  DataSourcesCard, 
  PatternLearningCard, 
  FeatureComputationCard, 
  ModelPerformanceCard, 
  LearningOverviewCard,
  SocialAwarenessCard,
  HistoricalLearningCard
} from "@/components/learning-stats-card";
import type { DashboardData } from "@shared/schema";
import { Loader2, RefreshCw, Bitcoin, Clock, Wifi, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { format } from "date-fns";
import { apiRequest, queryClient } from "@/lib/queryClient";

export default function Dashboard() {
  const { data, isLoading, error, refetch, isFetching } = useQuery<DashboardData>({
    queryKey: ["/api/dashboard"],
    refetchInterval: 15000,
  });

  const hasTriggeredAnalysis = useRef(false);
  
  const analyzeMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/ai/analyze"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] });
    },
  });

  useEffect(() => {
    if (data && !data.aiAnalysis && !hasTriggeredAnalysis.current && !analyzeMutation.isPending) {
      hasTriggeredAnalysis.current = true;
      analyzeMutation.mutate();
    }
  }, [data, analyzeMutation]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background" data-testid="loading-container">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" data-testid="loading-spinner" />
          <p className="text-muted-foreground" data-testid="text-loading">Loading trading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background" data-testid="error-container">
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="p-4 rounded-full bg-destructive/10">
            <Bitcoin className="h-8 w-8 text-destructive" />
          </div>
          <p className="text-destructive" data-testid="text-error">Failed to load dashboard data</p>
          <Button onClick={() => refetch()} data-testid="button-retry">
            Try Again
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background" data-testid="dashboard-container">
      <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60" data-testid="header">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="p-2 rounded-md bg-primary/10" data-testid="logo-container">
                  <Bitcoin className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h1 className="text-lg font-semibold" data-testid="text-app-title">BTC Futures Signal</h1>
                  <p className="text-xs text-muted-foreground" data-testid="text-app-subtitle">Kalman Trend Strategy</p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Badge 
                variant="secondary" 
                className={`text-xs flex items-center gap-1 ${data.isLiveData ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}`}
                data-testid="badge-data-source"
              >
                {data.isLiveData ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
                {data.dataSource === "coingecko" ? "CoinGecko" : data.dataSource === "cryptocompare" ? "CryptoCompare" : data.dataSource === "binance" ? "Binance" : "No Data"}
              </Badge>
              <div className="flex items-center gap-2 text-xs text-muted-foreground" data-testid="last-update-container">
                <Clock className="h-3.5 w-3.5" />
                <span data-testid="text-last-update">Updated {format(new Date(data.currentSignal.timestamp), "HH:mm:ss")}</span>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => refetch()}
                disabled={isFetching}
                data-testid="button-refresh"
              >
                <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
              </Button>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6" data-testid="main-content">
        {data.dataError && (
          <div className="mb-4 p-4 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400" data-testid="alert-data-error">
            <div className="flex items-center gap-2">
              <WifiOff className="h-5 w-5" />
              <span className="font-medium">No Market Data Available</span>
            </div>
            <p className="mt-1 text-sm text-red-400/80">{data.dataError}</p>
          </div>
        )}
        
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="mb-4" data-testid="tabs-list">
            <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
            <TabsTrigger value="signal" data-testid="tab-signal">Signal</TabsTrigger>
            <TabsTrigger value="learning" data-testid="tab-learning">Learning</TabsTrigger>
            <TabsTrigger value="analysis" data-testid="tab-analysis">AI Analysis</TabsTrigger>
            <TabsTrigger value="indicators" data-testid="tab-indicators">Indicators</TabsTrigger>
            <TabsTrigger value="performance" data-testid="tab-performance">Performance</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-0">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              <div className="lg:col-span-8 space-y-4">
                <PriceChart 
                  candles={data.candles} 
                  kalmanFast={data.kalmanFast}
                  kalmanSlow={data.kalmanSlow}
                  strategySignal={data.strategySignal}
                  activeTrade={data.activeTrade}
                  recentTrades={data.recentTrades}
                />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <FuturesMetricsCard data={data.futuresData} />
                  <FeaturesCard features={data.currentSignal.topFeatures} />
                </div>
              </div>

              <div className="lg:col-span-4 space-y-4">
                <StrategyControl 
                  strategyState={data.strategyState}
                  strategySignal={data.strategySignal}
                  activeTrade={data.activeTrade}
                />
                {data.activeTrade && (
                  <ActiveTradePanel 
                    trade={data.activeTrade}
                    currentPrice={data.candles[data.candles.length - 1]?.close ?? 0}
                  />
                )}
                <SignalCard signal={data.currentSignal} />
                <RegimeCard signal={data.currentSignal} />
                <StatsCard
                  equity={data.equity}
                  dailyPnl={data.dailyPnl}
                  winRate={data.winRate}
                  profitFactor={data.profitFactor}
                  totalTrades={data.totalTrades}
                />
                <RiskModeCard
                  riskMode={data.currentSignal.riskMode}
                  drawdown={data.drawdown}
                  maxDrawdown={data.maxDrawdown}
                  exposure={data.exposure}
                />
              </div>

              <div className="lg:col-span-12">
                <TradeHistory trades={data.recentTrades} />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="signal" className="mt-0">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              <div className="lg:col-span-8 space-y-4">
                <PriceChart 
                  candles={data.candles} 
                  kalmanFast={data.kalmanFast}
                  kalmanSlow={data.kalmanSlow}
                  strategySignal={data.strategySignal}
                  activeTrade={data.activeTrade}
                  recentTrades={data.recentTrades}
                />
                <AIAnalysisCard analysis={data.aiAnalysis} />
              </div>
              <div className="lg:col-span-4 space-y-4">
                <ShotPlanCard shotPlan={data.shotPlan} />
                <SentimentCard sentiment={data.sentiment} />
                <SignalCard signal={data.currentSignal} />
                <RegimeCard signal={data.currentSignal} />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="learning" className="mt-0">
            <div className="space-y-4">
              <LearningOverviewCard learningStats={data.learningStats} />
              
              {/* Social Awareness & Historical Learning - Key new sections */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <SocialAwarenessCard learningStats={data.learningStats} />
                <HistoricalLearningCard learningStats={data.learningStats} />
              </div>
              
              {/* Data Sources & Pattern Memory */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                <div className="lg:col-span-6 space-y-4">
                  <DataSourcesCard learningStats={data.learningStats} />
                  <PatternLearningCard learningStats={data.learningStats} />
                </div>
                <div className="lg:col-span-6 space-y-4">
                  <FeatureComputationCard learningStats={data.learningStats} />
                  <ModelPerformanceCard learningStats={data.learningStats} />
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="analysis" className="mt-0">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              <div className="lg:col-span-8 space-y-4">
                <AIAnalysisCard analysis={data.aiAnalysis} />
                <StrategySelectorCard 
                  indicators={data.indicators}
                  mtfScore={data.mtfScore}
                  strategySignal={data.strategySignal}
                />
              </div>
              <div className="lg:col-span-4 space-y-4">
                <MTFScoreCard mtfScore={data.mtfScore} />
                <WhaleActivityCard whaleActivity={data.whaleActivity} />
                <SignalCard signal={data.currentSignal} />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="indicators" className="mt-0">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              <div className="lg:col-span-8 space-y-4">
                <PriceChart 
                  candles={data.candles} 
                  kalmanFast={data.kalmanFast}
                  kalmanSlow={data.kalmanSlow}
                  strategySignal={data.strategySignal}
                  activeTrade={data.activeTrade}
                  recentTrades={data.recentTrades}
                />
                <IndicatorsCard indicators={data.indicators} />
              </div>
              <div className="lg:col-span-4 space-y-4">
                <MTFScoreCard mtfScore={data.mtfScore} />
                <WhaleActivityCard whaleActivity={data.whaleActivity} />
                <FuturesMetricsCard data={data.futuresData} />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="performance" className="mt-0">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              <div className="lg:col-span-8 space-y-4">
                <TradeHistory trades={data.recentTrades} />
              </div>
              <div className="lg:col-span-4 space-y-4">
                <PerformanceStatsCard stats={data.performanceStats} equity={data.equity} />
                <StatsCard
                  equity={data.equity}
                  dailyPnl={data.dailyPnl}
                  winRate={data.winRate}
                  profitFactor={data.profitFactor}
                  totalTrades={data.totalTrades}
                />
                <RiskModeCard
                  riskMode={data.currentSignal.riskMode}
                  drawdown={data.drawdown}
                  maxDrawdown={data.maxDrawdown}
                  exposure={data.exposure}
                />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </main>

      <footer className="border-t border-border py-4 mt-8" data-testid="footer">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span data-testid="text-footer-title">BTC Futures Trading Signal Dashboard</span>
            <div className="flex items-center gap-4">
              <Badge variant="secondary" className="text-xs" data-testid="badge-paper-trading">
                Paper Trading Mode
              </Badge>
              <span data-testid="text-footer-info">Kalman (70/250) | 15m timeframe</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
