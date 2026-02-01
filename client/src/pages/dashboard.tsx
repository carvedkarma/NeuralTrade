import { useQuery, useMutation } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { SignalCard } from "@/components/signal-card";
import { EnsembleSignalCard } from "@/components/ensemble-signal-card";
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
import { EnhancedShotPlanCard } from "@/components/enhanced-shot-plan-card";
import { SentimentCard } from "@/components/sentiment-card";
import { 
  DataSourcesCard, 
  PatternLearningCard, 
  FeatureComputationCard, 
  ModelPerformanceCard, 
  LearningOverviewCard,
  SocialAwarenessCard,
  HistoricalLearningCard,
  UnifiedLearningProgressCard
} from "@/components/learning-stats-card";
import { GPUTrainingSection, type GPUMetrics } from "@/components/gpu-training-card";
import { GPUConnectionSettings } from "@/components/gpu-connection-settings";
import { 
  CrossAssetOverviewCard, 
  CorrelationMatrixCard, 
  RelativeStrengthCard, 
  PriceComparisonChart,
  type CrossAssetData 
} from "@/components/cross-asset-card";
import {
  PerformanceCard,
  EquityPerformanceCard,
  RiskStatusCard,
  OpenPositionCard,
  PositionHistoryCard,
  AuditLogPanel
} from "@/components/paper-trading-card";
import { StrategyLearnerTab } from "@/components/strategy-learner-card";
import { DataManagementCard } from "@/components/data-management-card";
import { NeuralNetworkDataCard } from "@/components/nn-data-card";
import { NeuralNetworkPredictionCard, TrainingModeBadge, type QuantilePrediction, type PredictionTrace } from "@/components/neural-network-prediction";
import { QuantileFanChart, DerivedTradeLevels } from "@/components/quantile-fan-chart";
import { PremiumCandlestickChart } from "@/components/premium-candlestick-chart";
import { ConeSignalCard } from "@/components/cone-signal-card";
import type { DashboardData } from "@shared/schema";
import { Loader2, RefreshCw, Bitcoin, Clock, Wifi, WifiOff, Brain, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { format } from "date-fns";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface HistoricalStatus {
  totalCandles: number;
  daysOfData: number;
  startDate: string | null;
  endDate: string | null;
  backfillComplete: boolean;
  completionPct: number;
  expectedForTarget: number;
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

export default function Dashboard() {
  const [backfillInProgress, setBackfillInProgress] = useState(false);
  const [backfillProgress, setBackfillProgress] = useState(0);

  const { data, isLoading, error, refetch, isFetching } = useQuery<DashboardData>({
    queryKey: ["/api/dashboard"],
    refetchInterval: 5000,
  });

  const { data: historicalStatus } = useQuery<HistoricalStatus>({
    queryKey: ["/api/historical/status"],
    refetchInterval: backfillInProgress ? 2000 : 30000,
  });

  const { data: integrityReport } = useQuery<IntegrityReport>({
    queryKey: ["/api/historical/integrity"],
    refetchInterval: 60000,
  });

  const { data: gpuStatus } = useQuery<{ 
    connected: boolean; 
    isStale: boolean;
    gpuAvailable: boolean;
    gpuName: string | null;
    gpuMemoryUsed: number | null;
    gpuMemoryTotal: number | null;
    isTraining: boolean;
    trainingProgress: number;
    currentModel: string | null;
    currentEpoch: number;
    totalEpochs: number;
    trainLoss: number | null;
    valLoss: number | null;
    modelsCompleted: string[];
    modelStatus?: Record<string, {
      status: string;
      accuracy: number | null;
      loss: number | null;
      epochs: number;
      best_epoch: number;
    }>;
    trainingMode?: "quick" | "full" | null;
    trainingModeDescription?: string | null;
    inputDim?: number | null;
  }>({
    queryKey: ["/api/gpu/pushed-status"],
    refetchInterval: 5000,
  });

  const trainModelMutation = useMutation({
    mutationFn: (modelType: string) => apiRequest("POST", "/api/gpu/train", { modelType, epochs: 100 }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/gpu/pushed-status"] });
    },
  });

  const { data: crossAssetData } = useQuery<CrossAssetData>({
    queryKey: ["/api/cross-asset"],
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: unifiedProgress } = useQuery<{
    overallProgress: number;
    systems: { name: string; index: number; progress: number; complete: boolean }[];
    totalCandles: number;
    trainableCandles: number;
    allAligned: boolean;
  }>({
    queryKey: ["/api/unified-learning/progress"],
    refetchInterval: 10000,
  });

  const { data: dataSummary } = useQuery<{
    assets: { symbol: string; totalCandles: number }[];
    totalCandles: number;
  }>({
    queryKey: ["/api/data/summary"],
    refetchInterval: 10000,
  });

  const { data: trainingStatus } = useQuery<{
    deepLearning: { started: boolean; progress: number };
    strategyLearner: { started: boolean; epochs: number };
  }>({
    queryKey: ["/api/training/status"],
    refetchInterval: 5000,
  });

  const { data: ensembleStatus, isLoading: ensembleLoading } = useQuery<{
    available: boolean;
    status: {
      initialized: boolean;
      direction_models: string[];
      regime_models: string[];
      risk_models: string[];
    } | null;
  }>({
    queryKey: ["/api/gpu/ensemble/status"],
    refetchInterval: 10000,
  });

  const { data: ensemblePrediction } = useQuery<{
    available: boolean;
    prediction: {
      action: "LONG" | "SHORT" | "HOLD" | "NO_TRADE";
      confidence: number;
      confidence_margin: number;
      edge: number;
      market_regime: string;
      risk_regime: string;
      regime_confidence: number;
      agreement_pct: number;
      weighted_agreement: number;
      disagreement_score: number;
      position_size_pct: number;
      regime_adjusted_size: number;
      confidence_threshold_used: number;
      regime_adjustment: string;
      model_votes: Record<string, {
        action: string;
        confidence: number;
        confidence_margin: number;
        weight: number;
        probs: { SHORT: number; HOLD: number; LONG: number };
      }>;
      ensemble_probs: { SHORT: number; HOLD: number; LONG: number };
      reasons: string[];
    } | null;
    message?: string;
  }>({
    queryKey: ["/api/gpu/ensemble/current"],
    refetchInterval: 15000,
    enabled: ensembleStatus?.available === true,
  });

  // Neural Network Quantile Predictions
  const { data: nnPrediction, isLoading: nnPredictionLoading, refetch: refetchNnPrediction } = useQuery<{
    available: boolean;
    prediction: QuantilePrediction | null;
    predictedCandles: Array<{
      timestamp: number;
      q10: number;
      q25: number;
      q50: number;
      q75: number;
      q90: number;
      direction: "up" | "down";
    }>;
    trace?: PredictionTrace;
    error?: string;
  }>({
    queryKey: ["/api/gpu/nn-prediction"],
    queryFn: async () => {
      // Use cache-busting to ensure fresh request
      const res = await fetch(`/api/gpu/nn-prediction?_t=${Date.now()}`, { 
        cache: "no-store" 
      });
      if (!res.ok) throw new Error("Failed to fetch NN prediction");
      return res.json();
    },
    refetchInterval: 30000,
    enabled: gpuStatus?.connected === true,
  });

  // Check if historical data has been downloaded (minimum 1000 candles)
  const hasHistoricalData = dataSummary && dataSummary.totalCandles >= 1000;
  const hasDeepLearningStarted = trainingStatus?.deepLearning?.started || false;

  const startDeepLearningMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/deep-learning/start"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/training/status"] });
      queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] });
    },
  });

  const resetLearningMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/unified-learning/reset"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/unified-learning/progress"] });
      queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] });
    },
  });

  const handleResetLearning = () => {
    if (confirm("This will reset all 3 learning systems (Strategy Learner, Pattern Memory, GPU Trainer). Are you sure?")) {
      resetLearningMutation.mutate();
    }
  };

  const hasTriggeredAnalysis = useRef(false);
  
  const analyzeMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/ai/analyze"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] });
    },
  });

  const backfillMutation = useMutation({
    mutationFn: async () => {
      setBackfillInProgress(true);
      setBackfillProgress(0);
      const response = await apiRequest("POST", "/api/historical/backfill");
      return response.json();
    },
    onSuccess: () => {
      setBackfillInProgress(false);
      setBackfillProgress(100);
      queryClient.invalidateQueries({ queryKey: ["/api/historical/status"] });
      queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] });
    },
    onError: () => {
      setBackfillInProgress(false);
    },
  });

  useEffect(() => {
    if (backfillInProgress && historicalStatus) {
      const targetCandles = 35040;
      const progress = Math.min((historicalStatus.totalCandles / targetCandles) * 100, 99);
      setBackfillProgress(progress);
    }
  }, [historicalStatus, backfillInProgress]);

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
          <TabsList className="mb-4 flex-wrap" data-testid="tabs-list">
            <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
            <TabsTrigger value="signal" data-testid="tab-signal">Signal</TabsTrigger>
            <TabsTrigger value="neural-network" data-testid="tab-neural-network">Neural Network</TabsTrigger>
            <TabsTrigger value="paper" data-testid="tab-paper">Paper Trading</TabsTrigger>
            <TabsTrigger value="gpu-training" data-testid="tab-gpu-training">GPU Training</TabsTrigger>
            <TabsTrigger value="strategy-learner" data-testid="tab-strategy-learner">Strategy Learner</TabsTrigger>
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
                <EnsembleSignalCard 
                  prediction={ensemblePrediction?.prediction ?? null}
                  status={ensembleStatus?.status}
                  isLoading={ensembleLoading}
                />
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
            <div className="space-y-4">
              {/* Premium Chart with Strategy Signal Levels */}
              <PremiumCandlestickChart
                historicalCandles={data.candles.map(c => ({
                  timestamp: c.timestamp,
                  open: Number(c.open),
                  high: Number(c.high),
                  low: Number(c.low),
                  close: Number(c.close),
                  volume: c.volume ? Number(c.volume) : undefined
                }))}
                predictedCandles={[]}
                currentPrice={data.candles[data.candles.length - 1]?.close ? Number(data.candles[data.candles.length - 1].close) : 0}
                action={data.shotPlan?.signal || data.currentSignal?.signal || "HOLD"}
                tradeLevels={data.strategySignal?.entryZone ? {
                  entry: data.strategySignal.entryZone,
                  stopLoss: data.strategySignal.stopLoss ?? 0,
                  takeProfit: data.strategySignal.takeProfit1 ?? 0
                } : (data.shotPlan?.entryZone ? {
                  entry: (data.shotPlan.entryZone.low + data.shotPlan.entryZone.high) / 2,
                  stopLoss: data.shotPlan.stopLoss ?? 0,
                  takeProfit: data.shotPlan.takeProfit1 ?? 0
                } : undefined)}
                symbol="BTCUSDT"
                timeframe="15m"
              />
              
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                <div className="lg:col-span-8 space-y-4">
                  <AIAnalysisCard analysis={data.aiAnalysis} />
                </div>
                <div className="lg:col-span-4 space-y-4">
                  <EnhancedShotPlanCard shotPlan={data.shotPlan} />
                  <SentimentCard sentiment={data.sentiment} />
                  <EnsembleSignalCard 
                    prediction={ensemblePrediction?.prediction ?? null}
                    status={ensembleStatus?.status}
                    isLoading={ensembleLoading}
                  />
                  <SignalCard signal={data.currentSignal} />
                  <RegimeCard signal={data.currentSignal} />
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="neural-network" className="mt-0">
            <div className="space-y-4">
              {/* STALE data warning - show when chart data differs from live price */}
              {(() => {
                const lastCandleClose = data?.candles?.[data.candles.length - 1]?.close ? Number(data.candles[data.candles.length - 1].close) : 0;
                const tickerPrice = nnPrediction?.prediction?.currentPrice ?? lastCandleClose;
                const priceGapPct = tickerPrice > 0 && lastCandleClose > 0 
                  ? Math.abs(lastCandleClose - tickerPrice) / tickerPrice * 100 
                  : 0;
                const isStale = priceGapPct > 0.2;
                
                if (!isStale) return null;
                
                return (
                  <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 flex items-center gap-2 flex-wrap" data-testid="warning-stale-candles">
                    <Clock className="h-4 w-4 text-amber-500" />
                    <span className="text-amber-500 text-sm font-medium">
                      STALE CANDLES: Chart data (${lastCandleClose.toFixed(0)}) differs from live price (${tickerPrice.toFixed(0)}) by {priceGapPct.toFixed(2)}%
                    </span>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="ml-auto text-amber-500 border-amber-500/30 hover-elevate"
                      onClick={() => {
                        apiRequest('POST', '/api/self-learning/run-now')
                          .then(() => {
                            queryClient.invalidateQueries({ queryKey: ['/api/dashboard'] });
                            refetch();
                          })
                          .catch((err) => console.error('Failed to trigger data refresh:', err));
                      }}
                      data-testid="button-refresh-data"
                    >
                      Refresh Data
                    </Button>
                  </div>
                );
              })()}
              
              {/* Premium Candlestick Chart with Probability Cone */}
              <PremiumCandlestickChart
                historicalCandles={(data?.candles || []).map(c => ({
                  timestamp: c.timestamp,
                  open: Number(c.open),
                  high: Number(c.high),
                  low: Number(c.low),
                  close: Number(c.close),
                  volume: c.volume ? Number(c.volume) : undefined
                }))}
                horizonQuantiles={nnPrediction?.prediction?.quantiles}
                horizonBars={16}
                currentPrice={data?.candles?.[data.candles.length - 1]?.close ? Number(data.candles[data.candles.length - 1].close) : 0}
                action={nnPrediction?.prediction?.action || "HOLD"}
                tradeLevels={nnPrediction?.prediction ? {
                  entry: nnPrediction.prediction.entry,
                  stopLoss: nnPrediction.prediction.stopLoss,
                  takeProfit: nnPrediction.prediction.takeProfit
                } : undefined}
                symbol="BTCUSDT"
                timeframe="15m"
              />

              {/* Probabilistic Return Path & Derived Levels */}
              {nnPrediction?.prediction && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <QuantileFanChart 
                    currentPrice={data?.candles?.[data.candles.length - 1]?.close ? Number(data.candles[data.candles.length - 1].close) : 0}
                    quantiles={nnPrediction.prediction.quantiles}
                    action={nnPrediction.prediction.action}
                    horizonBars={10}
                    timeframeMinutes={15}
                  />
                  <DerivedTradeLevels
                    currentPrice={data?.candles?.[data.candles.length - 1]?.close ? Number(data.candles[data.candles.length - 1].close) : 0}
                    quantiles={nnPrediction.prediction.quantiles}
                    action={nnPrediction.prediction.action}
                  />
                </div>
              )}
              
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                {/* Cone Signal Generator */}
                <div className="lg:col-span-6 space-y-4">
                  <ConeSignalCard />
                </div>
                
                {/* Neural Network Prediction */}
                <div className="lg:col-span-6 space-y-4">
                  <NeuralNetworkPredictionCard 
                    prediction={nnPrediction?.prediction || null}
                    trace={nnPrediction?.trace || null}
                    isLoading={nnPredictionLoading}
                    onRefresh={() => refetchNnPrediction()}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                {/* Training Controls */}
                <div className="lg:col-span-4 space-y-4">
                  <TrainingModeBadge 
                    mode={gpuStatus?.trainingMode ?? null}
                    description={gpuStatus?.trainingModeDescription ?? null}
                    inputDim={gpuStatus?.inputDim ?? null}
                    connected={gpuStatus?.connected ?? false}
                  />
                  <EnsembleSignalCard 
                    prediction={ensemblePrediction?.prediction ?? null}
                    status={ensembleStatus?.status}
                    isLoading={ensembleLoading}
                  />
                </div>
                
                {/* Signal and Regime */}
                <div className="lg:col-span-4 space-y-4">
                  <SignalCard signal={data.currentSignal} />
                  <RegimeCard signal={data.currentSignal} />
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="paper" className="mt-0">
            <div className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                <div className="lg:col-span-4 space-y-4">
                  <PerformanceCard />
                </div>
                <div className="lg:col-span-4 space-y-4">
                  <EquityPerformanceCard />
                  <RiskStatusCard />
                </div>
                <div className="lg:col-span-4 space-y-4">
                  <OpenPositionCard />
                  <PositionHistoryCard />
                </div>
              </div>
              <AuditLogPanel />
            </div>
          </TabsContent>

          <TabsContent value="gpu-training" className="mt-0">
            <div className="space-y-4">
              {/* GPU Trainer Connection Settings */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="lg:col-span-1">
                  <GPUConnectionSettings />
                </div>
                <div className="lg:col-span-2">
                  {/* Neural Network Multi-Timeframe Data */}
                  <NeuralNetworkDataCard />
                </div>
              </div>
              
              {/* GPU Training Status */}
              <GPUTrainingSection 
                gpuMetrics={gpuStatus ? {
                  gpuAvailable: gpuStatus.connected && gpuStatus.gpuAvailable,
                  gpuName: gpuStatus.gpuName,
                  gpuMemoryUsed: gpuStatus.gpuMemoryUsed,
                  gpuMemoryTotal: gpuStatus.gpuMemoryTotal,
                  gpuMemoryPercent: gpuStatus.gpuMemoryTotal ? 
                    ((gpuStatus.gpuMemoryUsed || 0) / gpuStatus.gpuMemoryTotal) * 100 : 0,
                  modelsLoaded: gpuStatus.modelsCompleted || [],
                  uptime: 0,
                  isTraining: gpuStatus.isTraining,
                  trainingProgress: gpuStatus.trainingProgress,
                  currentModel: gpuStatus.currentModel,
                  trainingMetrics: {
                    epoch: gpuStatus.currentEpoch,
                    totalEpochs: gpuStatus.totalEpochs,
                    trainLoss: gpuStatus.trainLoss || 0,
                    valLoss: gpuStatus.valLoss || 0
                  }
                } : null}
                modelPerformance={gpuStatus?.modelStatus ? 
                  Object.entries(gpuStatus.modelStatus).map(([name, data]) => ({
                    name: name.charAt(0).toUpperCase() + name.slice(1),
                    accuracy: data.accuracy || 0,
                    loss: data.loss || 0,
                    epochs: data.epochs || 0,
                    status: data.status === "complete" ? "ready" as const : 
                            data.status === "training" ? "training" as const : "pending" as const
                  })) : undefined
                }
                onStartTraining={(modelType) => trainModelMutation.mutate(modelType)}
              />
              
              {/* Cross-Asset Analysis */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <CrossAssetOverviewCard data={crossAssetData} />
                <CorrelationMatrixCard data={crossAssetData} />
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <RelativeStrengthCard data={crossAssetData} />
                <PriceComparisonChart data={crossAssetData} />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="strategy-learner" className="mt-0">
            <StrategyLearnerTab />
          </TabsContent>

          <TabsContent value="learning" className="mt-0">
            <div className="space-y-4">
              {/* Data Management - Always show so user can download data */}
              <DataManagementCard />
              
              {/* Show waiting state when no historical data */}
              {!hasHistoricalData ? (
                <div className="p-6 rounded-lg border border-blue-500/30 bg-blue-500/10" data-testid="learning-waiting">
                  <div className="flex items-center gap-2 text-blue-400">
                    <Brain className="h-5 w-5" />
                    <span className="font-medium">Waiting for Historical Data</span>
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">
                    All learning systems require historical data to train. Use the 
                    <span className="text-primary font-medium"> Data Management </span>
                    panel above to download 1+ years of historical data for BTC, ETH, SOL, and BNB.
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Current data: {dataSummary?.totalCandles?.toLocaleString() || 0} candles (need 1,000+)
                  </p>
                </div>
              ) : !hasDeepLearningStarted ? (
                <div className="p-6 rounded-lg border border-emerald-500/30 bg-emerald-500/10" data-testid="learning-ready">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2 text-emerald-400">
                        <Brain className="h-5 w-5" />
                        <span className="font-medium">Ready to Train - Pattern Memory & Deep Learning</span>
                      </div>
                      <p className="mt-2 text-sm text-muted-foreground">
                        Historical data loaded: {dataSummary?.totalCandles?.toLocaleString()} candles.
                        Click Start Learning to begin training the Pattern Memory and Deep Learning systems.
                      </p>
                    </div>
                    <Button
                      onClick={() => startDeepLearningMutation.mutate()}
                      disabled={startDeepLearningMutation.isPending}
                      className="bg-emerald-600"
                      data-testid="button-start-deep-learning"
                    >
                      {startDeepLearningMutation.isPending ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Training...
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4 mr-2" />
                          Start Learning
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              ) : (
                <>
                  <LearningOverviewCard learningStats={data.learningStats} />
                  
                  {/* Unified Learning Progress - All 3 systems synchronized */}
                  <UnifiedLearningProgressCard 
                    progress={unifiedProgress}
                    onReset={handleResetLearning}
                  />
                  
                  {/* Social Awareness & Historical Learning - Key new sections */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <SocialAwarenessCard learningStats={data.learningStats} />
                    <HistoricalLearningCard 
                      learningStats={data.learningStats}
                      historicalStatus={historicalStatus}
                      integrityReport={integrityReport}
                      onBackfill={() => backfillMutation.mutate()}
                      backfillInProgress={backfillInProgress || backfillMutation.isPending}
                      backfillProgress={backfillProgress}
                    />
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
                </>
              )}
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
