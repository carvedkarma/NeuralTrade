import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { 
  Brain, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Target,
  Zap,
  BarChart3,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Loader2
} from "lucide-react";

interface ActionOutcome {
  action: "LONG" | "SHORT" | "HOLD";
  pnl: number;
  mae: number;
  mfe: number;
  winRate: number;
  expectancy: number;
  sampleCount: number;
}

interface PolicyPrediction {
  pLongProfitable: number;
  pShortProfitable: number;
  pHoldOptimal: number;
  recommendedAction: "LONG" | "SHORT" | "HOLD";
  confidence: number;
}

interface ExpectedValue {
  longEV: number;
  shortEV: number;
  holdEV: number;
  bestAction: "LONG" | "SHORT" | "HOLD";
  bestEV: number;
  variance: number;
  riskAdjustedScore: number;
}

interface ExpansionForecast {
  probability: number;
  expectedBars: number;
  currentATR: number;
  predictedATR: number;
  signal: "expanding" | "contracting" | "stable";
}

interface ActionPattern {
  patternId: string;
  regime: string;
  longWinRate: number;
  shortWinRate: number;
  longAvgReward: number;
  shortAvgReward: number;
  bestAction: "LONG" | "SHORT" | "HOLD";
  actionAdvantage: number;
  sampleCount: number;
}

interface StrategyLearnerData {
  actionOutcomes: ActionOutcome[];
  policyPrediction: PolicyPrediction;
  expectedValue: ExpectedValue;
  expansionForecast: ExpansionForecast;
  actionPatterns: ActionPattern[];
  trainingProgress: {
    totalSamples: number;
    epochsCompleted: number;
    lastTrainingTime: number | null;
    modelAccuracy: number;
    isTraining: boolean;
  };
  comparisonWithCurrent: {
    currentSignal: string;
    currentConfidence: number;
    learnerSignal: string;
    learnerEV: number;
    agreement: boolean;
  };
}

function getActionIcon(action: string) {
  switch (action) {
    case "LONG": return <TrendingUp className="h-4 w-4 text-emerald-400" />;
    case "SHORT": return <TrendingDown className="h-4 w-4 text-red-400" />;
    default: return <Minus className="h-4 w-4 text-yellow-400" />;
  }
}

function getActionColor(action: string) {
  switch (action) {
    case "LONG": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
    case "SHORT": return "bg-red-500/20 text-red-400 border-red-500/30";
    default: return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
  }
}

export function ActionOutcomesCard({ outcomes }: { outcomes: ActionOutcome[] }) {
  return (
    <Card data-testid="card-action-outcomes">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-primary" />
          Action Outcomes (Historical)
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {outcomes.map((outcome) => (
          <div 
            key={outcome.action} 
            className="p-3 rounded-lg border bg-card/50"
            data-testid={`action-outcome-${outcome.action.toLowerCase()}`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {getActionIcon(outcome.action)}
                <span className="font-medium">{outcome.action}</span>
              </div>
              <Badge variant="outline" className={getActionColor(outcome.action)}>
                {outcome.winRate.toFixed(1)}% win rate
              </Badge>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">Avg PnL</span>
                <p className={`font-mono ${outcome.pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {outcome.pnl >= 0 ? "+" : ""}{outcome.pnl.toFixed(2)}%
                </p>
              </div>
              <div>
                <span className="text-muted-foreground">MAE</span>
                <p className="font-mono text-red-400">-{outcome.mae.toFixed(2)}%</p>
              </div>
              <div>
                <span className="text-muted-foreground">MFE</span>
                <p className="font-mono text-emerald-400">+{outcome.mfe.toFixed(2)}%</p>
              </div>
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              Expectancy: <span className={outcome.expectancy >= 0 ? "text-emerald-400" : "text-red-400"}>
                {outcome.expectancy >= 0 ? "+" : ""}{outcome.expectancy.toFixed(3)}
              </span>
              <span className="ml-2">({outcome.sampleCount.toLocaleString()} samples)</span>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function PolicyPredictionCard({ prediction }: { prediction: PolicyPrediction }) {
  const probabilities = [
    { action: "LONG", prob: prediction.pLongProfitable, color: "bg-emerald-500" },
    { action: "SHORT", prob: prediction.pShortProfitable, color: "bg-red-500" },
    { action: "HOLD", prob: prediction.pHoldOptimal, color: "bg-yellow-500" },
  ];

  return (
    <Card data-testid="card-policy-prediction">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Brain className="h-4 w-4 text-primary" />
          Policy Model Prediction
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-center gap-2 p-4 rounded-lg border bg-card/50">
          {getActionIcon(prediction.recommendedAction)}
          <span className="text-2xl font-bold">{prediction.recommendedAction}</span>
          <Badge variant="secondary" className="ml-2">
            {(prediction.confidence * 100).toFixed(0)}% confident
          </Badge>
        </div>

        <div className="space-y-2">
          <p className="text-xs text-muted-foreground mb-2">P(Action Profitable)</p>
          {probabilities.map(({ action, prob, color }) => (
            <div key={action} className="space-y-1" data-testid={`policy-prob-${action.toLowerCase()}`}>
              <div className="flex items-center justify-between text-xs">
                <span>{action}</span>
                <span className="font-mono">{(prob * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all ${color}`} 
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function ExpectedValueCard({ ev }: { ev: ExpectedValue }) {
  const evItems = [
    { action: "LONG", value: ev.longEV },
    { action: "SHORT", value: ev.shortEV },
    { action: "HOLD", value: ev.holdEV },
  ];

  return (
    <Card data-testid="card-expected-value">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Target className="h-4 w-4 text-primary" />
          Expected Value Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-center gap-2 p-4 rounded-lg border bg-primary/10">
          <div className="text-center">
            <p className="text-xs text-muted-foreground mb-1">Best Action</p>
            <div className="flex items-center gap-2">
              {getActionIcon(ev.bestAction)}
              <span className="text-xl font-bold">{ev.bestAction}</span>
            </div>
            <p className={`text-lg font-mono mt-1 ${ev.bestEV >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              EV: {ev.bestEV >= 0 ? "+" : ""}{ev.bestEV.toFixed(4)}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {evItems.map(({ action, value }) => (
            <div 
              key={action} 
              className={`p-2 rounded-lg border text-center ${action === ev.bestAction ? "border-primary bg-primary/5" : ""}`}
              data-testid={`ev-${action.toLowerCase()}`}
            >
              <p className="text-xs text-muted-foreground">{action}</p>
              <p className={`font-mono text-sm ${value >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                {value >= 0 ? "+" : ""}{value.toFixed(4)}
              </p>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-4 pt-2 border-t">
          <div className="text-center">
            <p className="text-xs text-muted-foreground">Variance</p>
            <p className="font-mono">{ev.variance.toFixed(4)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground">Risk-Adj Score</p>
            <p className={`font-mono ${ev.riskAdjustedScore >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {ev.riskAdjustedScore.toFixed(3)}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function ExpansionForecastCard({ forecast }: { forecast: ExpansionForecast }) {
  const getSignalInfo = () => {
    switch (forecast.signal) {
      case "expanding":
        return { color: "text-emerald-400", bg: "bg-emerald-500/20", label: "Expansion Expected" };
      case "contracting":
        return { color: "text-red-400", bg: "bg-red-500/20", label: "Contraction Expected" };
      default:
        return { color: "text-yellow-400", bg: "bg-yellow-500/20", label: "Stable" };
    }
  };

  const signalInfo = getSignalInfo();

  return (
    <Card data-testid="card-expansion-forecast">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Zap className="h-4 w-4 text-primary" />
          Expansion Predictor
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className={`flex items-center justify-center gap-2 p-4 rounded-lg ${signalInfo.bg}`}>
          <Activity className={`h-5 w-5 ${signalInfo.color}`} />
          <span className={`font-medium ${signalInfo.color}`}>{signalInfo.label}</span>
          <Badge variant="secondary">{(forecast.probability * 100).toFixed(0)}%</Badge>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 rounded-lg border bg-card/50">
            <p className="text-xs text-muted-foreground">Current ATR</p>
            <p className="font-mono text-lg">${forecast.currentATR.toFixed(2)}</p>
          </div>
          <div className="p-3 rounded-lg border bg-card/50">
            <p className="text-xs text-muted-foreground">Predicted ATR</p>
            <p className={`font-mono text-lg ${forecast.predictedATR > forecast.currentATR ? "text-emerald-400" : "text-red-400"}`}>
              ${forecast.predictedATR.toFixed(2)}
            </p>
          </div>
        </div>

        <div className="text-center text-xs text-muted-foreground">
          Expected in next <span className="font-medium text-foreground">{forecast.expectedBars}</span> bars
        </div>
      </CardContent>
    </Card>
  );
}

export function ActionPatternsCard({ patterns }: { patterns: ActionPattern[] }) {
  return (
    <Card data-testid="card-action-patterns">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Brain className="h-4 w-4 text-primary" />
          Action-Aware Patterns
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {patterns.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              No patterns learned yet
            </p>
          ) : (
            patterns.map((pattern) => (
              <div 
                key={pattern.patternId} 
                className="p-2 rounded-lg border bg-card/50 text-xs"
                data-testid={`pattern-${pattern.patternId}`}
              >
                <div className="flex items-center justify-between mb-1">
                  <Badge variant="outline" className="text-xs">
                    {pattern.regime}
                  </Badge>
                  <div className="flex items-center gap-1">
                    {getActionIcon(pattern.bestAction)}
                    <span className="font-medium">{pattern.bestAction}</span>
                    <span className="text-emerald-400">+{(pattern.actionAdvantage * 100).toFixed(1)}%</span>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-1 text-muted-foreground">
                  <div>L: {(pattern.longWinRate * 100).toFixed(0)}%</div>
                  <div>S: {(pattern.shortWinRate * 100).toFixed(0)}%</div>
                  <div>L: {pattern.longAvgReward >= 0 ? "+" : ""}{pattern.longAvgReward.toFixed(2)}%</div>
                  <div>S: {pattern.shortAvgReward >= 0 ? "+" : ""}{pattern.shortAvgReward.toFixed(2)}%</div>
                </div>
                <div className="text-muted-foreground mt-1">
                  {pattern.sampleCount} samples
                </div>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export function TrainingStatusCard({ progress }: { progress: StrategyLearnerData["trainingProgress"] }) {
  return (
    <Card data-testid="card-training-status">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          Training Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Status</span>
          {progress.isTraining ? (
            <Badge variant="secondary" className="bg-blue-500/20 text-blue-400">
              <Loader2 className="h-3 w-3 animate-spin mr-1" />
              Training
            </Badge>
          ) : (
            <Badge variant="secondary" className="bg-emerald-500/20 text-emerald-400">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              Ready
            </Badge>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-muted-foreground">Total Samples</p>
            <p className="font-mono">{progress.totalSamples.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Epochs</p>
            <p className="font-mono">{progress.epochsCompleted}</p>
          </div>
        </div>

        <div>
          <p className="text-xs text-muted-foreground mb-1">Model Accuracy</p>
          <Progress value={progress.modelAccuracy * 100} className="h-2" />
          <p className="text-xs text-right mt-1 font-mono">
            {(progress.modelAccuracy * 100).toFixed(1)}%
          </p>
        </div>

        {progress.lastTrainingTime && (
          <p className="text-xs text-muted-foreground">
            Last trained: {new Date(progress.lastTrainingTime).toLocaleTimeString()}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export function ComparisonCard({ comparison }: { comparison: StrategyLearnerData["comparisonWithCurrent"] }) {
  return (
    <Card data-testid="card-comparison">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-primary" />
          Current vs Strategy Learner
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 rounded-lg border bg-card/50 text-center">
            <p className="text-xs text-muted-foreground mb-2">Current System</p>
            <div className="flex items-center justify-center gap-1">
              {getActionIcon(comparison.currentSignal)}
              <span className="font-medium">{comparison.currentSignal}</span>
            </div>
            <p className="text-xs mt-1">
              {(comparison.currentConfidence * 100).toFixed(0)}% confidence
            </p>
          </div>
          <div className="p-3 rounded-lg border bg-primary/10 text-center">
            <p className="text-xs text-muted-foreground mb-2">Strategy Learner</p>
            <div className="flex items-center justify-center gap-1">
              {getActionIcon(comparison.learnerSignal)}
              <span className="font-medium">{comparison.learnerSignal}</span>
            </div>
            <p className={`text-xs mt-1 ${comparison.learnerEV >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              EV: {comparison.learnerEV >= 0 ? "+" : ""}{comparison.learnerEV.toFixed(4)}
            </p>
          </div>
        </div>

        <div className="mt-3 flex items-center justify-center gap-2">
          {comparison.agreement ? (
            <Badge variant="secondary" className="bg-emerald-500/20 text-emerald-400">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              Systems Agree
            </Badge>
          ) : (
            <Badge variant="secondary" className="bg-yellow-500/20 text-yellow-400">
              <AlertTriangle className="h-3 w-3 mr-1" />
              Systems Disagree
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export function StrategyLearnerTab() {
  const { data, isLoading, error } = useQuery<StrategyLearnerData>({
    queryKey: ["/api/strategy-learner"],
    refetchInterval: 5000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12" data-testid="strategy-learner-loading">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-muted-foreground">Loading Strategy Learner...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-6 rounded-lg border border-yellow-500/30 bg-yellow-500/10" data-testid="strategy-learner-error">
        <div className="flex items-center gap-2 text-yellow-400">
          <AlertTriangle className="h-5 w-5" />
          <span className="font-medium">Strategy Learner Initializing</span>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          The strategy learner is building its action-labeled dataset from historical data. 
          This may take a few minutes on first load.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4" data-testid="strategy-learner-content">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <div className="lg:col-span-8 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <PolicyPredictionCard prediction={data.policyPrediction} />
            <ExpectedValueCard ev={data.expectedValue} />
          </div>
          <ActionOutcomesCard outcomes={data.actionOutcomes} />
          <ActionPatternsCard patterns={data.actionPatterns} />
        </div>

        <div className="lg:col-span-4 space-y-4">
          <ComparisonCard comparison={data.comparisonWithCurrent} />
          <ExpansionForecastCard forecast={data.expansionForecast} />
          <TrainingStatusCard progress={data.trainingProgress} />
        </div>
      </div>
    </div>
  );
}
