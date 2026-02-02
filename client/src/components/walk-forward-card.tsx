import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { 
  AlertTriangle, CheckCircle, Activity, TrendingUp, TrendingDown, 
  Play, RefreshCw, ChevronDown, ChevronUp, BarChart3, Target
} from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, LineChart, Line, CartesianGrid, Legend } from "recharts";

interface WalkForwardFold {
  fold_id: number;
  n_trades: number;
  win_rate: number;
  sharpe_ratio: number;
  expectancy: number;
  profit_factor: number;
  max_drawdown: number;
  total_return: number;
  regime_results?: Record<string, unknown>;
}

interface ModelResult {
  summary: {
    n_folds: number;
    total_trades: number;
    overall_win_rate: number;
    overall_sharpe: number;
    overall_expectancy: number;
    overall_profit_factor: number;
    avg_trades_per_fold: number;
    worst_drawdown: number;
    tail_risk: number;
    sharpe_stability: number;
    expectancy_stability: number;
  };
  folds: WalkForwardFold[];
  status: string;
  error?: string;
}

interface WalkForwardReport {
  status: string;
  timestamp: string;
  config?: {
    n_folds: number;
    train_periods: number;
    test_periods: number;
    purge_periods: number;
  };
  data_info?: {
    source: string;
    total_rows: number;
    features: number;
  };
  model_results?: Record<string, ModelResult>;
  overall_summary?: {
    models_evaluated: number;
    best_sharpe_model: string | null;
    best_expectancy_model: string | null;
    model_rankings_by_sharpe?: [string, number][];
    model_rankings_by_expectancy?: [string, number][];
    recommendations?: string[];
  };
  message?: string;
  error?: string;
}

const COLORS = {
  positive: "#22c55e",
  negative: "#ef4444",
  neutral: "#6b7280",
  warning: "#f59e0b"
};

export function WalkForwardCard() {
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [report, setReport] = useState<WalkForwardReport | null>(null);

  const { data: statusData, isLoading: statusLoading } = useQuery<{ status: string; message?: string }>({
    queryKey: ["/api/walk-forward/report"],
    refetchInterval: 60000
  });

  const runEvalMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", "/api/walk-forward/evaluate", {
        nFolds: 5,
        testPeriods: 500,
        trainPeriods: 2000,
        purgePeriods: 50
      });
      return response.json();
    },
    onSuccess: (data) => {
      setReport(data);
      queryClient.invalidateQueries({ queryKey: ["/api/walk-forward/report"] });
    }
  });

  const getMetricColor = (value: number, thresholds: { good: number; bad: number }, higherIsBetter = true) => {
    if (higherIsBetter) {
      if (value >= thresholds.good) return "text-green-500";
      if (value <= thresholds.bad) return "text-red-500";
      return "text-amber-500";
    } else {
      if (value <= thresholds.good) return "text-green-500";
      if (value >= thresholds.bad) return "text-red-500";
      return "text-amber-500";
    }
  };

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatDecimal = (value: number, decimals = 4) => value.toFixed(decimals);

  if (statusLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Walk-Forward Validation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-40">
            <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  }

  const displayReport = report || null;
  const isRunning = runEvalMutation.isPending;
  const gpuAvailable = statusData?.status === "ready";

  return (
    <div className="space-y-4">
      <Card data-testid="card-walk-forward-controls">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Walk-Forward Validation
            </div>
            <Button 
              onClick={() => runEvalMutation.mutate()} 
              disabled={isRunning || !gpuAvailable}
              data-testid="button-run-walk-forward"
            >
              {isRunning ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Evaluation
                </>
              )}
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!gpuAvailable && (
            <div className="flex items-center gap-2 text-amber-500 mb-4">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-sm">GPU trainer not connected - connect to run validation</span>
            </div>
          )}
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Folds</span>
              <div className="font-mono">5</div>
            </div>
            <div>
              <span className="text-muted-foreground">Train Periods</span>
              <div className="font-mono">2,000</div>
            </div>
            <div>
              <span className="text-muted-foreground">Test Periods</span>
              <div className="font-mono">500</div>
            </div>
            <div>
              <span className="text-muted-foreground">Purge Gap</span>
              <div className="font-mono">50</div>
            </div>
          </div>
          
          {displayReport?.status === "error" && (
            <div className="mt-4 p-3 rounded bg-red-500/10 border border-red-500/20">
              <div className="flex items-center gap-2 text-red-500 font-medium">
                <AlertTriangle className="h-4 w-4" />
                Evaluation Failed
              </div>
              <p className="text-sm mt-1 text-muted-foreground">{displayReport.error}</p>
            </div>
          )}
        </CardContent>
      </Card>

      {displayReport?.status === "success" && displayReport.overall_summary && (
        <>
          <Card data-testid="card-walk-forward-summary">
            <CardHeader>
              <CardTitle className="text-sm font-medium">Overall Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-3 rounded-lg bg-muted/50">
                  <span className="text-xs text-muted-foreground">Models Evaluated</span>
                  <div className="text-2xl font-bold">{displayReport.overall_summary.models_evaluated}</div>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <span className="text-xs text-muted-foreground">Best Sharpe Model</span>
                  <div className="text-lg font-bold truncate">{displayReport.overall_summary.best_sharpe_model || "N/A"}</div>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <span className="text-xs text-muted-foreground">Best Expectancy Model</span>
                  <div className="text-lg font-bold truncate">{displayReport.overall_summary.best_expectancy_model || "N/A"}</div>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <span className="text-xs text-muted-foreground">Data Source</span>
                  <div className="text-sm font-mono truncate">{displayReport.data_info?.source || "N/A"}</div>
                </div>
              </div>

              {displayReport.overall_summary.recommendations && displayReport.overall_summary.recommendations.length > 0 && (
                <div className="mt-4 space-y-2">
                  <span className="text-xs text-muted-foreground font-medium">Recommendations</span>
                  {displayReport.overall_summary.recommendations.map((rec, i) => (
                    <div key={i} className="flex items-start gap-2 p-2 rounded bg-amber-500/10 border border-amber-500/20">
                      <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                      <span className="text-sm">{rec}</span>
                    </div>
                  ))}
                </div>
              )}

              {displayReport.overall_summary.model_rankings_by_sharpe && (
                <div className="mt-4">
                  <span className="text-xs text-muted-foreground font-medium mb-2 block">Model Rankings by Sharpe</span>
                  <ResponsiveContainer width="100%" height={120}>
                    <BarChart 
                      data={displayReport.overall_summary.model_rankings_by_sharpe.map(([name, sharpe]) => ({
                        name: name.replace("multihead_", ""),
                        sharpe
                      }))}
                      layout="vertical"
                    >
                      <XAxis type="number" domain={["dataMin", "dataMax"]} />
                      <YAxis type="category" dataKey="name" width={80} fontSize={10} />
                      <Tooltip formatter={(v) => (v as number).toFixed(3)} />
                      <Bar dataKey="sharpe" radius={[0, 4, 4, 0]}>
                        {displayReport.overall_summary.model_rankings_by_sharpe.map(([, sharpe], i) => (
                          <Cell key={i} fill={sharpe >= 0 ? COLORS.positive : COLORS.negative} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>

          {displayReport.model_results && Object.entries(displayReport.model_results).map(([modelName, result]) => (
            result.status === "success" && (
              <Card key={modelName} data-testid={`card-model-${modelName}`}>
                <CardHeader 
                  className="cursor-pointer hover-elevate"
                  onClick={() => setExpandedModel(expandedModel === modelName ? null : modelName)}
                >
                  <CardTitle className="text-sm font-medium flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Target className="h-4 w-4" />
                      {modelName.replace("multihead_", "").toUpperCase()}
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <Badge variant={result.summary.overall_sharpe >= 0 ? "default" : "destructive"}>
                          Sharpe: {result.summary.overall_sharpe.toFixed(3)}
                        </Badge>
                        <Badge variant="secondary">
                          {result.summary.total_trades} trades
                        </Badge>
                      </div>
                      {expandedModel === modelName ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </div>
                  </CardTitle>
                </CardHeader>
                
                {expandedModel === modelName && (
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-4">
                      <div className="p-2 rounded bg-muted/50">
                        <span className="text-xs text-muted-foreground">Win Rate</span>
                        <div className={`font-mono font-bold ${getMetricColor(result.summary.overall_win_rate, { good: 0.55, bad: 0.45 })}`}>
                          {formatPercent(result.summary.overall_win_rate)}
                        </div>
                      </div>
                      <div className="p-2 rounded bg-muted/50">
                        <span className="text-xs text-muted-foreground">Sharpe</span>
                        <div className={`font-mono font-bold ${getMetricColor(result.summary.overall_sharpe, { good: 1.0, bad: 0 })}`}>
                          {formatDecimal(result.summary.overall_sharpe, 3)}
                        </div>
                      </div>
                      <div className="p-2 rounded bg-muted/50">
                        <span className="text-xs text-muted-foreground">Expectancy</span>
                        <div className={`font-mono font-bold ${getMetricColor(result.summary.overall_expectancy, { good: 0.001, bad: 0 })}`}>
                          {formatDecimal(result.summary.overall_expectancy, 5)}
                        </div>
                      </div>
                      <div className="p-2 rounded bg-muted/50">
                        <span className="text-xs text-muted-foreground">Profit Factor</span>
                        <div className={`font-mono font-bold ${getMetricColor(result.summary.overall_profit_factor, { good: 1.2, bad: 1.0 })}`}>
                          {formatDecimal(result.summary.overall_profit_factor, 2)}
                        </div>
                      </div>
                      <div className="p-2 rounded bg-muted/50">
                        <span className="text-xs text-muted-foreground">Max Drawdown</span>
                        <div className={`font-mono font-bold ${getMetricColor(Math.abs(result.summary.worst_drawdown), { good: 0.02, bad: 0.05 }, false)}`}>
                          {formatPercent(result.summary.worst_drawdown)}
                        </div>
                      </div>
                      <div className="p-2 rounded bg-muted/50">
                        <span className="text-xs text-muted-foreground">Sharpe Stability</span>
                        <div className={`font-mono font-bold ${getMetricColor(result.summary.sharpe_stability, { good: 0.3, bad: 0.8 }, false)}`}>
                          {formatDecimal(result.summary.sharpe_stability, 3)}
                        </div>
                      </div>
                    </div>

                    <div className="mt-4">
                      <span className="text-xs text-muted-foreground font-medium mb-2 block">Per-Fold Performance</span>
                      <ResponsiveContainer width="100%" height={180}>
                        <LineChart data={result.folds.map(f => ({
                          fold: `Fold ${f.fold_id + 1}`,
                          sharpe: f.sharpe_ratio,
                          winRate: f.win_rate * 100,
                          trades: f.n_trades
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis dataKey="fold" fontSize={10} />
                          <YAxis yAxisId="left" fontSize={10} />
                          <YAxis yAxisId="right" orientation="right" fontSize={10} />
                          <Tooltip />
                          <Legend />
                          <Line yAxisId="left" type="monotone" dataKey="sharpe" stroke="#22c55e" name="Sharpe" strokeWidth={2} />
                          <Line yAxisId="right" type="monotone" dataKey="winRate" stroke="#3b82f6" name="Win Rate %" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="mt-4 overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-2">Fold</th>
                            <th className="text-right py-2 px-2">Trades</th>
                            <th className="text-right py-2 px-2">Win Rate</th>
                            <th className="text-right py-2 px-2">Sharpe</th>
                            <th className="text-right py-2 px-2">Expectancy</th>
                            <th className="text-right py-2 px-2">Drawdown</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.folds.map((fold) => (
                            <tr key={fold.fold_id} className="border-b border-muted/50">
                              <td className="py-2 px-2 font-mono">Fold {fold.fold_id + 1}</td>
                              <td className="text-right py-2 px-2 font-mono">{fold.n_trades}</td>
                              <td className={`text-right py-2 px-2 font-mono ${getMetricColor(fold.win_rate, { good: 0.55, bad: 0.45 })}`}>
                                {formatPercent(fold.win_rate)}
                              </td>
                              <td className={`text-right py-2 px-2 font-mono ${getMetricColor(fold.sharpe_ratio, { good: 1.0, bad: 0 })}`}>
                                {formatDecimal(fold.sharpe_ratio, 3)}
                              </td>
                              <td className={`text-right py-2 px-2 font-mono ${getMetricColor(fold.expectancy, { good: 0.001, bad: 0 })}`}>
                                {formatDecimal(fold.expectancy, 5)}
                              </td>
                              <td className={`text-right py-2 px-2 font-mono ${getMetricColor(Math.abs(fold.max_drawdown), { good: 0.02, bad: 0.05 }, false)}`}>
                                {formatPercent(fold.max_drawdown)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                )}
              </Card>
            )
          ))}
        </>
      )}

      {!displayReport && !isRunning && (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No walk-forward report available</p>
            <p className="text-sm mt-1">Click "Run Evaluation" to generate a validation report</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
