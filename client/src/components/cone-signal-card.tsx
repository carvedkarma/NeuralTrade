import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Target, 
  Shield, 
  Crosshair, 
  TrendingUp, 
  TrendingDown,
  Minus,
  Clock,
  CheckCircle2,
  XCircle,
  Timer,
  Filter,
  BarChart3,
  RefreshCw,
  Loader2,
  AlertTriangle,
  Activity,
  Wind
} from "lucide-react";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend } from "recharts";

interface FlowForecast {
  volState: "contraction" | "neutral" | "expansion";
  volStateProbs: { contraction: number; neutral: number; expansion: number };
  acceleration: number;
  forecastMode: "QUANTILE_PATHS" | "NO_FORECAST";
  quantilePaths?: {
    q10: number[];
    q50: number[];
    q90: number[];
  };
}

interface ConeSignal {
  direction: "LONG" | "SHORT" | "HOLD";
  entryPrice: number;
  stopLoss: number | null;
  takeProfit: number | null;
  riskReward: number | null;
  edge: number;
  holdReasons: string[];
  quantiles: {
    q10: number;
    q25: number;
    q50: number;
    q75: number;
    q90: number;
  };
  probUp: number;
  probDown: number;
  probHold: number;
  mu: number;
  sigma: number;
  timestamp: number;
  edgeThreshold: number;
  cooldownBarsRemaining: number;
  flowForecast?: FlowForecast;
}

interface ConeStats {
  edgeThreshold: number;
  edgePercentile: number;
  recentTradeCount: number;
  edgeHistorySize: number;
  cooldownBarsRemaining: number;
  config: {
    horizonBars: number;
    cooldownBars: number;
    minMu: number;
    maxUncertainty: number;
    maxProbHold: number;
    minRR: number;
    targetTradesPerDay: number;
  };
}

interface SignalHistoryItem {
  id: number;
  timestamp: number;
  direction: string;
  entryPrice: number | string;
  stopLoss: number | string | null;
  takeProfit: number | string | null;
  edge: number | string;
  riskReward: number | string | null;
  outcome: string | null;
  exitPrice: number | string | null;
  pnlPercent: number | string | null;
}

const formatPrice = (val: number | string | null) => {
  if (val === null || val === undefined) return "-";
  const num = typeof val === "string" ? parseFloat(val) : val;
  return `$${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

const formatPercent = (val: number | string | null, decimals = 2) => {
  if (val === null || val === undefined) return "-";
  const num = typeof val === "string" ? parseFloat(val) : val;
  return `${num >= 0 ? "+" : ""}${(num * 100).toFixed(decimals)}%`;
};

function FlowForecastChart({ flowForecast, entryPrice }: { flowForecast: FlowForecast; entryPrice: number }) {
  if (flowForecast.forecastMode === "NO_FORECAST" || !flowForecast.quantilePaths) {
    return (
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4" data-testid="flow-forecast-gated">
        <div className="flex items-center gap-2 text-amber-400 mb-2">
          <Wind className="h-4 w-4" />
          <span className="text-sm font-medium">No Tradeable Flow</span>
        </div>
        <p className="text-xs text-muted-foreground">
          {flowForecast.volState === "contraction" 
            ? "Volatility compression regime detected. Price action too tight for confident projections."
            : "Quantile spread too narrow relative to trading costs."}
        </p>
        <div className="flex items-center gap-4 mt-3 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Vol State:</span>
            <Badge variant="outline" className="text-xs capitalize">
              {flowForecast.volState}
            </Badge>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Accel:</span>
            <span className={flowForecast.acceleration >= 0 ? "text-emerald-400" : "text-red-400"}>
              {flowForecast.acceleration >= 0 ? "+" : ""}{(flowForecast.acceleration * 100).toFixed(3)}%
            </span>
          </div>
        </div>
      </div>
    );
  }

  const { q10, q50, q90 } = flowForecast.quantilePaths;
  const chartData = q10.map((_, i) => ({
    bar: i + 1,
    q10: q10[i],
    q50: q50[i],
    q90: q90[i],
  }));

  const allPrices = [...q10, ...q50, ...q90, entryPrice];
  const minPrice = Math.min(...allPrices) * 0.9995;
  const maxPrice = Math.max(...allPrices) * 1.0005;

  const volStateColor = flowForecast.volState === "expansion" 
    ? "text-emerald-400" 
    : flowForecast.volState === "contraction" 
      ? "text-amber-400" 
      : "text-blue-400";

  return (
    <div className="space-y-2" data-testid="flow-forecast-chart">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          <span className="text-xs font-medium">Flow Forecast</span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Vol:</span>
            <Badge variant="outline" className={`text-xs capitalize ${volStateColor}`}>
              {flowForecast.volState}
            </Badge>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">α:</span>
            <span className="text-foreground">
              {flowForecast.volState === "expansion" ? "1.5" : flowForecast.volState === "contraction" ? "0.7" : "1.0"}
            </span>
          </div>
        </div>
      </div>

      <div className="h-32 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <XAxis 
              dataKey="bar" 
              tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              tickLine={{ stroke: 'hsl(var(--border))' }}
            />
            <YAxis 
              domain={[minPrice, maxPrice]}
              tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
              tickFormatter={(val) => `$${val.toLocaleString()}`}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              tickLine={{ stroke: 'hsl(var(--border))' }}
              width={70}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))', 
                border: '1px solid hsl(var(--border))',
                borderRadius: '6px',
                fontSize: '11px'
              }}
              formatter={(value: number) => [`$${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, '']}
              labelFormatter={(bar) => `Bar ${bar}`}
            />
            <Line 
              type="monotone" 
              dataKey="q90" 
              stroke="#10b981" 
              strokeWidth={1.5}
              dot={false}
              name="q90 (Upper)"
            />
            <Line 
              type="monotone" 
              dataKey="q50" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={false}
              name="q50 (Median)"
            />
            <Line 
              type="monotone" 
              dataKey="q10" 
              stroke="#ef4444" 
              strokeWidth={1.5}
              dot={false}
              name="q10 (Lower)"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-center gap-4 text-[10px]">
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-emerald-500 rounded" />
          <span className="text-muted-foreground">q90</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-blue-500 rounded" />
          <span className="text-muted-foreground">q50</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-red-500 rounded" />
          <span className="text-muted-foreground">q10</span>
        </div>
      </div>
    </div>
  );
}

export function ConeSignalCard() {
  const [isRecording, setIsRecording] = useState(false);

  const { data: coneSignal, isLoading, refetch, dataUpdatedAt } = useQuery<{ available: boolean; signal: ConeSignal }>({
    queryKey: ['/api/cone-signal/current'],
    refetchInterval: 15000, // Refresh every 15 seconds for more responsive updates
  });

  const { data: signalHistory } = useQuery<{ signals: SignalHistoryItem[] }>({
    queryKey: ['/api/cone-signals'],
    refetchInterval: 30000,
  });

  const handleRecordSignal = async () => {
    if (!coneSignal?.signal || coneSignal.signal.direction === "HOLD") return;
    
    setIsRecording(true);
    try {
      await apiRequest('POST', '/api/cone-signals/record', coneSignal.signal);
      queryClient.invalidateQueries({ queryKey: ['/api/cone-signals'] });
    } catch (err) {
      console.error('Failed to record signal:', err);
    } finally {
      setIsRecording(false);
    }
  };

  if (isLoading) {
    return (
      <Card data-testid="card-cone-signal-loading">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Target className="h-4 w-4" />
            Cone Signal Generator
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  const signal = coneSignal?.signal;
  if (!signal) {
    return (
      <Card data-testid="card-cone-signal-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Target className="h-4 w-4" />
            Cone Signal Generator
          </CardTitle>
        </CardHeader>
        <CardContent className="text-center py-6">
          <AlertTriangle className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">No signal available</p>
        </CardContent>
      </Card>
    );
  }

  const isHold = signal.direction === "HOLD";
  const directionColor = signal.direction === "LONG" 
    ? "text-emerald-400" 
    : signal.direction === "SHORT"
      ? "text-red-400"
      : "text-amber-400";

  const directionBg = signal.direction === "LONG"
    ? "bg-emerald-500/10"
    : signal.direction === "SHORT"
      ? "bg-red-500/10"
      : "bg-amber-500/10";

  const holdReasons = signal.holdReasons || [];
  const gates = {
    pHoldCheck: !holdReasons.some(r => r.includes("P(hold)")),
    minMoveCheck: !holdReasons.some(r => r.includes("|μ|") || r.includes("Expected move")),
    edgeCheck: !holdReasons.some(r => r.includes("Edge")),
    uncertaintyCheck: !holdReasons.some(r => r.includes("σ") || r.includes("Uncertainty")),
    cooldownCheck: !holdReasons.some(r => r.includes("Cooldown")),
    rrCheck: !holdReasons.some(r => r.includes("R:R") || r.includes("Risk/Reward")),
  };

  return (
    <div className="space-y-4">
      <Card data-testid="card-cone-signal">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2 flex-wrap">
            <Target className="h-4 w-4 text-primary" />
            Cone Signal Generator
            <Badge variant="outline" className="text-xs ml-auto">
              16-bar horizon
            </Badge>
            <span className="text-xs text-muted-foreground" data-testid="text-last-updated">
              Updated: {dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : '-'}
            </span>
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-6 w-6"
              onClick={() => refetch()}
              data-testid="button-refresh-cone"
            >
              <RefreshCw className="h-3 w-3" />
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className={`rounded-lg p-4 ${directionBg}`}>
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-3">
                {signal.direction === "LONG" ? (
                  <TrendingUp className={`h-8 w-8 ${directionColor}`} />
                ) : signal.direction === "SHORT" ? (
                  <TrendingDown className={`h-8 w-8 ${directionColor}`} />
                ) : (
                  <Minus className={`h-8 w-8 ${directionColor}`} />
                )}
                <div>
                  <div className={`text-2xl font-bold ${directionColor}`} data-testid="text-cone-direction">
                    {signal.direction}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Edge: {(signal.edge * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
              {!isHold && (
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={handleRecordSignal}
                  disabled={isRecording}
                  data-testid="button-record-signal"
                >
                  {isRecording ? (
                    <Loader2 className="h-3 w-3 animate-spin mr-1" />
                  ) : (
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                  )}
                  Record Signal
                </Button>
              )}
            </div>
          </div>

          {isHold && signal.holdReasons.length > 0 && (
            <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
              <div className="text-xs font-medium text-amber-400 mb-2 flex items-center gap-1">
                <Filter className="h-3 w-3" /> Hold Reasons
              </div>
              <ul className="text-xs text-muted-foreground space-y-1">
                {signal.holdReasons.map((reason, i) => (
                  <li key={i} className="flex items-start gap-1">
                    <XCircle className="h-3 w-3 text-amber-500 mt-0.5 flex-shrink-0" />
                    {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!isHold && signal.entryPrice !== null && (
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-muted/30 rounded-lg p-3 text-center">
                <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
                  <Crosshair className="h-3 w-3" />
                  Entry
                </div>
                <div className="text-sm font-medium" data-testid="text-cone-entry">
                  {formatPrice(signal.entryPrice)}
                </div>
              </div>
              <div className="bg-muted/30 rounded-lg p-3 text-center">
                <div className="flex items-center justify-center gap-1 text-xs text-red-400 mb-1">
                  <Shield className="h-3 w-3" />
                  Stop Loss
                </div>
                <div className="text-sm font-medium text-red-400" data-testid="text-cone-sl">
                  {formatPrice(signal.stopLoss)}
                </div>
              </div>
              <div className="bg-muted/30 rounded-lg p-3 text-center">
                <div className="flex items-center justify-center gap-1 text-xs text-emerald-400 mb-1">
                  <Target className="h-3 w-3" />
                  Take Profit
                </div>
                <div className="text-sm font-medium text-emerald-400" data-testid="text-cone-tp">
                  {formatPrice(signal.takeProfit)}
                </div>
              </div>
            </div>
          )}

          {signal.flowForecast && (
            <FlowForecastChart flowForecast={signal.flowForecast} entryPrice={signal.entryPrice} />
          )}

          <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center justify-between bg-muted/20 rounded-lg px-3 py-2">
              <span className="text-xs text-muted-foreground">Risk/Reward</span>
              <Badge variant="outline" className={signal.riskReward && signal.riskReward >= 1.5 ? "text-emerald-400" : "text-amber-400"}>
                {signal.riskReward ? `1:${signal.riskReward.toFixed(1)}` : "-"}
              </Badge>
            </div>
            <div className="flex items-center justify-between bg-muted/20 rounded-lg px-3 py-2">
              <span className="text-xs text-muted-foreground">Expected μ</span>
              <Badge variant="outline" className={signal.mu >= 0 ? "text-emerald-400" : "text-red-400"}>
                {formatPercent(signal.mu)}
              </Badge>
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-xs text-muted-foreground font-medium flex items-center gap-1">
              <Filter className="h-3 w-3" /> Trade Gates
            </div>
            <div className="grid grid-cols-3 gap-2">
              {[
                { label: "P(hold) < 60%", passed: gates.pHoldCheck },
                { label: "|μ| ≥ 0.2%", passed: gates.minMoveCheck },
                { label: `Edge ≥ ${(signal.edgeThreshold * 100).toFixed(1)}%`, passed: gates.edgeCheck },
                { label: "σ ≤ 3%", passed: gates.uncertaintyCheck },
                { label: "6-bar cool", passed: gates.cooldownCheck },
                { label: "R:R ≥ 1.2", passed: gates.rrCheck },
              ].map((gate, i) => (
                <div 
                  key={i}
                  className={`flex items-center gap-1 text-[10px] rounded px-2 py-1 ${
                    gate.passed 
                      ? "bg-emerald-500/10 text-emerald-400" 
                      : "bg-red-500/10 text-red-400"
                  }`}
                >
                  {gate.passed ? (
                    <CheckCircle2 className="h-2.5 w-2.5" />
                  ) : (
                    <XCircle className="h-2.5 w-2.5" />
                  )}
                  {gate.label}
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-between bg-muted/20 rounded-lg px-3 py-2">
            <span className="text-xs text-muted-foreground">Auto-Calibration</span>
            <div className="text-xs text-muted-foreground">
              Target: 3-4/day | 
              Edge threshold: {(signal.edgeThreshold * 100).toFixed(2)}%
            </div>
          </div>
        </CardContent>
      </Card>

      <Card data-testid="card-signal-history">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Signal History
            <Badge variant="outline" className="text-xs ml-auto">
              {signalHistory?.signals?.length || 0} signals
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!signalHistory?.signals?.length ? (
            <div className="text-center py-4 text-sm text-muted-foreground">
              No recorded signals yet
            </div>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {signalHistory.signals.slice(0, 10).map((sig) => (
                <div 
                  key={sig.id}
                  className="flex items-center justify-between bg-muted/20 rounded-lg px-3 py-2 text-xs"
                  data-testid={`signal-history-${sig.id}`}
                >
                  <div className="flex items-center gap-2">
                    <Badge 
                      variant="outline" 
                      className={sig.direction === "LONG" ? "text-emerald-400" : "text-red-400"}
                    >
                      {sig.direction}
                    </Badge>
                    <span className="text-muted-foreground">
                      {new Date(sig.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-muted-foreground">
                      Entry: {formatPrice(sig.entryPrice)}
                    </span>
                    {sig.outcome ? (
                      <Badge 
                        variant="outline" 
                        className={
                          sig.outcome === "HIT_TP" 
                            ? "text-emerald-400" 
                            : sig.outcome === "HIT_SL" 
                              ? "text-red-400" 
                              : "text-amber-400"
                        }
                      >
                        {sig.outcome === "HIT_TP" && <CheckCircle2 className="h-2.5 w-2.5 mr-1" />}
                        {sig.outcome === "HIT_SL" && <XCircle className="h-2.5 w-2.5 mr-1" />}
                        {sig.outcome === "EXPIRED" && <Timer className="h-2.5 w-2.5 mr-1" />}
                        {sig.outcome} {sig.pnlPercent !== null && (
                          <span className="ml-1">
                            ({typeof sig.pnlPercent === 'number' ? (sig.pnlPercent >= 0 ? '+' : '') + sig.pnlPercent.toFixed(2) : sig.pnlPercent}%)
                          </span>
                        )}
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="text-blue-400">
                        <Clock className="h-2.5 w-2.5 mr-1" />
                        PENDING
                      </Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
