import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Brain, TrendingUp, TrendingDown, Minus, Target, Shield, Crosshair, AlertTriangle, Loader2, Bug } from "lucide-react";

export interface QuantilePrediction {
  action: "LONG" | "SHORT" | "HOLD";
  confidence: number;
  entry: number;
  stopLoss: number;
  takeProfit: number;
  riskReward: number;
  expectedMove: number;  // Decimal return (e.g., 0.01 = 1%)
  uncertainty: number;   // Decimal (e.g., 0.02 = 2%)
  quantiles: {
    q10: number;  // Decimal returns
    q25: number;
    q50: number;
    q75: number;
    q90: number;
  };
  directionProbs: {
    LONG: number;
    SHORT: number;
    HOLD: number;
  };
  horizon: string;
  timestamp: number;
  currentPrice?: number;  // Base price used to compute predictions (for consistent upside/downside %)
  units?: "decimal_return" | "percent_return";  // API units indicator
  derived_low_price?: number;  // Backend-computed low price for verification
  derived_high_price?: number; // Backend-computed high price for verification
}

interface NeuralNetworkPredictionCardProps {
  prediction: QuantilePrediction | null;
  isLoading?: boolean;
  onRefresh?: () => void;
  showDebug?: boolean;
}

export function NeuralNetworkPredictionCard({ 
  prediction, 
  isLoading = false,
  onRefresh,
  showDebug: initialShowDebug = false
}: NeuralNetworkPredictionCardProps) {
  const [showDebug, setShowDebug] = useState(initialShowDebug);

  if (isLoading) {
    return (
      <Card data-testid="card-nn-prediction-loading">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Neural Network Prediction
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (!prediction) {
    return (
      <Card data-testid="card-nn-prediction-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Neural Network Prediction
          </CardTitle>
        </CardHeader>
        <CardContent className="text-center py-6">
          <AlertTriangle className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">No prediction available</p>
          <p className="text-xs text-muted-foreground mt-1">Train models first</p>
        </CardContent>
      </Card>
    );
  }

  const directionColor = prediction.action === "LONG" 
    ? "text-emerald-400" 
    : prediction.action === "SHORT"
      ? "text-red-400"
      : "text-amber-400";

  const directionBg = prediction.action === "LONG"
    ? "bg-emerald-500/10"
    : prediction.action === "SHORT"
      ? "bg-red-500/10"
      : "bg-amber-500/10";
  
  const isHold = prediction.action === "HOLD";

  const formatDecimalAsPercent = (val: number) => {
    const pct = val * 100;
    return `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
  };
  const formatPercent = formatDecimalAsPercent;
  const formatPrice = (val: number) => `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  const computedLowPrice = prediction.currentPrice ? prediction.currentPrice * (1 + prediction.quantiles.q10) : null;
  const computedHighPrice = prediction.currentPrice ? prediction.currentPrice * (1 + prediction.quantiles.q90) : null;

  return (
    <Card data-testid="card-nn-prediction">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Brain className="h-4 w-4 text-primary" />
          Neural Network Prediction
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-6 w-6 ml-auto" 
            onClick={() => setShowDebug(!showDebug)}
            data-testid="button-toggle-debug"
          >
            <Bug className={`h-3 w-3 ${showDebug ? "text-amber-400" : "text-muted-foreground"}`} />
          </Button>
          <Badge variant="outline" className="text-xs">
            {prediction.horizon}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Direction Signal */}
        <div className={`rounded-lg p-4 ${directionBg}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {prediction.action === "LONG" ? (
                <TrendingUp className={`h-8 w-8 ${directionColor}`} />
              ) : prediction.action === "SHORT" ? (
                <TrendingDown className={`h-8 w-8 ${directionColor}`} />
              ) : (
                <Minus className={`h-8 w-8 ${directionColor}`} />
              )}
              <div>
                <div className={`text-2xl font-bold ${directionColor}`} data-testid="text-nn-direction">
                  {prediction.action}
                </div>
                <div className="text-xs text-muted-foreground">
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-muted-foreground mb-1">Direction Probs</div>
              <div className="flex gap-2 text-xs">
                <span className="text-emerald-400">L: {(prediction.directionProbs.LONG * 100).toFixed(0)}%</span>
                <span className="text-red-400">S: {(prediction.directionProbs.SHORT * 100).toFixed(0)}%</span>
                <span className="text-amber-400">H: {(prediction.directionProbs.HOLD * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Entry / SL / TP */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
              <Crosshair className="h-3 w-3" />
              {isHold ? "Current" : "Entry"}
            </div>
            <div className="text-sm font-medium" data-testid="text-nn-entry">
              {formatPrice(prediction.entry)}
            </div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className={`flex items-center justify-center gap-1 text-xs ${isHold ? "text-muted-foreground" : "text-red-400"} mb-1`}>
              <Shield className="h-3 w-3" />
              {isHold ? "Low Range" : "Stop Loss"}
            </div>
            <div className={`text-sm font-medium ${isHold ? "" : "text-red-400"}`} data-testid="text-nn-sl">
              {formatPrice(prediction.stopLoss)}
            </div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className={`flex items-center justify-center gap-1 text-xs ${isHold ? "text-muted-foreground" : "text-emerald-400"} mb-1`}>
              <Target className="h-3 w-3" />
              {isHold ? "High Range" : "Take Profit"}
            </div>
            <div className={`text-sm font-medium ${isHold ? "" : "text-emerald-400"}`} data-testid="text-nn-tp">
              {formatPrice(prediction.takeProfit)}
            </div>
          </div>
        </div>

        {/* Risk/Reward and Expected Move */}
        <div className="grid grid-cols-2 gap-3">
          <div className="flex items-center justify-between bg-muted/20 rounded-lg px-3 py-2">
            <span className="text-xs text-muted-foreground">Risk/Reward</span>
            <Badge variant="outline" className={prediction.riskReward >= 2 ? "text-emerald-400" : "text-amber-400"}>
              1:{prediction.riskReward.toFixed(1)}
            </Badge>
          </div>
          <div className="flex items-center justify-between bg-muted/20 rounded-lg px-3 py-2">
            <span className="text-xs text-muted-foreground">Expected Move</span>
            <Badge variant="outline" className={prediction.expectedMove >= 0 ? "text-emerald-400" : "text-red-400"}>
              {formatPercent(prediction.expectedMove)}
            </Badge>
          </div>
        </div>

        {/* Quantile Predictions */}
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground font-medium">Expected Return Quantiles</div>
          <div className="grid grid-cols-5 gap-1">
            {[
              { label: "q10", value: prediction.quantiles.q10, color: "text-red-400" },
              { label: "q25", value: prediction.quantiles.q25, color: "text-orange-400" },
              { label: "q50", value: prediction.quantiles.q50, color: "text-blue-400" },
              { label: "q75", value: prediction.quantiles.q75, color: "text-cyan-400" },
              { label: "q90", value: prediction.quantiles.q90, color: "text-emerald-400" },
            ].map(q => (
              <div key={q.label} className="bg-muted/20 rounded p-2 text-center">
                <div className="text-[10px] text-muted-foreground">{q.label}</div>
                <div className={`text-xs font-medium ${q.color}`}>
                  {formatPercent(q.value)}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Uncertainty */}
        <div className="flex items-center justify-between bg-muted/20 rounded-lg px-3 py-2">
          <span className="text-xs text-muted-foreground">Uncertainty (q90-q10 spread)</span>
          <Badge variant="outline" className={(prediction.uncertainty * 100) > 5 ? "text-amber-400" : "text-muted-foreground"}>
            {(prediction.uncertainty * 100).toFixed(2)}%
          </Badge>
        </div>

        {/* DEBUG Panel - Shows raw values for unit verification */}
        {showDebug && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 space-y-2" data-testid="debug-panel">
            <div className="text-xs font-medium text-amber-400 flex items-center gap-1">
              <Bug className="h-3 w-3" /> Debug: Raw API Values
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
              <div>
                <span className="text-muted-foreground">units:</span>
                <span className="ml-1 text-amber-300">{prediction.units || "unknown"}</span>
              </div>
              <div>
                <span className="text-muted-foreground">currentPrice:</span>
                <span className="ml-1">${prediction.currentPrice?.toFixed(2) || "N/A"}</span>
              </div>
              <div>
                <span className="text-muted-foreground">raw mu:</span>
                <span className="ml-1">{prediction.expectedMove?.toFixed(6) || "N/A"}</span>
              </div>
              <div>
                <span className="text-muted-foreground">raw σ:</span>
                <span className="ml-1">{prediction.uncertainty?.toFixed(6) || "N/A"}</span>
              </div>
              <div>
                <span className="text-muted-foreground">raw q10:</span>
                <span className="ml-1">{prediction.quantiles.q10.toFixed(6)}</span>
              </div>
              <div>
                <span className="text-muted-foreground">raw q50:</span>
                <span className="ml-1">{prediction.quantiles.q50.toFixed(6)}</span>
              </div>
              <div>
                <span className="text-muted-foreground">raw q90:</span>
                <span className="ml-1">{prediction.quantiles.q90.toFixed(6)}</span>
              </div>
              <div>
                <span className="text-muted-foreground">displayed q10:</span>
                <span className="ml-1 text-amber-300">{formatPercent(prediction.quantiles.q10)}</span>
              </div>
            </div>
            <div className="border-t border-amber-500/20 pt-2 mt-2">
              <div className="text-[10px] text-muted-foreground mb-1">Price Range Verification:</div>
              <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                <div>
                  <span className="text-muted-foreground">computed low:</span>
                  <span className="ml-1">${computedLowPrice?.toFixed(2) || "N/A"}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">derived_low:</span>
                  <span className="ml-1">${prediction.derived_low_price?.toFixed(2) || "N/A"}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">computed high:</span>
                  <span className="ml-1">${computedHighPrice?.toFixed(2) || "N/A"}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">derived_high:</span>
                  <span className="ml-1">${prediction.derived_high_price?.toFixed(2) || "N/A"}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">displayed SL:</span>
                  <span className="ml-1">${prediction.stopLoss?.toFixed(2) || "N/A"}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">displayed TP:</span>
                  <span className="ml-1">${prediction.takeProfit?.toFixed(2) || "N/A"}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {onRefresh && (
          <Button 
            variant="outline" 
            size="sm" 
            onClick={onRefresh} 
            className="w-full"
            data-testid="button-refresh-nn-prediction"
          >
            Refresh Prediction
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

interface TrainingModeBadgeProps {
  mode: "quick" | "full" | null;
  description: string | null;
  inputDim: number | null;
  connected: boolean;
}

export function TrainingModeBadge({ mode, description, inputDim, connected }: TrainingModeBadgeProps) {
  const isQuick = mode === "quick";
  const isFull = mode === "full";
  
  return (
    <Card data-testid="card-training-mode">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Brain className="h-4 w-4" />
          Training Mode
          <Badge 
            variant={connected ? "default" : "outline"} 
            className={`ml-auto ${connected ? "bg-emerald-500/20 text-emerald-400" : ""}`}
          >
            {connected ? "GPU Connected" : "Not Connected"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {!connected ? (
          <div className="text-center py-4">
            <AlertTriangle className="h-6 w-6 mx-auto text-amber-400 mb-2" />
            <p className="text-sm text-muted-foreground">GPU trainer not connected</p>
            <p className="text-xs text-muted-foreground mt-1">Training mode is set on your local GPU machine</p>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-center">
              <Badge 
                variant="outline" 
                className={`text-base px-4 py-2 ${
                  isQuick ? "bg-amber-500/10 text-amber-400 border-amber-500/30" :
                  isFull ? "bg-blue-500/10 text-blue-400 border-blue-500/30" :
                  ""
                }`}
                data-testid="badge-training-mode"
              >
                {isQuick ? "Quick (15m Only)" : isFull ? "Full MTF" : "Unknown"}
              </Badge>
            </div>
            
            <div className="text-xs text-muted-foreground space-y-1 text-center">
              {isQuick ? (
                <>
                  <p>Using 15-minute timeframe only</p>
                  <p>~{inputDim ?? 57} features</p>
                  <p className="text-amber-400/80">Good for quick testing</p>
                </>
              ) : isFull ? (
                <>
                  <p>Multi-timeframe fusion (5m, 15m, 1h, 4h)</p>
                  <p>~{inputDim ?? 81} features</p>
                  <p className="text-blue-400/80">Best for production signals</p>
                </>
              ) : (
                <p>Mode will be detected from GPU trainer</p>
              )}
            </div>
            
            {description && (
              <p className="text-xs text-center text-muted-foreground border-t border-border/50 pt-2 mt-2">
                {description}
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
