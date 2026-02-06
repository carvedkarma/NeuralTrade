import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Brain, TrendingUp, TrendingDown, Minus, Target, Shield, 
  Activity, BarChart3, Gauge, AlertTriangle, Zap, ArrowUp, ArrowDown 
} from "lucide-react";
import { motion } from "framer-motion";

export interface MultiheadPredictionData {
  action: string;
  confidence: number;
  direction_probs: { LONG: number; SHORT: number; HOLD: number };
  quantiles: { q10: number; q25: number; q50: number; q75: number; q90: number } | null;
  vol_state: string | null;
  vol_state_probs: { contraction: number; neutral: number; expansion: number } | null;
  mu: number | null;
  sigma: number | null;
  edge: number | null;
  entry_price: number | null;
  stop_loss_price: number | null;
  take_profit_price: number | null;
  stop_loss_pct: number | null;
  take_profit_pct: number | null;
  risk_reward_ratio: number | null;
  position_size_pct: number | null;
  current_price: number | null;
  model_name: string | null;
  is_multihead: boolean | null;
  urgency: string | null;
  suggested_order_type: string | null;
  reasons: string[] | null;
  timestamp: number;
  created_at: number;
}

interface MultiheadSignalCardProps {
  prediction: MultiheadPredictionData | null;
  isStale?: boolean;
}

const actionConfig: Record<string, { icon: typeof TrendingUp; color: string; bg: string; border: string }> = {
  LONG: { icon: TrendingUp, color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30" },
  SHORT: { icon: TrendingDown, color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/30" },
  HOLD: { icon: Minus, color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/30" },
};

const volStateConfig: Record<string, { color: string; bg: string; label: string }> = {
  contraction: { color: "text-blue-400", bg: "bg-blue-500/15", label: "Contraction" },
  neutral: { color: "text-slate-400", bg: "bg-slate-500/15", label: "Neutral" },
  expansion: { color: "text-orange-400", bg: "bg-orange-500/15", label: "Expansion" },
};

function HeadStatusDot({ active, label }: { active: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1.5" data-testid={`head-status-${label.toLowerCase()}`}>
      <div className={`w-2 h-2 rounded-full ${active ? "bg-emerald-500" : "bg-muted-foreground/30"}`} />
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}

export function MultiheadSignalCard({ prediction, isStale }: MultiheadSignalCardProps) {
  if (!prediction) {
    return (
      <Card data-testid="card-multihead-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Brain className="h-4 w-4" />
            5-Head Neural Network Signal
          </CardTitle>
        </CardHeader>
        <CardContent className="text-center py-6">
          <Brain className="h-8 w-8 mx-auto text-muted-foreground/50 mb-2" />
          <p className="text-sm text-muted-foreground">Waiting for GPU trainer prediction</p>
          <p className="text-xs text-muted-foreground mt-1">Connect your local trainer to see live signals</p>
        </CardContent>
      </Card>
    );
  }

  const config = actionConfig[prediction.action] || actionConfig.HOLD;
  const Icon = config.icon;
  const volState = prediction.vol_state ? volStateConfig[prediction.vol_state] : null;
  const hasQuantiles = prediction.quantiles != null;
  const hasMu = prediction.mu != null;
  const hasSigma = prediction.sigma != null;
  const hasVolState = prediction.vol_state != null;

  const uncertaintyLevel = (prediction.sigma ?? 0) < 0.005 ? "Low" 
    : (prediction.sigma ?? 0) < 0.015 ? "Medium" 
    : "High";
  const uncertaintyColor = uncertaintyLevel === "Low" ? "text-emerald-400" 
    : uncertaintyLevel === "Medium" ? "text-amber-400" 
    : "text-red-400";

  return (
    <Card className="overflow-visible" data-testid="card-multihead-signal">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Brain className="h-4 w-4" />
            5-Head Neural Network Signal
          </CardTitle>
          <div className="flex items-center gap-2">
            {isStale && (
              <Badge className="bg-amber-500/20 text-amber-400 border-0 text-xs" data-testid="badge-stale">
                STALE
              </Badge>
            )}
            {prediction.is_multihead && (
              <Badge className="bg-purple-500/20 text-purple-400 border-0 text-xs" data-testid="badge-multihead">
                Multi-Head
              </Badge>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3 mt-1 flex-wrap">
          <HeadStatusDot active={true} label="Classification" />
          <HeadStatusDot active={hasQuantiles} label="Quantile" />
          <HeadStatusDot active={hasVolState} label="VolState" />
          <HeadStatusDot active={hasMu} label="Mu" />
          <HeadStatusDot active={hasSigma} label="Sigma" />
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Direction Signal */}
        <motion.div
          key={prediction.action}
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 20 }}
          className={`flex items-center justify-between gap-3 p-3 rounded-md border ${config.bg} ${config.border}`}
          data-testid="multihead-signal-display"
        >
          <div className="flex items-center gap-3">
            <Icon className={`h-7 w-7 ${config.color}`} />
            <div>
              <span className={`text-2xl font-bold font-mono tracking-tight ${config.color}`} data-testid="text-multihead-action">
                {prediction.action}
              </span>
              <p className="text-xs text-muted-foreground">
                L:{(prediction.direction_probs.LONG * 100).toFixed(0)}% / 
                H:{(prediction.direction_probs.HOLD * 100).toFixed(0)}% / 
                S:{(prediction.direction_probs.SHORT * 100).toFixed(0)}%
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm font-mono font-medium" data-testid="text-multihead-confidence">
              {(prediction.confidence * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">confidence</p>
          </div>
        </motion.div>

        {/* Vol State Regime */}
        {hasVolState && volState && (
          <div className="flex items-center justify-between gap-2" data-testid="multihead-vol-state">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Market Regime</span>
            </div>
            <div className="flex items-center gap-2">
              <Badge className={`${volState.bg} ${volState.color} border-0 text-xs`} data-testid="badge-vol-state">
                {volState.label}
              </Badge>
              {prediction.vol_state_probs && (
                <div className="flex gap-1">
                  {Object.entries(prediction.vol_state_probs).map(([key, val]) => (
                    <span key={key} className="text-[10px] text-muted-foreground font-mono">
                      {key.charAt(0).toUpperCase()}:{(val * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Mu & Sigma Row */}
        {(hasMu || hasSigma) && (
          <div className="grid grid-cols-3 gap-3 pt-2 border-t border-border">
            {hasMu && (
              <div className="space-y-1" data-testid="multihead-mu">
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Target className="h-3 w-3" />
                  Expected (mu)
                </p>
                <span className={`text-sm font-mono font-medium ${(prediction.mu ?? 0) > 0 ? "text-emerald-400" : (prediction.mu ?? 0) < 0 ? "text-red-400" : ""}`}>
                  {(prediction.mu ?? 0) > 0 ? "+" : ""}{((prediction.mu ?? 0) * 100).toFixed(3)}%
                </span>
              </div>
            )}
            {hasSigma && (
              <div className="space-y-1" data-testid="multihead-sigma">
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Gauge className="h-3 w-3" />
                  Uncertainty (sigma)
                </p>
                <span className={`text-sm font-mono font-medium ${uncertaintyColor}`}>
                  {((prediction.sigma ?? 0) * 100).toFixed(3)}%
                  <span className="text-[10px] ml-1">({uncertaintyLevel})</span>
                </span>
              </div>
            )}
            {prediction.edge != null && (
              <div className="space-y-1" data-testid="multihead-edge">
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Zap className="h-3 w-3" />
                  Edge
                </p>
                <div className="flex items-center gap-1">
                  {prediction.edge > 0 ? (
                    <ArrowUp className="h-3 w-3 text-emerald-400" />
                  ) : (
                    <ArrowDown className="h-3 w-3 text-red-400" />
                  )}
                  <span className={`text-sm font-mono font-medium ${prediction.edge > 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {prediction.edge > 0 ? "+" : ""}{prediction.edge.toFixed(3)}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Quantile Spread */}
        {hasQuantiles && prediction.quantiles && (
          <div className="pt-2 border-t border-border" data-testid="multihead-quantiles">
            <p className="text-xs text-muted-foreground flex items-center gap-1 mb-2">
              <BarChart3 className="h-3 w-3" />
              Return Distribution (16-bar horizon)
            </p>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-red-400 font-mono w-14 text-right">
                {(prediction.quantiles.q10 * 100).toFixed(2)}%
              </span>
              <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden relative">
                <div 
                  className="absolute h-full bg-red-500/30 rounded-l-full"
                  style={{ width: `${Math.min(50, 50 + prediction.quantiles.q10 * 500)}%`, left: 0 }}
                />
                <div 
                  className="absolute h-full bg-red-500/20"
                  style={{ 
                    left: `${Math.min(50, 50 + prediction.quantiles.q10 * 500)}%`,
                    width: `${Math.max(0, (prediction.quantiles.q25 - prediction.quantiles.q10) * 500)}%`
                  }}
                />
                <div 
                  className="absolute h-full bg-emerald-500/20"
                  style={{ 
                    left: `${50 + prediction.quantiles.q75 * 500}%`,
                    width: `${Math.max(0, (prediction.quantiles.q90 - prediction.quantiles.q75) * 500)}%`
                  }}
                />
                <div 
                  className="absolute h-full bg-emerald-500/30 rounded-r-full"
                  style={{ 
                    left: `${50 + prediction.quantiles.q75 * 500}%`,
                    width: `${Math.max(0, 50 - (50 + prediction.quantiles.q75 * 500))}%`
                  }}
                />
                <div className="absolute h-full w-0.5 bg-foreground/50" style={{ left: "50%" }} />
                {prediction.mu != null && (
                  <div 
                    className="absolute h-full w-1 bg-primary rounded-full"
                    style={{ left: `${Math.max(2, Math.min(98, 50 + prediction.mu * 500))}%` }}
                  />
                )}
              </div>
              <span className="text-[10px] text-emerald-400 font-mono w-14">
                {(prediction.quantiles.q90 * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between text-[10px] text-muted-foreground font-mono mt-0.5 px-14">
              <span>q25: {(prediction.quantiles.q25 * 100).toFixed(2)}%</span>
              <span>q50: {(prediction.quantiles.q50 * 100).toFixed(2)}%</span>
              <span>q75: {(prediction.quantiles.q75 * 100).toFixed(2)}%</span>
            </div>
          </div>
        )}

        {/* Trade Levels */}
        {prediction.entry_price && prediction.action !== "HOLD" && (
          <div className="pt-2 border-t border-border" data-testid="multihead-trade-levels">
            <div className="grid grid-cols-3 gap-3">
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Entry</p>
                <span className="text-sm font-mono font-medium" data-testid="text-mh-entry">
                  ${prediction.entry_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              </div>
              {prediction.stop_loss_price && (
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Stop Loss</p>
                  <span className="text-sm font-mono text-red-400" data-testid="text-mh-sl">
                    ${prediction.stop_loss_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </span>
                  {prediction.stop_loss_pct != null && (
                    <span className="text-[10px] text-muted-foreground block">
                      ({(prediction.stop_loss_pct * 100).toFixed(2)}%)
                    </span>
                  )}
                </div>
              )}
              {prediction.take_profit_price && (
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Take Profit</p>
                  <span className="text-sm font-mono text-emerald-400" data-testid="text-mh-tp">
                    ${prediction.take_profit_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </span>
                  {prediction.take_profit_pct != null && (
                    <span className="text-[10px] text-muted-foreground block">
                      (+{(prediction.take_profit_pct * 100).toFixed(2)}%)
                    </span>
                  )}
                </div>
              )}
            </div>
            <div className="flex items-center justify-between mt-2 flex-wrap gap-1">
              {prediction.position_size_pct != null && (
                <div className="flex items-center gap-1">
                  <Shield className="h-3 w-3 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">Size:</span>
                  <span className="text-xs font-mono font-medium">{(prediction.position_size_pct * 100).toFixed(1)}%</span>
                </div>
              )}
              {prediction.risk_reward_ratio != null && (
                <span className="text-xs text-muted-foreground">R:R 1:{prediction.risk_reward_ratio.toFixed(2)}</span>
              )}
              {prediction.suggested_order_type && (
                <Badge variant="outline" className="text-[10px]" data-testid="badge-mh-order-type">
                  {prediction.suggested_order_type.toUpperCase()}
                </Badge>
              )}
            </div>
          </div>
        )}

        {/* Model Info */}
        <div className="flex items-center justify-between text-[10px] text-muted-foreground pt-2 border-t border-border flex-wrap gap-1">
          <span>Model: {prediction.model_name ?? "unknown"}</span>
          <span>{new Date(prediction.created_at).toLocaleTimeString()}</span>
        </div>
      </CardContent>
    </Card>
  );
}

export function MultiheadPredictionHistory({ predictions }: { predictions: MultiheadPredictionData[] }) {
  if (!predictions || predictions.length === 0) {
    return null;
  }

  return (
    <Card data-testid="card-multihead-history">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <BarChart3 className="h-4 w-4" />
          Prediction History
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-[300px] overflow-y-auto">
          {predictions.slice(0, 20).map((pred, i) => {
            const conf = actionConfig[pred.action] || actionConfig.HOLD;
            return (
              <div key={i} className="flex items-center justify-between gap-2 py-1.5 border-b border-border/50 last:border-0" data-testid={`prediction-history-${i}`}>
                <div className="flex items-center gap-2">
                  <Badge className={`${conf.bg} ${conf.color} border-0 text-xs min-w-[52px] justify-center`}>
                    {pred.action}
                  </Badge>
                  <span className="text-xs font-mono">{(pred.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="flex items-center gap-2">
                  {pred.vol_state && (
                    <span className="text-[10px] text-muted-foreground capitalize">{pred.vol_state}</span>
                  )}
                  {pred.mu != null && (
                    <span className={`text-[10px] font-mono ${pred.mu > 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {pred.mu > 0 ? "+" : ""}{(pred.mu * 100).toFixed(2)}%
                    </span>
                  )}
                  <span className="text-[10px] text-muted-foreground">
                    {new Date(pred.created_at).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
