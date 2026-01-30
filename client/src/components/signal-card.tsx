import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Signal, SignalType } from "@shared/schema";
import { TrendingUp, TrendingDown, Minus, ArrowUp, ArrowDown, Target, AlertTriangle, Zap } from "lucide-react";
import { motion } from "framer-motion";

interface SignalCardProps {
  signal: Signal;
}

const signalConfig: Record<SignalType, { icon: typeof TrendingUp; color: string; bg: string; border: string }> = {
  LONG: { icon: TrendingUp, color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30" },
  SHORT: { icon: TrendingDown, color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/30" },
  HOLD: { icon: Minus, color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/30" },
};

const urgencyConfig: Record<string, { color: string; bg: string }> = {
  low: { color: "text-slate-400", bg: "bg-slate-500/20" },
  medium: { color: "text-amber-400", bg: "bg-amber-500/20" },
  high: { color: "text-red-400", bg: "bg-red-500/20" },
};

export function SignalCard({ signal }: SignalCardProps) {
  const config = signalConfig[signal.signal];
  const Icon = config.icon;
  const hasRegressionData = signal.mu !== undefined && signal.sigma !== undefined;
  const urgency = signal.urgency || "low";
  const urgencyStyle = urgencyConfig[urgency];

  return (
    <Card className="overflow-visible" data-testid="card-signal">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <CardTitle className="text-sm font-medium text-muted-foreground" data-testid="text-signal-title">Current Signal</CardTitle>
          {signal.urgency && (
            <Badge className={`${urgencyStyle.bg} ${urgencyStyle.color} border-0`} data-testid="badge-urgency">
              <Zap className="h-3 w-3 mr-1" />
              {urgency.toUpperCase()}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <motion.div
          key={signal.signal}
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 20 }}
          className={`flex items-center justify-center gap-3 p-4 rounded-md border ${config.bg} ${config.border}`}
          data-testid="signal-display"
        >
          <Icon className={`h-8 w-8 ${config.color}`} />
          <span className={`text-3xl font-bold font-mono tracking-tight ${config.color}`} data-testid="text-signal-value">
            {signal.signal}
          </span>
        </motion.div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Confidence</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${signal.confidence * 100}%` }}
                  transition={{ duration: 0.5 }}
                  className={`h-full ${signal.signal === "LONG" ? "bg-emerald-500" : signal.signal === "SHORT" ? "bg-red-500" : "bg-amber-500"}`}
                  data-testid="progress-confidence"
                />
              </div>
              <span className="text-sm font-mono font-medium" data-testid="text-confidence">{(signal.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">
              {hasRegressionData ? "Edge (risk-adjusted)" : "Edge"}
            </p>
            <div className="flex items-center gap-1">
              {signal.edge > 0 ? (
                <ArrowUp className="h-4 w-4 text-emerald-400" />
              ) : (
                <ArrowDown className="h-4 w-4 text-red-400" />
              )}
              <span className={`text-sm font-mono font-medium ${signal.edge > 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-edge">
                {hasRegressionData 
                  ? `${signal.edge > 0 ? "+" : ""}${signal.edge.toFixed(2)}σ`
                  : `${signal.edge > 0 ? "+" : ""}${(signal.edge * 100).toFixed(2)}%`
                }
              </span>
            </div>
          </div>
        </div>

        {hasRegressionData && (
          <div className="grid grid-cols-2 gap-3 pt-2 border-t border-border">
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                <Target className="h-3 w-3" />
                Expected Return (μ)
              </p>
              <span className={`text-sm font-mono font-medium ${(signal.mu ?? 0) > 0 ? "text-emerald-400" : (signal.mu ?? 0) < 0 ? "text-red-400" : ""}`} data-testid="text-mu">
                {(signal.mu ?? 0) > 0 ? "+" : ""}{((signal.mu ?? 0) * 100).toFixed(3)}%
              </span>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                <AlertTriangle className="h-3 w-3" />
                Uncertainty (σ)
              </p>
              <span className="text-sm font-mono font-medium" data-testid="text-sigma">
                {((signal.sigma ?? 0) * 100).toFixed(3)}%
              </span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3 pt-2 border-t border-border">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Expected Move</p>
            <span className="text-sm font-mono" data-testid="text-expected-move">{(signal.expectedMove * 100).toFixed(2)}%</span>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Est. Costs</p>
            <span className="text-sm font-mono text-muted-foreground" data-testid="text-costs">{(signal.costs * 100).toFixed(3)}%</span>
          </div>
        </div>

        {signal.positionSizePct !== undefined && (
          <div className="space-y-3 pt-2 border-t border-border">
            {/* Learned levels indicator */}
            {signal.isLearnedLevels && (
              <div className="flex items-center gap-2">
                <Badge className="bg-purple-500/20 text-purple-400 border-0 text-xs" data-testid="badge-learned">
                  MFE/MAE Learned
                </Badge>
                {signal.riskRewardRatio !== undefined && (
                  <span className="text-xs text-muted-foreground">
                    R:R 1:{signal.riskRewardRatio.toFixed(2)}
                  </span>
                )}
              </div>
            )}
            
            <div className="grid grid-cols-3 gap-3">
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Position Size</p>
                <span className="text-sm font-mono font-medium text-primary" data-testid="text-position-size">
                  {(signal.positionSizePct * 100).toFixed(1)}%
                </span>
              </div>
              {signal.stopLossPct !== undefined && (
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Stop Loss</p>
                  <span className="text-sm font-mono text-red-400" data-testid="text-stop-loss">
                    -{(signal.stopLossPct * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              {signal.takeProfitPct !== undefined && (
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Take Profit</p>
                  <span className="text-sm font-mono text-emerald-400" data-testid="text-take-profit">
                    +{(signal.takeProfitPct * 100).toFixed(2)}%
                  </span>
                </div>
              )}
            </div>

            {/* Price levels if multihead */}
            {signal.isMultihead && signal.entryPrice && (
              <div className="grid grid-cols-3 gap-3 pt-2 border-t border-border/50">
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Entry</p>
                  <span className="text-sm font-mono font-medium" data-testid="text-entry-price">
                    ${signal.entryPrice.toLocaleString()}
                  </span>
                  {signal.entryOffsetPct !== undefined && signal.entryOffsetPct !== 0 && (
                    <span className="text-xs text-muted-foreground">
                      ({signal.entryOffsetPct > 0 ? "+" : ""}{(signal.entryOffsetPct * 100).toFixed(3)}%)
                    </span>
                  )}
                </div>
                {signal.stopLossPrice && (
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">SL Price</p>
                    <span className="text-sm font-mono text-red-400" data-testid="text-sl-price">
                      ${signal.stopLossPrice.toLocaleString()}
                    </span>
                  </div>
                )}
                {signal.takeProfitPrice && (
                  <div className="space-y-1">
                    <p className="text-xs text-muted-foreground">TP Price</p>
                    <span className="text-sm font-mono text-emerald-400" data-testid="text-tp-price">
                      ${signal.takeProfitPrice.toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {signal.suggestedOrderType && (
          <div className="pt-2 border-t border-border">
            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground">Order Type</p>
              <Badge variant="outline" className="text-xs" data-testid="badge-order-type">
                {signal.suggestedOrderType.toUpperCase()}
              </Badge>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
