import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Minus, Brain, Activity, Shield, Users, AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";

interface ModelVote {
  action: string;
  confidence: number;
  confidence_margin: number;
  weight: number;
  probs: {
    SHORT: number;
    HOLD: number;
    LONG: number;
  };
}

interface EnsemblePrediction {
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
  model_votes: Record<string, ModelVote>;
  ensemble_probs: {
    SHORT: number;
    HOLD: number;
    LONG: number;
  };
  reasons: string[];
}

interface EnsembleStatus {
  initialized: boolean;
  direction_models: string[];
  regime_models: string[];
  risk_models: string[];
}

const actionConfig = {
  LONG: { icon: TrendingUp, color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30" },
  SHORT: { icon: TrendingDown, color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/30" },
  HOLD: { icon: Minus, color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/30" },
  NO_TRADE: { icon: XCircle, color: "text-slate-400", bg: "bg-slate-500/10", border: "border-slate-500/30" },
};

const regimeConfig: Record<string, { color: string; bg: string }> = {
  TRENDING: { color: "text-emerald-400", bg: "bg-emerald-500/20" },
  RANGING: { color: "text-blue-400", bg: "bg-blue-500/20" },
  CHOPPY: { color: "text-amber-400", bg: "bg-amber-500/20" },
  HIGH_VOLATILITY: { color: "text-red-400", bg: "bg-red-500/20" },
  RISK_ON: { color: "text-emerald-400", bg: "bg-emerald-500/20" },
  RISK_OFF: { color: "text-red-400", bg: "bg-red-500/20" },
  NEUTRAL: { color: "text-slate-400", bg: "bg-slate-500/20" },
  CORRELATION_SHOCK: { color: "text-purple-400", bg: "bg-purple-500/20" },
  UNKNOWN: { color: "text-slate-500", bg: "bg-slate-500/10" },
};

interface EnsembleSignalCardProps {
  prediction?: EnsemblePrediction | null;
  status?: EnsembleStatus | null;
  isLoading?: boolean;
}

export function EnsembleSignalCard({ prediction, status, isLoading }: EnsembleSignalCardProps) {
  if (isLoading) {
    return (
      <Card className="overflow-visible" data-testid="card-ensemble-signal">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <CardTitle className="text-sm font-medium">Neural Ensemble Signal</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32 text-muted-foreground">
            <Activity className="h-5 w-5 animate-pulse mr-2" />
            Loading ensemble...
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!prediction) {
    return (
      <Card className="overflow-visible" data-testid="card-ensemble-signal">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <CardTitle className="text-sm font-medium">Neural Ensemble Signal</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center h-32 text-muted-foreground text-center">
            <AlertTriangle className="h-6 w-6 mb-2 text-amber-500" />
            <p className="text-sm">GPU ensemble not available</p>
            <p className="text-xs mt-1">Connect GPU trainer for neural predictions</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const config = actionConfig[prediction.action] || actionConfig.HOLD;
  const Icon = config.icon;
  const marketRegimeStyle = regimeConfig[prediction.market_regime] || regimeConfig.UNKNOWN;
  const riskRegimeStyle = regimeConfig[prediction.risk_regime] || regimeConfig.UNKNOWN;
  
  const modelVotes = Object.entries(prediction.model_votes || {});
  const agreementPercent = (prediction.agreement_pct * 100).toFixed(0);
  const weightedAgreement = (prediction.weighted_agreement * 100).toFixed(0);

  return (
    <Card className="overflow-visible" data-testid="card-ensemble-signal">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <CardTitle className="text-sm font-medium">Neural Ensemble Signal</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={`${marketRegimeStyle.bg} ${marketRegimeStyle.color} border-0`} data-testid="badge-market-regime">
              {prediction.market_regime}
            </Badge>
            <Badge className={`${riskRegimeStyle.bg} ${riskRegimeStyle.color} border-0`} data-testid="badge-risk-regime">
              <Shield className="h-3 w-3 mr-1" />
              {prediction.risk_regime}
            </Badge>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <motion.div
          key={prediction.action}
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 20 }}
          className={`flex items-center justify-center gap-3 p-4 rounded-md border ${config.bg} ${config.border}`}
          data-testid="ensemble-signal-display"
        >
          <Icon className={`h-8 w-8 ${config.color}`} />
          <span className={`text-3xl font-bold font-mono tracking-tight ${config.color}`} data-testid="text-ensemble-action">
            {prediction.action}
          </span>
        </motion.div>

        <div className="grid grid-cols-3 gap-3">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Confidence</p>
            <div className="flex items-center gap-1">
              <span className="text-sm font-mono font-medium" data-testid="text-ensemble-confidence">
                {(prediction.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Margin</p>
            <span className="text-sm font-mono font-medium" data-testid="text-confidence-margin">
              {(prediction.confidence_margin * 100).toFixed(1)}%
            </span>
          </div>
          
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Edge</p>
            <span className={`text-sm font-mono font-medium ${prediction.edge > 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-ensemble-edge">
              {prediction.edge > 0 ? "+" : ""}{(prediction.edge * 100).toFixed(2)}%
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-2 border-t border-border">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <Users className="h-3 w-3" />
              Model Agreement
            </p>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${prediction.weighted_agreement * 100}%` }}
                  transition={{ duration: 0.5 }}
                  className={prediction.weighted_agreement > 0.7 ? "bg-emerald-500" : prediction.weighted_agreement > 0.5 ? "bg-amber-500" : "bg-red-500"}
                  style={{ height: '100%' }}
                />
              </div>
              <span className="text-sm font-mono" data-testid="text-agreement">{weightedAgreement}%</span>
            </div>
          </div>
          
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Position Size</p>
            <span className="text-sm font-mono font-medium text-primary" data-testid="text-position-size">
              {(prediction.regime_adjusted_size * 100).toFixed(1)}%
            </span>
            {prediction.regime_adjustment !== "NONE" && (
              <p className="text-xs text-amber-400">{prediction.regime_adjustment}</p>
            )}
          </div>
        </div>

        {modelVotes.length > 0 && (
          <div className="pt-2 border-t border-border">
            <p className="text-xs text-muted-foreground mb-2">Model Votes</p>
            <div className="grid grid-cols-2 gap-2">
              {modelVotes.slice(0, 6).map(([name, vote]) => {
                const voteColor = vote.action === "LONG" ? "text-emerald-400" : 
                                  vote.action === "SHORT" ? "text-red-400" : "text-amber-400";
                return (
                  <div key={name} className="flex items-center justify-between text-xs bg-muted/50 rounded px-2 py-1">
                    <span className="text-muted-foreground truncate max-w-[80px]">{name.replace(/_/g, ' ')}</span>
                    <div className="flex items-center gap-1">
                      <span className={voteColor}>{vote.action}</span>
                      <span className="text-muted-foreground">({(vote.confidence * 100).toFixed(0)}%)</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <div className="pt-2 border-t border-border">
          <p className="text-xs text-muted-foreground mb-2">Probabilities</p>
          <div className="flex gap-2 h-6">
            <div 
              className="bg-red-500/50 rounded-l flex items-center justify-center text-xs font-mono"
              style={{ width: `${prediction.ensemble_probs.SHORT * 100}%` }}
              data-testid="bar-prob-short"
            >
              {prediction.ensemble_probs.SHORT > 0.15 && `${(prediction.ensemble_probs.SHORT * 100).toFixed(0)}%`}
            </div>
            <div 
              className="bg-amber-500/50 flex items-center justify-center text-xs font-mono"
              style={{ width: `${prediction.ensemble_probs.HOLD * 100}%` }}
              data-testid="bar-prob-hold"
            >
              {prediction.ensemble_probs.HOLD > 0.15 && `${(prediction.ensemble_probs.HOLD * 100).toFixed(0)}%`}
            </div>
            <div 
              className="bg-emerald-500/50 rounded-r flex items-center justify-center text-xs font-mono"
              style={{ width: `${prediction.ensemble_probs.LONG * 100}%` }}
              data-testid="bar-prob-long"
            >
              {prediction.ensemble_probs.LONG > 0.15 && `${(prediction.ensemble_probs.LONG * 100).toFixed(0)}%`}
            </div>
          </div>
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>SHORT</span>
            <span>HOLD</span>
            <span>LONG</span>
          </div>
        </div>

        {prediction.reasons.length > 0 && (
          <div className="pt-2 border-t border-border">
            <p className="text-xs text-muted-foreground mb-1">Reasoning</p>
            <div className="space-y-1">
              {prediction.reasons.slice(0, 3).map((reason, idx) => (
                <p key={idx} className="text-xs text-muted-foreground flex items-start gap-1">
                  <CheckCircle2 className="h-3 w-3 text-primary mt-0.5 flex-shrink-0" />
                  {reason}
                </p>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
