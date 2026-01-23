import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { ShotPlan } from "@shared/schema";
import { 
  Target, TrendingUp, TrendingDown, Minus, Shield, Clock, 
  AlertTriangle, CheckCircle2, XCircle, DollarSign, Percent,
  ArrowUpRight, ArrowDownRight
} from "lucide-react";

interface ShotPlanCardProps {
  shotPlan?: ShotPlan;
}

export function ShotPlanCard({ shotPlan }: ShotPlanCardProps) {
  if (!shotPlan) {
    return (
      <Card data-testid="card-shot-plan-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Target className="h-4 w-4 text-muted-foreground" />
            Shot Plan
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground" data-testid="text-no-shot-plan">
            Generating ML-based shot plan... Need 250+ candles for pattern analysis.
          </p>
        </CardContent>
      </Card>
    );
  }

  const signalColor = 
    shotPlan.signal === "LONG" ? "text-emerald-400" :
    shotPlan.signal === "SHORT" ? "text-red-400" :
    "text-amber-400";
  
  const signalBg = 
    shotPlan.signal === "LONG" ? "bg-emerald-500/20" :
    shotPlan.signal === "SHORT" ? "bg-red-500/20" :
    "bg-amber-500/20";

  const SignalIcon = 
    shotPlan.signal === "LONG" ? TrendingUp :
    shotPlan.signal === "SHORT" ? TrendingDown : Minus;

  const formatPrice = (price: number | null) => 
    price ? `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—";

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

  return (
    <Card data-testid="card-shot-plan">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Shot Plan
          </CardTitle>
          <Badge 
            className={`${signalBg} ${signalColor} flex items-center gap-1`}
            data-testid="badge-shot-signal"
          >
            <SignalIcon className="h-3 w-3" />
            {shotPlan.signal}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Confidence</span>
          <div className="flex items-center gap-2">
            <Progress 
              value={shotPlan.confidence * 100} 
              className="w-24 h-2" 
              data-testid="progress-confidence"
            />
            <span className="text-sm font-medium" data-testid="text-confidence">
              {(shotPlan.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Regime</span>
            <Badge variant="outline" className="text-xs capitalize" data-testid="badge-regime">
              {shotPlan.regime.replace("_", " ")}
            </Badge>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Strategy</span>
            <Badge variant="secondary" className="text-xs" data-testid="badge-strategy">
              {shotPlan.strategy}
            </Badge>
          </div>
        </div>

        {shotPlan.entryZone && (
          <div className="bg-muted/50 rounded-lg p-3 space-y-2">
            <div className="text-xs font-medium text-muted-foreground mb-2">Trade Levels</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex items-center gap-1.5">
                <ArrowUpRight className="h-3 w-3 text-emerald-400" />
                <span className="text-muted-foreground">Entry:</span>
                <span className="font-mono" data-testid="text-entry-zone">
                  {formatPrice(shotPlan.entryZone.low)} - {formatPrice(shotPlan.entryZone.high)}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <Shield className="h-3 w-3 text-red-400" />
                <span className="text-muted-foreground">Stop:</span>
                <span className="font-mono" data-testid="text-stop-loss">
                  {formatPrice(shotPlan.stopLoss)}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <Target className="h-3 w-3 text-emerald-400" />
                <span className="text-muted-foreground">TP1:</span>
                <span className="font-mono" data-testid="text-tp1">
                  {formatPrice(shotPlan.takeProfit1)}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <Target className="h-3 w-3 text-emerald-400" />
                <span className="text-muted-foreground">TP2:</span>
                <span className="font-mono" data-testid="text-tp2">
                  {formatPrice(shotPlan.takeProfit2)}
                </span>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-emerald-500/10 rounded p-2">
            <div className="text-xs text-muted-foreground">Up</div>
            <div className="text-sm font-medium text-emerald-400" data-testid="text-prob-up">
              {(shotPlan.probUp * 100).toFixed(0)}%
            </div>
          </div>
          <div className="bg-red-500/10 rounded p-2">
            <div className="text-xs text-muted-foreground">Down</div>
            <div className="text-sm font-medium text-red-400" data-testid="text-prob-down">
              {(shotPlan.probDown * 100).toFixed(0)}%
            </div>
          </div>
          <div className="bg-amber-500/10 rounded p-2">
            <div className="text-xs text-muted-foreground">Chop</div>
            <div className="text-sm font-medium text-amber-400" data-testid="text-prob-chop">
              {(shotPlan.probChop * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="flex items-center gap-2">
            <DollarSign className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">R:R</span>
            <span className="font-medium" data-testid="text-risk-reward">
              1:{shotPlan.riskReward.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Hold</span>
            <span className="font-medium" data-testid="text-hold-time">
              {shotPlan.expectedHoldTime}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Percent className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Edge</span>
            <span className={`font-medium ${shotPlan.edge > 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-edge">
              {formatPercent(shotPlan.edge)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Percent className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Costs</span>
            <span className="font-medium text-muted-foreground" data-testid="text-costs">
              {formatPercent(shotPlan.estimatedCosts)}
            </span>
          </div>
        </div>

        {shotPlan.reasons.length > 0 && (
          <div className="space-y-1">
            <div className="text-xs font-medium text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Reasons to Trade
            </div>
            <ul className="text-xs text-muted-foreground space-y-0.5">
              {shotPlan.reasons.map((reason, i) => (
                <li key={i} className="flex items-start gap-1.5" data-testid={`text-reason-${i}`}>
                  <ArrowUpRight className="h-3 w-3 text-emerald-400 mt-0.5 flex-shrink-0" />
                  {reason}
                </li>
              ))}
            </ul>
          </div>
        )}

        {shotPlan.vetoReasons.length > 0 && (
          <div className="space-y-1">
            <div className="text-xs font-medium text-amber-400 flex items-center gap-1">
              <AlertTriangle className="h-3 w-3" />
              Veto Reasons (HOLD)
            </div>
            <ul className="text-xs text-muted-foreground space-y-0.5">
              {shotPlan.vetoReasons.map((reason, i) => (
                <li key={i} className="flex items-start gap-1.5" data-testid={`text-veto-${i}`}>
                  <XCircle className="h-3 w-3 text-amber-400 mt-0.5 flex-shrink-0" />
                  {reason}
                </li>
              ))}
            </ul>
          </div>
        )}

        {(shotPlan.patternMatchCount !== undefined || shotPlan.modelConsensus !== undefined) && (
          <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
            {shotPlan.patternMatchCount !== undefined && (
              <span data-testid="text-pattern-count">
                {shotPlan.patternMatchCount} similar patterns
              </span>
            )}
            {shotPlan.modelConsensus !== undefined && (
              <span data-testid="text-consensus">
                Model consensus: {(shotPlan.modelConsensus * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
