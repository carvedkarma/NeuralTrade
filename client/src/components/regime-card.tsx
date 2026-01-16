import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Signal } from "@shared/schema";
import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, Activity } from "lucide-react";

interface RegimeCardProps {
  signal: Signal;
}

export function RegimeCard({ signal }: RegimeCardProps) {
  const regimes = [
    { 
      key: "trend_up", 
      label: "Trend Up", 
      prob: signal.probUp, 
      icon: TrendingUp, 
      color: "text-emerald-400",
      bg: "bg-emerald-500",
      bgLight: "bg-emerald-500/20"
    },
    { 
      key: "trend_down", 
      label: "Trend Down", 
      prob: signal.probDown, 
      icon: TrendingDown, 
      color: "text-red-400",
      bg: "bg-red-500",
      bgLight: "bg-red-500/20"
    },
    { 
      key: "chop", 
      label: "Chop", 
      prob: signal.probChop, 
      icon: Activity, 
      color: "text-amber-400",
      bg: "bg-amber-500",
      bgLight: "bg-amber-500/20"
    },
  ];

  const maxProb = Math.max(signal.probUp, signal.probDown, signal.probChop);

  return (
    <Card className="overflow-visible" data-testid="card-regime">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground" data-testid="text-regime-title">Regime Probabilities</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {regimes.map((regime) => {
          const Icon = regime.icon;
          const isActive = regime.prob === maxProb;
          
          return (
            <div key={regime.key} className="space-y-1.5" data-testid={`regime-${regime.key}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`p-1.5 rounded-md ${regime.bgLight}`}>
                    <Icon className={`h-3.5 w-3.5 ${regime.color}`} />
                  </div>
                  <span className={`text-sm ${isActive ? "font-medium" : "text-muted-foreground"}`} data-testid={`text-regime-label-${regime.key}`}>
                    {regime.label}
                  </span>
                </div>
                <span className={`text-sm font-mono ${isActive ? "font-semibold" : "text-muted-foreground"}`} data-testid={`text-regime-prob-${regime.key}`}>
                  {(regime.prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${regime.prob * 100}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className={`h-full ${regime.bg} ${isActive ? "opacity-100" : "opacity-50"}`}
                  data-testid={`progress-regime-${regime.key}`}
                />
              </div>
            </div>
          );
        })}

        <div className="pt-3 mt-3 border-t border-border">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Current Regime</span>
            <span className={`text-sm font-medium ${
              signal.regime === "trend_up" ? "text-emerald-400" :
              signal.regime === "trend_down" ? "text-red-400" : "text-amber-400"
            }`} data-testid="text-current-regime">
              {signal.regime === "trend_up" ? "Trending Up" :
               signal.regime === "trend_down" ? "Trending Down" : "Choppy"}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
