import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { RiskMode } from "@shared/schema";
import { Shield, AlertTriangle, XCircle } from "lucide-react";
import { motion } from "framer-motion";

interface RiskModeCardProps {
  riskMode: RiskMode;
  drawdown: number;
  maxDrawdown: number;
  exposure: number;
}

const modeConfig: Record<RiskMode, { icon: typeof Shield; color: string; bg: string; border: string; label: string; description: string }> = {
  normal: { 
    icon: Shield, 
    color: "text-emerald-400", 
    bg: "bg-emerald-500/10", 
    border: "border-emerald-500/30",
    label: "Normal",
    description: "Trading conditions are favorable"
  },
  high_vol: { 
    icon: AlertTriangle, 
    color: "text-amber-400", 
    bg: "bg-amber-500/10", 
    border: "border-amber-500/30",
    label: "High Volatility",
    description: "Reduced position sizes recommended"
  },
  no_trade: { 
    icon: XCircle, 
    color: "text-red-400", 
    bg: "bg-red-500/10", 
    border: "border-red-500/30",
    label: "No Trade",
    description: "Market conditions unfavorable"
  },
};

export function RiskModeCard({ riskMode, drawdown, maxDrawdown, exposure }: RiskModeCardProps) {
  const config = modeConfig[riskMode];
  const Icon = config.icon;
  const drawdownPercent = Math.abs(drawdown * 100);
  const maxDrawdownPercent = Math.abs(maxDrawdown * 100);

  return (
    <Card className="overflow-visible" data-testid="card-risk-mode">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground" data-testid="text-risk-title">Risk Status</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <motion.div
          key={riskMode}
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className={`flex items-center gap-3 p-3 rounded-md border ${config.bg} ${config.border}`}
          data-testid="risk-mode-display"
        >
          <Icon className={`h-5 w-5 ${config.color}`} />
          <div className="flex-1 min-w-0">
            <p className={`text-sm font-medium ${config.color}`} data-testid="text-risk-mode">{config.label}</p>
            <p className="text-xs text-muted-foreground truncate" data-testid="text-risk-description">{config.description}</p>
          </div>
        </motion.div>

        <div className="space-y-3">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Current Drawdown</span>
              <span className={`text-sm font-mono ${drawdownPercent > 5 ? "text-red-400" : drawdownPercent > 2 ? "text-amber-400" : "text-emerald-400"}`} data-testid="text-drawdown">
                -{drawdownPercent.toFixed(2)}%
              </span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(drawdownPercent * 5, 100)}%` }}
                className={`h-full ${drawdownPercent > 5 ? "bg-red-500" : drawdownPercent > 2 ? "bg-amber-500" : "bg-emerald-500"}`}
                data-testid="progress-drawdown"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Max Drawdown</span>
              <span className="text-sm font-mono text-red-400" data-testid="text-max-drawdown">
                -{maxDrawdownPercent.toFixed(2)}%
              </span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(maxDrawdownPercent * 5, 100)}%` }}
                className="h-full bg-red-500/50"
                data-testid="progress-max-drawdown"
              />
            </div>
          </div>
        </div>

        <div className="pt-3 border-t border-border">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Market Exposure</span>
            <Badge variant="secondary" className="font-mono" data-testid="badge-exposure">
              {(exposure * 100).toFixed(0)}%
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
