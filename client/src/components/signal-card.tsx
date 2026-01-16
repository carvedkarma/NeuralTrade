import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Signal, SignalType } from "@shared/schema";
import { TrendingUp, TrendingDown, Minus, ArrowUp, ArrowDown } from "lucide-react";
import { motion } from "framer-motion";

interface SignalCardProps {
  signal: Signal;
}

const signalConfig: Record<SignalType, { icon: typeof TrendingUp; color: string; bg: string; border: string }> = {
  LONG: { icon: TrendingUp, color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30" },
  SHORT: { icon: TrendingDown, color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/30" },
  HOLD: { icon: Minus, color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/30" },
};

export function SignalCard({ signal }: SignalCardProps) {
  const config = signalConfig[signal.signal];
  const Icon = config.icon;

  return (
    <Card className="overflow-visible" data-testid="card-signal">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground" data-testid="text-signal-title">Current Signal</CardTitle>
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
            <p className="text-xs text-muted-foreground">Edge</p>
            <div className="flex items-center gap-1">
              {signal.edge > 0 ? (
                <ArrowUp className="h-4 w-4 text-emerald-400" />
              ) : (
                <ArrowDown className="h-4 w-4 text-red-400" />
              )}
              <span className={`text-sm font-mono font-medium ${signal.edge > 0 ? "text-emerald-400" : "text-red-400"}`} data-testid="text-edge">
                {signal.edge > 0 ? "+" : ""}{(signal.edge * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>

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
      </CardContent>
    </Card>
  );
}
