import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import type { StrategyState, StrategySignal, Trade } from "@shared/schema";
import { Play, Square, TrendingUp, TrendingDown, Target, AlertTriangle } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface StrategyControlProps {
  strategyState: StrategyState;
  strategySignal: StrategySignal;
  activeTrade: Trade | null;
}

export function StrategyControl({ strategyState, strategySignal, activeTrade }: StrategyControlProps) {
  const startMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/strategy/start"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] }),
  });

  const stopMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/strategy/stop"),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] }),
  });

  const updateSettingsMutation = useMutation({
    mutationFn: (settings: Partial<StrategyState>) => 
      apiRequest("PATCH", "/api/strategy/settings", settings),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] }),
  });

  const handleToggleRetest = () => {
    updateSettingsMutation.mutate({ useRetestSignals: !strategyState.useRetestSignals });
  };

  return (
    <Card data-testid="card-strategy-control">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-4">
          <CardTitle className="text-base font-semibold">Kalman Strategy</CardTitle>
          {strategyState.isRunning ? (
            <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30" data-testid="badge-strategy-running">
              RUNNING
            </Badge>
          ) : (
            <Badge variant="secondary" data-testid="badge-strategy-stopped">
              STOPPED
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          {strategyState.isRunning ? (
            <Button
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
              variant="destructive"
              className="flex-1"
              data-testid="button-stop-strategy"
            >
              <Square className="w-4 h-4 mr-2" />
              Stop Trading
            </Button>
          ) : (
            <Button
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
              className="flex-1 bg-emerald-600 hover:bg-emerald-700"
              data-testid="button-start-strategy"
            >
              <Play className="w-4 h-4 mr-2" />
              Start Trading
            </Button>
          )}
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="retest-toggle" className="text-sm text-muted-foreground">Retest Signals Only</Label>
          <Switch
            id="retest-toggle"
            checked={strategyState.useRetestSignals}
            onCheckedChange={handleToggleRetest}
            data-testid="switch-retest-signals"
          />
        </div>

        <div className="border-t border-border pt-3 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Signal Type</span>
            <Badge variant="secondary" data-testid="text-signal-type">
              {strategySignal.type === "none" ? "WAITING" : strategySignal.type.toUpperCase()}
            </Badge>
          </div>
          
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Direction</span>
            <div className="flex items-center gap-1" data-testid="text-signal-direction">
              {strategySignal.direction === "LONG" ? (
                <>
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400 font-medium">LONG</span>
                </>
              ) : strategySignal.direction === "SHORT" ? (
                <>
                  <TrendingDown className="w-4 h-4 text-red-400" />
                  <span className="text-red-400 font-medium">SHORT</span>
                </>
              ) : (
                <span className="text-muted-foreground">HOLD</span>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">ATR(14)</span>
            <span className="font-mono" data-testid="text-atr">
              ${strategySignal.atr.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
          </div>
        </div>

        {activeTrade && (
          <div className="border-t border-border pt-3 space-y-2">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-primary" />
              <span className="font-medium text-sm">Active Trade</span>
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex flex-col">
                <span className="text-muted-foreground">Entry</span>
                <span className="font-mono" data-testid="text-active-entry">
                  ${activeTrade.entryPrice.toLocaleString()}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-muted-foreground">Side</span>
                <span className={`font-medium ${activeTrade.side === "LONG" ? "text-emerald-400" : "text-red-400"}`} data-testid="text-active-side">
                  {activeTrade.side}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-muted-foreground flex items-center gap-1">
                  <AlertTriangle className="w-3 h-3 text-red-400" /> Stop
                </span>
                <span className="font-mono text-red-400" data-testid="text-active-stop">
                  ${activeTrade.stopLoss.toLocaleString()}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-muted-foreground flex items-center gap-1">
                  <Target className="w-3 h-3 text-emerald-400" /> TP
                </span>
                <span className="font-mono text-emerald-400" data-testid="text-active-tp">
                  ${activeTrade.takeProfit.toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        )}

        {!activeTrade && strategySignal.entryZone && (
          <div className="border-t border-border pt-3 space-y-2">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-primary" />
              <span className="font-medium text-sm">Shot Plan</span>
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex flex-col">
                <span className="text-muted-foreground">Entry Zone</span>
                <span className="font-mono" data-testid="text-plan-entry">
                  ${strategySignal.entryZone.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-muted-foreground">Stop Loss</span>
                <span className="font-mono text-red-400" data-testid="text-plan-stop">
                  ${strategySignal.stopLoss?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-muted-foreground">TP1 (1R)</span>
                <span className="font-mono text-emerald-400" data-testid="text-plan-tp1">
                  ${strategySignal.takeProfit1?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-muted-foreground">TP2 (2R)</span>
                <span className="font-mono text-emerald-400" data-testid="text-plan-tp2">
                  ${strategySignal.takeProfit2?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
