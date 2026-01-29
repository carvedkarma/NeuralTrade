import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ShotPlan, ShotPlanHistoryEntry } from "@shared/schema";
import { useQuery } from "@tanstack/react-query";
import { 
  Target, TrendingUp, TrendingDown, Minus, Shield, Clock, 
  AlertTriangle, CheckCircle2, XCircle, DollarSign, Percent,
  ArrowUpRight, ArrowDownRight, Zap, Activity, History, 
  Trophy, Skull, Timer, BarChart3
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useMemo } from "react";

interface EnhancedShotPlanCardProps {
  shotPlan?: ShotPlan;
}

export function EnhancedShotPlanCard({ shotPlan }: EnhancedShotPlanCardProps) {
  const [activeTab, setActiveTab] = useState<"current" | "history">("current");

  const { data: historyData } = useQuery<{
    history: ShotPlanHistoryEntry[];
    stats: {
      totalTrades: number;
      winRate: number;
      avgPnl: number;
      bestTrade: number;
      worstTrade: number;
      tp1Hits: number;
      tp2Hits: number;
      slHits: number;
      expired: number;
    };
  }>({
    queryKey: ["/api/shot-plan/history"],
    refetchInterval: 60000,
  });

  const stats = historyData?.stats || {
    totalTrades: 0,
    winRate: 0,
    avgPnl: 0,
    bestTrade: 0,
    worstTrade: 0,
    tp1Hits: 0,
    tp2Hits: 0,
    slHits: 0,
    expired: 0
  };

  const recentHistory = useMemo(() => {
    return (historyData?.history || []).slice(0, 10);
  }, [historyData?.history]);

  const signalColor = 
    shotPlan?.signal === "LONG" ? "text-emerald-400" :
    shotPlan?.signal === "SHORT" ? "text-red-400" :
    "text-amber-400";
  
  const signalBg = 
    shotPlan?.signal === "LONG" ? "bg-emerald-500/20" :
    shotPlan?.signal === "SHORT" ? "bg-red-500/20" :
    "bg-amber-500/20";

  const SignalIcon = 
    shotPlan?.signal === "LONG" ? TrendingUp :
    shotPlan?.signal === "SHORT" ? TrendingDown : Minus;

  const formatPrice = (price: number | null | undefined) => 
    price ? `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—";

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

  const getOutcomeColor = (outcome: string | null | undefined) => {
    if (!outcome) return "text-muted-foreground";
    if (outcome === "HIT_TP1" || outcome === "HIT_TP2") return "text-emerald-400";
    if (outcome === "HIT_SL") return "text-red-400";
    if (outcome === "EXPIRED") return "text-amber-400";
    return "text-blue-400";
  };

  const getOutcomeIcon = (outcome: string | null | undefined) => {
    if (!outcome || outcome === "PENDING") return Timer;
    if (outcome === "HIT_TP1" || outcome === "HIT_TP2") return Trophy;
    if (outcome === "HIT_SL") return Skull;
    return Clock;
  };

  return (
    <Card data-testid="card-enhanced-shot-plan">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Shot Plan
          </CardTitle>
          {shotPlan && (
            <Badge 
              className={`${signalBg} ${signalColor} flex items-center gap-1`}
              data-testid="badge-shot-signal"
            >
              <SignalIcon className="h-3 w-3" />
              {shotPlan.signal}
            </Badge>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-3">
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "current" | "history")}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="current" data-testid="tab-current-plan">
              <Target className="h-3 w-3 mr-1" />
              Current
            </TabsTrigger>
            <TabsTrigger value="history" data-testid="tab-history">
              <History className="h-3 w-3 mr-1" />
              History ({stats.totalTrades})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="current" className="mt-3 space-y-3">
            {!shotPlan ? (
              <p className="text-sm text-muted-foreground" data-testid="text-no-shot-plan">
                Generating ML-based shot plan... Need 250+ candles for pattern analysis.
              </p>
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Confidence</span>
                  <div className="flex items-center gap-2">
                    <Progress value={shotPlan.confidence * 100} className="w-24 h-2" />
                    <span className="text-sm font-medium">{(shotPlan.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <span className="text-xs text-muted-foreground">Regime</span>
                    <Badge variant="outline" className="text-xs capitalize">
                      {shotPlan.regime.replace("_", " ")}
                    </Badge>
                  </div>
                  <div className="space-y-1">
                    <span className="text-xs text-muted-foreground">Strategy</span>
                    <Badge variant="secondary" className="text-xs">
                      {shotPlan.strategy}
                    </Badge>
                  </div>
                </div>

                {shotPlan.entryZone && (
                  <div className="bg-muted/50 rounded-lg p-3 space-y-2">
                    <div className="text-xs font-medium text-muted-foreground">Trade Levels</div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="flex items-center gap-1.5">
                        <ArrowUpRight className="h-3 w-3 text-emerald-400" />
                        <span className="text-muted-foreground">Entry:</span>
                        <span className="font-mono">
                          {formatPrice(shotPlan.entryZone.low)} - {formatPrice(shotPlan.entryZone.high)}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <Shield className="h-3 w-3 text-red-400" />
                        <span className="text-muted-foreground">Stop:</span>
                        <span className="font-mono">{formatPrice(shotPlan.stopLoss)}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <Target className="h-3 w-3 text-emerald-400" />
                        <span className="text-muted-foreground">TP1:</span>
                        <span className="font-mono">{formatPrice(shotPlan.takeProfit1)}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <Target className="h-3 w-3 text-emerald-400" />
                        <span className="text-muted-foreground">TP2:</span>
                        <span className="font-mono">{formatPrice(shotPlan.takeProfit2)}</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="bg-emerald-500/10 rounded p-2">
                    <div className="text-xs text-muted-foreground">Up</div>
                    <div className="text-sm font-medium text-emerald-400">
                      {(shotPlan.probUp * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="bg-red-500/10 rounded p-2">
                    <div className="text-xs text-muted-foreground">Down</div>
                    <div className="text-sm font-medium text-red-400">
                      {(shotPlan.probDown * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="bg-amber-500/10 rounded p-2">
                    <div className="text-xs text-muted-foreground">Chop</div>
                    <div className="text-sm font-medium text-amber-400">
                      {(shotPlan.probChop * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex items-center gap-2">
                    <DollarSign className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">R:R</span>
                    <span className="font-medium">1:{shotPlan.riskReward.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Percent className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Edge</span>
                    <span className={`font-medium ${shotPlan.edge > 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {formatPercent(shotPlan.edge)}
                    </span>
                  </div>
                </div>

                {shotPlan.vetoReasons.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-xs font-medium text-amber-400 flex items-center gap-1">
                      <AlertTriangle className="h-3 w-3" />
                      Veto Reasons
                    </div>
                    <ul className="text-xs text-muted-foreground space-y-0.5">
                      {shotPlan.vetoReasons.slice(0, 3).map((reason, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <XCircle className="h-3 w-3 text-amber-400 mt-0.5 flex-shrink-0" />
                          {reason}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </TabsContent>

          <TabsContent value="history" className="mt-3 space-y-3">
            {/* Performance Stats */}
            <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-lg p-3 border border-purple-500/20">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-purple-300 flex items-center gap-1.5">
                  <BarChart3 className="h-3.5 w-3.5" />
                  Performance Stats
                </span>
                <Badge 
                  variant="outline"
                  className={stats.winRate >= 50 ? "border-emerald-500/50 text-emerald-400" : "border-red-500/50 text-red-400"}
                  data-testid="badge-win-rate"
                >
                  {stats.winRate.toFixed(1)}% Win Rate
                </Badge>
              </div>
              
              <div className="grid grid-cols-4 gap-2 text-center text-xs">
                <div className="bg-emerald-500/10 rounded p-1.5">
                  <div className="text-muted-foreground">TP1</div>
                  <div className="font-medium text-emerald-400" data-testid="text-tp1-hits">
                    {stats.tp1Hits}
                  </div>
                </div>
                <div className="bg-emerald-500/10 rounded p-1.5">
                  <div className="text-muted-foreground">TP2</div>
                  <div className="font-medium text-emerald-400" data-testid="text-tp2-hits">
                    {stats.tp2Hits}
                  </div>
                </div>
                <div className="bg-red-500/10 rounded p-1.5">
                  <div className="text-muted-foreground">SL</div>
                  <div className="font-medium text-red-400" data-testid="text-sl-hits">
                    {stats.slHits}
                  </div>
                </div>
                <div className="bg-amber-500/10 rounded p-1.5">
                  <div className="text-muted-foreground">Exp</div>
                  <div className="font-medium text-amber-400" data-testid="text-expired">
                    {stats.expired}
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between mt-2 pt-2 border-t border-purple-500/20 text-xs">
                <div>
                  <span className="text-muted-foreground">Avg PnL: </span>
                  <span className={stats.avgPnl >= 0 ? "text-emerald-400" : "text-red-400"}>
                    {stats.avgPnl >= 0 ? "+" : ""}{stats.avgPnl.toFixed(2)}%
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Best: </span>
                  <span className="text-emerald-400">+{stats.bestTrade.toFixed(2)}%</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Worst: </span>
                  <span className="text-red-400">{stats.worstTrade.toFixed(2)}%</span>
                </div>
              </div>
            </div>

            {/* Recent Trades */}
            <div className="space-y-2">
              <div className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <History className="h-3 w-3" />
                Recent Signals
              </div>
              
              <AnimatePresence>
                {recentHistory.length === 0 ? (
                  <p className="text-xs text-muted-foreground py-2" data-testid="text-no-history">
                    No trade history yet. Signals will be tracked here.
                  </p>
                ) : (
                  <div className="space-y-1.5 max-h-48 overflow-y-auto">
                    {recentHistory.map((entry, idx) => {
                      const OutcomeIcon = getOutcomeIcon(entry.outcome);
                      return (
                        <motion.div
                          key={entry.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.05 }}
                          className="flex items-center justify-between bg-muted/30 rounded p-2"
                          data-testid={`history-entry-${entry.id}`}
                        >
                          <div className="flex items-center gap-2">
                            <Badge 
                              variant="outline"
                              className={`text-[10px] ${
                                entry.signal === "LONG" ? "border-emerald-500/50 text-emerald-400" :
                                entry.signal === "SHORT" ? "border-red-500/50 text-red-400" :
                                "border-amber-500/50 text-amber-400"
                              }`}
                            >
                              {entry.signal}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {new Date(entry.timestamp).toLocaleString(undefined, {
                                month: "short",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit"
                              })}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className={`text-xs font-medium ${
                              (entry.pnlPercent ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                            }`}>
                              {entry.pnlPercent !== null 
                                ? `${entry.pnlPercent >= 0 ? "+" : ""}${entry.pnlPercent.toFixed(2)}%`
                                : "—"
                              }
                            </span>
                            <OutcomeIcon className={`h-3.5 w-3.5 ${getOutcomeColor(entry.outcome)}`} />
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                )}
              </AnimatePresence>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
