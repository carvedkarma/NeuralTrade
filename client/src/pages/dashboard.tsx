import { useQuery } from "@tanstack/react-query";
import { SignalCard } from "@/components/signal-card";
import { RegimeCard } from "@/components/regime-card";
import { FeaturesCard } from "@/components/features-card";
import { FuturesMetricsCard } from "@/components/futures-metrics-card";
import { RiskModeCard } from "@/components/risk-mode-card";
import { StatsCard } from "@/components/stats-card";
import { TradeHistory } from "@/components/trade-history";
import { PriceChart } from "@/components/price-chart";
import { ThemeToggle } from "@/components/theme-toggle";
import type { DashboardData } from "@shared/schema";
import { Loader2, RefreshCw, Bitcoin, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";

export default function Dashboard() {
  const { data, isLoading, error, refetch, isFetching } = useQuery<DashboardData>({
    queryKey: ["/api/dashboard"],
    refetchInterval: 15000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background" data-testid="loading-container">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" data-testid="loading-spinner" />
          <p className="text-muted-foreground" data-testid="text-loading">Loading trading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background" data-testid="error-container">
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="p-4 rounded-full bg-destructive/10">
            <Bitcoin className="h-8 w-8 text-destructive" />
          </div>
          <p className="text-destructive" data-testid="text-error">Failed to load dashboard data</p>
          <Button onClick={() => refetch()} data-testid="button-retry">
            Try Again
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background" data-testid="dashboard-container">
      <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60" data-testid="header">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="p-2 rounded-md bg-primary/10" data-testid="logo-container">
                  <Bitcoin className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h1 className="text-lg font-semibold" data-testid="text-app-title">BTC Futures Signal</h1>
                  <p className="text-xs text-muted-foreground" data-testid="text-app-subtitle">BTCUSDT Perpetual</p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-xs text-muted-foreground" data-testid="last-update-container">
                <Clock className="h-3.5 w-3.5" />
                <span data-testid="text-last-update">Updated {format(new Date(data.currentSignal.timestamp), "HH:mm:ss")}</span>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => refetch()}
                disabled={isFetching}
                data-testid="button-refresh"
              >
                <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
              </Button>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6" data-testid="main-content">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          <div className="lg:col-span-8 space-y-4">
            <PriceChart candles={data.candles} signal={data.currentSignal} />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <FuturesMetricsCard data={data.futuresData} />
              <FeaturesCard features={data.currentSignal.topFeatures} />
            </div>
          </div>

          <div className="lg:col-span-4 space-y-4">
            <SignalCard signal={data.currentSignal} />
            <RegimeCard signal={data.currentSignal} />
            <StatsCard
              equity={data.equity}
              dailyPnl={data.dailyPnl}
              winRate={data.winRate}
              profitFactor={data.profitFactor}
              totalTrades={data.totalTrades}
            />
            <RiskModeCard
              riskMode={data.currentSignal.riskMode}
              drawdown={data.drawdown}
              maxDrawdown={data.maxDrawdown}
              exposure={data.exposure}
            />
          </div>

          <div className="lg:col-span-12">
            <TradeHistory trades={data.recentTrades} />
          </div>
        </div>
      </main>

      <footer className="border-t border-border py-4 mt-8" data-testid="footer">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span data-testid="text-footer-title">BTC Futures Trading Signal Dashboard</span>
            <div className="flex items-center gap-4">
              <Badge variant="secondary" className="text-xs" data-testid="badge-paper-trading">
                Paper Trading Mode
              </Badge>
              <span data-testid="text-footer-info">H=8 (2hr horizon) | 15m timeframe</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
