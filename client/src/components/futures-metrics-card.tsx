import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { FuturesData } from "@shared/schema";
import { Activity, TrendingUp, TrendingDown, Zap, DollarSign } from "lucide-react";

interface FuturesMetricsCardProps {
  data: FuturesData;
}

export function FuturesMetricsCard({ data }: FuturesMetricsCardProps) {
  const formatNumber = (n: number, decimals = 2) => {
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + "B";
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return n.toFixed(decimals);
  };

  const fundingAnnualized = data.fundingRate * 3 * 365 * 100;
  const isFundingPositive = data.fundingRate > 0;
  const isOiIncreasing = data.oiChange15m > 0;

  return (
    <Card className="overflow-visible" data-testid="card-futures-metrics">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground" data-testid="text-futures-title">Futures Metrics</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-1.5">
              <DollarSign className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Funding Rate</span>
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className={`text-lg font-mono font-semibold ${isFundingPositive ? "text-emerald-400" : "text-red-400"}`} data-testid="text-funding-rate">
                {isFundingPositive ? "+" : ""}{(data.fundingRate * 100).toFixed(4)}%
              </span>
              <span className="text-xs text-muted-foreground" data-testid="text-funding-apr">
                ({fundingAnnualized.toFixed(1)}% APR)
              </span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1.5">
              <Activity className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Open Interest</span>
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-lg font-mono font-semibold" data-testid="text-open-interest">
                ${formatNumber(data.openInterest)}
              </span>
              <Badge variant="secondary" className={`text-xs ${isOiIncreasing ? "text-emerald-400" : "text-red-400"}`} data-testid="badge-oi-change">
                {isOiIncreasing ? "+" : ""}{(data.oiChange15m * 100).toFixed(2)}%
              </Badge>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-3 border-t border-border">
          <div className="space-y-1">
            <div className="flex items-center gap-1.5">
              <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
              <TrendingDown className="h-3.5 w-3.5 text-red-400" />
              <span className="text-xs text-muted-foreground">L/S Ratio</span>
            </div>
            <span className={`text-sm font-mono ${data.longShortRatio > 1 ? "text-emerald-400" : data.longShortRatio < 1 ? "text-red-400" : ""}`} data-testid="text-ls-ratio">
              {data.longShortRatio.toFixed(2)}
            </span>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1.5">
              <Zap className="h-3.5 w-3.5 text-amber-400" />
              <span className="text-xs text-muted-foreground">Liquidations 1h</span>
            </div>
            <span className="text-sm font-mono" data-testid="text-liquidations">${formatNumber(data.liquidations1h)}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-3 border-t border-border">
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Mark Price</span>
            <span className="text-sm font-mono" data-testid="text-mark-price">${data.markPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Basis</span>
            <span className={`text-sm font-mono ${data.basis > 0 ? "text-emerald-400" : data.basis < 0 ? "text-red-400" : ""}`} data-testid="text-basis">
              {data.basis > 0 ? "+" : ""}{(data.basis * 100).toFixed(4)}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
