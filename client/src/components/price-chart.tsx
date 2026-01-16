import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Candle, StrategySignal } from "@shared/schema";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { format } from "date-fns";
import { useMemo } from "react";

interface PriceChartProps {
  candles: Candle[];
  kalmanFast: number[];
  kalmanSlow: number[];
  strategySignal: StrategySignal;
}

export function PriceChart({ candles, kalmanFast, kalmanSlow, strategySignal }: PriceChartProps) {
  const chartData = useMemo(() => {
    if (!candles || candles.length === 0) return [];
    return candles.map((candle, i) => {
      return {
        time: candle.timestamp,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
        kalmanFast: kalmanFast?.[i] ?? null,
        kalmanSlow: kalmanSlow?.[i] ?? null,
      };
    });
  }, [candles, kalmanFast, kalmanSlow]);

  const currentPrice = candles && candles.length > 0 ? candles[candles.length - 1].close : 0;
  const firstPrice = candles && candles.length > 0 ? candles[0].open : currentPrice;
  const priceChange = firstPrice !== 0 ? ((currentPrice - firstPrice) / firstPrice) * 100 : 0;
  const isPositive = priceChange >= 0;

  const priceRange = useMemo(() => {
    if (!candles || candles.length === 0) return { min: 0, max: 0 };
    const allPrices = [
      ...candles.map(c => c.high),
      ...candles.map(c => c.low),
      ...(kalmanFast?.filter(v => v !== null && v !== undefined) ?? []),
      ...(kalmanSlow?.filter(v => v !== null && v !== undefined) ?? []),
    ];
    if (allPrices.length === 0) return { min: 0, max: 0 };
    const min = Math.min(...allPrices);
    const max = Math.max(...allPrices);
    const padding = (max - min) * 0.05;
    return { min: min - padding, max: max + padding };
  }, [candles, kalmanFast, kalmanSlow]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const candleIsUp = data.close >= data.open;
      return (
        <div className="bg-popover border border-popover-border rounded-md p-2 shadow-lg" data-testid="chart-tooltip">
          <p className="text-xs text-muted-foreground mb-1">
            {format(new Date(data.time), "MMM d, HH:mm")}
          </p>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs font-mono">
            <span className="text-muted-foreground">O:</span>
            <span>${data.open.toLocaleString()}</span>
            <span className="text-muted-foreground">H:</span>
            <span>${data.high.toLocaleString()}</span>
            <span className="text-muted-foreground">L:</span>
            <span>${data.low.toLocaleString()}</span>
            <span className="text-muted-foreground">C:</span>
            <span className={candleIsUp ? "text-emerald-400" : "text-red-400"}>
              ${data.close.toLocaleString()}
            </span>
            {data.kalmanFast && (
              <>
                <span className="text-cyan-400">Fast:</span>
                <span className="text-cyan-400">${data.kalmanFast.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
              </>
            )}
            {data.kalmanSlow && (
              <>
                <span className="text-orange-400">Slow:</span>
                <span className="text-orange-400">${data.kalmanSlow.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
              </>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="overflow-visible h-full" data-testid="card-price-chart">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <CardTitle className="text-lg font-semibold" data-testid="text-chart-title">BTCUSDT</CardTitle>
            <Badge variant="secondary" className="font-mono" data-testid="badge-timeframe">15m</Badge>
            <Badge 
              variant="secondary"
              className={strategySignal.regime === "bull" ? "text-emerald-400" : "text-red-400"}
              data-testid="badge-regime"
            >
              {strategySignal.regime === "bull" ? "BULL" : "BEAR"}
            </Badge>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-mono font-bold" data-testid="text-current-price">
              ${currentPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </span>
            <Badge 
              variant="secondary"
              className={`font-mono ${isPositive ? "text-emerald-400" : "text-red-400"}`}
              data-testid="badge-price-change"
            >
              {isPositive ? "+" : ""}{priceChange.toFixed(2)}%
            </Badge>
          </div>
        </div>
        <div className="flex items-center gap-4 mt-2 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-cyan-400 rounded" />
            <span className="text-muted-foreground">Fast (70)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-orange-400 rounded" />
            <span className="text-muted-foreground">Slow (250)</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0 pr-2">
        <div className="h-[320px] w-full" data-testid="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, bottom: 10, left: 60 }}>
              <defs>
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={isPositive ? "#10b981" : "#ef4444"} stopOpacity={0.2} />
                  <stop offset="100%" stopColor={isPositive ? "#10b981" : "#ef4444"} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="time"
                tickFormatter={(time) => format(new Date(time), "HH:mm")}
                stroke="hsl(var(--muted-foreground))"
                fontSize={10}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                domain={[priceRange.min, priceRange.max]}
                tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
                stroke="hsl(var(--muted-foreground))"
                fontSize={10}
                tickLine={false}
                axisLine={false}
                width={50}
              />
              <Tooltip content={<CustomTooltip />} />
              
              {strategySignal.stopLoss && (
                <ReferenceLine
                  y={strategySignal.stopLoss}
                  stroke="#ef4444"
                  strokeDasharray="3 3"
                  strokeOpacity={0.7}
                />
              )}
              {strategySignal.takeProfit1 && (
                <ReferenceLine
                  y={strategySignal.takeProfit1}
                  stroke="#10b981"
                  strokeDasharray="3 3"
                  strokeOpacity={0.7}
                />
              )}

              <Area
                type="monotone"
                dataKey="close"
                stroke={isPositive ? "#10b981" : "#ef4444"}
                strokeWidth={1.5}
                fill="url(#priceGradient)"
                isAnimationActive={false}
              />

              <Line
                type="monotone"
                dataKey="kalmanFast"
                stroke="#22d3ee"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />

              <Line
                type="monotone"
                dataKey="kalmanSlow"
                stroke="#fb923c"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
