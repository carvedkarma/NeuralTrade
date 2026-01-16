import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Candle, Signal } from "@shared/schema";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { format } from "date-fns";
import { useMemo } from "react";

interface PriceChartProps {
  candles: Candle[];
  signal: Signal;
}

export function PriceChart({ candles, signal }: PriceChartProps) {
  const chartData = useMemo(() => {
    return candles.map((candle) => {
      return {
        time: candle.timestamp,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
        priceRange: [candle.low, candle.high],
      };
    });
  }, [candles]);

  const currentPrice = candles.length > 0 ? candles[candles.length - 1].close : 0;
  const firstPrice = candles.length > 0 ? candles[0].open : currentPrice;
  const priceChange = firstPrice !== 0 ? ((currentPrice - firstPrice) / firstPrice) * 100 : 0;
  const isPositive = priceChange >= 0;

  const priceRange = useMemo(() => {
    if (candles.length === 0) return { min: 0, max: 0 };
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);
    const min = Math.min(...lows);
    const max = Math.max(...highs);
    const padding = (max - min) * 0.05;
    return { min: min - padding, max: max + padding };
  }, [candles]);

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
      </CardHeader>
      <CardContent className="p-0 pr-2">
        <div className="h-[320px] w-full" data-testid="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, bottom: 10, left: 60 }}>
              <defs>
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={isPositive ? "#10b981" : "#ef4444"} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={isPositive ? "#10b981" : "#ef4444"} stopOpacity={0.05} />
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
              
              <ReferenceLine
                y={currentPrice}
                stroke="hsl(var(--primary))"
                strokeDasharray="3 3"
                strokeOpacity={0.7}
              />

              <Area
                type="monotone"
                dataKey="close"
                stroke={isPositive ? "#10b981" : "#ef4444"}
                strokeWidth={2}
                fill="url(#priceGradient)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
