import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Candle, StrategySignal, Trade } from "@shared/schema";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Scatter,
  Cell,
} from "recharts";
import { format } from "date-fns";
import { useMemo } from "react";

interface PriceChartProps {
  candles: Candle[];
  kalmanFast: number[];
  kalmanSlow: number[];
  strategySignal: StrategySignal;
  activeTrade: Trade | null;
  recentTrades: Trade[];
}

export function PriceChart({ candles, kalmanFast, kalmanSlow, strategySignal, activeTrade, recentTrades }: PriceChartProps) {
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

  const tradeMarkers = useMemo(() => {
    if (!candles || candles.length === 0) return [];
    const markers: Array<{ time: number; price: number; type: 'entry' | 'exit'; side: 'LONG' | 'SHORT'; pnl?: number }> = [];
    
    const minTime = candles[0].timestamp;
    const maxTime = candles[candles.length - 1].timestamp;
    
    (recentTrades ?? []).forEach(trade => {
      if (trade.timestamp >= minTime && trade.timestamp <= maxTime) {
        markers.push({
          time: trade.timestamp,
          price: trade.entryPrice,
          type: 'entry',
          side: trade.side,
        });
      }
      if (trade.status === 'closed' && trade.exitPrice) {
        const exitTime = trade.timestamp + 15 * 60 * 1000 * 2;
        if (exitTime >= minTime && exitTime <= maxTime) {
          markers.push({
            time: exitTime,
            price: trade.exitPrice,
            type: 'exit',
            side: trade.side,
            pnl: trade.pnlPercent ?? 0,
          });
        }
      }
    });
    
    if (activeTrade) {
      markers.push({
        time: activeTrade.timestamp,
        price: activeTrade.entryPrice,
        type: 'entry',
        side: activeTrade.side,
      });
    }
    
    return markers;
  }, [candles, recentTrades, activeTrade]);

  const chartDataWithMarkers = useMemo(() => {
    return chartData.map(point => {
      const entryMarker = tradeMarkers.find(m => Math.abs(m.time - point.time) < 15 * 60 * 1000 && m.type === 'entry');
      const exitMarker = tradeMarkers.find(m => Math.abs(m.time - point.time) < 15 * 60 * 1000 && m.type === 'exit');
      return {
        ...point,
        entryPrice: entryMarker ? entryMarker.price : null,
        entrySide: entryMarker ? entryMarker.side : null,
        exitPrice: exitMarker ? exitMarker.price : null,
        exitPnl: exitMarker ? exitMarker.pnl : null,
      };
    });
  }, [chartData, tradeMarkers]);

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
    if (activeTrade) {
      allPrices.push(activeTrade.stopLoss, activeTrade.takeProfit, activeTrade.entryPrice);
    }
    if (allPrices.length === 0) return { min: 0, max: 0 };
    const min = Math.min(...allPrices);
    const max = Math.max(...allPrices);
    const padding = (max - min) * 0.08;
    return { min: min - padding, max: max + padding };
  }, [candles, kalmanFast, kalmanSlow, activeTrade]);

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
            {data.entryPrice && (
              <>
                <span className="text-yellow-400">Entry:</span>
                <span className="text-yellow-400">${data.entryPrice.toLocaleString()} ({data.entrySide})</span>
              </>
            )}
            {data.exitPrice && (
              <>
                <span className={data.exitPnl >= 0 ? "text-emerald-400" : "text-red-400"}>Exit:</span>
                <span className={data.exitPnl >= 0 ? "text-emerald-400" : "text-red-400"}>
                  ${data.exitPrice.toLocaleString()} ({data.exitPnl >= 0 ? '+' : ''}{data.exitPnl?.toFixed(2)}%)
                </span>
              </>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  const EntryMarkerShape = (props: any) => {
    const { cx, cy, payload } = props;
    if (!payload.entryPrice) return null;
    const isLong = payload.entrySide === 'LONG';
    return (
      <g>
        <polygon
          points={isLong 
            ? `${cx},${cy - 8} ${cx - 6},${cy + 4} ${cx + 6},${cy + 4}` 
            : `${cx},${cy + 8} ${cx - 6},${cy - 4} ${cx + 6},${cy - 4}`
          }
          fill={isLong ? "#10b981" : "#ef4444"}
          stroke="#fff"
          strokeWidth={1}
        />
        <text x={cx} y={cy - 14} textAnchor="middle" fill="#fff" fontSize={9} fontWeight="bold">
          {isLong ? 'BUY' : 'SELL'}
        </text>
      </g>
    );
  };

  const ExitMarkerShape = (props: any) => {
    const { cx, cy, payload } = props;
    if (!payload.exitPrice) return null;
    const isProfit = (payload.exitPnl ?? 0) >= 0;
    return (
      <g>
        <circle cx={cx} cy={cy} r={6} fill={isProfit ? "#10b981" : "#ef4444"} stroke="#fff" strokeWidth={1} />
        <text x={cx} y={cy + 3} textAnchor="middle" fill="#fff" fontSize={8} fontWeight="bold">
          X
        </text>
        <text x={cx} y={cy - 12} textAnchor="middle" fill={isProfit ? "#10b981" : "#ef4444"} fontSize={9} fontWeight="bold">
          {isProfit ? '+' : ''}{payload.exitPnl?.toFixed(1)}%
        </text>
      </g>
    );
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
            {activeTrade && (
              <Badge 
                className={`animate-pulse ${activeTrade.side === "LONG" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "bg-red-500/20 text-red-400 border-red-500/30"}`}
                data-testid="badge-active-trade"
              >
                {activeTrade.side} OPEN @ ${activeTrade.entryPrice.toLocaleString()}
              </Badge>
            )}
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
        <div className="flex items-center gap-4 mt-2 text-xs flex-wrap">
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-cyan-400 rounded" />
            <span className="text-muted-foreground">Fast (70)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-orange-400 rounded" />
            <span className="text-muted-foreground">Slow (250)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-0 h-0 border-l-4 border-r-4 border-b-8 border-l-transparent border-r-transparent border-b-emerald-400" />
            <span className="text-muted-foreground">Buy Entry</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-0 h-0 border-l-4 border-r-4 border-t-8 border-l-transparent border-r-transparent border-t-red-400" />
            <span className="text-muted-foreground">Sell Entry</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-emerald-400 flex items-center justify-center text-[6px] text-white font-bold">X</div>
            <span className="text-muted-foreground">Exit</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0 pr-2">
        <div className="h-[360px] w-full" data-testid="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartDataWithMarkers} margin={{ top: 20, right: 10, bottom: 10, left: 60 }}>
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
              
              {activeTrade && (
                <>
                  <ReferenceLine
                    y={activeTrade.entryPrice}
                    stroke="#fbbf24"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    label={{ value: `Entry $${activeTrade.entryPrice.toLocaleString()}`, position: 'right', fill: '#fbbf24', fontSize: 10 }}
                  />
                  <ReferenceLine
                    y={activeTrade.stopLoss}
                    stroke="#ef4444"
                    strokeWidth={2}
                    strokeDasharray="3 3"
                    label={{ value: `SL $${activeTrade.stopLoss.toLocaleString()}`, position: 'right', fill: '#ef4444', fontSize: 10 }}
                  />
                  <ReferenceLine
                    y={activeTrade.takeProfit}
                    stroke="#10b981"
                    strokeWidth={2}
                    strokeDasharray="3 3"
                    label={{ value: `TP $${activeTrade.takeProfit.toLocaleString()}`, position: 'right', fill: '#10b981', fontSize: 10 }}
                  />
                </>
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

              <Scatter
                dataKey="entryPrice"
                shape={<EntryMarkerShape />}
                isAnimationActive={false}
              />

              <Scatter
                dataKey="exitPrice"
                shape={<ExitMarkerShape />}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
