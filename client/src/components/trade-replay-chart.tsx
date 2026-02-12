import { useRef, useEffect, useState, useCallback } from "react";
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  HistogramSeries,
  ColorType,
  CrosshairMode,
  LineStyle,
} from "lightweight-charts";
import type { IChartApi, ISeriesApi, CandlestickData, HistogramData, Time, SeriesMarker } from "lightweight-charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BarChart3,
  Crosshair,
  Eye,
  EyeOff,
  BarChart,
} from "lucide-react";

interface ReplayCandle {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

interface ReplayMarker {
  t: number;
  price: number;
  type: "ENTRY" | "EXIT";
  side?: string;
  outcome?: string;
}

interface ReplayLine {
  name: string;
  price: number;
}

interface ReplayBand {
  from: number;
  to: number;
  label: string;
}

interface ReplayTrade {
  id: number;
  symbol: string;
  side: string;
  entry_at: number;
  exit_at: number | null;
  entry_price: number;
  exit_price: number | null;
  sl_price: number | null;
  tp_price: number | null;
  outcome: string | null;
  net_r: number | null;
  pnl_usd: number | null;
  bars_held: number | null;
  status: string;
}

export interface ReplayData {
  trade: ReplayTrade;
  timeframe: string;
  candles: ReplayCandle[];
  markers: ReplayMarker[];
  lines: ReplayLine[];
  bands: ReplayBand[];
}

interface TmAction {
  ts: number;
  action: string;
  reason: string;
  price?: number;
  sl?: number;
  ur?: number;
}

interface TradeReplayChartProps {
  data: ReplayData | null | undefined;
  isLoading: boolean;
  tmActions?: TmAction[];
}

function msToChartTime(ms: number): Time {
  return Math.floor(ms / 1000) as Time;
}

function fmtR(val: number | null | undefined): string {
  if (val === null || val === undefined) return "N/A";
  return `${val >= 0 ? "+" : ""}${val.toFixed(2)}R`;
}

function fmtUsd(val: number | null | undefined): string {
  if (val === null || val === undefined) return "N/A";
  return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export default function TradeReplayChart({ data, isLoading, tmActions }: TradeReplayChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  const [showLevels, setShowLevels] = useState(true);
  const [showVolume, setShowVolume] = useState(false);

  const jumpToEntry = useCallback(() => {
    if (!chartRef.current || !data?.trade?.entry_at) return;
    const entryTime = msToChartTime(data.trade.entry_at);
    const range = chartRef.current.timeScale().getVisibleRange();
    if (range) {
      const halfRange = ((range.to as number) - (range.from as number)) / 2;
      chartRef.current.timeScale().setVisibleRange({
        from: ((entryTime as number) - halfRange) as Time,
        to: ((entryTime as number) + halfRange) as Time,
      });
    }
  }, [data]);

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.candles.length === 0) return;

    const container = chartContainerRef.current;
    const isDark = document.documentElement.classList.contains("dark");

    const bgColor = isDark ? "#0a0a0b" : "#ffffff";
    const textColor = isDark ? "#9ca3af" : "#6b7280";
    const gridColor = isDark ? "#1f2937" : "#f3f4f6";
    const crosshairColor = isDark ? "#4b5563" : "#d1d5db";

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: bgColor },
        textColor: textColor,
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: crosshairColor, width: 1, style: LineStyle.Dashed },
        horzLine: { color: crosshairColor, width: 1, style: LineStyle.Dashed },
      },
      rightPriceScale: {
        borderColor: gridColor,
        scaleMargins: { top: 0.1, bottom: showVolume ? 0.25 : 0.05 },
      },
      timeScale: {
        borderColor: gridColor,
        timeVisible: true,
        secondsVisible: false,
      },
      width: container.clientWidth,
      height: 360,
    });

    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e80",
      wickDownColor: "#ef444480",
    });

    const candleData: CandlestickData[] = data.candles.map((c) => ({
      time: msToChartTime(c.t),
      open: c.o,
      high: c.h,
      low: c.l,
      close: c.c,
    }));
    candleSeries.setData(candleData);

    if (showVolume) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });
      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
      const volumeData: HistogramData[] = data.candles.map((c) => ({
        time: msToChartTime(c.t),
        value: c.v,
        color: c.c >= c.o ? "#22c55e30" : "#ef444430",
      }));
      volumeSeries.setData(volumeData);
    }

    if (showLevels) {
      for (const line of data.lines) {
        const color =
          line.name === "ENTRY" ? "#3b82f6" :
          line.name === "SL" ? "#ef4444" :
          line.name === "TP" ? "#22c55e" : "#9ca3af";

        candleSeries.createPriceLine({
          price: line.price,
          color: color,
          lineWidth: 1,
          lineStyle: line.name === "ENTRY" ? LineStyle.Dashed : LineStyle.Dotted,
          axisLabelVisible: true,
          title: line.name,
        });
      }
    }

    const isLong = data.trade.side?.toUpperCase() === "LONG";
    const chartMarkers: SeriesMarker<Time>[] = data.markers.map((m) => {
      if (m.type === "ENTRY") {
        return {
          time: msToChartTime(m.t),
          position: (isLong ? "belowBar" : "aboveBar") as "belowBar" | "aboveBar",
          color: isLong ? "#22c55e" : "#ef4444",
          shape: (isLong ? "arrowUp" : "arrowDown") as "arrowUp" | "arrowDown",
          text: `ENTRY ${data.trade.entry_price?.toFixed(2) ?? ""}`,
        };
      } else {
        const outcomeColor =
          m.outcome === "TP" ? "#22c55e" :
          m.outcome === "SL" ? "#ef4444" : "#f59e0b";
        return {
          time: msToChartTime(m.t),
          position: (isLong ? "aboveBar" : "belowBar") as "aboveBar" | "belowBar",
          color: outcomeColor,
          shape: "circle" as "circle",
          text: m.outcome || "EXIT",
        };
      }
    });

    chartMarkers.sort((a, b) => (a.time as number) - (b.time as number));
    const markersPlugin = createSeriesMarkers(candleSeries, chartMarkers);

    if (data.bands.length > 0) {
      const band = data.bands[0];
      const fromTime = msToChartTime(band.from);
      const toTime = msToChartTime(band.to);
      const bandColor = "rgba(59, 130, 246, 0.06)";

      const allPrices = data.candles.flatMap(c => [c.h, c.l]);
      const minPrice = Math.min(...allPrices);
      const maxPrice = Math.max(...allPrices);
      const range = maxPrice - minPrice;

      const bandCandles: CandlestickData[] = [];
      for (const c of data.candles) {
        const ct = msToChartTime(c.t);
        if ((ct as number) >= (fromTime as number) && (ct as number) <= (toTime as number)) {
          bandCandles.push({
            time: ct,
            open: minPrice - range * 0.02,
            high: maxPrice + range * 0.02,
            low: minPrice - range * 0.02,
            close: maxPrice + range * 0.02,
          });
        }
      }

      if (bandCandles.length > 0) {
        const bandSeries = chart.addSeries(CandlestickSeries, {
          upColor: bandColor,
          downColor: bandColor,
          borderUpColor: "transparent",
          borderDownColor: "transparent",
          wickUpColor: "transparent",
          wickDownColor: "transparent",
          priceScaleId: "band",
        });
        chart.priceScale("band").applyOptions({
          visible: false,
          scaleMargins: { top: 0, bottom: 0 },
        });
        bandSeries.setData(bandCandles);
      }
    }

    if (showLevels && tmActions && tmActions.length > 0) {
      const slActions = tmActions.filter(a => (a.action === "TRAIL_SL" || a.action === "MOVE_SL") && a.sl != null);
      for (let i = 0; i < slActions.length; i++) {
        const act = slActions[i];
        const color = act.action === "TRAIL_SL" ? "#3b82f6" : "#22c55e";
        const label = act.action === "TRAIL_SL" ? `Trail ${i + 1}` : "BE";
        candleSeries.createPriceLine({
          price: act.sl!,
          color: color,
          lineWidth: 1,
          lineStyle: LineStyle.SparseDotted,
          axisLabelVisible: false,
          title: label,
        });
      }

      const tmMarkers: SeriesMarker<Time>[] = tmActions
        .filter(a => a.action === "CLOSE_FULL" && a.price != null)
        .map(a => ({
          time: msToChartTime(a.ts),
          position: "aboveBar" as "aboveBar",
          color: "#f59e0b",
          shape: "square" as "square",
          text: a.reason.split("_").slice(0, 2).join(" "),
        }));
      if (tmMarkers.length > 0) {
        const allMarkersSorted = [...chartMarkers, ...tmMarkers].sort((a, b) => (a.time as number) - (b.time as number));
        markersPlugin.setMarkers(allMarkersSorted);
      }
    }

    if (data.trade.entry_at && candleData.length > 0) {
      const paddingBars = 5;
      const barWidth = 15 * 60;
      const firstCandle = candleData[0].time as number;
      const lastCandle = candleData[candleData.length - 1].time as number;
      chart.timeScale().setVisibleRange({
        from: (firstCandle - paddingBars * barWidth) as Time,
        to: (lastCandle + paddingBars * barWidth) as Time,
      });
    }

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      markersPlugin.detach();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, showLevels, showVolume, tmActions]);

  const hasData = data && data.candles.length > 0;
  const entryInRange = hasData && data.candles.some(c => Math.abs(c.t - data.trade.entry_at) < 15 * 60 * 1000);
  const exitInRange = !data?.trade?.exit_at || (hasData && data.candles.some(c => Math.abs(c.t - (data.trade.exit_at ?? 0)) < 15 * 60 * 1000));

  return (
    <Card className="rounded-md" data-testid="card-replay-chart">
      <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-1.5">
          <BarChart3 className="h-4 w-4 text-muted-foreground" />
          Trade Replay
        </CardTitle>
        <div className="flex items-center gap-1.5 flex-wrap">
          {hasData && data.trade.bars_held !== null && (
            <Badge variant="secondary" className="text-xs" data-testid="badge-bars-held">
              {data.trade.bars_held} bars
            </Badge>
          )}
          {hasData && data.trade.net_r !== null && (
            <Badge
              variant={data.trade.net_r >= 0 ? "default" : "destructive"}
              className="text-xs"
              data-testid="badge-replay-net-r"
            >
              {fmtR(data.trade.net_r)}
            </Badge>
          )}
          {hasData && data.trade.pnl_usd !== null && (
            <Badge variant="outline" className="text-xs" data-testid="badge-replay-pnl-usd">
              {fmtUsd(data.trade.pnl_usd)}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-[360px] w-full" data-testid="skeleton-replay" />
        ) : !hasData ? (
          <div className="h-[360px] flex items-center justify-center" data-testid="replay-no-data">
            <p className="text-sm text-muted-foreground">
              {data ? "Replay data missing — no candles in database for this trade's time range" : "No replay data available"}
            </p>
          </div>
        ) : (
          <>
            {(!entryInRange || !exitInRange) && (
              <div className="mb-2 text-xs text-amber-500" data-testid="replay-range-warning">
                {!entryInRange && "Entry time not found in candle range. "}
                {!exitInRange && "Exit time not found in candle range."}
              </div>
            )}
            <div className="flex items-center gap-1.5 mb-2 flex-wrap">
              <Button
                size="sm"
                variant={showLevels ? "default" : "outline"}
                onClick={() => setShowLevels(!showLevels)}
                data-testid="button-toggle-levels"
                className="toggle-elevate"
              >
                {showLevels ? <Eye className="h-3.5 w-3.5 mr-1" /> : <EyeOff className="h-3.5 w-3.5 mr-1" />}
                SL/TP
              </Button>
              <Button
                size="sm"
                variant={showVolume ? "default" : "outline"}
                onClick={() => setShowVolume(!showVolume)}
                data-testid="button-toggle-volume"
                className="toggle-elevate"
              >
                <BarChart className="h-3.5 w-3.5 mr-1" />
                Volume
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={jumpToEntry}
                data-testid="button-jump-entry"
              >
                <Crosshair className="h-3.5 w-3.5 mr-1" />
                Jump to Entry
              </Button>
            </div>
            <div
              ref={chartContainerRef}
              className="w-full rounded-md overflow-hidden"
              data-testid="chart-container"
              style={{ minHeight: 360 }}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
}
