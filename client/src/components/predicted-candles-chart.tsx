import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, AlertTriangle } from "lucide-react";
import { useMemo } from "react";

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface PredictedCandle {
  timestamp: number;
  q10: number;
  q25: number;
  q50: number;
  q75: number;
  q90: number;
  direction: "up" | "down";
}

interface PredictedCandlesChartProps {
  historicalCandles: Candle[];
  predictedCandles: PredictedCandle[];
  currentPrice: number;
  horizon?: number;
}

export function PredictedCandlesChart({
  historicalCandles,
  predictedCandles,
  currentPrice,
  horizon = 10
}: PredictedCandlesChartProps) {
  const chartData = useMemo(() => {
    const displayHistorical = historicalCandles.slice(-30);
    
    const allPrices = [
      ...displayHistorical.map(c => c.high),
      ...displayHistorical.map(c => c.low),
      ...predictedCandles.map(c => c.q90),
      ...predictedCandles.map(c => c.q10)
    ];
    
    const minPrice = Math.min(...allPrices) * 0.999;
    const maxPrice = Math.max(...allPrices) * 1.001;
    const priceRange = maxPrice - minPrice;
    
    return { displayHistorical, minPrice, maxPrice, priceRange };
  }, [historicalCandles, predictedCandles]);

  const { displayHistorical, minPrice, maxPrice, priceRange } = chartData;

  const chartWidth = 800;
  const chartHeight = 400;
  const padding = { top: 20, right: 60, bottom: 40, left: 10 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  const totalCandles = displayHistorical.length + predictedCandles.length;
  const candleWidth = Math.min(12, (innerWidth / totalCandles) * 0.8);
  const candleGap = candleWidth * 0.3;

  const priceToY = (price: number) => {
    return padding.top + innerHeight - ((price - minPrice) / priceRange) * innerHeight;
  };

  const formatPrice = (price: number) => {
    return price.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 });
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
  };

  const priceGridLines = useMemo(() => {
    const lines = [];
    const step = priceRange / 5;
    for (let i = 0; i <= 5; i++) {
      const price = minPrice + step * i;
      lines.push(price);
    }
    return lines;
  }, [minPrice, priceRange]);

  if (!historicalCandles.length) {
    return (
      <Card data-testid="card-predicted-candles-empty">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Predicted Candles
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <AlertTriangle className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground">No candle data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="card-predicted-candles">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <TrendingUp className="h-4 w-4" />
          Price Chart with Neural Network Predictions
          <Badge variant="outline" className="ml-auto text-xs">
            {predictedCandles.length} bars predicted
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="w-full overflow-x-auto">
          <svg
            viewBox={`0 0 ${chartWidth} ${chartHeight}`}
            className="w-full h-auto min-h-[300px]"
            style={{ maxHeight: "450px" }}
            data-testid="svg-predicted-candles"
          >
            {/* Background */}
            <rect
              x={padding.left}
              y={padding.top}
              width={innerWidth}
              height={innerHeight}
              fill="hsl(var(--muted) / 0.1)"
              rx="4"
            />

            {/* Price grid lines */}
            {priceGridLines.map((price, i) => (
              <g key={i}>
                <line
                  x1={padding.left}
                  y1={priceToY(price)}
                  x2={chartWidth - padding.right}
                  y2={priceToY(price)}
                  stroke="hsl(var(--muted-foreground) / 0.2)"
                  strokeDasharray="4,4"
                />
                <text
                  x={chartWidth - padding.right + 5}
                  y={priceToY(price) + 4}
                  fontSize="10"
                  fill="hsl(var(--muted-foreground))"
                >
                  {formatPrice(price)}
                </text>
              </g>
            ))}

            {/* Separator line between historical and predicted */}
            {predictedCandles.length > 0 && (
              <line
                x1={padding.left + displayHistorical.length * (candleWidth + candleGap)}
                y1={padding.top}
                x2={padding.left + displayHistorical.length * (candleWidth + candleGap)}
                y2={padding.top + innerHeight}
                stroke="hsl(var(--primary) / 0.5)"
                strokeWidth="2"
                strokeDasharray="6,4"
              />
            )}

            {/* Label for prediction zone */}
            {predictedCandles.length > 0 && (
              <text
                x={padding.left + (displayHistorical.length + predictedCandles.length / 2) * (candleWidth + candleGap)}
                y={padding.top + 15}
                fontSize="11"
                fill="hsl(var(--primary))"
                textAnchor="middle"
                fontWeight="500"
              >
                NN Predicted
              </text>
            )}

            {/* Historical candles */}
            {displayHistorical.map((candle, i) => {
              const x = padding.left + i * (candleWidth + candleGap) + candleGap / 2;
              const isUp = candle.close >= candle.open;
              const bodyTop = priceToY(Math.max(candle.open, candle.close));
              const bodyBottom = priceToY(Math.min(candle.open, candle.close));
              const bodyHeight = Math.max(1, bodyBottom - bodyTop);

              return (
                <g key={`hist-${i}`}>
                  {/* Wick */}
                  <line
                    x1={x + candleWidth / 2}
                    y1={priceToY(candle.high)}
                    x2={x + candleWidth / 2}
                    y2={priceToY(candle.low)}
                    stroke={isUp ? "hsl(142.1 76.2% 36.3%)" : "hsl(0 72.2% 50.6%)"}
                    strokeWidth="1"
                  />
                  {/* Body */}
                  <rect
                    x={x}
                    y={bodyTop}
                    width={candleWidth}
                    height={bodyHeight}
                    fill={isUp ? "hsl(142.1 76.2% 36.3%)" : "hsl(0 72.2% 50.6%)"}
                    rx="1"
                  />
                </g>
              );
            })}

            {/* Predicted candles (probability bands as candle shapes) */}
            {predictedCandles.map((pred, i) => {
              const x = padding.left + (displayHistorical.length + i) * (candleWidth + candleGap) + candleGap / 2;
              const isUp = pred.direction === "up";
              
              const wickTop = priceToY(pred.q90);
              const wickBottom = priceToY(pred.q10);
              const bodyTop = priceToY(pred.q75);
              const bodyBottom = priceToY(pred.q25);
              const bodyHeight = Math.max(1, bodyBottom - bodyTop);
              
              const fillColor = isUp 
                ? "hsl(142.1 76.2% 36.3% / 0.5)" 
                : "hsl(0 72.2% 50.6% / 0.5)";
              const strokeColor = isUp 
                ? "hsl(142.1 76.2% 36.3%)" 
                : "hsl(0 72.2% 50.6%)";

              return (
                <g key={`pred-${i}`}>
                  {/* Uncertainty band (q10-q90 as wick) */}
                  <line
                    x1={x + candleWidth / 2}
                    y1={wickTop}
                    x2={x + candleWidth / 2}
                    y2={wickBottom}
                    stroke={strokeColor}
                    strokeWidth="1"
                    strokeDasharray="2,2"
                    opacity="0.7"
                  />
                  
                  {/* Core range (q25-q75 as body) */}
                  <rect
                    x={x}
                    y={bodyTop}
                    width={candleWidth}
                    height={bodyHeight}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth="1"
                    strokeDasharray="3,2"
                    rx="1"
                  />
                  
                  {/* Median line (q50) */}
                  <line
                    x1={x}
                    y1={priceToY(pred.q50)}
                    x2={x + candleWidth}
                    y2={priceToY(pred.q50)}
                    stroke={strokeColor}
                    strokeWidth="2"
                  />
                </g>
              );
            })}

            {/* Current price line */}
            <g>
              <line
                x1={padding.left}
                y1={priceToY(currentPrice)}
                x2={padding.left + displayHistorical.length * (candleWidth + candleGap)}
                y2={priceToY(currentPrice)}
                stroke="hsl(var(--primary))"
                strokeWidth="1"
                strokeDasharray="4,2"
              />
              <rect
                x={padding.left}
                y={priceToY(currentPrice) - 8}
                width="50"
                height="16"
                fill="hsl(var(--primary))"
                rx="3"
              />
              <text
                x={padding.left + 25}
                y={priceToY(currentPrice) + 4}
                fontSize="9"
                fill="hsl(var(--primary-foreground))"
                textAnchor="middle"
                fontWeight="500"
              >
                {formatPrice(currentPrice)}
              </text>
            </g>

            {/* X-axis labels (time) */}
            {displayHistorical.filter((_, i) => i % 5 === 0).map((candle, idx) => {
              const i = idx * 5;
              const x = padding.left + i * (candleWidth + candleGap) + candleWidth / 2;
              return (
                <text
                  key={i}
                  x={x}
                  y={chartHeight - 10}
                  fontSize="9"
                  fill="hsl(var(--muted-foreground))"
                  textAnchor="middle"
                >
                  {formatTime(candle.timestamp)}
                </text>
              );
            })}

            {/* Legend */}
            <g transform={`translate(${padding.left + 10}, ${padding.top + 10})`}>
              <rect
                x="0"
                y="0"
                width="150"
                height="50"
                fill="hsl(var(--background) / 0.9)"
                rx="4"
                stroke="hsl(var(--border))"
              />
              <circle cx="12" cy="14" r="5" fill="hsl(142.1 76.2% 36.3%)" />
              <text x="22" y="18" fontSize="10" fill="hsl(var(--foreground))">Historical (actual)</text>
              <circle cx="12" cy="32" r="5" fill="hsl(142.1 76.2% 36.3% / 0.5)" stroke="hsl(142.1 76.2% 36.3%)" strokeDasharray="2,2" />
              <text x="22" y="36" fontSize="10" fill="hsl(var(--foreground))">Predicted (q10-q90)</text>
            </g>
          </svg>
        </div>

        {/* Prediction Summary */}
        {predictedCandles.length > 0 && (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-muted/20 rounded-lg p-2 text-center">
              <div className="text-xs text-muted-foreground">Predicted Range</div>
              <div className="text-sm font-medium">
                ${formatPrice(Math.min(...predictedCandles.map(c => c.q10)))} - ${formatPrice(Math.max(...predictedCandles.map(c => c.q90)))}
              </div>
            </div>
            <div className="bg-muted/20 rounded-lg p-2 text-center">
              <div className="text-xs text-muted-foreground">Expected (q50)</div>
              <div className="text-sm font-medium">
                ${formatPrice(predictedCandles[predictedCandles.length - 1]?.q50 || currentPrice)}
              </div>
            </div>
            <div className="bg-muted/20 rounded-lg p-2 text-center">
              <div className="text-xs text-muted-foreground">Upside (q90)</div>
              <div className="text-sm font-medium text-emerald-400">
                +{((predictedCandles[predictedCandles.length - 1]?.q90 / currentPrice - 1) * 100).toFixed(2)}%
              </div>
            </div>
            <div className="bg-muted/20 rounded-lg p-2 text-center">
              <div className="text-xs text-muted-foreground">Downside (q10)</div>
              <div className="text-sm font-medium text-red-400">
                {((predictedCandles[predictedCandles.length - 1]?.q10 / currentPrice - 1) * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
