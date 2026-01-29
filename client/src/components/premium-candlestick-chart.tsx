import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Minus, Activity, BarChart3, Eye, EyeOff } from "lucide-react";
import { useMemo, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";

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

interface TradeLevels {
  entry: number;
  stopLoss: number;
  takeProfit: number;
}

interface PremiumCandlestickChartProps {
  historicalCandles: Candle[];
  predictedCandles?: PredictedCandle[];
  currentPrice: number;
  action?: "LONG" | "SHORT" | "HOLD";
  tradeLevels?: TradeLevels;
  symbol?: string;
  timeframe?: string;
}

const calculateEMA = (candles: Candle[], period: number): number[] => {
  const closes = candles.map(c => c.close);
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);
  
  if (closes.length < period) return ema;
  
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += closes[i];
  }
  ema.push(sum / period);
  
  for (let i = period; i < closes.length; i++) {
    ema.push((closes[i] - ema[ema.length - 1]) * multiplier + ema[ema.length - 1]);
  }
  
  return ema;
};

export function PremiumCandlestickChart({
  historicalCandles,
  predictedCandles = [],
  currentPrice,
  action = "HOLD",
  tradeLevels,
  symbol = "BTCUSDT",
  timeframe = "15m"
}: PremiumCandlestickChartProps) {
  const [showEMA9, setShowEMA9] = useState(true);
  const [showEMA21, setShowEMA21] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [hoveredCandle, setHoveredCandle] = useState<number | null>(null);
  const [crosshairPos, setCrosshairPos] = useState<{ x: number; y: number } | null>(null);

  const chartData = useMemo(() => {
    const displayHistorical = historicalCandles.slice(-40);
    
    const allPrices = [
      ...displayHistorical.map(c => c.high),
      ...displayHistorical.map(c => c.low),
      ...predictedCandles.map(c => c.q90),
      ...predictedCandles.map(c => c.q10),
      tradeLevels?.stopLoss || 0,
      tradeLevels?.takeProfit || 0,
    ].filter(p => p > 0);
    
    const minPrice = Math.min(...allPrices) * 0.998;
    const maxPrice = Math.max(...allPrices) * 1.002;
    const priceRange = maxPrice - minPrice;
    
    const volumes = displayHistorical.map(c => c.volume || 0);
    const maxVolume = Math.max(...volumes) || 1;
    
    const ema9 = calculateEMA(displayHistorical, 9);
    const ema21 = calculateEMA(displayHistorical, 21);
    
    return { displayHistorical, minPrice, maxPrice, priceRange, maxVolume, ema9, ema21 };
  }, [historicalCandles, predictedCandles, tradeLevels]);

  const { displayHistorical, minPrice, maxPrice, priceRange, maxVolume, ema9, ema21 } = chartData;

  const chartWidth = 900;
  const chartHeight = 500;
  const volumeHeight = 80;
  const padding = { top: 30, right: 75, bottom: 50, left: 15 };
  const mainChartHeight = chartHeight - volumeHeight - padding.top - padding.bottom;
  const innerWidth = chartWidth - padding.left - padding.right;

  const totalCandles = displayHistorical.length + predictedCandles.length;
  const candleWidth = Math.min(16, (innerWidth / totalCandles) * 0.75);
  const candleGap = candleWidth * 0.35;
  const totalCandleWidth = candleWidth + candleGap;

  const priceToY = useCallback((price: number) => {
    const safeRange = priceRange > 0 ? priceRange : 1;
    return padding.top + mainChartHeight - ((price - minPrice) / safeRange) * mainChartHeight;
  }, [minPrice, priceRange, mainChartHeight]);

  const volumeToY = useCallback((volume: number) => {
    const volumeTop = chartHeight - padding.bottom - volumeHeight;
    return volumeTop + volumeHeight - (volume / maxVolume) * volumeHeight * 0.9;
  }, [maxVolume, chartHeight, volumeHeight]);

  const formatPrice = (price: number) => {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  const formatVolume = (vol: number) => {
    if (vol >= 1e9) return `${(vol / 1e9).toFixed(1)}B`;
    if (vol >= 1e6) return `${(vol / 1e6).toFixed(1)}M`;
    if (vol >= 1e3) return `${(vol / 1e3).toFixed(1)}K`;
    return vol.toFixed(0);
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return `${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getDate().toString().padStart(2, '0')}`;
  };

  const priceGridLines = useMemo(() => {
    const lines = [];
    const step = priceRange / 6;
    for (let i = 0; i <= 6; i++) {
      const price = minPrice + step * i;
      lines.push(price);
    }
    return lines;
  }, [minPrice, priceRange]);

  const getActionColors = () => {
    if (action === "LONG") return { primary: "#10b981", secondary: "#059669", glow: "rgba(16, 185, 129, 0.3)" };
    if (action === "SHORT") return { primary: "#ef4444", secondary: "#dc2626", glow: "rgba(239, 68, 68, 0.3)" };
    return { primary: "#f59e0b", secondary: "#d97706", glow: "rgba(245, 158, 11, 0.3)" };
  };

  const actionColors = getActionColors();

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const scaleX = chartWidth / rect.width;
    const scaleY = chartHeight / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    if (x > padding.left && x < chartWidth - padding.right && y > padding.top && y < chartHeight - padding.bottom) {
      setCrosshairPos({ x, y });
      const candleIndex = Math.floor((x - padding.left) / totalCandleWidth);
      if (candleIndex >= 0 && candleIndex < displayHistorical.length) {
        setHoveredCandle(candleIndex);
      } else {
        setHoveredCandle(null);
      }
    } else {
      setCrosshairPos(null);
      setHoveredCandle(null);
    }
  };

  const handleMouseLeave = () => {
    setCrosshairPos(null);
    setHoveredCandle(null);
  };

  const hoveredCandleData = hoveredCandle !== null ? displayHistorical[hoveredCandle] : null;

  if (!historicalCandles.length) {
    return (
      <Card data-testid="card-premium-chart-empty" className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Premium Price Chart
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-16">
          <div className="text-center">
            <Activity className="h-10 w-10 mx-auto text-muted-foreground/50 mb-3" />
            <p className="text-sm text-muted-foreground">Waiting for candle data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="card-premium-chart" className="border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
      <CardHeader className="pb-2 border-b border-border/30">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-sm font-medium flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-semibold">{symbol}</span>
              <Badge variant="outline" className="text-xs font-normal">{timeframe}</Badge>
            </div>
            <div className="h-4 w-px bg-border/50" />
            <motion.div
              key={currentPrice}
              initial={{ scale: 1.1, opacity: 0.7 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex items-center gap-1"
            >
              <span className="text-lg font-bold tabular-nums">${formatPrice(currentPrice)}</span>
              {action !== "HOLD" && (
                <Badge 
                  className={`ml-1 ${
                    action === "LONG" 
                      ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" 
                      : "bg-red-500/20 text-red-400 border-red-500/30"
                  }`}
                >
                  {action === "LONG" ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
                  {action}
                </Badge>
              )}
            </motion.div>
          </CardTitle>
          
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              className={`text-xs ${showEMA9 ? 'text-cyan-400' : 'text-muted-foreground'}`}
              onClick={() => setShowEMA9(!showEMA9)}
              data-testid="button-toggle-ema9"
            >
              {showEMA9 ? <Eye className="h-3 w-3 mr-1" /> : <EyeOff className="h-3 w-3 mr-1" />}
              EMA9
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className={`text-xs ${showEMA21 ? 'text-orange-400' : 'text-muted-foreground'}`}
              onClick={() => setShowEMA21(!showEMA21)}
              data-testid="button-toggle-ema21"
            >
              {showEMA21 ? <Eye className="h-3 w-3 mr-1" /> : <EyeOff className="h-3 w-3 mr-1" />}
              EMA21
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className={`text-xs ${showVolume ? 'text-purple-400' : 'text-muted-foreground'}`}
              onClick={() => setShowVolume(!showVolume)}
              data-testid="button-toggle-volume"
            >
              {showVolume ? <Eye className="h-3 w-3 mr-1" /> : <EyeOff className="h-3 w-3 mr-1" />}
              VOL
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        <div className="w-full overflow-x-auto">
          <svg
            viewBox={`0 0 ${chartWidth} ${chartHeight}`}
            className="w-full h-auto"
            style={{ minHeight: "400px", maxHeight: "550px" }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            data-testid="svg-premium-chart"
          >
            <defs>
              <linearGradient id="bullishGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="1" />
                <stop offset="100%" stopColor="#059669" stopOpacity="1" />
              </linearGradient>
              <linearGradient id="bearishGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity="1" />
                <stop offset="100%" stopColor="#dc2626" stopOpacity="1" />
              </linearGradient>
              <linearGradient id="volumeBullGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.6" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.2" />
              </linearGradient>
              <linearGradient id="volumeBearGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity="0.6" />
                <stop offset="100%" stopColor="#ef4444" stopOpacity="0.2" />
              </linearGradient>
              <linearGradient id="predictionZoneGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={actionColors.primary} stopOpacity="0.15" />
                <stop offset="50%" stopColor={actionColors.primary} stopOpacity="0.05" />
                <stop offset="100%" stopColor={actionColors.primary} stopOpacity="0.15" />
              </linearGradient>
              <filter id="candleGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="priceGlow" x="-100%" y="-100%" width="300%" height="300%">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            <rect x="0" y="0" width={chartWidth} height={chartHeight} fill="transparent" />

            {priceGridLines.map((price, i) => (
              <g key={`grid-${i}`}>
                <line
                  x1={padding.left}
                  y1={priceToY(price)}
                  x2={chartWidth - padding.right}
                  y2={priceToY(price)}
                  stroke="hsl(var(--muted-foreground))"
                  strokeOpacity="0.1"
                  strokeWidth="1"
                />
                <text
                  x={chartWidth - padding.right + 8}
                  y={priceToY(price) + 4}
                  fontSize="10"
                  fill="hsl(var(--muted-foreground))"
                  fontFamily="ui-monospace, monospace"
                >
                  {formatPrice(price)}
                </text>
              </g>
            ))}

            {tradeLevels && action !== "HOLD" && (
              <>
                <rect
                  x={padding.left}
                  y={Math.min(priceToY(tradeLevels.takeProfit), priceToY(tradeLevels.entry))}
                  width={innerWidth}
                  height={Math.abs(priceToY(tradeLevels.takeProfit) - priceToY(tradeLevels.entry))}
                  fill={action === "LONG" ? "rgba(16, 185, 129, 0.05)" : "rgba(239, 68, 68, 0.05)"}
                />
                <rect
                  x={padding.left}
                  y={Math.min(priceToY(tradeLevels.stopLoss), priceToY(tradeLevels.entry))}
                  width={innerWidth}
                  height={Math.abs(priceToY(tradeLevels.stopLoss) - priceToY(tradeLevels.entry))}
                  fill={action === "LONG" ? "rgba(239, 68, 68, 0.05)" : "rgba(16, 185, 129, 0.05)"}
                />
                
                <line
                  x1={padding.left}
                  y1={priceToY(tradeLevels.takeProfit)}
                  x2={chartWidth - padding.right}
                  y2={priceToY(tradeLevels.takeProfit)}
                  stroke="#10b981"
                  strokeWidth="1.5"
                  strokeDasharray="8,4"
                />
                <rect
                  x={chartWidth - padding.right - 55}
                  y={priceToY(tradeLevels.takeProfit) - 10}
                  width="55"
                  height="20"
                  fill="#10b981"
                  rx="3"
                />
                <text
                  x={chartWidth - padding.right - 27}
                  y={priceToY(tradeLevels.takeProfit) + 4}
                  fontSize="9"
                  fill="white"
                  textAnchor="middle"
                  fontWeight="600"
                >
                  TP
                </text>
                
                <line
                  x1={padding.left}
                  y1={priceToY(tradeLevels.stopLoss)}
                  x2={chartWidth - padding.right}
                  y2={priceToY(tradeLevels.stopLoss)}
                  stroke="#ef4444"
                  strokeWidth="1.5"
                  strokeDasharray="8,4"
                />
                <rect
                  x={chartWidth - padding.right - 55}
                  y={priceToY(tradeLevels.stopLoss) - 10}
                  width="55"
                  height="20"
                  fill="#ef4444"
                  rx="3"
                />
                <text
                  x={chartWidth - padding.right - 27}
                  y={priceToY(tradeLevels.stopLoss) + 4}
                  fontSize="9"
                  fill="white"
                  textAnchor="middle"
                  fontWeight="600"
                >
                  SL
                </text>
              </>
            )}

            {predictedCandles.length > 0 && (
              <>
                <line
                  x1={padding.left + displayHistorical.length * totalCandleWidth - candleGap/2}
                  y1={padding.top}
                  x2={padding.left + displayHistorical.length * totalCandleWidth - candleGap/2}
                  y2={padding.top + mainChartHeight}
                  stroke={actionColors.primary}
                  strokeWidth="2"
                  strokeDasharray="6,4"
                  strokeOpacity="0.6"
                />
                <text
                  x={padding.left + (displayHistorical.length + predictedCandles.length / 2) * totalCandleWidth}
                  y={padding.top + 18}
                  fontSize="11"
                  fill={actionColors.primary}
                  textAnchor="middle"
                  fontWeight="600"
                  letterSpacing="0.5"
                >
                  NEURAL NETWORK PREDICTION
                </text>
              </>
            )}

            {showEMA9 && ema9.length > 1 && (
              <path
                d={ema9.map((val, i) => {
                  const x = padding.left + (i + displayHistorical.length - ema9.length) * totalCandleWidth + candleWidth / 2;
                  const y = priceToY(val);
                  return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                }).join(' ')}
                fill="none"
                stroke="#22d3ee"
                strokeWidth="1.5"
                strokeOpacity="0.8"
              />
            )}

            {showEMA21 && ema21.length > 1 && (
              <path
                d={ema21.map((val, i) => {
                  const x = padding.left + (i + displayHistorical.length - ema21.length) * totalCandleWidth + candleWidth / 2;
                  const y = priceToY(val);
                  return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                }).join(' ')}
                fill="none"
                stroke="#fb923c"
                strokeWidth="1.5"
                strokeOpacity="0.8"
              />
            )}

            {displayHistorical.map((candle, i) => {
              const x = padding.left + i * totalCandleWidth;
              const isUp = candle.close >= candle.open;
              const bodyTop = priceToY(Math.max(candle.open, candle.close));
              const bodyBottom = priceToY(Math.min(candle.open, candle.close));
              const bodyHeight = Math.max(2, bodyBottom - bodyTop);
              const isHovered = hoveredCandle === i;

              return (
                <g key={`hist-${i}`} filter={isHovered ? "url(#candleGlow)" : undefined}>
                  <line
                    x1={x + candleWidth / 2}
                    y1={priceToY(candle.high)}
                    x2={x + candleWidth / 2}
                    y2={priceToY(candle.low)}
                    stroke={isUp ? "#10b981" : "#ef4444"}
                    strokeWidth={isHovered ? 2 : 1}
                  />
                  <rect
                    x={x}
                    y={bodyTop}
                    width={candleWidth}
                    height={bodyHeight}
                    fill={isUp ? "url(#bullishGradient)" : "url(#bearishGradient)"}
                    rx="1"
                    style={{ transition: 'all 0.15s ease' }}
                  />
                  {isHovered && (
                    <rect
                      x={x - 2}
                      y={bodyTop - 2}
                      width={candleWidth + 4}
                      height={bodyHeight + 4}
                      fill="none"
                      stroke={isUp ? "#10b981" : "#ef4444"}
                      strokeWidth="2"
                      rx="2"
                      opacity="0.5"
                    />
                  )}
                </g>
              );
            })}

            {predictedCandles.map((pred, i) => {
              const x = padding.left + (displayHistorical.length + i) * totalCandleWidth;
              const isUp = pred.direction === "up";
              
              return (
                <g key={`pred-${i}`}>
                  <rect
                    x={x - 2}
                    y={priceToY(pred.q90)}
                    width={candleWidth + 4}
                    height={priceToY(pred.q10) - priceToY(pred.q90)}
                    fill="url(#predictionZoneGradient)"
                    rx="2"
                  />
                  <line
                    x1={x + candleWidth / 2}
                    y1={priceToY(pred.q90)}
                    x2={x + candleWidth / 2}
                    y2={priceToY(pred.q10)}
                    stroke={isUp ? "#10b981" : "#ef4444"}
                    strokeWidth="1"
                    strokeDasharray="3,3"
                    opacity="0.6"
                  />
                  <rect
                    x={x}
                    y={priceToY(pred.q75)}
                    width={candleWidth}
                    height={Math.max(2, priceToY(pred.q25) - priceToY(pred.q75))}
                    fill={isUp ? "rgba(16, 185, 129, 0.4)" : "rgba(239, 68, 68, 0.4)"}
                    stroke={isUp ? "#10b981" : "#ef4444"}
                    strokeWidth="1"
                    strokeDasharray="4,2"
                    rx="1"
                  />
                  <line
                    x1={x}
                    y1={priceToY(pred.q50)}
                    x2={x + candleWidth}
                    y2={priceToY(pred.q50)}
                    stroke={isUp ? "#10b981" : "#ef4444"}
                    strokeWidth="2.5"
                  />
                </g>
              );
            })}

            <g filter="url(#priceGlow)">
              <line
                x1={padding.left}
                y1={priceToY(currentPrice)}
                x2={chartWidth - padding.right}
                y2={priceToY(currentPrice)}
                stroke={actionColors.primary}
                strokeWidth="1"
                strokeDasharray="6,3"
              />
            </g>
            <motion.g
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <rect
                x={padding.left}
                y={priceToY(currentPrice) - 11}
                width="72"
                height="22"
                fill={actionColors.primary}
                rx="4"
              />
              <text
                x={padding.left + 36}
                y={priceToY(currentPrice) + 4}
                fontSize="10"
                fill="white"
                textAnchor="middle"
                fontWeight="600"
                fontFamily="ui-monospace, monospace"
              >
                ${formatPrice(currentPrice)}
              </text>
            </motion.g>

            {showVolume && (
              <g>
                <line
                  x1={padding.left}
                  y1={chartHeight - padding.bottom - volumeHeight}
                  x2={chartWidth - padding.right}
                  y2={chartHeight - padding.bottom - volumeHeight}
                  stroke="hsl(var(--muted-foreground))"
                  strokeOpacity="0.2"
                />
                {displayHistorical.map((candle, i) => {
                  const x = padding.left + i * totalCandleWidth;
                  const vol = candle.volume || 0;
                  const isUp = candle.close >= candle.open;
                  const barHeight = (vol / maxVolume) * volumeHeight * 0.85;
                  const y = chartHeight - padding.bottom - barHeight;
                  
                  return (
                    <rect
                      key={`vol-${i}`}
                      x={x}
                      y={y}
                      width={candleWidth}
                      height={barHeight}
                      fill={isUp ? "url(#volumeBullGradient)" : "url(#volumeBearGradient)"}
                      rx="1"
                    />
                  );
                })}
                <text
                  x={chartWidth - padding.right + 8}
                  y={chartHeight - padding.bottom - volumeHeight / 2}
                  fontSize="9"
                  fill="hsl(var(--muted-foreground))"
                  dominantBaseline="middle"
                >
                  {formatVolume(maxVolume)}
                </text>
              </g>
            )}

            {displayHistorical.filter((_, i) => i % 6 === 0).map((candle, idx) => {
              const i = idx * 6;
              const x = padding.left + i * totalCandleWidth + candleWidth / 2;
              return (
                <g key={`time-${i}`}>
                  <text
                    x={x}
                    y={chartHeight - padding.bottom + 15}
                    fontSize="9"
                    fill="hsl(var(--muted-foreground))"
                    textAnchor="middle"
                    fontFamily="ui-monospace, monospace"
                  >
                    {formatTime(candle.timestamp)}
                  </text>
                  <text
                    x={x}
                    y={chartHeight - padding.bottom + 27}
                    fontSize="8"
                    fill="hsl(var(--muted-foreground))"
                    textAnchor="middle"
                    opacity="0.6"
                  >
                    {formatDate(candle.timestamp)}
                  </text>
                </g>
              );
            })}

            {crosshairPos && (
              <g>
                <line
                  x1={crosshairPos.x}
                  y1={padding.top}
                  x2={crosshairPos.x}
                  y2={chartHeight - padding.bottom}
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth="1"
                  strokeDasharray="4,4"
                  opacity="0.4"
                />
                <line
                  x1={padding.left}
                  y1={crosshairPos.y}
                  x2={chartWidth - padding.right}
                  y2={crosshairPos.y}
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth="1"
                  strokeDasharray="4,4"
                  opacity="0.4"
                />
                {crosshairPos.y > padding.top && crosshairPos.y < padding.top + mainChartHeight && (
                  <>
                    <rect
                      x={chartWidth - padding.right + 3}
                      y={crosshairPos.y - 10}
                      width="65"
                      height="20"
                      fill="hsl(var(--muted))"
                      rx="3"
                    />
                    <text
                      x={chartWidth - padding.right + 35}
                      y={crosshairPos.y + 4}
                      fontSize="9"
                      fill="hsl(var(--foreground))"
                      textAnchor="middle"
                      fontFamily="ui-monospace, monospace"
                    >
                      {formatPrice(minPrice + (1 - (crosshairPos.y - padding.top) / mainChartHeight) * priceRange)}
                    </text>
                  </>
                )}
              </g>
            )}
          </svg>
        </div>

        <AnimatePresence>
          {hoveredCandleData && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-16 left-4 bg-card/95 backdrop-blur-sm border border-border/50 rounded-lg p-3 shadow-lg z-10"
            >
              <div className="text-xs space-y-1 font-mono">
                <div className="text-muted-foreground mb-2">
                  {new Date(hoveredCandleData.timestamp).toLocaleString()}
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                  <span className="text-muted-foreground">O:</span>
                  <span>${formatPrice(hoveredCandleData.open)}</span>
                  <span className="text-muted-foreground">H:</span>
                  <span className="text-emerald-400">${formatPrice(hoveredCandleData.high)}</span>
                  <span className="text-muted-foreground">L:</span>
                  <span className="text-red-400">${formatPrice(hoveredCandleData.low)}</span>
                  <span className="text-muted-foreground">C:</span>
                  <span className={hoveredCandleData.close >= hoveredCandleData.open ? 'text-emerald-400' : 'text-red-400'}>
                    ${formatPrice(hoveredCandleData.close)}
                  </span>
                  {hoveredCandleData.volume && (
                    <>
                      <span className="text-muted-foreground">V:</span>
                      <span className="text-purple-400">{formatVolume(hoveredCandleData.volume)}</span>
                    </>
                  )}
                </div>
                <div className={`mt-2 pt-2 border-t border-border/30 ${
                  hoveredCandleData.close >= hoveredCandleData.open ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {((hoveredCandleData.close / hoveredCandleData.open - 1) * 100).toFixed(2)}%
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {predictedCandles.length > 0 && (
          <div className="px-4 pb-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-muted/20 rounded-lg p-3 text-center border border-border/30">
                <div className="text-xs text-muted-foreground mb-1">Predicted Range</div>
                <div className="text-sm font-semibold tabular-nums">
                  ${formatPrice(Math.min(...predictedCandles.map(c => c.q10)))} - ${formatPrice(Math.max(...predictedCandles.map(c => c.q90)))}
                </div>
              </div>
              <div className="bg-muted/20 rounded-lg p-3 text-center border border-border/30">
                <div className="text-xs text-muted-foreground mb-1">Expected (Median)</div>
                <div className="text-sm font-semibold tabular-nums">
                  ${formatPrice(predictedCandles[predictedCandles.length - 1]?.q50 || currentPrice)}
                </div>
              </div>
              <div className="bg-muted/20 rounded-lg p-3 text-center border border-border/30">
                <div className="text-xs text-muted-foreground mb-1">Upside (90%)</div>
                <div className="text-sm font-semibold text-emerald-400 tabular-nums">
                  +{((predictedCandles[predictedCandles.length - 1]?.q90 / currentPrice - 1) * 100).toFixed(2)}%
                </div>
              </div>
              <div className="bg-muted/20 rounded-lg p-3 text-center border border-border/30">
                <div className="text-xs text-muted-foreground mb-1">Downside (10%)</div>
                <div className="text-sm font-semibold text-red-400 tabular-nums">
                  {((predictedCandles[predictedCandles.length - 1]?.q10 / currentPrice - 1) * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
