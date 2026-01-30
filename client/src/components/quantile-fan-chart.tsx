import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, ReferenceLine, Tooltip } from "recharts";

interface QuantileValues {
  q10: number;
  q25: number;
  q50: number;
  q75: number;
  q90: number;
}

interface QuantileFanChartProps {
  currentPrice: number;
  quantiles: QuantileValues;
  action: "LONG" | "SHORT" | "HOLD";
  horizonBars: number;
  timeframeMinutes: number;
}

/**
 * Sanitize quantile values to reasonable bounds.
 * Quantiles should be return percentages (e.g., 0.02 = 2%, -0.03 = -3%)
 * Clamp to [-0.5, 0.5] which is [-50%, +50%] - extreme but plausible for crypto
 */
function sanitizeQuantiles(quantiles: QuantileValues): QuantileValues {
  const clamp = (val: number, min: number, max: number) => 
    Math.max(min, Math.min(max, val));
  
  // Reasonable bounds for 10-bar horizon return distribution
  const MIN_RETURN = -0.5;  // -50%
  const MAX_RETURN = 0.5;   // +50%
  
  return {
    q10: clamp(quantiles.q10, MIN_RETURN, MAX_RETURN),
    q25: clamp(quantiles.q25, MIN_RETURN, MAX_RETURN),
    q50: clamp(quantiles.q50, MIN_RETURN, MAX_RETURN),
    q75: clamp(quantiles.q75, MIN_RETURN, MAX_RETURN),
    q90: clamp(quantiles.q90, MIN_RETURN, MAX_RETURN),
  };
}

export function QuantileFanChart({
  currentPrice,
  quantiles: rawQuantiles,
  action,
  horizonBars = 10,
  timeframeMinutes = 15
}: QuantileFanChartProps) {
  // Sanitize quantiles to prevent absurd values from breaking the UI
  const quantiles = sanitizeQuantiles(rawQuantiles);
  
  const generateProbabilisticPath = () => {
    const data = [];
    
    for (let i = 0; i <= horizonBars; i++) {
      const progress = i / horizonBars;
      
      const sqrtProgress = Math.sqrt(progress);
      
      const q10Price = currentPrice * (1 + quantiles.q10 * sqrtProgress);
      const q25Price = currentPrice * (1 + quantiles.q25 * sqrtProgress);
      const q50Price = currentPrice * (1 + quantiles.q50 * sqrtProgress);
      const q75Price = currentPrice * (1 + quantiles.q75 * sqrtProgress);
      const q90Price = currentPrice * (1 + quantiles.q90 * sqrtProgress);
      
      data.push({
        bar: i,
        time: `+${i * timeframeMinutes}m`,
        q10: q10Price,
        q25: q25Price,
        q50: q50Price,
        q75: q75Price,
        q90: q90Price,
        range_outer: [q10Price, q90Price],
        range_inner: [q25Price, q75Price]
      });
    }
    
    return data;
  };
  
  const data = generateProbabilisticPath();
  
  const allPrices = data.flatMap(d => [d.q10, d.q90]);
  const minPrice = Math.min(...allPrices) * 0.999;
  const maxPrice = Math.max(...allPrices) * 1.001;
  
  const formatPrice = (val: number) => 
    val.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 });
  
  const actionColor = action === "LONG" ? "#10b981" : action === "SHORT" ? "#ef4444" : "#f59e0b";
  
  return (
    <Card data-testid="card-quantile-fan-chart">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Brain className="h-4 w-4 text-primary" />
          Probabilistic Price Forecast
          <Badge 
            variant="outline" 
            className={`ml-auto ${
              action === "LONG" ? "text-emerald-400 border-emerald-500/30" :
              action === "SHORT" ? "text-red-400 border-red-500/30" :
              "text-amber-400 border-amber-500/30"
            }`}
          >
            {action === "LONG" ? <TrendingUp className="h-3 w-3 mr-1" /> :
             action === "SHORT" ? <TrendingDown className="h-3 w-3 mr-1" /> :
             <Minus className="h-3 w-3 mr-1" />}
            {action}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[200px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="outerBand" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={actionColor} stopOpacity={0.15} />
                  <stop offset="50%" stopColor={actionColor} stopOpacity={0.05} />
                  <stop offset="100%" stopColor={actionColor} stopOpacity={0.15} />
                </linearGradient>
                <linearGradient id="innerBand" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={actionColor} stopOpacity={0.3} />
                  <stop offset="50%" stopColor={actionColor} stopOpacity={0.1} />
                  <stop offset="100%" stopColor={actionColor} stopOpacity={0.3} />
                </linearGradient>
              </defs>
              
              <XAxis 
                dataKey="time" 
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
              />
              <YAxis 
                domain={[minPrice, maxPrice]}
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
                tickFormatter={formatPrice}
                width={60}
              />
              
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))', 
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                  fontSize: '12px'
                }}
                formatter={(value: number, name: string) => [
                  `$${formatPrice(value)}`,
                  name === 'q90' ? '90th %ile' :
                  name === 'q75' ? '75th %ile' :
                  name === 'q50' ? 'Median' :
                  name === 'q25' ? '25th %ile' :
                  '10th %ile'
                ]}
              />
              
              <Area
                type="monotone"
                dataKey="q90"
                stroke="transparent"
                fill="url(#outerBand)"
                stackId="outer-top"
              />
              <Area
                type="monotone"
                dataKey="q10"
                stroke="transparent"
                fill="transparent"
                stackId="outer-bottom"
              />
              
              <Area
                type="monotone"
                dataKey="q75"
                stroke="transparent"
                fill="url(#innerBand)"
              />
              <Area
                type="monotone"
                dataKey="q25"
                stroke="transparent"
                fill="hsl(var(--card))"
              />
              
              <Area
                type="monotone"
                dataKey="q50"
                stroke={actionColor}
                strokeWidth={2}
                fill="transparent"
                dot={false}
              />
              
              <ReferenceLine 
                y={currentPrice} 
                stroke="hsl(var(--muted-foreground))" 
                strokeDasharray="3 3"
                strokeOpacity={0.5}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        
        <div className="flex justify-between items-center mt-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: actionColor, opacity: 0.15 }} />
              <span>10-90% range</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: actionColor, opacity: 0.3 }} />
              <span>25-75% range</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-8 h-0.5" style={{ backgroundColor: actionColor }} />
              <span>Median</span>
            </div>
          </div>
          <div>
            Horizon: {horizonBars * timeframeMinutes}m ({horizonBars} bars)
          </div>
        </div>
        
        <div className="mt-4 p-3 bg-muted/20 rounded-lg">
          <div className="text-xs font-medium text-muted-foreground mb-2">Return Distribution at Horizon</div>
          <div className="grid grid-cols-5 gap-1 text-center">
            {[
              { label: "10%", value: quantiles.q10, color: "text-red-400" },
              { label: "25%", value: quantiles.q25, color: "text-orange-400" },
              { label: "50%", value: quantiles.q50, color: "text-blue-400" },
              { label: "75%", value: quantiles.q75, color: "text-cyan-400" },
              { label: "90%", value: quantiles.q90, color: "text-emerald-400" },
            ].map(q => (
              <div key={q.label} className="bg-muted/30 rounded p-2">
                <div className="text-[10px] text-muted-foreground">{q.label}</div>
                <div className={`text-xs font-medium ${q.color}`}>
                  {q.value >= 0 ? '+' : ''}{(q.value * 100).toFixed(2)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface DerivedLevelsProps {
  currentPrice: number;
  quantiles: QuantileValues;
  action: "LONG" | "SHORT" | "HOLD";
}

export function DerivedTradeLevels({ currentPrice, quantiles: rawQuantiles, action }: DerivedLevelsProps) {
  // Sanitize quantiles to prevent absurd values from breaking the UI
  const quantiles = sanitizeQuantiles(rawQuantiles);
  
  const deriveLevels = () => {
    // Prevent division by zero in R:R calculation
    const safeDiv = (num: number, den: number) => 
      den === 0 ? 1 : Math.abs(num) / Math.abs(den);
    
    if (action === "LONG") {
      return {
        entry: currentPrice,
        stopLoss: currentPrice * (1 + quantiles.q10),
        takeProfit: currentPrice * (1 + quantiles.q90),
        riskReward: safeDiv(quantiles.q90, quantiles.q10),
        description: "SL at 10th percentile, TP at 90th percentile"
      };
    } else if (action === "SHORT") {
      return {
        entry: currentPrice,
        stopLoss: currentPrice * (1 + quantiles.q90),
        takeProfit: currentPrice * (1 + quantiles.q10),
        riskReward: safeDiv(quantiles.q10, quantiles.q90),
        description: "SL at 90th percentile, TP at 10th percentile"
      };
    } else {
      return {
        entry: currentPrice,
        stopLoss: currentPrice * (1 + quantiles.q10),
        takeProfit: currentPrice * (1 + quantiles.q90),
        riskReward: 1,
        description: "No trade recommended - showing price range"
      };
    }
  };
  
  const levels = deriveLevels();
  const formatPrice = (val: number) => 
    `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  
  const isHold = action === "HOLD";
  
  return (
    <Card data-testid="card-derived-levels">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">
          {isHold ? "Price Range (No Trade)" : "Derived Trade Levels"}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground mb-1">
              {isHold ? "Current" : "Entry"}
            </div>
            <div className="text-sm font-medium" data-testid="text-derived-entry">
              {formatPrice(levels.entry)}
            </div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className={`text-xs mb-1 ${isHold ? "text-muted-foreground" : "text-red-400"}`}>
              {isHold ? "Low (q10)" : "Stop Loss"}
            </div>
            <div className={`text-sm font-medium ${isHold ? "" : "text-red-400"}`} data-testid="text-derived-sl">
              {formatPrice(levels.stopLoss)}
            </div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className={`text-xs mb-1 ${isHold ? "text-muted-foreground" : "text-emerald-400"}`}>
              {isHold ? "High (q90)" : "Take Profit"}
            </div>
            <div className={`text-sm font-medium ${isHold ? "" : "text-emerald-400"}`} data-testid="text-derived-tp">
              {formatPrice(levels.takeProfit)}
            </div>
          </div>
        </div>
        
        {!isHold && (
          <div className="flex items-center justify-between mt-3 bg-muted/20 rounded-lg px-3 py-2">
            <span className="text-xs text-muted-foreground">Risk/Reward Ratio</span>
            <Badge 
              variant="outline" 
              className={levels.riskReward >= 2 ? "text-emerald-400" : "text-amber-400"}
            >
              1:{levels.riskReward.toFixed(2)}
            </Badge>
          </div>
        )}
        
        <p className="text-xs text-muted-foreground text-center mt-3 italic">
          {levels.description}
        </p>
      </CardContent>
    </Card>
  );
}
