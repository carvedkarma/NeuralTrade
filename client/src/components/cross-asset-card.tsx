import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, TrendingDown, Minus, Activity, BarChart3, Link2, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend, AreaChart, Area } from "recharts";

export interface CrossAssetData {
  symbols: Array<{
    symbol: string;
    price: number;
    change24h: number;
    volume24h: number;
    lastUpdate: number;
  }>;
  correlations: Array<{
    pair: string;
    correlation20: number;
    correlation60: number;
    leadLag: string;
  }>;
  marketMomentum: {
    allUp: boolean;
    allDown: boolean;
    mixed: boolean;
    avgChange: number;
    btcDominance: number;
  };
  relativeStrength: Array<{
    symbol: string;
    rsVsBtc: number;
  }>;
  priceHistory: Array<{
    timestamp: number;
    BTC: number;
    ETH: number;
    SOL: number;
    BNB: number;
  }>;
}

interface CrossAssetCardProps {
  data: CrossAssetData | null | undefined;
}

export function CrossAssetOverviewCard({ data }: CrossAssetCardProps) {
  if (!data) {
    return (
      <Card data-testid="card-cross-asset-offline">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-muted-foreground" />
            Cross-Asset Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <Activity className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p className="font-medium">No Cross-Asset Data</p>
            <p className="text-sm mt-1">Fetch ETH, SOL, BNB data using the GPU Trainer GUI</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getTrendIcon = (change: number) => {
    if (change > 0.5) return <TrendingUp className="h-4 w-4 text-emerald-400" />;
    if (change < -0.5) return <TrendingDown className="h-4 w-4 text-red-400" />;
    return <Minus className="h-4 w-4 text-yellow-400" />;
  };

  return (
    <Card data-testid="card-cross-asset-overview">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center justify-between">
          <span className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            Cross-Asset Overview
          </span>
          <Badge variant="outline" className="text-xs">
            {data.symbols.length} Symbols
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {data.symbols.map((s) => (
            <div 
              key={s.symbol} 
              className="p-3 rounded-lg bg-muted/50 border border-border"
              data-testid={`cross-asset-symbol-${s.symbol}`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium text-sm">{s.symbol.replace("USDT", "")}</span>
                {getTrendIcon(s.change24h)}
              </div>
              <div className="text-lg font-bold">
                ${s.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
              <div className={`text-xs ${s.change24h >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                {s.change24h >= 0 ? "+" : ""}{s.change24h.toFixed(2)}%
              </div>
            </div>
          ))}
        </div>

        <div className="p-3 rounded-lg bg-muted/30 border border-border">
          <div className="text-sm font-medium mb-2 flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Market Momentum
          </div>
          <div className="flex items-center gap-4 flex-wrap">
            <Badge 
              variant={data.marketMomentum.allUp ? "default" : data.marketMomentum.allDown ? "destructive" : "secondary"}
              className="text-xs"
            >
              {data.marketMomentum.allUp ? "All Up" : data.marketMomentum.allDown ? "All Down" : "Mixed"}
            </Badge>
            <span className="text-sm">
              Avg Change: <span className={data.marketMomentum.avgChange >= 0 ? "text-emerald-400" : "text-red-400"}>
                {data.marketMomentum.avgChange >= 0 ? "+" : ""}{data.marketMomentum.avgChange.toFixed(2)}%
              </span>
            </span>
            <span className="text-sm text-muted-foreground">
              BTC Dominance: {data.marketMomentum.btcDominance.toFixed(1)}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function CorrelationMatrixCard({ data }: CrossAssetCardProps) {
  if (!data || !data.correlations.length) {
    return (
      <Card data-testid="card-correlation-offline">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <Link2 className="h-5 w-5 text-muted-foreground" />
            Correlation Matrix
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-muted-foreground">
            <p className="text-sm">No correlation data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getCorrelationColor = (corr: number) => {
    if (corr > 0.7) return "text-emerald-400";
    if (corr > 0.4) return "text-yellow-400";
    if (corr > 0) return "text-orange-400";
    return "text-red-400";
  };

  const getCorrelationBg = (corr: number) => {
    if (corr > 0.7) return "bg-emerald-500/20";
    if (corr > 0.4) return "bg-yellow-500/20";
    if (corr > 0) return "bg-orange-500/20";
    return "bg-red-500/20";
  };

  return (
    <Card data-testid="card-correlation-matrix">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <Link2 className="h-5 w-5 text-primary" />
          Correlation Matrix (vs BTC)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {data.correlations.map((c) => (
            <div 
              key={c.pair}
              className="flex items-center justify-between p-2 rounded-lg bg-muted/30"
              data-testid={`correlation-${c.pair}`}
            >
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm w-24">{c.pair}</span>
                <Badge variant="outline" className="text-xs">
                  {c.leadLag}
                </Badge>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-xs text-muted-foreground">20-period</div>
                  <div className={`font-mono text-sm ${getCorrelationColor(c.correlation20)}`}>
                    {c.correlation20.toFixed(3)}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-muted-foreground">60-period</div>
                  <div className={`font-mono text-sm ${getCorrelationColor(c.correlation60)}`}>
                    {c.correlation60.toFixed(3)}
                  </div>
                </div>
                <div className={`w-16 h-2 rounded-full ${getCorrelationBg(c.correlation20)}`}>
                  <div 
                    className={`h-full rounded-full ${c.correlation20 > 0.5 ? "bg-emerald-500" : "bg-yellow-500"}`}
                    style={{ width: `${Math.abs(c.correlation20) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function RelativeStrengthCard({ data }: CrossAssetCardProps) {
  if (!data || !data.relativeStrength.length) {
    return (
      <Card data-testid="card-rs-offline">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-muted-foreground" />
            Relative Strength
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-muted-foreground">
            <p className="text-sm">No relative strength data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="card-relative-strength">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          Relative Strength vs BTC
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {data.relativeStrength.map((rs) => (
            <div 
              key={rs.symbol}
              className="flex items-center justify-between"
              data-testid={`rs-${rs.symbol}`}
            >
              <span className="font-medium text-sm">{rs.symbol.replace("USDT", "")}</span>
              <div className="flex items-center gap-2 flex-1 mx-4">
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${rs.rsVsBtc > 0 ? "bg-emerald-500" : "bg-red-500"}`}
                    style={{ 
                      width: `${Math.min(Math.abs(rs.rsVsBtc) * 10, 100)}%`,
                      marginLeft: rs.rsVsBtc < 0 ? "auto" : 0,
                      marginRight: rs.rsVsBtc > 0 ? "auto" : 0
                    }}
                  />
                </div>
              </div>
              <div className="flex items-center gap-1">
                {rs.rsVsBtc > 0 ? (
                  <ArrowUpRight className="h-4 w-4 text-emerald-400" />
                ) : (
                  <ArrowDownRight className="h-4 w-4 text-red-400" />
                )}
                <span className={`font-mono text-sm ${rs.rsVsBtc > 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {rs.rsVsBtc > 0 ? "+" : ""}{rs.rsVsBtc.toFixed(2)}%
                </span>
              </div>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          Positive = outperforming BTC | Negative = underperforming BTC
        </p>
      </CardContent>
    </Card>
  );
}

export function PriceComparisonChart({ data }: CrossAssetCardProps) {
  if (!data || !data.priceHistory.length) {
    return (
      <Card data-testid="card-price-comparison-offline">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-muted-foreground" />
            Normalized Price Chart
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[200px] flex items-center justify-center text-muted-foreground">
            <p className="text-sm">No price history available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const normalizedData = data.priceHistory.map((point, idx) => {
    const first = data.priceHistory[0];
    return {
      time: new Date(point.timestamp).toLocaleTimeString(),
      BTC: ((point.BTC / first.BTC) - 1) * 100,
      ETH: ((point.ETH / first.ETH) - 1) * 100,
      SOL: ((point.SOL / first.SOL) - 1) * 100,
      BNB: ((point.BNB / first.BNB) - 1) * 100,
    };
  });

  return (
    <Card data-testid="card-price-comparison">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary" />
          Normalized Price Change (%)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={normalizedData}>
              <XAxis 
                dataKey="time" 
                stroke="#666"
                tick={{ fill: '#888', fontSize: 10 }}
                tickLine={false}
              />
              <YAxis 
                stroke="#666"
                tick={{ fill: '#888', fontSize: 10 }}
                tickLine={false}
                tickFormatter={(v) => `${v.toFixed(1)}%`}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))', 
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px'
                }}
                formatter={(value: number) => [`${value.toFixed(2)}%`, '']}
              />
              <Legend />
              <Line type="monotone" dataKey="BTC" stroke="#f7931a" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="ETH" stroke="#627eea" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="SOL" stroke="#00ffa3" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="BNB" stroke="#f0b90b" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
