import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Minus, Activity } from "lucide-react";
import type { TechnicalIndicator } from "@shared/schema";

interface IndicatorsCardProps {
  indicators?: {
    rsi: TechnicalIndicator;
    macd: TechnicalIndicator;
    bollingerBands: TechnicalIndicator;
    obv: TechnicalIndicator;
    vwap: TechnicalIndicator;
    atr: TechnicalIndicator;
    adx: TechnicalIndicator;
    stochastic: TechnicalIndicator;
  };
}

export function IndicatorsCard({ indicators }: IndicatorsCardProps) {
  if (!indicators) {
    return (
      <Card className="overflow-visible" data-testid="card-indicators">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <CardTitle className="text-sm font-medium">Technical Indicators</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4 text-muted-foreground text-sm">
            Loading indicators...
          </div>
        </CardContent>
      </Card>
    );
  }

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case "bullish": return <TrendingUp className="w-3 h-3 text-emerald-400" />;
      case "bearish": return <TrendingDown className="w-3 h-3 text-red-400" />;
      default: return <Minus className="w-3 h-3 text-yellow-400" />;
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case "bullish": return "text-emerald-400";
      case "bearish": return "text-red-400";
      default: return "text-yellow-400";
    }
  };

  const formatValue = (name: string, value: number) => {
    if (name === "VWAP" || name === "ATR") return `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
    if (name === "OBV") return value > 1e9 ? `${(value / 1e9).toFixed(1)}B` : value > 1e6 ? `${(value / 1e6).toFixed(1)}M` : value.toFixed(0);
    return value.toFixed(1);
  };

  const indicatorList = [
    { key: "rsi", label: "RSI (14)", data: indicators.rsi },
    { key: "macd", label: "MACD", data: indicators.macd },
    { key: "stochastic", label: "Stochastic", data: indicators.stochastic },
    { key: "bollingerBands", label: "Bollinger %B", data: indicators.bollingerBands },
    { key: "adx", label: "ADX", data: indicators.adx },
    { key: "obv", label: "OBV", data: indicators.obv },
    { key: "vwap", label: "VWAP", data: indicators.vwap },
    { key: "atr", label: "ATR (14)", data: indicators.atr },
  ];

  const bullishCount = indicatorList.filter(i => i.data.signal === "bullish").length;
  const bearishCount = indicatorList.filter(i => i.data.signal === "bearish").length;

  return (
    <Card className="overflow-visible" data-testid="card-indicators">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <CardTitle className="text-sm font-medium">Technical Indicators</CardTitle>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <Badge variant="secondary" className="bg-emerald-500/20 text-emerald-400">
              {bullishCount} Bull
            </Badge>
            <Badge variant="secondary" className="bg-red-500/20 text-red-400">
              {bearishCount} Bear
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-2">
          {indicatorList.map(({ key, label, data }) => (
            <div 
              key={key} 
              className="p-2 rounded-md bg-muted/50 border border-border"
              data-testid={`indicator-${key}`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-muted-foreground">{label}</span>
                {getSignalIcon(data.signal)}
              </div>
              <div className="flex items-center justify-between">
                <span className={`text-sm font-mono font-medium ${getSignalColor(data.signal)}`}>
                  {formatValue(data.name, data.value)}
                </span>
                <div 
                  className="w-12 h-1.5 bg-muted rounded-full overflow-hidden"
                  title={`Strength: ${(data.strength * 100).toFixed(0)}%`}
                >
                  <div 
                    className={`h-full transition-all ${
                      data.signal === "bullish" ? "bg-emerald-500" : 
                      data.signal === "bearish" ? "bg-red-500" : "bg-yellow-500"
                    }`}
                    style={{ width: `${data.strength * 100}%` }}
                  />
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground mt-1 truncate" title={data.description}>
                {data.description}
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
