import { useQuery } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import { AreaChart, Area, ResponsiveContainer } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Target,
  Wallet,
  BarChart3,
  Brain,
  Radio,
  Crosshair,
} from "lucide-react";

const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"] as const;

function MetricCard({
  label,
  value,
  glow,
  icon: Icon,
  testId,
}: {
  label: string;
  value: string;
  glow?: "green" | "red" | "cyan";
  icon: typeof Activity;
  testId: string;
}) {
  const glowClass = glow === "green" ? "glow-green" : glow === "red" ? "glow-red" : glow === "cyan" ? "glow-cyan" : "";
  return (
    <div
      className={`glass-card rounded-lg p-4 animate-fade-in-up ${glowClass}`}
      data-testid={testId}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-muted-foreground" />
        <span className="text-xs text-muted-foreground uppercase tracking-wider">{label}</span>
      </div>
      <div className="number-mono text-2xl font-bold">{value}</div>
    </div>
  );
}

function Sparkline({ data }: { data: Array<{ close: number }> }) {
  if (!data || data.length === 0) return <div className="h-12 shimmer rounded" />;
  return (
    <ResponsiveContainer width="100%" height={48}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22c55e" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey="close"
          stroke="#22c55e"
          strokeWidth={1.5}
          fill="url(#sparkGrad)"
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function MarketCard({
  symbol,
  priceData,
}: {
  symbol: string;
  priceData?: { price: number; change24h: number; high24h: number; low24h: number };
}) {
  const { data: candles } = useQuery<Array<{ close: number }>>({
    queryKey: [`/api/market/candles?symbol=${symbol}&interval=15m&limit=96`],
    refetchInterval: 15000,
  });

  const change = priceData?.change24h ?? 0;
  const isPositive = change >= 0;

  return (
    <div
      className="glass-card rounded-lg p-4 animate-fade-in-up scanline"
      data-testid={`market-card-${symbol}`}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="font-bold text-sm">{symbol.replace("USDT", "")}</span>
        <Badge
          className={`no-default-hover-elevate no-default-active-elevate text-xs ${isPositive ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}`}
          data-testid={`badge-change-${symbol}`}
        >
          {isPositive ? "+" : ""}{change.toFixed(2)}%
        </Badge>
      </div>
      <div className="number-mono text-xl font-bold mb-2" data-testid={`price-${symbol}`}>
        {priceData ? `$${priceData.price.toLocaleString()}` : "—"}
      </div>
      <Sparkline data={candles ?? []} />
    </div>
  );
}

export default function CommandCenter() {
  const { data: systemStatus, isLoading: statusLoading } = useQuery<{
    moneyConfig?: { account_equity_usd?: number };
    dailyPnl?: number;
    winRate?: number;
    profitFactor?: number;
    activePositions?: number;
    avgConfidence?: number;
  }>({
    queryKey: ["/api/system/status"],
  });

  const { data: prices } = useQuery<
    Record<string, { price: number; change24h: number; high24h: number; low24h: number }>
  >({
    queryKey: ["/api/market/prices"],
    refetchInterval: 15000,
  });

  const { data: signals, isLoading: signalsLoading } = useQuery<
    Array<{
      id?: number;
      signalTs: number;
      symbol?: string;
      direction?: string;
      confidence: number;
      score?: number;
      lane?: string;
    }>
  >({
    queryKey: ["/api/v5/signals?limit=20"],
  });

  const { data: positions } = useQuery<
    Array<{
      id?: number;
      symbol?: string;
      side?: string;
      entryPrice?: number;
      currentPrice?: number;
      pnlR?: number;
      pnl?: number;
      openedAt?: number;
      timestamp?: number;
    }>
  >({
    queryKey: ["/api/paper/positions?status=OPEN"],
  });

  const { data: equityCurve } = useQuery<Array<{ ts: number; r: number }>>({
    queryKey: ["/api/v5/equity-curve"],
  });

  const equity = systemStatus?.moneyConfig?.account_equity_usd ?? 0;
  const dailyPnl = systemStatus?.dailyPnl ?? 0;
  const winRate = systemStatus?.winRate ?? 0;
  const profitFactor = systemStatus?.profitFactor ?? 0;
  const activeCount = systemStatus?.activePositions ?? positions?.length ?? 0;
  const avgConfidence = systemStatus?.avgConfidence ?? 0;

  return (
    <div className="p-4 space-y-4" data-testid="command-center">
      <div className="grid grid-cols-6 gap-3" data-testid="metrics-bar">
        {statusLoading ? (
          Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="glass-card rounded-lg p-4 h-24 shimmer" />
          ))
        ) : (
          <>
            <MetricCard
              label="Account Equity"
              value={`$${equity.toLocaleString()}`}
              glow="cyan"
              icon={Wallet}
              testId="metric-equity"
            />
            <MetricCard
              label="Daily P&L"
              value={`${dailyPnl >= 0 ? "+" : ""}$${dailyPnl.toLocaleString()}`}
              glow={dailyPnl >= 0 ? "green" : "red"}
              icon={dailyPnl >= 0 ? TrendingUp : TrendingDown}
              testId="metric-daily-pnl"
            />
            <MetricCard
              label="Win Rate"
              value={`${(winRate * 100).toFixed(1)}%`}
              glow={winRate >= 0.5 ? "green" : "red"}
              icon={Target}
              testId="metric-win-rate"
            />
            <MetricCard
              label="Profit Factor"
              value={profitFactor.toFixed(2)}
              glow={profitFactor >= 1 ? "green" : "red"}
              icon={BarChart3}
              testId="metric-profit-factor"
            />
            <MetricCard
              label="Active Positions"
              value={String(activeCount)}
              glow={activeCount > 0 ? "cyan" : undefined}
              icon={Crosshair}
              testId="metric-active-positions"
            />
            <MetricCard
              label="Model Confidence"
              value={`${(avgConfidence * 100).toFixed(0)}%`}
              glow={avgConfidence >= 0.7 ? "green" : avgConfidence >= 0.4 ? "cyan" : "red"}
              icon={Brain}
              testId="metric-confidence"
            />
          </>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3" data-testid="market-grid">
        {SYMBOLS.map((sym) => (
          <MarketCard
            key={sym}
            symbol={sym}
            priceData={prices?.[sym]}
          />
        ))}
      </div>

      <div className="glass-card rounded-lg p-4" data-testid="signal-feed">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
          <span className="text-sm font-semibold uppercase tracking-wider">Live Signal Feed</span>
          <Radio className="w-4 h-4 text-emerald-400 ml-auto" />
        </div>

        {signalsLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-10 shimmer rounded" />
            ))}
          </div>
        ) : !signals || signals.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <div className="shimmer rounded px-6 py-3 text-sm">Awaiting signals...</div>
          </div>
        ) : (
          <div className="space-y-1">
            <div className="grid grid-cols-6 gap-2 px-2 py-1 text-xs text-muted-foreground uppercase tracking-wider">
              <span>Time</span>
              <span>Symbol</span>
              <span>Direction</span>
              <span>Confidence</span>
              <span>Score</span>
              <span>Lane</span>
            </div>
            {signals.map((sig, idx) => {
              const dir = sig.direction ?? "HOLD";
              const dirColor =
                dir === "LONG" ? "bg-emerald-500/20 text-emerald-400" :
                dir === "SHORT" ? "bg-red-500/20 text-red-400" :
                "bg-amber-500/20 text-amber-400";
              const laneColor =
                sig.lane === "CORE" ? "bg-cyan-500/20 text-cyan-400" :
                sig.lane === "FLOW" ? "bg-violet-500/20 text-violet-400" :
                "bg-muted text-muted-foreground";

              return (
                <div
                  key={sig.id ?? idx}
                  className="grid grid-cols-6 gap-2 px-2 py-2 rounded hover-elevate animate-signal-arrive items-center"
                  data-testid={`signal-row-${idx}`}
                >
                  <span className="text-xs text-muted-foreground number-mono">
                    {sig.signalTs
                      ? formatDistanceToNow(new Date(sig.signalTs), { addSuffix: true })
                      : "—"}
                  </span>
                  <Badge className="no-default-hover-elevate no-default-active-elevate text-xs w-fit bg-muted text-foreground" data-testid={`signal-symbol-${idx}`}>
                    {sig.symbol ?? "BTC"}
                  </Badge>
                  <Badge className={`no-default-hover-elevate no-default-active-elevate text-xs w-fit ${dirColor}`} data-testid={`signal-dir-${idx}`}>
                    {dir}
                  </Badge>
                  <div className="flex items-center gap-2">
                    <Progress value={sig.confidence * 100} className="h-2 flex-1" />
                    <span className="number-mono text-xs">{(sig.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <span className="number-mono text-sm" data-testid={`signal-score-${idx}`}>
                    {sig.score?.toFixed(2) ?? "—"}
                  </span>
                  <Badge className={`no-default-hover-elevate no-default-active-elevate text-xs w-fit ${laneColor}`}>
                    {sig.lane ?? "—"}
                  </Badge>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="grid grid-cols-5 gap-3" data-testid="bottom-section">
        <div className="col-span-3 glass-card rounded-lg p-4" data-testid="active-positions">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold uppercase tracking-wider">Active Positions</span>
          </div>

          {!positions || positions.length === 0 ? (
            <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
              No open positions
            </div>
          ) : (
            <div className="space-y-1">
              <div className="grid grid-cols-6 gap-2 px-2 py-1 text-xs text-muted-foreground uppercase tracking-wider">
                <span>Symbol</span>
                <span>Side</span>
                <span>Entry</span>
                <span>Current</span>
                <span>P&L (R)</span>
                <span>Duration</span>
              </div>
              {positions.map((pos, idx) => {
                const pnl = pos.pnlR ?? pos.pnl ?? 0;
                const isProfit = pnl >= 0;
                const rowBg = isProfit ? "bg-emerald-500/5" : "bg-red-500/5";

                return (
                  <div
                    key={pos.id ?? idx}
                    className={`grid grid-cols-6 gap-2 px-2 py-2 rounded ${rowBg} items-center`}
                    data-testid={`position-row-${idx}`}
                  >
                    <span className="text-sm font-medium">{pos.symbol ?? "—"}</span>
                    <Badge
                      className={`no-default-hover-elevate no-default-active-elevate text-xs w-fit ${
                        pos.side === "LONG" ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {pos.side ?? "—"}
                    </Badge>
                    <span className="number-mono text-sm">
                      ${pos.entryPrice?.toLocaleString() ?? "—"}
                    </span>
                    <span className="number-mono text-sm">
                      ${pos.currentPrice?.toLocaleString() ?? "—"}
                    </span>
                    <span
                      className={`number-mono text-sm font-medium ${isProfit ? "text-emerald-400" : "text-red-400"}`}
                    >
                      {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}R
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {pos.openedAt || pos.timestamp
                        ? formatDistanceToNow(new Date((pos.openedAt ?? pos.timestamp ?? 0) * 1000), { addSuffix: false })
                        : "—"}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="col-span-2 glass-card rounded-lg p-4" data-testid="equity-curve">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            <span className="text-sm font-semibold uppercase tracking-wider">Equity Curve</span>
          </div>

          {!equityCurve || equityCurve.length === 0 ? (
            <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
              No equity data
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={equityCurve}>
                <defs>
                  <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22c55e" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area
                  type="monotone"
                  dataKey="r"
                  stroke="#22c55e"
                  strokeWidth={2}
                  fill="url(#eqGrad)"
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}