import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Globe,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  Clock,
  ChevronRight,
  Activity,
  DollarSign,
  Zap,
  Shield,
  BarChart2,
  MessageSquare,
  Calendar,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from "date-fns";

// ─── Types ──────────────────────────────────────────────────────────────────

interface WorldIntelSnapshot {
  id: number;
  macroClimateScore: number;
  direction: string;
  confidence: number;
  heroVerdict: string | null;
  narrative: string;
  prediction24h: string | null;
  keyCatalysts: string[];
  categoryScores: Record<string, number>;
  createdAt: number;
}

interface WorldEvent {
  id: number;
  title: string;
  source: string;
  category: string;
  url: string | null;
  relevanceScore: number;
  sentiment: string;
  cryptoImpactExplanation: string;
  publishedAt: number | null;
  fetchedAt: number;
}

interface MacroIndicators {
  id: number;
  dxy: number | null;
  sp500: number | null;
  gold: number | null;
  oil: number | null;
  btcDominance: number | null;
  fearGreedIndex: number | null;
  fearGreedLabel: string | null;
  recordedAt: number;
}

interface RiskCalendarEvent {
  event: string;
  date: string;
  category: string;
  btcImpact: string;
  direction: string;
}

interface SnapshotResponse {
  snapshot: WorldIntelSnapshot | null;
  isRunning: boolean;
  lastRunAt: number | null;
  nextRunAt: number | null;
}

interface MacroResponse {
  macro: MacroIndicators | null;
  riskCalendar: RiskCalendarEvent[];
}

interface EventsResponse {
  events: WorldEvent[];
  count: number;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function sentimentColor(s: string | null | undefined) {
  if (!s) return "text-muted-foreground";
  const l = s.toLowerCase();
  if (l === "bullish") return "text-emerald-400";
  if (l === "bearish") return "text-red-400";
  return "text-yellow-400";
}

function sentimentBg(s: string | null | undefined) {
  if (!s) return "bg-muted/30";
  const l = s.toLowerCase();
  if (l === "bullish") return "bg-emerald-400/10 border-emerald-400/20";
  if (l === "bearish") return "bg-red-400/10 border-red-400/20";
  return "bg-yellow-400/10 border-yellow-400/20";
}

function SentimentIcon({ sentiment, size = 4 }: { sentiment: string; size?: number }) {
  const s = sentiment?.toLowerCase();
  const cls = `w-${size} h-${size}`;
  if (s === "bullish") return <TrendingUp className={cn(cls, "text-emerald-400")} />;
  if (s === "bearish") return <TrendingDown className={cn(cls, "text-red-400")} />;
  return <Minus className={cn(cls, "text-yellow-400")} />;
}

function directionColor(d: string) {
  const l = d?.toLowerCase();
  if (l === "bullish") return "text-emerald-400";
  if (l === "bearish") return "text-red-400";
  return "text-yellow-400";
}

function scoreGradient(score: number) {
  if (score > 40) return "from-emerald-500 to-emerald-400";
  if (score > 10) return "from-emerald-600 to-yellow-500";
  if (score > -10) return "from-yellow-500 to-yellow-400";
  if (score > -40) return "from-yellow-500 to-red-500";
  return "from-red-600 to-red-400";
}

function scoreBgColor(score: number) {
  if (score > 40) return "border-emerald-500/30 bg-emerald-500/5";
  if (score > 10) return "border-yellow-500/30 bg-yellow-500/5";
  if (score > -10) return "border-yellow-400/20 bg-yellow-400/5";
  if (score > -40) return "border-orange-500/30 bg-orange-500/5";
  return "border-red-500/30 bg-red-500/5";
}

function categoryIcon(cat: string) {
  const map: Record<string, React.ReactNode> = {
    "Monetary Policy": <DollarSign className="w-3.5 h-3.5" />,
    "Regulatory": <Shield className="w-3.5 h-3.5" />,
    "Geopolitical": <Globe className="w-3.5 h-3.5" />,
    "Markets": <BarChart2 className="w-3.5 h-3.5" />,
    "Crypto": <Zap className="w-3.5 h-3.5" />,
    "Tech": <Activity className="w-3.5 h-3.5" />,
    "Social Sentiment": <MessageSquare className="w-3.5 h-3.5" />,
    "Market Structure": <BarChart2 className="w-3.5 h-3.5" />,
    "Tech/Innovation": <Activity className="w-3.5 h-3.5" />,
  };
  return map[cat] ?? <Globe className="w-3.5 h-3.5" />;
}

// ─── Macro Climate Gauge ─────────────────────────────────────────────────────

function MacroGauge({ score }: { score: number }) {
  const normalized = clamp((score + 100) / 200, 0, 1);
  const angle = normalized * 180 - 90;
  const radius = 80;
  const cx = 100;
  const cy = 100;

  const arcPath = (startAngle: number, endAngle: number, r: number) => {
    const s = ((startAngle - 90) * Math.PI) / 180;
    const e = ((endAngle - 90) * Math.PI) / 180;
    const x1 = cx + r * Math.cos(s);
    const y1 = cy + r * Math.sin(s);
    const x2 = cx + r * Math.cos(e);
    const y2 = cy + r * Math.sin(e);
    const large = endAngle - startAngle > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };

  const needleRad = ((angle - 90) * Math.PI) / 180;
  const needleX = cx + 65 * Math.cos(needleRad);
  const needleY = cy + 65 * Math.sin(needleRad);

  return (
    <div className="flex flex-col items-center gap-2" data-testid="macro-gauge">
      <svg viewBox="0 0 200 120" className="w-52 h-36">
        <path d={arcPath(0, 60, radius)} stroke="#ef4444" strokeWidth="10" fill="none" strokeLinecap="round" opacity="0.7" />
        <path d={arcPath(60, 120, radius)} stroke="#eab308" strokeWidth="10" fill="none" strokeLinecap="round" opacity="0.7" />
        <path d={arcPath(120, 180, radius)} stroke="#22c55e" strokeWidth="10" fill="none" strokeLinecap="round" opacity="0.7" />
        <line x1={cx} y1={cy} x2={needleX} y2={needleY} stroke="white" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="5" fill="white" />
        <text x={cx} y={115} textAnchor="middle" fill="white" fontSize="22" fontWeight="bold" className="font-mono">
          {score > 0 ? "+" : ""}{Math.round(score)}
        </text>
        <text x="20" y="108" fill="#ef4444" fontSize="9" opacity="0.8">BEARISH</text>
        <text x="160" y="108" fill="#22c55e" fontSize="9" opacity="0.8">BULLISH</text>
      </svg>
    </div>
  );
}

// ─── Countdown Timer ─────────────────────────────────────────────────────────

function CountdownTimer({ nextRunAt }: { nextRunAt: number | null }) {
  const [remaining, setRemaining] = useState<string>("");

  useEffect(() => {
    if (!nextRunAt) { setRemaining(""); return; }
    const update = () => {
      const diff = nextRunAt - Date.now();
      if (diff <= 0) { setRemaining("Refreshing..."); return; }
      const m = Math.floor(diff / 60000);
      const s = Math.floor((diff % 60000) / 1000);
      setRemaining(`${m}m ${s.toString().padStart(2, "0")}s`);
    };
    update();
    const t = setInterval(update, 1000);
    return () => clearInterval(t);
  }, [nextRunAt]);

  if (!remaining) return null;
  return (
    <div className="flex items-center gap-1.5 text-xs text-muted-foreground" data-testid="countdown-timer">
      <Clock className="w-3 h-3" />
      <span>Next refresh in <span className="font-mono text-foreground">{remaining}</span></span>
    </div>
  );
}

// ─── Category Scores Bar ─────────────────────────────────────────────────────

function CategoryBar({ label, score }: { label: string; score: number }) {
  const pct = clamp((score + 100) / 2, 0, 100);
  const color = score > 20 ? "bg-emerald-500" : score < -20 ? "bg-red-500" : "bg-yellow-500";
  return (
    <div className="space-y-1" data-testid={`category-bar-${label.toLowerCase().replace(/\s+/g, "-")}`}>
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 text-muted-foreground">
          {categoryIcon(label)}
          <span>{label}</span>
        </div>
        <span className={cn("font-mono font-semibold text-[11px]", score > 0 ? "text-emerald-400" : score < 0 ? "text-red-400" : "text-yellow-400")}>
          {score > 0 ? "+" : ""}{Math.round(score)}
        </span>
      </div>
      <div className="h-1.5 bg-muted/30 rounded-full overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-700", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ─── Macro Indicator Card ────────────────────────────────────────────────────

function MacroCard({ label, value, unit, impact, icon, trend }: {
  label: string;
  value: number | null;
  unit?: string;
  impact?: string;
  icon?: React.ReactNode;
  trend?: "up" | "down" | "neutral";
}) {
  return (
    <div className="flex flex-col gap-1 p-3 rounded-lg bg-card border border-border" data-testid={`macro-card-${label.toLowerCase()}`}>
      <div className="flex items-center justify-between gap-1">
        <div className="flex items-center gap-1.5 text-muted-foreground text-xs">
          {icon}
          <span>{label}</span>
        </div>
        {trend === "up" && <TrendingUp className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
        {trend === "down" && <TrendingDown className="w-3.5 h-3.5 text-red-400 shrink-0" />}
        {trend === "neutral" && <Minus className="w-3.5 h-3.5 text-yellow-400 shrink-0" />}
      </div>
      <div className={cn("font-mono text-lg font-bold",
        trend === "up" ? "text-emerald-400" : trend === "down" ? "text-red-400" : "text-foreground"
      )}>
        {value !== null ? `${unit ?? ""}${value.toLocaleString()}` : <span className="text-muted-foreground text-sm">—</span>}
      </div>
      {impact && <div className="text-[10px] text-muted-foreground leading-tight">{impact}</div>}
    </div>
  );
}

// ─── Fear & Greed Card ───────────────────────────────────────────────────────

function FearGreedCard({ index, label }: { index: number | null; label: string | null }) {
  const val = index ?? 50;
  const color = val < 25 ? "text-red-400" : val < 45 ? "text-orange-400" : val < 55 ? "text-yellow-400" : val < 75 ? "text-emerald-400" : "text-emerald-300";
  const bg = val < 25 ? "border-red-500/30 bg-red-500/5" : val < 45 ? "border-orange-500/30 bg-orange-500/5" : val < 55 ? "border-yellow-400/20" : "border-emerald-500/30 bg-emerald-500/5";
  const impact = val < 25 ? "Extreme fear → historically bullish reversal signal" :
    val < 45 ? "Fear zone → accumulation opportunity" :
    val < 55 ? "Neutral — no directional signal" :
    val < 75 ? "Greed zone → momentum intact" :
    "Extreme greed → distribution risk, watch funding rates";

  return (
    <div className={cn("flex flex-col gap-1 p-3 rounded-lg border", bg)} data-testid="fear-greed-card">
      <div className="flex items-center gap-1.5 text-muted-foreground text-xs">
        <Activity className="w-3.5 h-3.5" />
        <span>Fear & Greed</span>
      </div>
      <div className={cn("font-mono text-2xl font-bold", color)}>
        {index !== null ? index : "—"}
      </div>
      {label && <Badge variant="outline" className="w-fit text-[10px] py-0">{label}</Badge>}
      <div className="text-[10px] text-muted-foreground leading-tight mt-0.5">{impact}</div>
    </div>
  );
}

// ─── Event Card ───────────────────────────────────────────────────────────────

const CATEGORY_COLORS: Record<string, string> = {
  "Monetary Policy": "border-blue-500/30 text-blue-400 bg-blue-500/10",
  "Regulatory": "border-purple-500/30 text-purple-400 bg-purple-500/10",
  "Geopolitical": "border-orange-500/30 text-orange-400 bg-orange-500/10",
  "Markets": "border-cyan-500/30 text-cyan-400 bg-cyan-500/10",
  "Crypto": "border-yellow-500/30 text-yellow-400 bg-yellow-500/10",
  "Tech": "border-emerald-500/30 text-emerald-400 bg-emerald-500/10",
};

function EventCard({ event }: { event: WorldEvent }) {
  const catColor = CATEGORY_COLORS[event.category] ?? "border-muted text-muted-foreground bg-muted/10";
  const rel = Math.round(event.relevanceScore ?? 0);

  return (
    <div
      className={cn("p-3 rounded-lg border transition-colors hover:bg-muted/5", sentimentBg(event.sentiment))}
      data-testid={`event-card-${event.id}`}
    >
      <div className="flex items-start gap-2.5">
        <div className="mt-0.5 shrink-0">
          <SentimentIcon sentiment={event.sentiment} size={4} />
        </div>
        <div className="flex-1 min-w-0 space-y-1">
          <div className="flex items-start justify-between gap-2">
            <p className="text-sm font-medium text-foreground leading-snug line-clamp-2">
              {event.url ? (
                <a href={event.url} target="_blank" rel="noopener noreferrer" className="hover:underline">
                  {event.title}
                </a>
              ) : event.title}
            </p>
            <div className="shrink-0 flex flex-col items-end gap-1">
              <div className="flex">
                {Array.from({ length: 10 }).map((_, i) => (
                  <div
                    key={i}
                    className={cn("w-1.5 h-3 rounded-sm mx-[1px]", i < rel ? "bg-primary" : "bg-muted/30")}
                  />
                ))}
              </div>
              <span className="text-[9px] text-muted-foreground">{rel}/10</span>
            </div>
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            <Badge className={cn("text-[10px] py-0 px-1.5 border", catColor)}>{event.category}</Badge>
            {event.source && <span className="text-[10px] text-muted-foreground">{event.source}</span>}
            {event.fetchedAt && (
              <span className="text-[10px] text-muted-foreground">
                {formatDistanceToNow(new Date(event.fetchedAt), { addSuffix: true })}
              </span>
            )}
          </div>
          {event.cryptoImpactExplanation && (
            <p className="text-xs text-muted-foreground leading-relaxed border-l-2 border-primary/30 pl-2 mt-1">
              <span className="text-primary font-medium">Crypto impact: </span>
              {event.cryptoImpactExplanation}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Risk Calendar ───────────────────────────────────────────────────────────

function RiskCalendar({ events }: { events: RiskCalendarEvent[] }) {
  return (
    <div className="space-y-2">
      {events.map((ev, i) => {
        const daysUntil = Math.ceil((new Date(ev.date).getTime() - Date.now()) / 86400000);
        const isNear = daysUntil <= 7;
        return (
          <div key={i} className={cn("flex items-center gap-3 p-2.5 rounded-lg border", isNear ? "border-yellow-500/30 bg-yellow-500/5" : "border-border bg-card")} data-testid={`risk-event-${i}`}>
            <div className="text-center shrink-0 w-10">
              <div className={cn("text-xs font-mono font-bold", isNear ? "text-yellow-400" : "text-muted-foreground")}>
                {daysUntil > 0 ? `${daysUntil}d` : "Today"}
              </div>
              <div className="text-[9px] text-muted-foreground">{new Date(ev.date).toLocaleDateString("en", { month: "short", day: "numeric" })}</div>
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-foreground truncate">{ev.event}</div>
              <div className="text-[10px] text-muted-foreground">{ev.btcImpact}</div>
            </div>
            <Badge variant="outline" className={cn("text-[10px] shrink-0", ev.direction === "bullish" ? "border-emerald-500/30 text-emerald-400" : ev.direction === "bearish" ? "border-red-500/30 text-red-400" : "border-yellow-500/30 text-yellow-400")}>
              {ev.direction}
            </Badge>
          </div>
        );
      })}
    </div>
  );
}

// ─── Skeleton Loader ─────────────────────────────────────────────────────────

function DashboardSkeleton() {
  return (
    <div className="space-y-6 p-6">
      <Skeleton className="h-8 w-64" />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Skeleton className="h-64 col-span-1" />
        <Skeleton className="h-64 col-span-2" />
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />)}
      </div>
    </div>
  );
}

// ─── Empty State ──────────────────────────────────────────────────────────────

function EmptyState({ onRefresh, isLoading }: { onRefresh: () => void; isLoading: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4 text-center">
      <Globe className="w-16 h-16 text-muted-foreground/30" />
      <div>
        <p className="text-lg font-semibold text-foreground">No intelligence data yet</p>
        <p className="text-sm text-muted-foreground mt-1">The world intelligence engine hasn't run its first cycle yet.</p>
        <p className="text-xs text-muted-foreground mt-0.5">Trigger a manual refresh to fetch the latest global signals.</p>
      </div>
      <Button onClick={onRefresh} disabled={isLoading} data-testid="button-trigger-refresh">
        {isLoading ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Globe className="w-4 h-4 mr-2" />}
        {isLoading ? "Analyzing..." : "Run First Analysis"}
      </Button>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function WorldIntelligence() {
  const qc = useQueryClient();
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const snapshotQ = useQuery<SnapshotResponse>({
    queryKey: ["/api/world-intel/snapshot"],
    refetchInterval: 60_000,
  });

  const eventsQ = useQuery<EventsResponse>({
    queryKey: ["/api/world-intel/events", selectedCategory],
    queryFn: () => {
      const base = selectedCategory
        ? `/api/world-intel/events?limit=20&sort=relevance&category=${encodeURIComponent(selectedCategory)}`
        : "/api/world-intel/events?limit=20&sort=relevance";
      return fetch(base).then(r => r.json());
    },
    refetchInterval: 60_000,
  });

  const macroQ = useQuery<MacroResponse>({
    queryKey: ["/api/world-intel/macro"],
    refetchInterval: 120_000,
  });

  const refreshMut = useMutation({
    mutationFn: () => apiRequest("POST", "/api/world-intel/refresh"),
    onSuccess: () => {
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ["/api/world-intel/snapshot"] });
        qc.invalidateQueries({ queryKey: ["/api/world-intel/events"] });
        qc.invalidateQueries({ queryKey: ["/api/world-intel/macro"] });
      }, 3000);
    },
  });

  const snapshot = snapshotQ.data?.snapshot;
  const macro = macroQ.data?.macro;
  const events = eventsQ.data?.events ?? [];
  const riskCalendar = macroQ.data?.riskCalendar ?? [];
  const isRunning = snapshotQ.data?.isRunning ?? refreshMut.isPending;
  const lastRunAt = snapshotQ.data?.lastRunAt ?? null;
  const nextRunAt = snapshotQ.data?.nextRunAt ?? null;

  const isLoading = snapshotQ.isLoading && eventsQ.isLoading;

  const categoryScores = snapshot?.categoryScores ?? {};
  const CATEGORIES = ["Monetary Policy", "Regulatory", "Geopolitical", "Market Structure", "Tech/Innovation", "Social Sentiment"];
  const EVENT_CATEGORIES = Array.from(new Set(events.map(e => e.category))).filter(Boolean);

  if (isLoading) return <DashboardSkeleton />;

  const score = snapshot?.macroClimateScore ?? 0;
  const direction = snapshot?.direction ?? "Neutral";
  const confidence = snapshot?.confidence ?? 0;
  const heroVerdict = snapshot?.heroVerdict;
  const narrative = snapshot?.narrative;
  const prediction24h = snapshot?.prediction24h;
  const keyCatalysts = snapshot?.keyCatalysts ?? [];

  return (
    <div className="min-h-screen bg-background" data-testid="world-intelligence-page">
      {/* ─── Header ─────────────────────────────────────────────────── */}
      <div className="border-b border-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
              <Globe className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-foreground tracking-tight">World Intelligence</h1>
              <p className="text-xs text-muted-foreground">Macro Oracle — Global signals → Crypto impact</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <CountdownTimer nextRunAt={nextRunAt} />
            {lastRunAt && (
              <span className="text-xs text-muted-foreground hidden sm:block" data-testid="text-last-updated">
                Updated {formatDistanceToNow(new Date(lastRunAt), { addSuffix: true })}
              </span>
            )}
            <Button
              size="sm"
              variant="outline"
              onClick={() => refreshMut.mutate()}
              disabled={isRunning || refreshMut.isPending}
              data-testid="button-manual-refresh"
            >
              <RefreshCw className={cn("w-3.5 h-3.5 mr-1.5", (isRunning || refreshMut.isPending) && "animate-spin")} />
              {isRunning || refreshMut.isPending ? "Analyzing..." : "Refresh"}
            </Button>
          </div>
        </div>
      </div>

      {/* ─── Empty State ──────────────────────────────────────────── */}
      {!snapshot && !isLoading && (
        <EmptyState
          onRefresh={() => refreshMut.mutate()}
          isLoading={isRunning || refreshMut.isPending}
        />
      )}

      {/* ─── Dashboard ────────────────────────────────────────────── */}
      {snapshot && (
        <div className="p-6 space-y-5">
          {/* Hero Verdict Banner */}
          {heroVerdict && (
            <div className={cn("rounded-lg border px-4 py-3 flex items-start gap-3", scoreBgColor(score))} data-testid="hero-verdict-banner">
              <div className="shrink-0 mt-0.5">
                <SentimentIcon sentiment={direction} size={5} />
              </div>
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-0.5">AI Verdict</p>
                <p className={cn("text-sm font-medium leading-snug", directionColor(direction))} data-testid="text-hero-verdict">
                  {heroVerdict}
                </p>
              </div>
            </div>
          )}

          {/* Row 1: Hero + AI Narrative */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            {/* Macro Climate Score Gauge */}
            <Card className={cn("border col-span-2", scoreBgColor(score))} data-testid="card-macro-climate">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                  <Activity className="w-4 h-4" />
                  Global Macro Climate
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <MacroGauge score={score} />
                <div className="text-center space-y-1">
                  <div className={cn("text-2xl font-bold tracking-tight", directionColor(direction))} data-testid="text-direction">
                    {direction}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    <span className="font-mono font-semibold text-foreground">{confidence.toFixed(0)}%</span> confidence
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* AI Narrative + Key Catalysts */}
            <Card className="col-span-3 border border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                  <Zap className="w-4 h-4 text-primary" />
                  AI Macro Narrative
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {narrative ? (
                  <p className="text-sm text-foreground leading-relaxed" data-testid="text-narrative">
                    {narrative}
                  </p>
                ) : (
                  <p className="text-sm text-muted-foreground italic">AI narrative unavailable.</p>
                )}

                {keyCatalysts.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Key Catalysts</p>
                    <ul className="space-y-1.5">
                      {keyCatalysts.map((c, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm" data-testid={`catalyst-${i}`}>
                          <ChevronRight className="w-3.5 h-3.5 text-primary mt-0.5 shrink-0" />
                          <span>{c}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* 24-Hour Prediction Card */}
          {prediction24h && (
            <Card className="border border-primary/20 bg-primary/5" data-testid="card-prediction-24h">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-between gap-1.5">
                  <div className="flex items-center gap-1.5">
                    <Clock className="w-4 h-4 text-primary" />
                    24-Hour Crypto Prediction
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={cn("text-xs font-semibold", directionColor(direction))}>
                      {direction}
                    </span>
                    <span className="text-xs text-muted-foreground">{confidence.toFixed(0)}% conf.</span>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-foreground leading-relaxed" data-testid="text-prediction-24h">
                  {prediction24h}
                </p>
                {keyCatalysts.length > 0 && (
                  <div className="border-t border-border/50 pt-3 space-y-1.5">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Primary Catalysts</p>
                    {keyCatalysts.slice(0, 3).map((c, i) => (
                      <div key={i} className="flex items-start gap-2 text-xs" data-testid={`prediction-catalyst-${i}`}>
                        <ChevronRight className="w-3 h-3 text-primary mt-0.5 shrink-0" />
                        <span className="text-muted-foreground">{c}</span>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Row 2: Macro Indicators */}
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Live Macro Indicators</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              <FearGreedCard index={macro?.fearGreedIndex ?? null} label={macro?.fearGreedLabel ?? null} />
              <MacroCard
                label="DXY"
                value={macro?.dxy ?? null}
                trend={macro?.dxy ? (macro.dxy > 105 ? "down" : macro.dxy < 100 ? "up" : "neutral") : undefined}
                impact={macro?.dxy ? (macro.dxy > 105 ? "Risk-off pressure on crypto" : macro.dxy < 100 ? "Risk-on, positive for crypto" : "Neutral DXY reading") : undefined}
                icon={<DollarSign className="w-3.5 h-3.5" />}
              />
              <MacroCard
                label="S&P 500"
                value={macro?.sp500 ?? null}
                trend={macro?.sp500 ? (macro.sp500 > 5000 ? "up" : macro.sp500 < 4000 ? "down" : "neutral") : undefined}
                impact="Crypto typically follows with 1.2-1.5x leverage"
                icon={<BarChart2 className="w-3.5 h-3.5" />}
              />
              <MacroCard
                label="Gold"
                value={macro?.gold ?? null}
                unit="$"
                trend={macro?.gold ? (macro.gold > 2000 ? "up" : "neutral") : undefined}
                impact="Safe-haven signal — BTC correlation rising"
                icon={<Activity className="w-3.5 h-3.5" />}
              />
              <MacroCard
                label="Oil (WTI)"
                value={macro?.oil ?? null}
                unit="$"
                trend={macro?.oil ? (macro.oil > 85 ? "down" : macro.oil < 70 ? "up" : "neutral") : undefined}
                impact="Inflation proxy — high oil → Fed hawkish risk"
                icon={<Activity className="w-3.5 h-3.5" />}
              />
              <MacroCard
                label="BTC Dominance"
                value={macro?.btcDominance ?? null}
                trend={macro?.btcDominance ? (macro.btcDominance > 55 ? "down" : macro.btcDominance < 45 ? "up" : "neutral") : undefined}
                impact={macro?.btcDominance ? (macro.btcDominance > 55 ? "Capital in BTC — altcoin pressure" : "Alt season conditions building") : undefined}
                icon={<Zap className="w-3.5 h-3.5" />}
              />
            </div>
          </div>

          {/* Row 3: Category Scores + Risk Calendar */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Category Intelligence Bars */}
            <Card className="col-span-2 border border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                  <BarChart2 className="w-4 h-4" />
                  Category Intelligence
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3.5">
                {CATEGORIES.map(cat => (
                  <CategoryBar
                    key={cat}
                    label={cat}
                    score={categoryScores[cat] ?? 0}
                  />
                ))}
              </CardContent>
            </Card>

            {/* Risk Calendar */}
            <Card className="border border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5">
                  <Calendar className="w-4 h-4" />
                  Risk Calendar
                </CardTitle>
              </CardHeader>
              <CardContent>
                {riskCalendar.length > 0 ? (
                  <RiskCalendar events={riskCalendar} />
                ) : (
                  <p className="text-xs text-muted-foreground text-center py-4">No upcoming events</p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Row 4: World Events Feed */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5" />
                Live World Events Feed
                {events.length > 0 && (
                  <span className="text-[10px] font-mono text-primary ml-1">({events.length})</span>
                )}
              </h2>

              {/* Category Filter */}
              <div className="flex items-center gap-1.5 flex-wrap">
                <button
                  onClick={() => setSelectedCategory(null)}
                  className={cn("text-[10px] px-2 py-0.5 rounded-full border transition-colors", selectedCategory === null ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground hover:border-foreground/30")}
                  data-testid="filter-all"
                >
                  All
                </button>
                {EVENT_CATEGORIES.map(cat => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat === selectedCategory ? null : cat)}
                    className={cn("text-[10px] px-2 py-0.5 rounded-full border transition-colors", selectedCategory === cat ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground hover:border-foreground/30")}
                    data-testid={`filter-${cat.toLowerCase().replace(/\s+/g, "-")}`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            {eventsQ.isLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
              </div>
            ) : events.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <Globe className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p className="text-sm">No events in the last 6 hours.</p>
                <p className="text-xs mt-1">Trigger a refresh to fetch the latest global signals.</p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1" data-testid="events-feed">
                {events.map(ev => (
                  <EventCard key={ev.id} event={ev} />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
