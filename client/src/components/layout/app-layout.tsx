import { useLocation, Link } from "wouter";
import {
  LayoutDashboard,
  Activity,
  FileText,
  BarChart3,
  Settings,
  Zap,
  Moon,
  Sun,
  ChevronLeft,
  Radio,
  Brain,
  ClipboardList,
} from "lucide-react";
import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useTradingWs } from "@/hooks/use-trading-ws";
import { useToast } from "@/hooks/use-toast";
import { queryClient } from "@/lib/queryClient";
import RiskAlertBar from "@/components/risk-alert-bar";

const NAV_ITEMS = [
  { path: "/", label: "Command Center", icon: LayoutDashboard },
  { path: "/live", label: "Live Trading", icon: Activity },
  { path: "/paper", label: "Paper Trading", icon: FileText },
  { path: "/trade-history", label: "Trade History", icon: ClipboardList },
  { path: "/analytics", label: "Analytics", icon: BarChart3 },
  { path: "/training", label: "Training Monitor", icon: Brain },
  { path: "/settings", label: "Settings", icon: Settings },
];

interface SystemStatusData {
  gpu: { isAvailable: boolean; latencyMs: number; lastActivity: number | null };
  paper: { portfolioExists: boolean; openPositions: number };
  cyclesToday: number;
  lastCycleTs: number | null;
  lastSignal: { signalTs: number } | null;
}

function useSystemStatus() {
  return useQuery<SystemStatusData>({
    queryKey: ["/api/system/status"],
    refetchInterval: 15000,
  });
}

function GpuStatusDot({ className }: { className?: string }) {
  const { data } = useSystemStatus();
  const isOnline = data?.gpu?.isAvailable ?? false;
  return (
    <span
      data-testid="gpu-status-dot"
      className={cn(
        "inline-block w-2 h-2 rounded-full",
        isOnline ? "bg-emerald-400" : "bg-red-400",
        isOnline && "pulse-dot",
        className,
      )}
    />
  );
}

function SystemLiveIndicator({ collapsed }: { collapsed: boolean }) {
  const { data } = useSystemStatus();
  const gpuOnline = data?.gpu?.isAvailable ?? false;
  const paperEnabled = data?.paper?.portfolioExists ?? false;
  const isLive = gpuOnline && paperEnabled;
  const cyclesToday = data?.cyclesToday ?? 0;
  const lastCycleTs = data?.lastCycleTs ?? data?.gpu?.lastActivity ?? null;

  if (collapsed) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center justify-center py-1" data-testid="system-live-indicator">
            <span
              className={cn(
                "inline-block w-2 h-2 rounded-full",
                isLive ? "bg-emerald-400 pulse-dot" : "bg-muted-foreground",
              )}
            />
          </div>
        </TooltipTrigger>
        <TooltipContent side="right" className="text-xs">
          {isLive ? "System LIVE" : "System Offline"}
          {lastCycleTs && ` — Last scan: ${formatDistanceToNow(new Date(lastCycleTs), { addSuffix: true })}`}
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <div className="px-2.5 py-1.5 space-y-1" data-testid="system-live-indicator">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "inline-block w-2 h-2 rounded-full shrink-0",
            isLive ? "bg-emerald-400 pulse-dot" : "bg-muted-foreground",
          )}
        />
        <span className={cn("text-xs font-semibold tracking-wider uppercase", isLive ? "text-emerald-400" : "text-muted-foreground")}>
          {isLive ? "LIVE" : "OFFLINE"}
        </span>
        {isLive && <Radio className="w-3 h-3 text-emerald-400 ml-auto" />}
      </div>
      {lastCycleTs && (
        <div className="text-[10px] text-muted-foreground pl-4 truncate" data-testid="text-last-scan">
          Last scan: {formatDistanceToNow(new Date(lastCycleTs), { addSuffix: true })}
        </div>
      )}
      {cyclesToday > 0 && (
        <div className="text-[10px] text-muted-foreground pl-4" data-testid="text-cycles-today">
          {cyclesToday} cycles today
        </div>
      )}
    </div>
  );
}

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return true;
    return localStorage.getItem("theme") !== "light";
  });
  const { connectionState, subscribe } = useTradingWs();
  const { toast } = useToast();

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [dark]);

  useEffect(() => {
    const unsubs = [
      subscribe("TRADE_OPEN", (payload) => {
        const p = payload as { symbol?: string; side?: string; entryPrice?: number; riskUsd?: number; manual?: boolean };
        toast({
          title: `${p.manual ? "Manual " : ""}Opened ${p.side} ${p.symbol}`,
          description: `Entry: $${p.entryPrice?.toLocaleString() ?? "?"} | Risk: $${p.riskUsd?.toFixed(2) ?? "?"}`,
        });
        queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
        queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
        queryClient.invalidateQueries({ queryKey: ["/api/paper/risk-alerts"] });
      }),
      subscribe("TRADE_CLOSE", (payload) => {
        const p = payload as { symbol?: string; side?: string; pnl?: number; reason?: string };
        const isWin = (p.pnl ?? 0) >= 0;
        toast({
          title: `Closed ${p.symbol} ${p.side}`,
          description: `P&L: ${isWin ? "+" : ""}$${p.pnl?.toFixed(2) ?? "0"} (${p.reason ?? ""})`,
          variant: isWin ? "default" : "destructive",
        });
        queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
        queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
      }),
      subscribe("TRADE_UPDATE", (payload) => {
        const p = payload as { symbol?: string; action?: string };
        toast({ title: `${p.symbol} — ${p.action}` });
        queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
        queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
        queryClient.invalidateQueries({ queryKey: ["/api/paper/risk-alerts"] });
      }),
    ];
    return () => unsubs.forEach((u) => u());
  }, [subscribe, toast]);

  return (
    <div className="flex h-screen overflow-hidden bg-background" data-testid="app-layout">
      <aside
        className={cn(
          "flex flex-col h-full border-r border-sidebar-border bg-sidebar transition-all duration-200",
          collapsed ? "w-[52px]" : "w-[220px]",
        )}
        data-testid="sidebar"
      >
        <div className={cn("flex items-center gap-2 px-3 h-14 border-b border-sidebar-border", collapsed && "justify-center")}>
          <div className="flex items-center justify-center w-7 h-7 rounded-md bg-primary/10">
            <Zap className="w-4 h-4 text-primary" />
          </div>
          {!collapsed && (
            <span className="text-sm font-semibold text-sidebar-foreground tracking-tight truncate">
              Neural Terminal
            </span>
          )}
        </div>

        <nav className="flex-1 py-2 px-2 space-y-0.5 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const isActive = location === item.path || (item.path !== "/" && location.startsWith(item.path));
            const Icon = item.icon;
            const linkContent = (
              <Link
                href={item.path}
                data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, "-")}`}
                className={cn(
                  "flex items-center gap-2.5 px-2.5 py-2 rounded-md text-sm transition-colors",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-foreground font-medium"
                    : "text-muted-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent/50",
                  collapsed && "justify-center px-0",
                )}
              >
                <Icon className="w-4 h-4 shrink-0" />
                {!collapsed && <span className="truncate">{item.label}</span>}
              </Link>
            );

            if (collapsed) {
              return (
                <Tooltip key={item.path}>
                  <TooltipTrigger asChild>{linkContent}</TooltipTrigger>
                  <TooltipContent side="right" className="text-xs">
                    {item.label}
                  </TooltipContent>
                </Tooltip>
              );
            }
            return <div key={item.path}>{linkContent}</div>;
          })}
        </nav>

        <div className="border-t border-sidebar-border p-2 space-y-1">
          <SystemLiveIndicator collapsed={collapsed} />
          <div className={cn("flex items-center gap-2 px-2.5 py-1.5 text-xs text-muted-foreground", collapsed && "justify-center")}>
            <GpuStatusDot />
            {!collapsed && (
              <span className="truncate">
                GPU {connectionState === "connected" ? "Online" : "Offline"}
              </span>
            )}
          </div>

          <div className={cn("flex gap-1", collapsed ? "flex-col items-center" : "items-center")}>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => setDark(!dark)}
              data-testid="theme-toggle"
            >
              {dark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => setCollapsed(!collapsed)}
              data-testid="sidebar-toggle"
            >
              <ChevronLeft className={cn("w-3.5 h-3.5 transition-transform", collapsed && "rotate-180")} />
            </Button>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto" data-testid="main-content">
        <RiskAlertBar />
        {children}
      </main>
    </div>
  );
}
