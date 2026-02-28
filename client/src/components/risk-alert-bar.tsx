import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ShieldAlert, Info, X, ChevronDown, ChevronUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface RiskAlert {
  id: string;
  type: string;
  severity: "info" | "warning" | "critical";
  symbol?: string;
  message: string;
  data: Record<string, unknown>;
}

export default function RiskAlertBar() {
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState(false);

  const { data: alerts } = useQuery<RiskAlert[]>({
    queryKey: ["/api/paper/risk-alerts"],
    refetchInterval: 10000,
  });

  const activeAlerts = (alerts ?? []).filter((a) => !dismissed.has(a.id));
  if (activeAlerts.length === 0) return null;

  const criticals = activeAlerts.filter((a) => a.severity === "critical");
  const warnings = activeAlerts.filter((a) => a.severity === "warning");
  const infos = activeAlerts.filter((a) => a.severity === "info");

  const topAlert = criticals[0] || warnings[0] || infos[0];
  if (!topAlert) return null;

  const bgClass = topAlert.severity === "critical"
    ? "bg-red-500/15 border-red-500/30"
    : topAlert.severity === "warning"
      ? "bg-amber-500/15 border-amber-500/30"
      : "bg-cyan-500/15 border-cyan-500/30";

  const textClass = topAlert.severity === "critical"
    ? "text-red-400"
    : topAlert.severity === "warning"
      ? "text-amber-400"
      : "text-cyan-400";

  const IconComp = topAlert.severity === "critical"
    ? ShieldAlert
    : topAlert.severity === "warning"
      ? AlertTriangle
      : Info;

  return (
    <div className={`border rounded-md mx-4 mt-3 ${bgClass}`} data-testid="risk-alert-bar">
      <div className="flex items-center gap-2 px-3 py-2">
        <IconComp className={`w-4 h-4 shrink-0 ${textClass} ${topAlert.severity === "critical" ? "animate-pulse" : ""}`} />
        <span className={`text-sm flex-1 ${textClass}`} data-testid="risk-alert-message">
          {topAlert.message}
        </span>
        {activeAlerts.length > 1 && (
          <Badge
            variant="secondary"
            className="cursor-pointer text-xs"
            onClick={() => setExpanded(!expanded)}
            data-testid="risk-alert-count"
          >
            {activeAlerts.length} alerts
            {expanded ? <ChevronUp className="w-3 h-3 ml-1" /> : <ChevronDown className="w-3 h-3 ml-1" />}
          </Badge>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={() => setDismissed((prev) => new Set(prev).add(topAlert.id))}
          data-testid="button-dismiss-alert"
        >
          <X className="w-3 h-3" />
        </Button>
      </div>
      {expanded && activeAlerts.length > 1 && (
        <div className="px-3 pb-2 space-y-1" data-testid="risk-alert-expanded">
          {activeAlerts.slice(1).map((alert) => {
            const sevClass = alert.severity === "critical"
              ? "text-red-400"
              : alert.severity === "warning"
                ? "text-amber-400"
                : "text-cyan-400";
            return (
              <div key={alert.id} className="flex items-center gap-2 text-xs">
                <span className={`w-1.5 h-1.5 rounded-full ${alert.severity === "critical" ? "bg-red-400" : alert.severity === "warning" ? "bg-amber-400" : "bg-cyan-400"}`} />
                <span className={sevClass}>{alert.message}</span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-5 w-5 ml-auto"
                  onClick={() => setDismissed((prev) => new Set(prev).add(alert.id))}
                >
                  <X className="w-2.5 h-2.5" />
                </Button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
