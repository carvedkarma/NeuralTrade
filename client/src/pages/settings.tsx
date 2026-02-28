import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Cpu,
  Wallet,
  Brain,
  Shield,
  Database,
  AlertTriangle,
} from "lucide-react";

const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"];

export default function SettingsPage() {
  const [equity, setEquity] = useState<number>(0);
  const [riskPct, setRiskPct] = useState<number>(0);

  const { data: systemStatus } = useQuery<any>({
    queryKey: ["/api/system/status"],
    refetchInterval: 10000,
  });

  const { data: paperConfig } = useQuery<any>({
    queryKey: ["/api/paper/config"],
  });

  const { data: prices } = useQuery<any>({
    queryKey: ["/api/market/prices"],
    refetchInterval: 15000,
  });

  useEffect(() => {
    if (systemStatus?.moneyConfig) {
      setEquity(systemStatus.moneyConfig.account_equity_usd ?? 0);
      setRiskPct(systemStatus.moneyConfig.risk_per_trade_pct ?? 0);
    }
  }, [systemStatus]);

  const saveConfigMutation = useMutation({
    mutationFn: async () => {
      await apiRequest("POST", "/api/paper/config", {
        account_equity_usd: equity,
        risk_per_trade_pct: riskPct,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper/config"] });
      queryClient.invalidateQueries({ queryKey: ["/api/system/status"] });
    },
  });

  const resetMutation = useMutation({
    mutationFn: async () => {
      await apiRequest("POST", "/api/paper/reset", {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paper"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/config"] });
      queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
      queryClient.invalidateQueries({ queryKey: ["/api/system/status"] });
    },
  });

  const gpuConnected = systemStatus?.gpu?.isAvailable ?? false;
  const gpuUrl = systemStatus?.gpu?.url ?? "";
  const lastSync = systemStatus?.sync?.lastSync;

  return (
    <div className="p-4 space-y-4 max-w-4xl" data-testid="settings">

      <div
        className={`glass-card rounded-md p-4 ${gpuConnected ? "glow-green" : ""}`}
        data-testid="gpu-connection-card"
      >
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-5 h-5 text-cyan-500" />
          <h2 className="font-semibold text-lg">GPU Connection</h2>
        </div>
        <div className="flex items-center gap-3 mb-3">
          {gpuConnected ? (
            <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30" data-testid="gpu-status-badge">
              Connected
            </Badge>
          ) : (
            <Badge className="bg-red-500/20 text-red-400 border-red-500/30" data-testid="gpu-status-badge">
              Disconnected
            </Badge>
          )}
        </div>
        {gpuUrl && (
          <div className="mb-2">
            <Label className="text-muted-foreground text-xs">GPU URL</Label>
            <p className="number-mono text-sm" data-testid="gpu-url-display">{gpuUrl}</p>
          </div>
        )}
        {!gpuConnected && (
          <p className="text-sm text-muted-foreground" data-testid="gpu-info-text">
            GPU trainer runs locally. Configure tunnel URL to connect.
          </p>
        )}
      </div>

      <div className="glass-card rounded-md p-4" data-testid="account-settings-card">
        <div className="flex items-center gap-2 mb-3">
          <Wallet className="w-5 h-5 text-amber-400" />
          <h2 className="font-semibold text-lg">Account Settings</h2>
        </div>
        <div className="space-y-3">
          <div>
            <Label htmlFor="equity-input" className="text-sm text-muted-foreground">
              Account Equity (USD)
            </Label>
            <Input
              id="equity-input"
              type="number"
              value={equity}
              onChange={(e) => setEquity(parseFloat(e.target.value) || 0)}
              data-testid="input-equity"
            />
          </div>
          <div>
            <Label htmlFor="risk-input" className="text-sm text-muted-foreground">
              Risk Per Trade (%)
            </Label>
            <Input
              id="risk-input"
              type="number"
              step={0.1}
              value={riskPct}
              onChange={(e) => setRiskPct(parseFloat(e.target.value) || 0)}
              data-testid="input-risk-pct"
            />
          </div>
          <Button
            onClick={() => saveConfigMutation.mutate()}
            disabled={saveConfigMutation.isPending}
            data-testid="button-save-config"
          >
            {saveConfigMutation.isPending ? "Saving..." : "Save"}
          </Button>
        </div>
      </div>

      <div className="glass-card rounded-md p-4" data-testid="model-info-card">
        <div className="flex items-center gap-2 mb-3">
          <Brain className="w-5 h-5 text-cyan-500" />
          <h2 className="font-semibold text-lg">Model Information</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
          <InfoItem label="Architecture" value="EnhancedMultiHeadMLP" />
          <InfoItem label="Features" value="85 (STF:44, ENH:24, HTF:12, REGIME:5)" />
          <InfoItem label="Timeframe" value="15m with adaptive horizon (8-48 bars)" />
          <InfoItem label="Symbols" value="BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, AVAXUSDT" />
          <InfoItem label="Engine" value="Triple-Lane Aggression (CORE/FLOW/SCALP)" />
          <InfoItem label="Version" value="v5.4.0" />
        </div>
      </div>

      <div className="glass-card rounded-md p-4" data-testid="risk-params-card">
        <div className="flex items-center gap-2 mb-3">
          <Shield className="w-5 h-5 text-amber-400" />
          <h2 className="font-semibold text-lg">Risk Parameters</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
          <InfoItem
            label="Daily Loss Cap"
            value={paperConfig?.daily_loss_cap_r != null ? `${paperConfig.daily_loss_cap_r}R` : "-3R"}
          />
          <InfoItem
            label="Per-Symbol R Kill"
            value={paperConfig?.per_symbol_r_kill != null ? `${paperConfig.per_symbol_r_kill}R` : "-8R"}
          />
          <InfoItem
            label="Max Exposure"
            value={paperConfig?.max_exposure != null ? `${paperConfig.max_exposure}%` : "100%"}
          />
        </div>
      </div>

      <div className="glass-card rounded-md p-4" data-testid="data-freshness-card">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-5 h-5 text-cyan-500" />
          <h2 className="font-semibold text-lg">Data Freshness</h2>
        </div>
        {lastSync && (
          <p className="text-xs text-muted-foreground mb-2" data-testid="last-sync-time">
            Last sync: {new Date(lastSync).toLocaleString()}
          </p>
        )}
        <div className="space-y-1">
          {SYMBOLS.map((symbol) => {
            const price = prices?.[symbol] ?? prices?.[symbol.toLowerCase()];
            const hasFresh = price != null && price > 0;
            return (
              <div
                key={symbol}
                className="flex items-center justify-between gap-2 py-1 border-b border-border/30 last:border-0"
                data-testid={`data-row-${symbol}`}
              >
                <span className="text-sm font-medium">{symbol}</span>
                <div className="flex items-center gap-2">
                  {hasFresh && (
                    <span className="number-mono text-sm text-muted-foreground" data-testid={`price-${symbol}`}>
                      ${typeof price === "number" ? price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : price}
                    </span>
                  )}
                  <Badge
                    className={
                      hasFresh
                        ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                        : "bg-amber-500/20 text-amber-400 border-amber-500/30"
                    }
                    data-testid={`status-${symbol}`}
                  >
                    {hasFresh ? "Fresh" : "Stale"}
                  </Badge>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div
        className="glass-card rounded-md p-4 border-red-500/40"
        data-testid="danger-zone-card"
      >
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle className="w-5 h-5 text-red-400" />
          <h2 className="font-semibold text-lg text-red-400">Danger Zone</h2>
        </div>
        <p className="text-sm text-muted-foreground mb-3">
          Resetting the paper portfolio will delete all trades and reset equity to the configured starting balance.
        </p>
        <Button
          variant="destructive"
          onClick={() => {
            if (window.confirm("Are you sure you want to reset the paper portfolio? This cannot be undone.")) {
              resetMutation.mutate();
            }
          }}
          disabled={resetMutation.isPending}
          data-testid="button-reset-portfolio"
        >
          {resetMutation.isPending ? "Resetting..." : "Reset Paper Portfolio"}
        </Button>
      </div>
    </div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-muted-foreground text-xs">{label}</span>
      <p className="text-sm" data-testid={`info-${label.toLowerCase().replace(/\s+/g, "-")}`}>
        {value}
      </p>
    </div>
  );
}
