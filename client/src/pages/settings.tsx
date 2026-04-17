import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Cpu,
  Wallet,
  Brain,
  Shield,
  Database,
  AlertTriangle,
  Zap,
  CheckCircle,
  XCircle,
  Loader2,
  Key,
  Eye,
  EyeOff,
  DollarSign,
} from "lucide-react";
import { TRADING_SYMBOLS } from "@shared/symbols";

const SYMBOLS = TRADING_SYMBOLS;

export default function SettingsPage() {
  const [equity, setEquity] = useState<number>(0);
  const [riskPct, setRiskPct] = useState<number>(0);
  const [liveRiskPct, setLiveRiskPct] = useState<number>(0.5);
  const [maxDailyLoss, setMaxDailyLoss] = useState<number>(500);

  const [bgApiKey, setBgApiKey] = useState("");
  const [bgSecretKey, setBgSecretKey] = useState("");
  const [bgPassphrase, setBgPassphrase] = useState("");
  const [showBgSecret, setShowBgSecret] = useState(false);
  const [showBgPassphrase, setShowBgPassphrase] = useState(false);
  const [bgLiveRiskPct, setBgLiveRiskPct] = useState<number>(0.5);
  const [bgMaxDailyLoss, setBgMaxDailyLoss] = useState<number>(500);
  const [bgMakerEntry, setBgMakerEntry] = useState<boolean>(true);

  const { data: systemStatus } = useQuery<any>({
    queryKey: ["/api/system/status"],
    refetchInterval: 10000,
  });

  const { data: paperConfig } = useQuery<any>({
    queryKey: ["/api/paper/config"],
  });

  const { data: bybitStatus, refetch: refetchBybit } = useQuery<any>({
    queryKey: ["/api/bybit/status"],
    refetchInterval: 30000,
  });

  const testConnectionMutation = useMutation({
    mutationFn: () => fetch("/api/bybit/status").then(r => r.json()),
    onSuccess: () => { refetchBybit(); },
  });

  const toggleLiveMutation = useMutation({
    mutationFn: (enabled: boolean) => apiRequest("POST", "/api/bybit/toggle", { enabled }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bybit/status"] });
    },
  });

  const saveLiveConfigMutation = useMutation({
    mutationFn: () => apiRequest("PATCH", "/api/bybit/config", { riskPerTradePct: liveRiskPct, maxDailyLossUsdt: maxDailyLoss }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bybit/status"] });
    },
  });

  useEffect(() => {
    if (bybitStatus?.config) {
      setLiveRiskPct(bybitStatus.config.riskPerTradePct ?? 0.5);
      setMaxDailyLoss(bybitStatus.config.maxDailyLossUsdt ?? 500);
    }
  }, [bybitStatus]);

  const { data: bitgetStatus, refetch: refetchBitget } = useQuery<any>({
    queryKey: ["/api/bitget/status"],
    refetchInterval: 30000,
  });

  const saveBitgetCredsMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bitget/credentials", { apiKey: bgApiKey, secretKey: bgSecretKey, passphrase: bgPassphrase }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bitget/status"] });
      refetchBitget();
      setBgApiKey("");
      setBgSecretKey("");
      setBgPassphrase("");
    },
  });

  const testBitgetConnectionMutation = useMutation({
    mutationFn: () => fetch("/api/bitget/status").then(r => r.json()),
    onSuccess: () => { refetchBitget(); },
  });

  const saveBitgetConfigMutation = useMutation({
    mutationFn: () => apiRequest("PATCH", "/api/bitget/config", { riskPerTradePct: bgLiveRiskPct, maxDailyLossUsdt: bgMaxDailyLoss, makerEntry: bgMakerEntry }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bitget/status"] });
    },
  });

  useEffect(() => {
    if (bitgetStatus?.config) {
      setBgLiveRiskPct(bitgetStatus.config.riskPerTradePct ?? 0.5);
      setBgMaxDailyLoss(bitgetStatus.config.maxDailyLossUsdt ?? 500);
      setBgMakerEntry(bitgetStatus.config.makerEntry ?? true);
    }
  }, [bitgetStatus]);

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
  const gpuLastActivity = systemStatus?.gpu?.lastActivity ?? null;
  const lastSync = systemStatus?.sync?.lastSync;

  const formatTimeAgo = (ts: number | null): string => {
    if (!ts) return "";
    const diff = Date.now() - ts;
    if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return `${Math.floor(diff / 86400000)}d ago`;
  };

  const isViaPush = gpuConnected && gpuLastActivity && (Date.now() - gpuLastActivity) < 5 * 60 * 1000;

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
              {isViaPush ? "Connected (via push)" : "Connected"}
            </Badge>
          ) : (
            <Badge className="bg-red-500/20 text-red-400 border-red-500/30" data-testid="gpu-status-badge">
              Disconnected
            </Badge>
          )}
        </div>
        {gpuLastActivity && (
          <div className="mb-2">
            <Label className="text-muted-foreground text-xs">Last Activity</Label>
            <p className="text-sm" data-testid="gpu-last-activity">{formatTimeAgo(gpuLastActivity)}</p>
          </div>
        )}
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

      <div
        className={`glass-card rounded-md p-4 ${bybitStatus?.connected ? "glow-green" : ""}`}
        data-testid="bybit-connection-card"
      >
        <div className="flex items-center gap-2 mb-3">
          <Zap className="w-5 h-5 text-amber-400" />
          <h2 className="font-semibold text-lg">Exchange Connection (Bybit)</h2>
        </div>
        <div className="flex items-center gap-3 mb-3">
          {bybitStatus?.connected ? (
            <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30" data-testid="bybit-status-badge">
              <CheckCircle className="w-3 h-3 mr-1" /> Connected
            </Badge>
          ) : bybitStatus?.configured ? (
            <Badge className="bg-red-500/20 text-red-400 border-red-500/30" data-testid="bybit-status-badge">
              <XCircle className="w-3 h-3 mr-1" /> Connection Error
            </Badge>
          ) : (
            <Badge className="bg-gray-500/20 text-gray-400 border-gray-500/30" data-testid="bybit-status-badge">
              Not Configured
            </Badge>
          )}
          {bybitStatus?.liveTradingEnabled && (
            <Badge className="bg-red-500/20 text-red-400 border-red-500/50 animate-pulse" data-testid="live-trading-badge">
              LIVE TRADING ACTIVE
            </Badge>
          )}
        </div>

        {bybitStatus?.connected && (
          <div className="mb-3">
            <Label className="text-muted-foreground text-xs">Account Balance (USDT)</Label>
            <p className="number-mono text-lg text-emerald-400" data-testid="bybit-balance">
              ${parseFloat(bybitStatus.balance || "0").toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
        )}

        {bybitStatus?.error && !bybitStatus?.connected && (
          <p className="text-sm text-red-400 mb-3" data-testid="bybit-error">{bybitStatus.error}</p>
        )}

        <div className="flex gap-2 mb-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => testConnectionMutation.mutate()}
            disabled={testConnectionMutation.isPending}
            data-testid="button-test-bybit"
          >
            {testConnectionMutation.isPending ? (
              <><Loader2 className="w-3 h-3 mr-1 animate-spin" /> Testing...</>
            ) : "Test Connection"}
          </Button>
        </div>

        {bybitStatus?.connected && (
          <div className="border-t border-border/30 pt-3 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-sm font-medium">Live Trading</Label>
                <p className="text-xs text-muted-foreground">Execute real trades on Bybit when V5 signals fire</p>
              </div>
              <Switch
                checked={bybitStatus?.liveTradingEnabled || false}
                onCheckedChange={(checked) => {
                  if (checked) {
                    if (window.confirm("⚠️ ENABLE LIVE TRADING?\n\nThis will execute REAL trades with REAL money on Bybit when V5 signals fire.\n\nAre you sure?")) {
                      toggleLiveMutation.mutate(true);
                    }
                  } else {
                    toggleLiveMutation.mutate(false);
                  }
                }}
                disabled={toggleLiveMutation.isPending}
                data-testid="switch-live-trading"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label htmlFor="live-risk-input" className="text-xs text-muted-foreground">Risk Per Trade (%)</Label>
                <Input
                  id="live-risk-input"
                  type="number"
                  step={0.1}
                  value={liveRiskPct}
                  onChange={(e) => setLiveRiskPct(parseFloat(e.target.value) || 0)}
                  data-testid="input-live-risk"
                />
              </div>
              <div>
                <Label htmlFor="max-loss-input" className="text-xs text-muted-foreground">Max Daily Loss (USDT)</Label>
                <Input
                  id="max-loss-input"
                  type="number"
                  value={maxDailyLoss}
                  onChange={(e) => setMaxDailyLoss(parseFloat(e.target.value) || 0)}
                  data-testid="input-max-daily-loss"
                />
              </div>
            </div>
            <Button
              size="sm"
              onClick={() => saveLiveConfigMutation.mutate()}
              disabled={saveLiveConfigMutation.isPending}
              data-testid="button-save-live-config"
            >
              {saveLiveConfigMutation.isPending ? "Saving..." : "Save Live Config"}
            </Button>
          </div>
        )}

        {!bybitStatus?.configured && (
          <p className="text-sm text-muted-foreground" data-testid="bybit-not-configured">
            Add BYBIT_API_KEY and BYBIT_API_SECRET as environment secrets to connect.
          </p>
        )}
        {bybitStatus?.configured && !bybitStatus?.connected && bybitStatus?.error && (
          <div className="mt-2 space-y-2" data-testid="bybit-setup-guidance">
            <p className="text-xs text-red-400/80" data-testid="bybit-error-detail">{bybitStatus.error}</p>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-amber-400" />
              <p className="text-xs text-muted-foreground">
                Mode: <span className="number-mono text-amber-400">{bybitStatus.proxy?.mode === "proxy" ? `Proxy (${bybitStatus.proxy.url})` : "Direct"}</span>
              </p>
            </div>
          </div>
        )}
        {bybitStatus?.configured && bybitStatus?.connected && (
          <div className="flex items-center gap-2 mt-1" data-testid="bybit-connected-info">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <p className="text-xs text-emerald-400/70">
              {bybitStatus.proxy?.mode === "proxy" ? `Connected via GPU proxy — ${bybitStatus.proxy?.url}` : "Connected directly to Bybit API"}
            </p>
          </div>
        )}
        {bybitStatus?.executionService && (
          <div className="flex items-center gap-2 mt-1" data-testid="execution-service-status">
            <div className={`w-2 h-2 rounded-full ${bybitStatus.executionService.connected ? "bg-emerald-400 pulse-dot" : "bg-muted-foreground"}`} />
            <p className={`text-xs ${bybitStatus.executionService.connected ? "text-emerald-400/70" : "text-muted-foreground"}`}>
              {bybitStatus.executionService.connected
                ? `Execution Service connected — ${bybitStatus.executionService.positionCount} positions, last push ${Math.round((bybitStatus.executionService.lastPushAgo || 0) / 1000)}s ago`
                : "Execution Service not connected — start GPU trainer with Bybit keys to enable"}
            </p>
          </div>
        )}
      </div>

      <div className={`glass-card rounded-md p-4 ${bitgetStatus?.connected ? "glow-green" : ""}`} data-testid="bitget-settings-card">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Key className="w-5 h-5 text-orange-400" />
            <h2 className="font-semibold text-lg">Exchange Connection (Bitget)</h2>
          </div>
          {bitgetStatus?.connected ? (
            <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30" data-testid="bitget-connected-badge">
              <CheckCircle className="w-3 h-3 mr-1" /> Connected
            </Badge>
          ) : bitgetStatus?.configured ? (
            <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/30">
              <AlertTriangle className="w-3 h-3 mr-1" /> Configured
            </Badge>
          ) : (
            <Badge className="bg-gray-500/20 text-gray-400 border-gray-500/30">Not Configured</Badge>
          )}
        </div>

        {bitgetStatus?.connected && (
          <div className="mb-3 text-sm text-emerald-400/70 flex items-center gap-1">
            <DollarSign className="w-3.5 h-3.5" /> Available Balance: ${parseFloat(bitgetStatus.balance || "0").toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        )}

        {bitgetStatus?.error && !bitgetStatus?.connected && (
          <p className="text-sm text-red-400 mb-3" data-testid="bitget-error">{bitgetStatus.error}</p>
        )}

        <div className="space-y-3 mb-3">
          <div>
            <Label className="text-sm text-muted-foreground">API Key</Label>
            <Input value={bgApiKey} onChange={(e) => setBgApiKey(e.target.value)} placeholder={bitgetStatus?.configured ? "••••••••" : "Enter Bitget API Key"} data-testid="input-bitget-api-key" />
          </div>
          <div>
            <Label className="text-sm text-muted-foreground">Secret Key</Label>
            <div className="relative">
              <Input type={showBgSecret ? "text" : "password"} value={bgSecretKey} onChange={(e) => setBgSecretKey(e.target.value)} placeholder={bitgetStatus?.configured ? "••••••••" : "Enter Bitget Secret Key"} data-testid="input-bitget-secret-key" />
              <button type="button" className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" onClick={() => setShowBgSecret(!showBgSecret)}>
                {showBgSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>
          <div>
            <Label className="text-sm text-muted-foreground">Passphrase</Label>
            <div className="relative">
              <Input type={showBgPassphrase ? "text" : "password"} value={bgPassphrase} onChange={(e) => setBgPassphrase(e.target.value)} placeholder={bitgetStatus?.configured ? "••••••••" : "Enter Bitget Passphrase"} data-testid="input-bitget-passphrase" />
              <button type="button" className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground" onClick={() => setShowBgPassphrase(!showBgPassphrase)}>
                {showBgPassphrase ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>
          <Button
            onClick={() => saveBitgetCredsMutation.mutate()}
            disabled={saveBitgetCredsMutation.isPending || !bgApiKey || !bgSecretKey || !bgPassphrase}
            data-testid="button-save-bitget-creds"
          >
            {saveBitgetCredsMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : <Key className="w-4 h-4 mr-1" />}
            {saveBitgetCredsMutation.isPending ? "Saving..." : "Save Bitget Credentials"}
          </Button>
          {bitgetStatus?.configured && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => testBitgetConnectionMutation.mutate()}
              disabled={testBitgetConnectionMutation.isPending}
              data-testid="button-test-bitget-connection"
            >
              {testBitgetConnectionMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : <Zap className="w-4 h-4 mr-1" />}
              {testBitgetConnectionMutation.isPending ? "Testing..." : "Test Connection"}
            </Button>
          )}
        </div>

        {bitgetStatus?.connected && (
          <div className="space-y-3 border-t border-border/20 pt-3">
            <h3 className="text-sm font-medium">Bitget Live Trading Config</h3>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">Risk Per Trade (%)</Label>
                <Input type="number" step={0.1} min={0.1} max={5} value={bgLiveRiskPct} onChange={(e) => setBgLiveRiskPct(parseFloat(e.target.value) || 0.5)} data-testid="input-bitget-risk" />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Max Daily Loss (USDT)</Label>
                <Input type="number" step={50} min={50} max={100000} value={bgMaxDailyLoss} onChange={(e) => setBgMaxDailyLoss(parseFloat(e.target.value) || 500)} data-testid="input-bitget-daily-loss" />
              </div>
            </div>
            <div className="flex items-center justify-between py-1.5 px-2 rounded-md bg-background/30 border border-border/20">
              <div>
                <Label className="text-sm font-medium">Maker Entry Mode</Label>
                <p className="text-xs text-muted-foreground mt-0.5">Limit orders at candle close price (0.02% vs 0.06% taker fee)</p>
              </div>
              <Switch
                checked={bgMakerEntry}
                onCheckedChange={setBgMakerEntry}
                data-testid="switch-maker-entry"
              />
            </div>
            <Button
              size="sm"
              onClick={() => saveBitgetConfigMutation.mutate()}
              disabled={saveBitgetConfigMutation.isPending}
              data-testid="button-save-bitget-config"
            >
              {saveBitgetConfigMutation.isPending ? "Saving..." : "Save Bitget Config"}
            </Button>
          </div>
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
          <InfoItem label="Architecture" value="V5Forecaster (multi-head)" />
          <InfoItem label="Features" value="85 (STF:44, ENH:24, HTF:12, REGIME:5)" />
          <InfoItem label="Timeframe" value="15m with adaptive horizon (8-48 bars)" />
          <InfoItem label="Symbols" value={SYMBOLS.map(s => s.replace("USDT", "")).join(", ")} />
          <InfoItem label="Engine" value="V5 Composite Scoring (score ≥ 0.02)" />
          <InfoItem label="Version" value="v5.0" />
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

      {/* ── Leverage Tiers ───────────────────────────────────── */}
      <div className="glass-card rounded-md p-4" data-testid="leverage-tiers-card">
        <div className="flex items-center gap-2 mb-3">
          <Zap className="w-5 h-5 text-amber-400" />
          <h2 className="font-semibold text-lg">Leverage Tiers</h2>
          {paperConfig?.leverageEnabled != null && (
            <Badge className={paperConfig.leverageEnabled ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30 text-[10px]" : "bg-red-500/20 text-red-400 border-red-500/30 text-[10px]"} data-testid="leverage-enabled-badge">
              {paperConfig.leverageEnabled ? "Enabled" : "Disabled"}
            </Badge>
          )}
        </div>
        <p className="text-[10px] text-muted-foreground mb-3">
          V5 score threshold → exchange leverage applied to both qty and initialRiskUsdt. Max: {paperConfig?.maxLeverage ?? 50}x
        </p>
        {paperConfig?.leverageTiers && paperConfig.leverageTiers.length > 0 ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Min V5 Score</TableHead>
                <TableHead className="text-xs text-center">Leverage</TableHead>
                <TableHead className="text-xs text-right">At 1.5% Base Risk</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paperConfig.leverageTiers.map((tier: { minScore: number; leverage: number }, i: number) => {
                const nextTier = paperConfig.leverageTiers[i + 1];
                const rangeLabel = nextTier ? `${tier.minScore} – ${nextTier.minScore}` : `≥ ${tier.minScore}`;
                const effectiveRisk = (tier.leverage * 1.5).toFixed(1);
                const riskColor = tier.leverage >= 50 ? "text-red-400"
                  : tier.leverage >= 35 ? "text-orange-400"
                  : tier.leverage >= 25 ? "text-amber-400"
                  : tier.leverage >= 15 ? "text-yellow-400"
                  : "text-muted-foreground";
                return (
                  <TableRow key={i} data-testid={`leverage-tier-${i}`}>
                    <TableCell className="text-xs font-mono text-muted-foreground">{rangeLabel}</TableCell>
                    <TableCell className={`text-sm font-bold text-center ${riskColor}`} data-testid={`leverage-value-${i}`}>{tier.leverage}x</TableCell>
                    <TableCell className={`text-xs text-right font-mono ${riskColor}`}>{effectiveRisk}%</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        ) : (
          <p className="text-xs text-muted-foreground/50">No leverage tiers configured</p>
        )}
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
