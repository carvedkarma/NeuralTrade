import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { 
  Wifi, 
  WifiOff, 
  Settings, 
  CheckCircle2, 
  XCircle, 
  Loader2, 
  ExternalLink,
  Copy,
  Terminal,
  Cpu,
  HardDrive,
  Layers
} from "lucide-react";

interface GPUSettings {
  url: string;
  defaultUrl: string;
  predictionMode: "stf" | "mtf";
}

interface TestConnectionResult {
  connected: boolean;
  health?: {
    status: string;
    gpu_available: boolean;
    gpu_name: string;
    gpu_memory_used: number;
    gpu_memory_total: number;
    models_loaded: string[];
  };
  message?: string;
}

interface GPUStatus {
  connected: boolean;
  isStale: boolean;
  gpuAvailable: boolean;
  gpuName: string | null;
  gpuMemoryUsed: number | null;
  gpuMemoryTotal: number | null;
  modelsLoaded: string[];
}

export function GPUConnectionSettings() {
  const [inputUrl, setInputUrl] = useState("");
  const [predictionMode, setPredictionMode] = useState<"stf" | "mtf">("stf");
  const [testResult, setTestResult] = useState<TestConnectionResult | null>(null);
  const [copied, setCopied] = useState(false);

  const { data: settings, isLoading: settingsLoading } = useQuery<GPUSettings>({
    queryKey: ["/api/gpu/settings"],
  });

  useEffect(() => {
    if (settings?.predictionMode) {
      setPredictionMode(settings.predictionMode);
    }
  }, [settings]);

  const { data: gpuStatus } = useQuery<GPUStatus>({
    queryKey: ["/api/gpu/pushed-status"],
    refetchInterval: 5000,
  });

  const testConnectionMutation = useMutation({
    mutationFn: async (url: string) => {
      const response = await apiRequest("POST", "/api/gpu/test-connection", { url });
      return response.json() as Promise<TestConnectionResult>;
    },
    onSuccess: (data) => {
      setTestResult(data);
    },
  });

  const saveSettingsMutation = useMutation({
    mutationFn: (data: { url: string; predictionMode: "stf" | "mtf" }) => 
      apiRequest("POST", "/api/gpu/settings", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/gpu/settings"] });
      queryClient.invalidateQueries({ queryKey: ["/api/gpu/pushed-status"] });
      queryClient.invalidateQueries({ queryKey: ["/api/gpu/ensemble/status"] });
    },
  });

  const handleTestConnection = () => {
    const urlToTest = inputUrl || settings?.url || settings?.defaultUrl || "";
    if (urlToTest) {
      setTestResult(null);
      testConnectionMutation.mutate(urlToTest);
    }
  };

  const handleSaveAndConnect = () => {
    const urlToSave = inputUrl || settings?.url || settings?.defaultUrl || "";
    if (urlToSave) {
      saveSettingsMutation.mutate({ url: urlToSave, predictionMode });
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isConnected = gpuStatus?.connected && !gpuStatus?.isStale;
  const currentUrl = inputUrl || settings?.url || "";

  return (
    <div className="space-y-4">
      <Card className="overflow-visible" data-testid="card-gpu-connection">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-2">
              {isConnected ? (
                <Wifi className="w-5 h-5 text-emerald-400" />
              ) : (
                <WifiOff className="w-5 h-5 text-red-400" />
              )}
              <CardTitle className="text-lg">GPU Trainer Connection</CardTitle>
            </div>
            <Badge 
              variant={isConnected ? "default" : "destructive"}
              className={isConnected ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : ""}
              data-testid="badge-connection-status"
            >
              {isConnected ? "Connected" : "Disconnected"}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {isConnected && gpuStatus && (
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-md">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-medium text-emerald-400">Connected to GPU Trainer</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Cpu className="w-3 h-3" />
                  <span>{gpuStatus.gpuName || "Unknown GPU"}</span>
                </div>
                {gpuStatus.gpuMemoryUsed && gpuStatus.gpuMemoryTotal && (
                  <div className="flex items-center gap-1">
                    <HardDrive className="w-3 h-3" />
                    <span>
                      {(gpuStatus.gpuMemoryUsed / 1024).toFixed(1)}GB / {(gpuStatus.gpuMemoryTotal / 1024).toFixed(1)}GB
                    </span>
                  </div>
                )}
              </div>
              {gpuStatus.modelsLoaded && gpuStatus.modelsLoaded.length > 0 && (
                <div className="mt-2 text-xs text-muted-foreground">
                  <span className="font-medium">Models loaded:</span> {gpuStatus.modelsLoaded.join(", ")}
                </div>
              )}
            </div>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium">GPU Trainer URL</label>
            <div className="flex gap-2">
              <Input
                placeholder={settings?.defaultUrl || "http://localhost:8000"}
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                className="flex-1"
                data-testid="input-gpu-url"
              />
              <Button
                variant="outline"
                onClick={handleTestConnection}
                disabled={testConnectionMutation.isPending}
                data-testid="button-test-connection"
              >
                {testConnectionMutation.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  "Test"
                )}
              </Button>
            </div>
            {currentUrl && (
              <p className="text-xs text-muted-foreground">
                Current: {settings?.url || settings?.defaultUrl}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-muted-foreground" />
              <label className="text-sm font-medium">Prediction Mode</label>
            </div>
            <Select 
              value={predictionMode} 
              onValueChange={(value: "stf" | "mtf") => setPredictionMode(value)}
            >
              <SelectTrigger className="w-full" data-testid="select-prediction-mode">
                <SelectValue placeholder="Select prediction mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="stf">
                  <div className="flex flex-col">
                    <span className="font-medium">STF - Single Timeframe</span>
                    <span className="text-xs text-muted-foreground">15m only, 41 features (recommended)</span>
                  </div>
                </SelectItem>
                <SelectItem value="mtf">
                  <div className="flex flex-col">
                    <span className="font-medium">MTF - Multi-Timeframe</span>
                    <span className="text-xs text-muted-foreground">5m/15m/1h/4h fusion, 66 features</span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {predictionMode === "stf" 
                ? "Uses compute_technical_features (41 features) - matches 15m-trained models"
                : "Uses MTF fusion pipeline (66 features) - requires MTF-trained models"
              }
            </p>
          </div>

          {testResult && (
            <div className={`p-3 rounded-md border ${
              testResult.connected 
                ? "bg-emerald-500/10 border-emerald-500/30" 
                : "bg-red-500/10 border-red-500/30"
            }`}>
              <div className="flex items-center gap-2">
                {testResult.connected ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-400" />
                )}
                <span className={`text-sm font-medium ${
                  testResult.connected ? "text-emerald-400" : "text-red-400"
                }`}>
                  {testResult.message}
                </span>
              </div>
              {testResult.health && (
                <div className="mt-2 text-xs text-muted-foreground">
                  <div>GPU: {testResult.health.gpu_name}</div>
                  <div>Models: {testResult.health.models_loaded?.join(", ") || "None loaded"}</div>
                </div>
              )}
            </div>
          )}

          <Button
            onClick={handleSaveAndConnect}
            disabled={saveSettingsMutation.isPending || (!inputUrl && !settings?.url)}
            className="w-full"
            data-testid="button-save-connect"
          >
            {saveSettingsMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Connecting...
              </>
            ) : (
              <>
                <Settings className="w-4 h-4 mr-2" />
                Save & Connect
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <Card className="overflow-visible" data-testid="card-ngrok-setup">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Terminal className="w-5 h-5 text-blue-400" />
            <CardTitle className="text-lg">Setup Instructions</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              To connect your local GPU trainer (running on your RTX 4070) to this dashboard, 
              you'll need to expose it using ngrok:
            </p>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">Step 1</Badge>
                <span>Install ngrok if you haven't already</span>
              </div>
              <div className="bg-muted/50 p-2 rounded-md font-mono text-xs flex items-center justify-between">
                <code>pip install ngrok</code>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={() => copyToClipboard("pip install ngrok")}
                >
                  <Copy className="w-3 h-3" />
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">Step 2</Badge>
                <span>Start your GPU trainer (port 8000)</span>
              </div>
              <div className="bg-muted/50 p-2 rounded-md font-mono text-xs flex items-center justify-between">
                <code>python main.py api</code>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={() => copyToClipboard("python main.py api")}
                >
                  <Copy className="w-3 h-3" />
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">Step 3</Badge>
                <span>Expose it with ngrok</span>
              </div>
              <div className="bg-muted/50 p-2 rounded-md font-mono text-xs flex items-center justify-between">
                <code>ngrok http 8000</code>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={() => copyToClipboard("ngrok http 8000")}
                >
                  <Copy className="w-3 h-3" />
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">Step 4</Badge>
                <span>Copy the HTTPS URL from ngrok (e.g., https://abc123.ngrok.io)</span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">Step 5</Badge>
                <span>Paste the URL above and click "Save & Connect"</span>
              </div>
            </div>
          </div>

          <div className="pt-2 border-t">
            <a 
              href="https://ngrok.com/download" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sm text-blue-400 hover:underline flex items-center gap-1"
            >
              <ExternalLink className="w-3 h-3" />
              Download ngrok
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
