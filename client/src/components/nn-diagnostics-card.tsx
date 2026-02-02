import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle, Activity, TrendingUp, TrendingDown, Minus, Wifi, WifiOff, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, LineChart, Line, CartesianGrid, Legend } from "recharts";

interface DiagnosticsData {
  timestamp: string;
  gpuConnected: boolean;
  predictionDistribution: { SHORT: number; HOLD: number; LONG: number };
  calibration: { ece: number; bins: { confidence: number; accuracy: number; count: number }[] };
  uncertainty: { mean: number; std: number; histogram: { bin: string; count: number }[] };
  trainingHealth: { status: string; alerts: string[]; lastCheck: string | null; labelDistribution?: Record<string, number> };
  driftStatus: { psi: number; ece: number; alert: boolean; status: string; nPredictions?: number; psiPerClass?: Record<string, number> };
  modelSensitivity?: { overall_status: string; models?: Record<string, { status: string; response_delta: number }> };
  gpuHealth?: { status: string; models_loaded?: number };
}

const COLORS = {
  SHORT: "#ef4444",
  HOLD: "#f59e0b", 
  LONG: "#22c55e"
};

export function NNDiagnosticsCard() {
  const { data, isLoading, error, refetch } = useQuery<DiagnosticsData>({
    queryKey: ["/api/gpu/diagnostics/dashboard"],
    refetchInterval: 30000
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Neural Network Diagnostics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-40">
            <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Neural Network Diagnostics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            Failed to load diagnostics
          </div>
        </CardContent>
      </Card>
    );
  }

  const predictionData = [
    { name: "SHORT", value: data.predictionDistribution.SHORT || 0, color: COLORS.SHORT },
    { name: "HOLD", value: data.predictionDistribution.HOLD || 0, color: COLORS.HOLD },
    { name: "LONG", value: data.predictionDistribution.LONG || 0, color: COLORS.LONG }
  ].filter(d => d.value > 0);

  const hasNoPredictions = predictionData.length === 0 || predictionData.every(d => d.value === 0);

  const labelData = data.trainingHealth.labelDistribution 
    ? Object.entries(data.trainingHealth.labelDistribution).map(([name, value]) => ({
        name,
        value: Math.round((value as number) * 100),
        fill: COLORS[name as keyof typeof COLORS] || "#6b7280"
      }))
    : [];

  const calibrationData = data.calibration.bins?.map((bin, i) => ({
    bin: `${(bin.confidence * 100).toFixed(0)}%`,
    accuracy: bin.accuracy * 100,
    perfect: bin.confidence * 100,
    count: bin.count
  })) || [];

  const getHealthColor = (status: string) => {
    if (status === "healthy" || status === "ok") return "text-green-500";
    if (status === "warning") return "text-amber-500";
    if (status === "critical" || status === "error") return "text-red-500";
    return "text-muted-foreground";
  };

  const getHealthIcon = (status: string) => {
    if (status === "healthy" || status === "ok") return <CheckCircle className="h-4 w-4 text-green-500" />;
    if (status === "warning") return <AlertTriangle className="h-4 w-4 text-amber-500" />;
    if (status === "critical" || status === "error") return <AlertTriangle className="h-4 w-4 text-red-500" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };

  const psiThreshold = 0.25;
  const eceThreshold = 0.15;
  const psiAlert = data.driftStatus.psi > psiThreshold;
  const eceAlert = data.driftStatus.ece > eceThreshold;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card data-testid="card-gpu-status">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              {data.gpuConnected ? <Wifi className="h-4 w-4 text-green-500" /> : <WifiOff className="h-4 w-4 text-red-500" />}
              GPU Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {data.gpuConnected ? "Connected" : "Disconnected"}
            </div>
            <p className="text-xs text-muted-foreground">
              {data.gpuHealth?.models_loaded ?? 0} models loaded
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-training-health">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              {getHealthIcon(data.trainingHealth.status)}
              Training Health
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold capitalize ${getHealthColor(data.trainingHealth.status)}`}>
              {data.trainingHealth.status}
            </div>
            <p className="text-xs text-muted-foreground">
              {data.trainingHealth.alerts.length} alert{data.trainingHealth.alerts.length !== 1 ? 's' : ''}
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-drift-psi">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Drift (PSI)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${psiAlert ? 'text-red-500' : 'text-green-500'}`}>
              {data.driftStatus.psi.toFixed(3)}
            </div>
            <Progress value={Math.min(data.driftStatus.psi / psiThreshold * 100, 100)} className="h-1 mt-2" />
            <p className="text-xs text-muted-foreground mt-1">
              Threshold: {psiThreshold}
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-drift-ece">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Calibration (ECE)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${eceAlert ? 'text-red-500' : 'text-green-500'}`}>
              {(data.driftStatus.ece * 100).toFixed(1)}%
            </div>
            <Progress value={Math.min(data.driftStatus.ece / eceThreshold * 100, 100)} className="h-1 mt-2" />
            <p className="text-xs text-muted-foreground mt-1">
              Threshold: {(eceThreshold * 100).toFixed(0)}%
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card data-testid="card-prediction-distribution">
          <CardHeader>
            <CardTitle className="text-sm font-medium">Prediction Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {hasNoPredictions ? (
              <div className="flex items-center justify-center h-48 text-muted-foreground">
                No predictions recorded yet
              </div>
            ) : (
              <div className="flex items-center gap-4">
                <ResponsiveContainer width="50%" height={160}>
                  <PieChart>
                    <Pie
                      data={predictionData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={60}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      labelLine={false}
                    >
                      {predictionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex-1 space-y-2">
                  {predictionData.map((entry) => (
                    <div key={entry.name} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                        <span className="text-sm">{entry.name}</span>
                      </div>
                      <span className="font-mono text-sm">{entry.value}</span>
                    </div>
                  ))}
                  <div className="pt-2 border-t text-xs text-muted-foreground">
                    Total: {data.driftStatus.nPredictions ?? predictionData.reduce((a, b) => a + b.value, 0)}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-label-distribution">
          <CardHeader>
            <CardTitle className="text-sm font-medium">Training Label Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {labelData.length === 0 ? (
              <div className="flex items-center justify-center h-48 text-muted-foreground">
                No label data available
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={labelData} layout="vertical">
                  <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="name" width={60} />
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {labelData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card data-testid="card-calibration-curve">
          <CardHeader>
            <CardTitle className="text-sm font-medium">Calibration Curve</CardTitle>
          </CardHeader>
          <CardContent>
            {calibrationData.length === 0 ? (
              <div className="flex items-center justify-center h-48 text-muted-foreground">
                Not enough predictions for calibration
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={calibrationData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="bin" fontSize={10} />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} fontSize={10} />
                  <Tooltip 
                    formatter={(value: number) => `${value.toFixed(1)}%`}
                    labelFormatter={(label) => `Confidence: ${label}`}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#22c55e" name="Actual Accuracy" strokeWidth={2} dot />
                  <Line type="monotone" dataKey="perfect" stroke="#6b7280" name="Perfect Calibration" strokeDasharray="5 5" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-alerts">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium">System Alerts</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => refetch()} data-testid="button-refresh-diagnostics">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {data.trainingHealth.alerts.length === 0 && !psiAlert && !eceAlert ? (
                <div className="flex items-center gap-2 text-green-500">
                  <CheckCircle className="h-4 w-4" />
                  <span className="text-sm">All systems healthy</span>
                </div>
              ) : (
                <>
                  {psiAlert && (
                    <div className="flex items-start gap-2 p-2 rounded bg-red-500/10 border border-red-500/20">
                      <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5" />
                      <div>
                        <span className="text-sm font-medium text-red-500">Prediction Drift Detected</span>
                        <p className="text-xs text-muted-foreground">PSI {data.driftStatus.psi.toFixed(3)} exceeds threshold</p>
                      </div>
                    </div>
                  )}
                  {eceAlert && (
                    <div className="flex items-start gap-2 p-2 rounded bg-amber-500/10 border border-amber-500/20">
                      <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5" />
                      <div>
                        <span className="text-sm font-medium text-amber-500">Calibration Warning</span>
                        <p className="text-xs text-muted-foreground">ECE {(data.driftStatus.ece * 100).toFixed(1)}% exceeds {(eceThreshold * 100).toFixed(0)}%</p>
                      </div>
                    </div>
                  )}
                  {data.trainingHealth.alerts.map((alert, i) => (
                    <div key={i} className="flex items-start gap-2 p-2 rounded bg-amber-500/10 border border-amber-500/20">
                      <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span className="text-sm">{alert}</span>
                    </div>
                  ))}
                </>
              )}
            </div>
            <div className="mt-3 pt-3 border-t text-xs text-muted-foreground">
              Last updated: {new Date(data.timestamp).toLocaleTimeString()}
            </div>
          </CardContent>
        </Card>
      </div>

      {data.modelSensitivity?.models && (
        <Card data-testid="card-model-sensitivity">
          <CardHeader>
            <CardTitle className="text-sm font-medium">Model Sensitivity Check</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(data.modelSensitivity.models).map(([name, info]) => (
                <div key={name} className="p-3 rounded-lg bg-muted/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium uppercase">{name}</span>
                    <Badge 
                      variant={info.status === "OK" ? "default" : "destructive"} 
                      className="text-xs"
                    >
                      {info.status}
                    </Badge>
                  </div>
                  <div className="text-lg font-mono">
                    Δ {(info.response_delta * 100).toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-3 border-t">
              <Badge 
                variant={data.modelSensitivity.overall_status.includes("CRITICAL") ? "destructive" : "secondary"}
                className="text-xs"
              >
                {data.modelSensitivity.overall_status}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
