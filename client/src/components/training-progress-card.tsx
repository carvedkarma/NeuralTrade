import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { 
  Activity, AlertTriangle, CheckCircle, Clock, Cpu, 
  TrendingDown, RefreshCw, Square, Zap
} from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

interface EpochMetrics {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc?: number;
  val_acc?: number;
  timestamp: string;
}

interface TrainingStatus {
  is_training: boolean;
  current_epoch: number;
  total_epochs: number;
  current_model: string | null;
  progress: number;
  metrics: Record<string, unknown>;
  epoch_history: EpochMetrics[];
  start_time: string | null;
  eta_seconds: number | null;
  health_warnings: string[];
  last_update: string | null;
  per_head_losses: Record<string, number>;
  learning_rate: number | null;
  best_val_loss: number | null;
  early_stop_counter: number;
  connected?: boolean;
  error?: string;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function formatTime(isoString: string | null): string {
  if (!isoString) return "N/A";
  return new Date(isoString).toLocaleTimeString();
}

export function TrainingProgressCard() {
  const { data, isLoading, error, refetch } = useQuery<TrainingStatus>({
    queryKey: ["/api/gpu/training/status"],
    refetchInterval: (query) => {
      const data = query.state.data as TrainingStatus | undefined;
      return data?.is_training ? 2000 : 10000;
    }
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            Training Progress
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
            <Cpu className="h-5 w-5" />
            Training Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            <p>Unable to fetch training status</p>
            <Button variant="outline" size="sm" className="mt-2" onClick={() => refetch()} data-testid="button-retry-training-status">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  const isConnected = data.connected !== false;

  const progressPct = data.total_epochs > 0 
    ? (data.current_epoch / data.total_epochs) * 100 
    : 0;

  const lossChartData = data.epoch_history?.map(e => ({
    epoch: e.epoch,
    trainLoss: e.train_loss,
    valLoss: e.val_loss,
    trainAcc: e.train_acc ? e.train_acc * 100 : undefined,
    valAcc: e.val_acc ? e.val_acc * 100 : undefined
  })) || [];

  return (
    <Card data-testid="card-training-progress">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            Training Progress
          </div>
          <div className="flex items-center gap-2">
            {data.is_training ? (
              <Badge variant="default" className="animate-pulse" data-testid="badge-training-active">
                <Activity className="h-3 w-3 mr-1" />
                Training
              </Badge>
            ) : (
              <Badge variant="secondary" data-testid="badge-training-idle">
                Idle
              </Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {data.is_training && (
          <>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">{data.current_model || "Model"}</span>
                <span className="text-muted-foreground">
                  Epoch {data.current_epoch}/{data.total_epochs}
                </span>
              </div>
              <Progress value={progressPct} className="h-2" />
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>{progressPct.toFixed(1)}% complete</span>
                {data.eta_seconds && (
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    ETA: {formatDuration(data.eta_seconds)}
                  </span>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div className="p-2 rounded bg-muted/50">
                <span className="text-xs text-muted-foreground block">Learning Rate</span>
                <span className="font-mono font-bold">{data.learning_rate?.toExponential(2) || "N/A"}</span>
              </div>
              <div className="p-2 rounded bg-muted/50">
                <span className="text-xs text-muted-foreground block">Best Val Loss</span>
                <span className="font-mono font-bold">{data.best_val_loss?.toFixed(4) || "N/A"}</span>
              </div>
              <div className="p-2 rounded bg-muted/50">
                <span className="text-xs text-muted-foreground block">Early Stop</span>
                <span className="font-mono font-bold">{data.early_stop_counter}/10</span>
              </div>
              <div className="p-2 rounded bg-muted/50">
                <span className="text-xs text-muted-foreground block">Started</span>
                <span className="font-mono text-xs">{formatTime(data.start_time)}</span>
              </div>
            </div>

            {data.per_head_losses && Object.keys(data.per_head_losses).length > 0 && (
              <div className="space-y-2">
                <span className="text-xs text-muted-foreground font-medium">Per-Head Losses</span>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                  {Object.entries(data.per_head_losses).map(([head, loss]) => (
                    <div key={head} className="p-2 rounded bg-muted/30 text-center">
                      <span className="text-xs text-muted-foreground block truncate">{head}</span>
                      <span className="font-mono text-xs font-bold">{(loss as number).toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {data.health_warnings && data.health_warnings.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs text-muted-foreground font-medium">Health Warnings</span>
                {data.health_warnings.map((warning, i) => (
                  <div key={i} className="flex items-start gap-2 p-2 rounded bg-amber-500/10 border border-amber-500/20">
                    <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm">{warning}</span>
                  </div>
                ))}
              </div>
            )}

            {lossChartData.length > 0 && (
              <div>
                <span className="text-xs text-muted-foreground font-medium block mb-2">Loss Curves</span>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={lossChartData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="epoch" fontSize={10} />
                    <YAxis fontSize={10} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="trainLoss" stroke="#3b82f6" name="Train Loss" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="valLoss" stroke="#ef4444" name="Val Loss" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </>
        )}

        {!data.is_training && (
          <div className="text-center py-8 text-muted-foreground">
            <Cpu className="h-12 w-12 mx-auto mb-4 opacity-50" />
            {isConnected ? (
              <>
                <p>No training in progress</p>
                <p className="text-sm mt-1">Start training from the GPU Training tab</p>
              </>
            ) : (
              <>
                <p>GPU trainer not connected</p>
                <p className="text-sm mt-1">{data.error || "Connect to local GPU trainer to start training"}</p>
              </>
            )}
          </div>
        )}

        <div className="pt-2 border-t text-xs text-muted-foreground flex items-center justify-between">
          <span>Last update: {formatTime(data.last_update)}</span>
          <Button variant="ghost" size="sm" onClick={() => refetch()} data-testid="button-refresh-training">
            <RefreshCw className="h-3 w-3" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
