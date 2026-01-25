import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { 
  Cpu, Zap, Activity, TrendingDown, TrendingUp, 
  Server, Layers, PlayCircle, StopCircle, RefreshCw,
  CheckCircle2, XCircle, AlertTriangle, HardDrive
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend } from "recharts";

interface GPUMetrics {
  gpuAvailable: boolean;
  gpuName: string | null;
  gpuMemoryUsed: number | null;
  gpuMemoryTotal: number | null;
  gpuMemoryPercent: number;
  modelsLoaded: string[];
  uptime: number;
  isTraining: boolean;
  trainingProgress: number;
  currentModel: string | null;
  trainingMetrics: Record<string, number>;
}

interface ModelPerformance {
  name: string;
  accuracy: number;
  loss: number;
  epochs: number;
  status: "training" | "ready" | "pending";
}

interface LossHistory {
  epoch: number;
  trainLoss: number;
  valLoss: number;
}

interface GPUTrainingCardProps {
  gpuMetrics?: GPUMetrics | null;
  modelPerformance?: ModelPerformance[];
  lossHistory?: LossHistory[];
  onStartTraining?: (modelType: string) => void;
  onStopTraining?: () => void;
}

function formatBytes(bytes: number | null): string {
  if (bytes === null) return "N/A";
  const gb = bytes / (1024 * 1024 * 1024);
  return `${gb.toFixed(1)} GB`;
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export function GPUStatusCard({ gpuMetrics }: { gpuMetrics?: GPUMetrics | null }) {
  if (!gpuMetrics || !gpuMetrics.gpuAvailable) {
    return (
      <Card data-testid="card-gpu-status-offline">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Cpu className="h-4 w-4 text-muted-foreground" />
            GPU Trainer
            <Badge variant="outline" className="ml-auto text-amber-400 bg-amber-500/10">
              <XCircle className="h-3 w-3 mr-1" />
              Offline
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-muted/30 rounded-lg p-4 text-center">
            <Server className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
            <p className="text-sm font-medium">GPU Trainer Not Connected</p>
            <p className="text-xs text-muted-foreground mt-1">
              Start the local GPU trainer to enable deep learning
            </p>
          </div>
          
          <div className="bg-muted/20 rounded p-3 text-xs space-y-2">
            <p className="font-medium">To connect your GPU:</p>
            <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
              <li>Navigate to <code className="bg-muted px-1 rounded">gpu_trainer/</code></li>
              <li>Run <code className="bg-muted px-1 rounded">pip install -r requirements.txt</code></li>
              <li>Start with <code className="bg-muted px-1 rounded">python main.py serve</code></li>
            </ol>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="card-gpu-status">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Cpu className="h-4 w-4 text-primary" />
          GPU Trainer
          <Badge variant="outline" className="ml-auto text-emerald-400 bg-emerald-500/10">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            Connected
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-muted/30 rounded p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <Zap className="h-3 w-3" />
              GPU
            </div>
            <div className="text-sm font-medium truncate" data-testid="text-gpu-name">
              {gpuMetrics.gpuName || "Unknown GPU"}
            </div>
          </div>
          <div className="bg-muted/30 rounded p-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
              <Activity className="h-3 w-3" />
              Uptime
            </div>
            <div className="text-sm font-medium" data-testid="text-gpu-uptime">
              {formatUptime(gpuMetrics.uptime)}
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground flex items-center gap-1">
              <HardDrive className="h-3 w-3" />
              VRAM Usage
            </span>
            <span className="font-medium" data-testid="text-vram-usage">
              {formatBytes(gpuMetrics.gpuMemoryUsed)} / {formatBytes(gpuMetrics.gpuMemoryTotal)}
            </span>
          </div>
          <Progress 
            value={gpuMetrics.gpuMemoryPercent} 
            className="h-2"
            data-testid="progress-vram"
          />
          <div className="text-xs text-muted-foreground text-right">
            {gpuMetrics.gpuMemoryPercent.toFixed(1)}% used
          </div>
        </div>

        {gpuMetrics.modelsLoaded.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground flex items-center gap-1">
              <Layers className="h-3 w-3" />
              Loaded Models ({gpuMetrics.modelsLoaded.length})
            </div>
            <div className="flex flex-wrap gap-1">
              {gpuMetrics.modelsLoaded.map((model) => (
                <Badge key={model} variant="secondary" className="text-xs">
                  {model}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function TrainingProgressCard({ 
  gpuMetrics, 
  lossHistory,
  onStartTraining,
  onStopTraining 
}: GPUTrainingCardProps) {
  const isTraining = gpuMetrics?.isTraining || false;
  const progress = gpuMetrics?.trainingProgress || 0;
  const currentModel = gpuMetrics?.currentModel;
  const metrics = gpuMetrics?.trainingMetrics || {};

  return (
    <Card data-testid="card-training-progress">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          Training Progress
          {isTraining && (
            <Badge variant="outline" className="ml-auto text-amber-400 bg-amber-500/10 animate-pulse">
              Training...
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {isTraining ? (
          <>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">
                  Training: <span className="font-medium text-foreground">{currentModel}</span>
                </span>
                <span className="font-medium">{progress.toFixed(1)}%</span>
              </div>
              <Progress value={progress} className="h-2" data-testid="progress-training" />
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div className="bg-muted/30 rounded p-2 text-center">
                <div className="text-xs text-muted-foreground">Epoch</div>
                <div className="text-lg font-bold" data-testid="text-current-epoch">
                  {metrics.epoch || 0}
                </div>
              </div>
              <div className="bg-muted/30 rounded p-2 text-center">
                <div className="text-xs text-muted-foreground">Train Loss</div>
                <div className="text-lg font-bold text-amber-400" data-testid="text-train-loss">
                  {metrics.train_loss?.toFixed(4) || "—"}
                </div>
              </div>
              <div className="bg-muted/30 rounded p-2 text-center">
                <div className="text-xs text-muted-foreground">Val Loss</div>
                <div className="text-lg font-bold text-emerald-400" data-testid="text-val-loss">
                  {metrics.val_loss?.toFixed(4) || "—"}
                </div>
              </div>
            </div>

            {onStopTraining && (
              <Button 
                variant="destructive" 
                size="sm" 
                className="w-full"
                onClick={onStopTraining}
                data-testid="button-stop-training"
              >
                <StopCircle className="h-4 w-4 mr-2" />
                Stop Training
              </Button>
            )}
          </>
        ) : (
          <>
            <div className="bg-muted/30 rounded-lg p-4 text-center">
              <RefreshCw className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
              <p className="text-sm font-medium">No Active Training</p>
              <p className="text-xs text-muted-foreground mt-1">
                Start training a neural network model
              </p>
            </div>

            {onStartTraining && (
              <div className="grid grid-cols-2 gap-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => onStartTraining("transformer")}
                  data-testid="button-train-transformer"
                >
                  <Zap className="h-3 w-3 mr-1" />
                  Transformer
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => onStartTraining("lstm")}
                  data-testid="button-train-lstm"
                >
                  <Layers className="h-3 w-3 mr-1" />
                  LSTM
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => onStartTraining("cnn")}
                  data-testid="button-train-cnn"
                >
                  <Activity className="h-3 w-3 mr-1" />
                  CNN
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => onStartTraining("tft")}
                  data-testid="button-train-tft"
                >
                  <TrendingUp className="h-3 w-3 mr-1" />
                  TFT
                </Button>
              </div>
            )}
          </>
        )}

        {lossHistory && lossHistory.length > 0 && (
          <div className="h-40 mt-4">
            <div className="text-xs font-medium text-muted-foreground mb-2">Loss Curve</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossHistory}>
                <XAxis 
                  dataKey="epoch" 
                  tick={{ fontSize: 10 }} 
                  stroke="hsl(var(--muted-foreground))"
                />
                <YAxis 
                  tick={{ fontSize: 10 }} 
                  stroke="hsl(var(--muted-foreground))"
                  width={40}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--card))', 
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px',
                    fontSize: '12px'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Line 
                  type="monotone" 
                  dataKey="trainLoss" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={2}
                  dot={false}
                  name="Train"
                />
                <Line 
                  type="monotone" 
                  dataKey="valLoss" 
                  stroke="hsl(var(--chart-2))" 
                  strokeWidth={2}
                  dot={false}
                  name="Val"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function ModelComparisonCard({ modelPerformance }: { modelPerformance?: ModelPerformance[] }) {
  const defaultModels: ModelPerformance[] = [
    { name: "Transformer", accuracy: 0, loss: 0, epochs: 0, status: "pending" },
    { name: "TFT", accuracy: 0, loss: 0, epochs: 0, status: "pending" },
    { name: "LSTM", accuracy: 0, loss: 0, epochs: 0, status: "pending" },
    { name: "CNN", accuracy: 0, loss: 0, epochs: 0, status: "pending" },
    { name: "VAE", accuracy: 0, loss: 0, epochs: 0, status: "pending" },
    { name: "GNN", accuracy: 0, loss: 0, epochs: 0, status: "pending" },
    { name: "PPO", accuracy: 0, loss: 0, epochs: 0, status: "pending" },
  ];

  const models = modelPerformance || defaultModels;

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "ready":
        return <Badge variant="outline" className="text-emerald-400 bg-emerald-500/10">Ready</Badge>;
      case "training":
        return <Badge variant="outline" className="text-amber-400 bg-amber-500/10 animate-pulse">Training</Badge>;
      default:
        return <Badge variant="outline" className="text-muted-foreground">Pending</Badge>;
    }
  };

  return (
    <Card data-testid="card-model-comparison">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Layers className="h-4 w-4 text-primary" />
          Neural Network Models
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {models.map((model) => (
            <div 
              key={model.name}
              className="flex items-center justify-between p-2 bg-muted/30 rounded"
              data-testid={`model-row-${model.name.toLowerCase()}`}
            >
              <div className="flex items-center gap-3">
                <div className="w-24">
                  <span className="text-sm font-medium">{model.name}</span>
                </div>
                {getStatusBadge(model.status)}
              </div>
              <div className="flex items-center gap-4 text-xs">
                <div className="text-center">
                  <div className="text-muted-foreground">Accuracy</div>
                  <div className={`font-medium ${model.accuracy > 0 ? 'text-emerald-400' : 'text-muted-foreground'}`}>
                    {model.accuracy > 0 ? `${(model.accuracy * 100).toFixed(1)}%` : "—"}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-muted-foreground">Loss</div>
                  <div className={`font-medium ${model.loss > 0 ? 'text-amber-400' : 'text-muted-foreground'}`}>
                    {model.loss > 0 ? model.loss.toFixed(4) : "—"}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-muted-foreground">Epochs</div>
                  <div className="font-medium">
                    {model.epochs > 0 ? model.epochs : "—"}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 p-3 bg-primary/5 rounded text-xs text-muted-foreground">
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle className="h-3 w-3 text-amber-400" />
            <span className="font-medium">Local GPU Required</span>
          </div>
          <p>
            These models run on your local RTX 4070 GPU. Start the trainer to enable training and inference.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

export function GPUTrainingSection(props: GPUTrainingCardProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <GPUStatusCard gpuMetrics={props.gpuMetrics} />
      <TrainingProgressCard {...props} />
      <div className="md:col-span-2">
        <ModelComparisonCard modelPerformance={props.modelPerformance} />
      </div>
    </div>
  );
}

export type { GPUMetrics, ModelPerformance, LossHistory };
