import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Database, Download, FileJson, Loader2, Clock, CheckCircle, AlertCircle, Trash2, XCircle } from "lucide-react";
import { useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";

interface TimeframeData {
  timeframe: string;
  assets: { symbol: string; candles: number }[];
}

interface NNDataSummary {
  timeframes: TimeframeData[];
  totalCandles: number;
}

interface DownloadProgress {
  symbol: string;
  timeframe: string;
  status: "pending" | "downloading" | "complete" | "error";
  progress: number;
  candlesFetched: number;
}

export function NeuralNetworkDataCard() {
  const [selectedYears, setSelectedYears] = useState("3");
  const { toast } = useToast();

  const { data: summary, isLoading } = useQuery<NNDataSummary>({
    queryKey: ["/api/nn-data/summary"],
    refetchInterval: 10000,
  });

  const { data: progressData } = useQuery<{ progress: DownloadProgress[] }>({
    queryKey: ["/api/nn-data/progress"],
    refetchInterval: 2000,
  });

  const downloadMutation = useMutation({
    mutationFn: (years: number) => apiRequest("POST", "/api/nn-data/download", { years }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/nn-data/summary"] });
      queryClient.invalidateQueries({ queryKey: ["/api/nn-data/progress"] });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/nn-data/cancel"),
    onSuccess: () => {
      toast({ title: "Download Cancelled", description: "Download cancellation requested" });
      queryClient.invalidateQueries({ queryKey: ["/api/nn-data/progress"] });
    },
  });

  const clearMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/nn-data/clear"),
    onSuccess: (data: unknown) => {
      const result = data as { deletedCandles?: number };
      toast({ 
        title: "Data Cleared", 
        description: `Cleared ${result.deletedCandles?.toLocaleString() || 0} candles` 
      });
      queryClient.invalidateQueries({ queryKey: ["/api/nn-data/summary"] });
    },
  });

  const isDownloading = progressData?.progress?.some(p => p.status === "downloading") || false;
  const downloadProgress = progressData?.progress || [];

  const getTimeframeTotal = (tf: TimeframeData): number => {
    return tf.assets.reduce((sum, a) => sum + a.candles, 0);
  };

  const hasAnyData = summary && summary.totalCandles > 0;

  return (
    <Card className="border-purple-500/30" data-testid="card-nn-data">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-purple-400" />
            <CardTitle className="text-lg">Neural Network Data</CardTitle>
          </div>
          <Badge variant="outline" className="text-purple-400 border-purple-400/30">
            Multi-Timeframe
          </Badge>
        </div>
        <CardDescription>
          Separate data for GPU training: 1m, 5m, 1h, 4h timeframes (Strategy Learner uses 15m)
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {summary?.timeframes.map(tf => {
                const total = getTimeframeTotal(tf);
                const hasData = total > 0;
                return (
                  <div
                    key={tf.timeframe}
                    className={`p-3 rounded-lg border ${
                      hasData ? "border-emerald-500/30 bg-emerald-500/5" : "border-border"
                    }`}
                    data-testid={`tf-${tf.timeframe}`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-sm">{tf.timeframe.toUpperCase()}</span>
                      {hasData ? (
                        <CheckCircle className="h-4 w-4 text-emerald-400" />
                      ) : (
                        <AlertCircle className="h-4 w-4 text-muted-foreground" />
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {total.toLocaleString()} candles
                    </div>
                    <div className="mt-1 text-xs">
                      {tf.assets.map(a => (
                        <span key={a.symbol} className="mr-2">
                          {a.symbol.replace("USDT", "")}: {a.candles.toLocaleString()}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>Total: {summary?.totalCandles?.toLocaleString() || 0} candles across all timeframes</span>
            </div>

            {isDownloading && (
              <div className="space-y-2 p-3 bg-muted/30 rounded-lg">
                <div className="text-sm font-medium flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Downloading...
                </div>
                {downloadProgress
                  .filter(p => p.status === "downloading")
                  .map(p => (
                    <div key={`${p.symbol}_${p.timeframe}`} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span>
                          {p.symbol} {p.timeframe}
                        </span>
                        <span>{p.progress.toFixed(0)}%</span>
                      </div>
                      <Progress value={p.progress} className="h-1" />
                    </div>
                  ))}
              </div>
            )}

            <div className="flex items-center gap-3 pt-2">
              <Select
                value={selectedYears}
                onValueChange={setSelectedYears}
                disabled={isDownloading}
              >
                <SelectTrigger className="w-32" data-testid="select-nn-years">
                  <SelectValue placeholder="Years" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 Year</SelectItem>
                  <SelectItem value="2">2 Years</SelectItem>
                  <SelectItem value="3">3 Years</SelectItem>
                  <SelectItem value="5">5 Years</SelectItem>
                </SelectContent>
              </Select>

              <Button
                onClick={() => downloadMutation.mutate(Number(selectedYears))}
                disabled={isDownloading || downloadMutation.isPending}
                className="bg-purple-600"
                data-testid="button-download-nn-data"
              >
                {isDownloading || downloadMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 mr-2" />
                    Download All Timeframes
                  </>
                )}
              </Button>

              {isDownloading && (
                <Button
                  variant="outline"
                  onClick={() => cancelMutation.mutate()}
                  disabled={cancelMutation.isPending}
                  className="border-red-500/50 text-red-400"
                  data-testid="button-cancel-nn-download"
                >
                  <XCircle className="h-4 w-4 mr-2" />
                  Cancel
                </Button>
              )}

              {hasAnyData && !isDownloading && (
                <>
                  <Button
                    variant="outline"
                    onClick={() => window.open("/api/nn-data/export", "_blank")}
                    data-testid="button-export-nn-data"
                  >
                    <FileJson className="h-4 w-4 mr-2" />
                    Export JSON
                  </Button>
                  
                  <Button
                    variant="outline"
                    onClick={() => clearMutation.mutate()}
                    disabled={clearMutation.isPending}
                    className="border-red-500/50 text-red-400"
                    data-testid="button-clear-nn-data"
                  >
                    {clearMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4 mr-2" />
                    )}
                    Clear Data
                  </Button>
                </>
              )}
            </div>

            <p className="text-xs text-muted-foreground">
              This data is separate from 15m data used by Strategy Learner and Pattern Memory.
              Multi-timeframe data allows the GPU neural networks to learn patterns across different time scales.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
