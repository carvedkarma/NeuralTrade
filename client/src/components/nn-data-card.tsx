import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Database, Download, FileJson, Loader2, Clock, CheckCircle, AlertCircle, Trash2, XCircle, PlayCircle, Zap, HardDrive } from "lucide-react";
import { useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

interface ResumableStatus {
  canResume: boolean;
  details: { symbol: string; timeframe: string; lastTimestamp: number | null; candleCount: number }[];
}

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
      queryClient.invalidateQueries({ queryKey: ["/api/nn-data/resumable"] });
    },
  });

  // Query for resumable status
  const { data: resumableStatus } = useQuery<ResumableStatus>({
    queryKey: ["/api/nn-data/resumable"],
    refetchInterval: 30000,
  });

  // Resume mutation - passes the selected years to ensure consistent data range
  const resumeMutation = useMutation({
    mutationFn: (years: number) => apiRequest("POST", "/api/nn-data/resume", { years }),
    onSuccess: () => {
      toast({ title: "Resume Started", description: "Continuing download from where it stopped" });
      queryClient.invalidateQueries({ queryKey: ["/api/nn-data/summary"] });
      queryClient.invalidateQueries({ queryKey: ["/api/nn-data/progress"] });
    },
  });

  const isDownloading = progressData?.progress?.some(p => p.status === "downloading") || false;
  const canResume = resumableStatus?.canResume && !isDownloading;
  const downloadProgress = progressData?.progress || [];

  const getTimeframeTotal = (tf: TimeframeData): number => {
    return tf.assets.reduce((sum, a) => sum + a.candles, 0);
  };

  const hasAnyData = summary && summary.totalCandles > 0;
  
  const completedStreams = downloadProgress.filter(p => p.status === "complete").length;
  const activeStreams = downloadProgress.filter(p => p.status === "downloading");
  const totalStreams = downloadProgress.length || 20;
  
  const activeProgress = activeStreams.reduce((sum, p) => sum + p.progress, 0) / 100;
  const overallProgress = totalStreams > 0 
    ? Math.round(((completedStreams + activeProgress) / totalStreams) * 100) 
    : 0;
  
  const estimatedCandles: Record<string, number> = {
    "1": 2100000,
    "2": 4200000,
    "3": 6300000,
    "5": 10500000
  };

  return (
    <Card className="border-purple-500/30" data-testid="card-nn-data">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-purple-400" />
            <CardTitle className="text-lg">Neural Network Data</CardTitle>
            <Badge variant="outline" className="text-purple-400 border-purple-400/30">
              Multi-Timeframe
            </Badge>
          </div>
          
          {!isDownloading ? (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  size="lg"
                  disabled={downloadMutation.isPending}
                  className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-semibold shadow-lg"
                  data-testid="button-download-all-prominent"
                >
                  <HardDrive className="h-5 w-5 mr-2" />
                  Download All GPU Data
                  <Zap className="h-4 w-4 ml-2 text-yellow-300" />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle className="flex items-center gap-2">
                    <HardDrive className="h-5 w-5 text-purple-400" />
                    Download GPU Training Data
                  </AlertDialogTitle>
                  <AlertDialogDescription asChild>
                    <div className="space-y-3">
                      <p>This will download historical market data for GPU neural network training:</p>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="p-2 bg-muted rounded">
                          <div className="font-medium">Timeframes</div>
                          <div className="text-muted-foreground">1m, 5m, 15m, 1h, 4h</div>
                        </div>
                        <div className="p-2 bg-muted rounded">
                          <div className="font-medium">Assets</div>
                          <div className="text-muted-foreground">BTC, ETH, SOL, BNB</div>
                        </div>
                        <div className="p-2 bg-muted rounded">
                          <div className="font-medium">Data Streams</div>
                          <div className="text-muted-foreground">20 parallel downloads</div>
                        </div>
                        <div className="p-2 bg-muted rounded">
                          <div className="font-medium">Est. Candles</div>
                          <div className="text-muted-foreground">~{(estimatedCandles[selectedYears] / 1000000).toFixed(1)}M ({selectedYears} years)</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 mt-2">
                        <span className="text-sm">Download period:</span>
                        <Select value={selectedYears} onValueChange={setSelectedYears}>
                          <SelectTrigger className="w-28" data-testid="select-download-years-dialog">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="1">1 Year</SelectItem>
                            <SelectItem value="2">2 Years</SelectItem>
                            <SelectItem value="3">3 Years</SelectItem>
                            <SelectItem value="5">5 Years</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel data-testid="button-cancel-download-dialog">Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => {
                      downloadMutation.mutate(Number(selectedYears));
                      toast({ 
                        title: "Download Started", 
                        description: `Downloading ${selectedYears} years of GPU training data...` 
                      });
                    }}
                    className="bg-purple-600"
                    data-testid="button-start-download-confirm"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Start Download
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          ) : (
            <div className="flex items-center gap-2">
              <Badge className="bg-purple-600 animate-pulse">
                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                Downloading...
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={() => cancelMutation.mutate()}
                disabled={cancelMutation.isPending}
                className="border-red-500/50 text-red-400"
                data-testid="button-cancel-download-header"
              >
                <XCircle className="h-4 w-4 mr-1" />
                Cancel
              </Button>
            </div>
          )}
        </div>
        <CardDescription>
          GPU training data: 1m, 5m, 15m, 1h, 4h timeframes across BTC, ETH, SOL, BNB
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
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

            {(isDownloading || downloadProgress.length > 0) && (
              <div className="space-y-3 p-4 bg-gradient-to-r from-purple-500/10 to-indigo-500/10 rounded-lg border border-purple-500/20">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium flex items-center gap-2">
                    {isDownloading ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
                        <span>Downloading {downloadProgress.filter(p => p.status === "downloading").length} streams...</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-4 w-4 text-emerald-400" />
                        <span>Download Complete</span>
                      </>
                    )}
                  </div>
                  <Badge variant="secondary" className="text-xs">
                    {completedStreams}/{totalStreams} streams
                  </Badge>
                </div>
                
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Overall Progress</span>
                    <span>{overallProgress}%</span>
                  </div>
                  <Progress value={overallProgress} className="h-2" />
                </div>
                
                {isDownloading && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
                    {downloadProgress
                      .filter(p => p.status === "downloading")
                      .map(p => (
                        <div key={`${p.symbol}_${p.timeframe}`} className="p-2 bg-background/50 rounded text-xs">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-emerald-400 font-medium">
                              {p.symbol.replace("USDT", "")} {p.timeframe}
                            </span>
                            <span>{p.progress.toFixed(0)}%</span>
                          </div>
                          <Progress value={p.progress} className="h-1" />
                          <div className="text-muted-foreground mt-1">
                            {p.candlesFetched.toLocaleString()} candles
                          </div>
                        </div>
                      ))}
                  </div>
                )}
                
                {downloadProgress.filter(p => p.status === "pending").length > 0 && (
                  <div className="text-xs text-muted-foreground">
                    {downloadProgress.filter(p => p.status === "pending").length} streams queued...
                  </div>
                )}
              </div>
            )}

            <div className="flex items-center gap-3 pt-2 flex-wrap">
              {canResume && (
                <Button
                  onClick={() => resumeMutation.mutate(Number(selectedYears))}
                  disabled={resumeMutation.isPending}
                  className="bg-emerald-600"
                  data-testid="button-resume-nn-download"
                >
                  {resumeMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <PlayCircle className="h-4 w-4 mr-2" />
                  )}
                  Resume Download
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
                  
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="outline"
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
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Clear All GPU Training Data?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will permanently delete {summary?.totalCandles?.toLocaleString()} candles across all timeframes. 
                          You will need to download the data again.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel data-testid="button-cancel-clear-dialog">Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={() => clearMutation.mutate()}
                          className="bg-red-600"
                          data-testid="button-confirm-clear-data"
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Clear All Data
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </>
              )}
            </div>

            <p className="text-xs text-muted-foreground">
              5 timeframes × 4 assets = 20 data streams for GPU neural network training.
              Strategy Learner and Pattern Memory share the 15m BTC data.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
