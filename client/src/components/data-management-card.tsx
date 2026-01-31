import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Database, Download, Trash2, RefreshCw, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { queryClient, apiRequest } from "@/lib/queryClient";

interface AssetDataSummary {
  symbol: string;
  totalCandles: number;
  startTs: number | null;
  endTs: number | null;
  daysOfData: number;
  yearsOfData: number;
  startDate: string | null;
  endDate: string | null;
}

interface MultiAssetDataSummary {
  assets: AssetDataSummary[];
  totalCandles: number;
  alignedTimeRange: {
    startTs: number | null;
    endTs: number | null;
    startDate: string | null;
    endDate: string | null;
  };
  allAssetsAvailable: boolean;
}

interface DownloadProgress {
  symbol: string;
  status: "pending" | "downloading" | "complete" | "error";
  progress: number;
  candlesFetched: number;
  totalExpected: number;
  error?: string;
}

interface DownloadStatus {
  inProgress: boolean;
  progress: DownloadProgress[];
}

const TIMEFRAME_OPTIONS = [
  { value: "all", label: "All Timeframes (1m, 5m, 15m, 1h, 4h)" },
  { value: "1m", label: "1 Minute" },
  { value: "5m", label: "5 Minutes" },
  { value: "15m", label: "15 Minutes" },
  { value: "1h", label: "1 Hour" },
  { value: "4h", label: "4 Hours" },
];

export function DataManagementCard() {
  const [selectedYears, setSelectedYears] = useState("1");
  const [selectedTimeframe, setSelectedTimeframe] = useState("15m");
  const [showDownloadSection, setShowDownloadSection] = useState(false);

  const { data: summary, isLoading: summaryLoading, refetch: refetchSummary } = useQuery<MultiAssetDataSummary>({
    queryKey: ["/api/data/summary"],
    refetchInterval: 5000,
  });

  const { data: downloadStatus, refetch: refetchDownloadStatus } = useQuery<DownloadStatus>({
    queryKey: ["/api/data/download/status"],
    refetchInterval: 2000,
  });

  const downloadMutation = useMutation({
    mutationFn: async ({ years, timeframe }: { years: number; timeframe: string }) => {
      return apiRequest("POST", "/api/data/download", { years, timeframe });
    },
    onSuccess: () => {
      refetchDownloadStatus();
    },
  });

  const clearMutation = useMutation({
    mutationFn: async () => {
      return apiRequest("POST", "/api/data/clear");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/data/summary"] });
      queryClient.invalidateQueries({ queryKey: ["/api/unified-learning/progress"] });
    },
  });

  const formatNumber = (n: number) => n.toLocaleString();

  const getAssetDisplayName = (symbol: string) => {
    return symbol.replace("USDT", "");
  };

  const getStatusIcon = (status: DownloadProgress["status"]) => {
    switch (status) {
      case "complete":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case "downloading":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      default:
        return <div className="h-4 w-4 rounded-full bg-muted" />;
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Database className="h-4 w-4" />
          Historical Data Management
        </CardTitle>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => refetchSummary()}
          disabled={summaryLoading}
          data-testid="button-refresh-data"
        >
          <RefreshCw className={`h-4 w-4 ${summaryLoading ? "animate-spin" : ""}`} />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {summaryLoading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : summary ? (
          <>
            <div className="grid gap-2">
              {summary.assets.map((asset) => (
                <div
                  key={asset.symbol}
                  className="flex items-center justify-between p-2 rounded-md bg-muted/50"
                  data-testid={`asset-row-${asset.symbol}`}
                >
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="font-mono">
                      {getAssetDisplayName(asset.symbol)}
                    </Badge>
                    {asset.totalCandles > 0 ? (
                      <span className="text-xs text-muted-foreground">
                        {asset.startDate} - {asset.endDate}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">No data</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {asset.totalCandles > 0 ? (
                      <>
                        <span className="text-sm font-medium">
                          {asset.yearsOfData.toFixed(1)}y
                        </span>
                        <span className="text-xs text-muted-foreground">
                          ({formatNumber(asset.totalCandles)} candles)
                        </span>
                      </>
                    ) : (
                      <span className="text-xs text-muted-foreground">-</span>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <div className="flex items-center justify-between pt-2 border-t">
              <div className="text-sm">
                <span className="text-muted-foreground">Total: </span>
                <span className="font-medium">{formatNumber(summary.totalCandles)} candles</span>
              </div>
              {summary.allAssetsAvailable && summary.alignedTimeRange.startDate && (
                <Badge variant="secondary">
                  Aligned: {summary.alignedTimeRange.startDate} - {summary.alignedTimeRange.endDate}
                </Badge>
              )}
            </div>

            {downloadStatus?.inProgress && (
              <div className="space-y-2 p-3 rounded-md bg-blue-500/10 border border-blue-500/20">
                <div className="text-sm font-medium text-blue-500">Downloading...</div>
                {downloadStatus.progress.map((p: DownloadProgress) => (
                  <div key={p.symbol} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(p.status)}
                        <span>{getAssetDisplayName(p.symbol)}</span>
                      </div>
                      <span>
                        {p.status === "complete"
                          ? formatNumber(p.candlesFetched)
                          : `${p.progress.toFixed(0)}%`}
                      </span>
                    </div>
                    {p.status === "downloading" && (
                      <Progress value={p.progress} className="h-1" />
                    )}
                  </div>
                ))}
              </div>
            )}

            <div className="flex gap-2">
              {!showDownloadSection ? (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowDownloadSection(true)}
                  disabled={downloadStatus?.inProgress}
                  className="flex-1"
                  data-testid="button-show-download"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download Data
                </Button>
              ) : (
                <div className="flex-1 space-y-2">
                  <div className="flex flex-wrap gap-2 items-center">
                    <span className="text-sm text-muted-foreground">Years:</span>
                    <Select value={selectedYears} onValueChange={setSelectedYears}>
                      <SelectTrigger className="w-28" data-testid="select-years">
                        <SelectValue placeholder="Select years">{selectedYears} year{parseInt(selectedYears) > 1 ? "s" : ""}</SelectValue>
                      </SelectTrigger>
                      <SelectContent className="z-50">
                        {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].map((y) => (
                          <SelectItem key={y} value={y.toString()}>
                            {y} year{y > 1 ? "s" : ""}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex flex-wrap gap-2 items-center">
                    <span className="text-sm text-muted-foreground">Timeframe:</span>
                    <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
                      <SelectTrigger className="w-52" data-testid="select-timeframe">
                        <SelectValue placeholder="Select timeframe">
                          {TIMEFRAME_OPTIONS.find(t => t.value === selectedTimeframe)?.label || selectedTimeframe}
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent className="z-50">
                        {TIMEFRAME_OPTIONS.map((tf) => (
                          <SelectItem key={tf.value} value={tf.value}>
                            {tf.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      onClick={() => {
                        downloadMutation.mutate({ 
                          years: parseInt(selectedYears), 
                          timeframe: selectedTimeframe 
                        });
                        setShowDownloadSection(false);
                      }}
                      disabled={downloadMutation.isPending || downloadStatus?.inProgress}
                      data-testid="button-start-download"
                    >
                      {downloadMutation.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <>
                          <Download className="h-4 w-4 mr-1" />
                          Start Download
                        </>
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowDownloadSection(false)}
                      data-testid="button-cancel-download"
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}
              <Button
                variant="destructive"
                size="sm"
                onClick={() => {
                  if (confirm("Clear all historical data and reset learning? This cannot be undone.")) {
                    clearMutation.mutate();
                  }
                }}
                disabled={clearMutation.isPending || downloadStatus?.inProgress}
                data-testid="button-clear-data"
              >
                {clearMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <>
                    <Trash2 className="h-4 w-4 mr-1" />
                    Clear All
                  </>
                )}
              </Button>
            </div>

            <div className="text-xs text-muted-foreground">
              All 3 learning systems (Strategy Learner, Pattern Memory, GPU Trainer) share this data.
            </div>
          </>
        ) : (
          <div className="text-sm text-muted-foreground">Failed to load data summary</div>
        )}
      </CardContent>
    </Card>
  );
}