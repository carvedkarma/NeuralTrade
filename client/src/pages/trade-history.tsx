import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ClipboardList } from "lucide-react";

interface TradeRecord {
  id: number;
  positionId: number;
  symbol: string;
  side: string;
  entryTs: number;
  entryPrice: number;
  exitTs: number;
  exitPrice: number;
  grossR: number;
  netR: number;
  costR: number;
  pnlUsdt: number;
  riskUsdt: number;
  barsHeld: number;
  exitReason: string;
  maxFavorableR: number;
  fillType?: string | null;
}

function formatUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(value);
}

function formatDateTime(ts: number): string {
  return new Date(ts).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(ms: number): string {
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  if (hours > 24) {
    const days = Math.floor(hours / 24);
    return `${days}d ${hours % 24}h`;
  }
  return `${hours}h ${minutes}m`;
}

export default function TradeHistory() {
  const { data: tradeHistory, isLoading } = useQuery<TradeRecord[]>({
    queryKey: ["/api/paper/trade-history", "?limit=500"],
    refetchInterval: 30000,
  });

  const sorted = tradeHistory
    ? [...tradeHistory].sort((a, b) => (b.exitTs ?? 0) - (a.exitTs ?? 0))
    : [];

  const totalR = sorted.reduce((s, t) => s + (t.netR ?? 0), 0);
  const wins = sorted.filter((t) => (t.netR ?? 0) > 0).length;
  const winRate = sorted.length > 0 ? ((wins / sorted.length) * 100).toFixed(1) : "0";

  return (
    <div className="p-4 space-y-4" data-testid="trade-history-page">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ClipboardList className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold">Trade History</h2>
          {sorted.length > 0 && (
            <Badge variant="outline" className="text-[10px] text-muted-foreground" data-testid="badge-trade-count">
              {sorted.length} trades
            </Badge>
          )}
        </div>
        {sorted.length > 0 && (
          <div className="flex items-center gap-4 text-sm">
            <span className="text-muted-foreground">
              Win Rate: <span className="font-semibold text-foreground" data-testid="text-win-rate">{winRate}%</span>
            </span>
            <span className="text-muted-foreground">
              Total:{" "}
              <span
                className={`font-semibold number-mono ${totalR >= 0 ? "text-emerald-400" : "text-red-400"}`}
                data-testid="text-total-r"
              >
                {totalR >= 0 ? "+" : ""}{totalR.toFixed(2)}R
              </span>
            </span>
          </div>
        )}
      </div>

      <Card>
        <CardContent className="p-0">
          {isLoading ? (
            <p className="text-sm text-muted-foreground text-center py-12">Loading trades...</p>
          ) : sorted.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-12" data-testid="text-no-trades">
              No completed trades yet
            </p>
          ) : (
            <div className="overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Date</TableHead>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Side</TableHead>
                    <TableHead>Fill</TableHead>
                    <TableHead>Entry</TableHead>
                    <TableHead>Exit</TableHead>
                    <TableHead>P&L R</TableHead>
                    <TableHead>P&L USD</TableHead>
                    <TableHead>MFE</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Exit Type</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sorted.map((trade, i) => {
                    const pnl = trade.netR ?? 0;
                    const dur = trade.entryTs && trade.exitTs ? trade.exitTs - trade.entryTs : 0;
                    return (
                      <TableRow key={trade.id ?? i} data-testid={`row-trade-history-${i}`}>
                        <TableCell className="text-muted-foreground whitespace-nowrap">
                          {trade.exitTs ? formatDateTime(trade.exitTs) : "-"}
                        </TableCell>
                        <TableCell className="font-medium">{trade.symbol}</TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className={trade.side === "LONG" ? "text-emerald-400 border-emerald-400/30" : "text-red-400 border-red-400/30"}
                          >
                            {trade.side}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {trade.fillType ? (
                            <Badge
                              variant="outline"
                              className={trade.fillType === "MAKER"
                                ? "text-cyan-400 border-cyan-400/30 font-mono text-[10px] px-1"
                                : "text-amber-400 border-amber-400/30 font-mono text-[10px] px-1"}
                              data-testid={`badge-fill-type-${i}`}
                            >
                              {trade.fillType === "MAKER" ? "M" : "T"}
                            </Badge>
                          ) : (
                            <span className="text-muted-foreground/40 text-xs">—</span>
                          )}
                        </TableCell>
                        <TableCell className="number-mono">{trade.entryPrice?.toFixed(2)}</TableCell>
                        <TableCell className="number-mono">{trade.exitPrice?.toFixed(2) ?? "-"}</TableCell>
                        <TableCell className={`number-mono ${pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                          {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}R
                        </TableCell>
                        <TableCell className={`number-mono ${(trade.pnlUsdt ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                          {trade.pnlUsdt != null ? formatUsd(trade.pnlUsdt) : "-"}
                        </TableCell>
                        <TableCell className="number-mono text-cyan-400/70">
                          {trade.maxFavorableR != null ? `${trade.maxFavorableR.toFixed(2)}R` : "-"}
                        </TableCell>
                        <TableCell className="text-muted-foreground whitespace-nowrap">
                          {dur > 0 ? formatDuration(dur) : "-"}
                        </TableCell>
                        <TableCell>
                          <span className="text-xs text-muted-foreground">{trade.exitReason ?? "-"}</span>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
