import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { X, Scissors, Pencil, Loader2 } from "lucide-react";

function invalidatePositions() {
  queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
  queryClient.invalidateQueries({ queryKey: ["/api/paper/portfolio"] });
  queryClient.invalidateQueries({ queryKey: ["/api/paper/risk-alerts"] });
}

export function CloseButton({ positionId, symbol, side, livePrice }: { positionId: number; symbol: string; side: string; livePrice?: number }) {
  const { toast } = useToast();
  const mutation = useMutation({
    mutationFn: () => apiRequest("POST", `/api/paper/positions/${positionId}/close`, livePrice ? { exitPrice: livePrice } : {}),
    onSuccess: async (res) => {
      const data = await res.json();
      invalidatePositions();
      toast({
        title: `Closed ${symbol} ${side}`,
        description: `P&L: ${data.pnl >= 0 ? "+" : ""}$${data.pnl?.toFixed(2) ?? "0.00"}`,
        variant: data.pnl >= 0 ? "default" : "destructive",
      });
    },
    onError: (err: Error) => {
      toast({ title: "Close failed", description: err.message, variant: "destructive" });
    },
  });

  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-7 w-7 text-red-400 hover:text-red-300 hover:bg-red-500/20"
      onClick={() => mutation.mutate()}
      disabled={mutation.isPending}
      data-testid={`button-close-position-${positionId}`}
    >
      {mutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <X className="w-4 h-4" />}
    </Button>
  );
}

export function PartialCloseButton({ positionId, symbol }: { positionId: number; symbol: string }) {
  const { toast } = useToast();
  const mutation = useMutation({
    mutationFn: () => apiRequest("POST", `/api/paper/positions/${positionId}/partial-close`, { percent: 50 }),
    onSuccess: async (res) => {
      const data = await res.json();
      invalidatePositions();
      toast({
        title: `Partial close ${symbol} (50%)`,
        description: `P&L: ${data.pnl >= 0 ? "+" : ""}$${data.pnl?.toFixed(2) ?? "0.00"}`,
      });
    },
    onError: (err: Error) => {
      toast({ title: "Partial close failed", description: err.message, variant: "destructive" });
    },
  });

  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-7 w-7 text-amber-400 hover:text-amber-300 hover:bg-amber-500/20"
      onClick={() => mutation.mutate()}
      disabled={mutation.isPending}
      data-testid={`button-partial-close-${positionId}`}
    >
      {mutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Scissors className="w-4 h-4" />}
    </Button>
  );
}

export function EditSLTPDialog({
  positionId,
  symbol,
  side,
  currentSL,
  currentTP,
  entryPrice,
}: {
  positionId: number;
  symbol: string;
  side: string;
  currentSL: number | null;
  currentTP: number | null;
  entryPrice: number;
}) {
  const { toast } = useToast();
  const [sl, setSl] = useState(currentSL?.toString() ?? "");
  const [tp, setTp] = useState(currentTP?.toString() ?? "");
  const [open, setOpen] = useState(false);

  const slMutation = useMutation({
    mutationFn: (stopLoss: number) => apiRequest("PATCH", `/api/paper/positions/${positionId}/sl`, { stopLoss }),
    onSuccess: () => {
      invalidatePositions();
      toast({ title: `SL updated for ${symbol}` });
    },
    onError: (err: Error) => {
      toast({ title: "SL update failed", description: err.message, variant: "destructive" });
    },
  });

  const tpMutation = useMutation({
    mutationFn: (tp1: number) => apiRequest("PATCH", `/api/paper/positions/${positionId}/tp`, { tp1 }),
    onSuccess: () => {
      invalidatePositions();
      toast({ title: `TP updated for ${symbol}` });
    },
    onError: (err: Error) => {
      toast({ title: "TP update failed", description: err.message, variant: "destructive" });
    },
  });

  const handleSave = () => {
    const slVal = parseFloat(sl);
    const tpVal = parseFloat(tp);
    if (!isNaN(slVal) && slVal !== currentSL) {
      slMutation.mutate(slVal);
    }
    if (!isNaN(tpVal) && tpVal !== currentTP) {
      tpMutation.mutate(tpVal);
    }
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/20"
          data-testid={`button-edit-sltp-${positionId}`}
        >
          <Pencil className="w-3.5 h-3.5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Edit SL/TP — {symbol} {side}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="text-sm text-muted-foreground">
            Entry: <span className="number-mono text-foreground">${entryPrice.toLocaleString()}</span>
          </div>
          <div className="space-y-2">
            <Label htmlFor="sl-input">Stop Loss</Label>
            <Input
              id="sl-input"
              type="number"
              step="any"
              value={sl}
              onChange={(e) => setSl(e.target.value)}
              placeholder={`Current: ${currentSL ?? "none"}`}
              data-testid="input-stop-loss"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="tp-input">Take Profit</Label>
            <Input
              id="tp-input"
              type="number"
              step="any"
              value={tp}
              onChange={(e) => setTp(e.target.value)}
              placeholder={`Current: ${currentTP ?? "none"}`}
              data-testid="input-take-profit"
            />
          </div>
        </div>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="ghost" data-testid="button-cancel-sltp">Cancel</Button>
          </DialogClose>
          <Button
            onClick={handleSave}
            disabled={slMutation.isPending || tpMutation.isPending}
            data-testid="button-save-sltp"
          >
            {(slMutation.isPending || tpMutation.isPending) && <Loader2 className="w-4 h-4 animate-spin mr-1" />}
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function SLTPProgressBar({
  entryPrice,
  currentPrice,
  stopLoss,
  takeProfit,
  side,
}: {
  entryPrice: number;
  currentPrice: number | null | undefined;
  stopLoss: number | null;
  takeProfit: number | null;
  side: string;
}) {
  if (!currentPrice || !stopLoss || !takeProfit) return null;

  const totalRange = Math.abs(takeProfit - stopLoss);
  if (totalRange === 0) return null;

  const progress = side === "LONG"
    ? ((currentPrice - stopLoss) / totalRange) * 100
    : ((stopLoss - currentPrice) / totalRange) * 100;

  const clamped = Math.max(0, Math.min(100, progress));
  const barColor = clamped > 50 ? "bg-emerald-500" : clamped > 25 ? "bg-amber-500" : "bg-red-500";

  return (
    <div className="w-full flex items-center gap-1" data-testid="sltp-progress-bar">
      <span className="text-[9px] text-red-400 number-mono">SL</span>
      <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${clamped}%` }} />
      </div>
      <span className="text-[9px] text-emerald-400 number-mono">TP</span>
    </div>
  );
}
