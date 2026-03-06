import { Badge } from "@/components/ui/badge";

export function PingBadge({ ping }: { ping: number | null }) {
  if (ping === null) return null;
  const color = ping < 100 ? "text-emerald-400 border-emerald-500/30" : ping < 300 ? "text-amber-400 border-amber-500/30" : "text-red-400 border-red-500/30";
  return (
    <Badge variant="outline" className={`text-[10px] number-mono ${color}`} data-testid="ping-badge">
      {ping}ms
    </Badge>
  );
}
