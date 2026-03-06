import { useState, useEffect } from "react";

export function usePingMonitor(intervalMs = 5000) {
  const [ping, setPing] = useState<number | null>(null);
  useEffect(() => {
    let mounted = true;
    const measure = async () => {
      try {
        const t0 = performance.now();
        await fetch("/api/ping");
        const t1 = performance.now();
        if (mounted) setPing(Math.round(t1 - t0));
      } catch {
        if (mounted) setPing(null);
      }
    };
    measure();
    const id = setInterval(measure, intervalMs);
    return () => { mounted = false; clearInterval(id); };
  }, [intervalMs]);
  return ping;
}
