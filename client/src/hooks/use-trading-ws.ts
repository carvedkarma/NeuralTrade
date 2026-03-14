import { useEffect, useRef, useState, useCallback } from "react";

export type WsEventType =
  | "CONNECTED"
  | "CYCLE_UPDATE"
  | "TRADE_OPEN"
  | "TRADE_OPENED"
  | "TRADE_UPDATE"
  | "TRADE_CLOSE"
  | "GPU_STATUS_UPDATE"
  | "SIGNAL_UPDATE"
  | "HEALTH_STATUS"
  | "PRICE_TICK";

export type WsConnectionState = "connecting" | "connected" | "disconnected" | "error";

interface WsMessage {
  type: WsEventType;
  payload: Record<string, unknown>;
  ts: number;
}

type WsListener = (payload: Record<string, unknown>, ts: number) => void;

export function useTradingWs() {
  const [connectionState, setConnectionState] = useState<WsConnectionState>("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const listenersRef = useRef<Map<string, Set<WsListener>>>(new Map());
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptRef = useRef(0);
  const mountedRef = useRef(true);

  const emit = useCallback((type: string, payload: Record<string, unknown>, ts: number) => {
    const listeners = listenersRef.current.get(type);
    if (listeners) {
      listeners.forEach((fn) => fn(payload, ts));
    }
    const allListeners = listenersRef.current.get("*");
    if (allListeners) {
      allListeners.forEach((fn) => fn({ type, ...payload }, ts));
    }
  }, []);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${proto}//${window.location.host}/ws`;
      setConnectionState("connecting");

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnectionState("connected");
        reconnectAttemptRef.current = 0;
      };

      ws.onmessage = (evt) => {
        if (!mountedRef.current) return;
        try {
          const msg: WsMessage = JSON.parse(evt.data);
          emit(msg.type, msg.payload ?? {}, msg.ts);
        } catch {}
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnectionState("disconnected");
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 30000);
        reconnectAttemptRef.current++;
        reconnectTimerRef.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        setConnectionState("error");
        ws.close();
      };
    } catch {
      setConnectionState("error");
    }
  }, [emit]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const subscribe = useCallback((type: string, listener: WsListener) => {
    if (!listenersRef.current.has(type)) {
      listenersRef.current.set(type, new Set());
    }
    listenersRef.current.get(type)!.add(listener);
    return () => {
      listenersRef.current.get(type)?.delete(listener);
    };
  }, []);

  return { connectionState, subscribe };
}
