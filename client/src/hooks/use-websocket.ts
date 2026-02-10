import { useState, useEffect, useRef, useCallback } from "react";

export type WsEvent = {
  type: string;
  payload?: unknown;
  ts: number;
  clients?: number;
};

type WsStatus = "connecting" | "connected" | "disconnected";

type UseWebSocketReturn = {
  status: WsStatus;
  lastEvent: WsEvent | null;
  lastUpdated: number | null;
  clientCount: number;
  subscribe: (eventType: string, handler: (payload: unknown) => void) => () => void;
};

export function useWebSocket(): UseWebSocketReturn {
  const [status, setStatus] = useState<WsStatus>("connecting");
  const [lastEvent, setLastEvent] = useState<WsEvent | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [clientCount, setClientCount] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const handlersRef = useRef<Map<string, Set<(payload: unknown) => void>>>(new Map());
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    setStatus("connecting");
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setStatus("connected");
      console.log("[WS] Connected");
    };

    ws.onmessage = (event) => {
      if (!mountedRef.current) return;
      try {
        const data: WsEvent = JSON.parse(event.data);
        setLastEvent(data);
        setLastUpdated(Date.now());
        if (data.clients !== undefined) setClientCount(data.clients);

        const handlers = handlersRef.current.get(data.type);
        if (handlers) {
          handlers.forEach((handler) => handler(data.payload));
        }
        const allHandlers = handlersRef.current.get("*");
        if (allHandlers) {
          allHandlers.forEach((handler) => handler(data));
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setStatus("disconnected");
      console.log("[WS] Disconnected, reconnecting in 3s...");
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const subscribe = useCallback((eventType: string, handler: (payload: unknown) => void) => {
    if (!handlersRef.current.has(eventType)) {
      handlersRef.current.set(eventType, new Set());
    }
    handlersRef.current.get(eventType)!.add(handler);
    return () => {
      handlersRef.current.get(eventType)?.delete(handler);
    };
  }, []);

  return { status, lastEvent, lastUpdated, clientCount, subscribe };
}
