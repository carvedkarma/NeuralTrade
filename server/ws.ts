import { WebSocketServer, WebSocket } from "ws";
import type { Server } from "http";

let wss: WebSocketServer | null = null;
const clients = new Set<WebSocket>();

export function setupWebSocket(httpServer: Server): void {
  wss = new WebSocketServer({ server: httpServer, path: "/ws" });

  wss.on("connection", (ws) => {
    clients.add(ws);
    console.log(`[WS] Client connected (${clients.size} total)`);

    ws.send(JSON.stringify({ type: "CONNECTED", ts: Date.now(), clients: clients.size }));

    ws.on("close", () => {
      clients.delete(ws);
      console.log(`[WS] Client disconnected (${clients.size} total)`);
    });

    ws.on("error", (err) => {
      console.error("[WS] Client error:", err.message);
      clients.delete(ws);
    });
  });

  console.log("[WS] WebSocket server ready on /ws");
}

export function broadcast(eventType: string, payload: unknown): void {
  if (!wss) return;
  const msg = JSON.stringify({ type: eventType, payload, ts: Date.now() });
  let sent = 0;
  clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
      sent++;
    }
  });
  if (sent > 0 && eventType !== "PRICE_TICK") {
    console.log(`[WS] Broadcast ${eventType} to ${sent} clients`);
  }
}

export function getClientCount(): number {
  return clients.size;
}
