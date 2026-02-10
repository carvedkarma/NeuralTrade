import { db } from "../server/db";
import { liveCycleLogs, liveTradeRecords, learningRuns, healthStatus, tradeEvents, modelLearningStats } from "../shared/schema";

const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
const DECISIONS = ["HOLD", "GATE_FAIL", "ENTER", "COOLDOWN"];
const DIRECTIONS = ["LONG", "SHORT"];
const OUTCOMES = ["TP", "SL", "EXPIRE"];

function rand(min: number, max: number) {
  return min + Math.random() * (max - min);
}
function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

async function seedCycles(count = 200) {
  const now = Date.now();
  const rows = [];
  for (let i = 0; i < count; i++) {
    const symbol = pick(SYMBOLS);
    const decision = pick(DECISIONS);
    const reasons: string[] = [];
    if (decision === "GATE_FAIL") reasons.push("HTF trend not aligned");
    if (decision === "HOLD") reasons.push("p_enter below threshold");
    if (decision === "COOLDOWN") reasons.push("Cooldown active (4 bars remaining)");

    rows.push({
      symbol,
      cycleTs: now - i * 15 * 60 * 1000,
      price: symbol === "BTCUSDT" ? rand(95000, 100000) : symbol === "ETHUSDT" ? rand(3200, 3600) : rand(80, 95),
      pEnter: rand(0.01, 0.95),
      htfH1Trend: rand(-1, 1),
      htfH4Trend: rand(-1, 1),
      slopeOk: Math.random() > 0.4,
      rangeOk: Math.random() > 0.3,
      direction: pick(DIRECTIONS),
      thresholdUsed: 0.55,
      decision,
      reasons,
      createdAt: now - i * 15 * 60 * 1000,
    });
  }
  await db.insert(liveCycleLogs).values(rows);
  console.log(`Seeded ${count} cycle logs`);
}

async function seedTrades(count = 50) {
  const now = Date.now();
  for (let i = 0; i < count; i++) {
    const symbol = pick(SYMBOLS);
    const side = pick(DIRECTIONS);
    const entryPrice = symbol === "BTCUSDT" ? rand(95000, 100000) : symbol === "ETHUSDT" ? rand(3200, 3600) : rand(80, 95);
    const atr = entryPrice * rand(0.005, 0.02);
    const sl = side === "LONG" ? entryPrice - atr * 1.5 : entryPrice + atr * 1.5;
    const tp = side === "LONG" ? entryPrice + atr * 3.0 : entryPrice - atr * 3.0;
    const outcome = pick(OUTCOMES);
    const grossR = outcome === "TP" ? rand(1.5, 3.5) : outcome === "SL" ? -1 : rand(-0.5, 0.5);
    const costR = rand(0.05, 0.15);
    const netR = grossR - costR;
    const barsHeld = Math.floor(rand(2, 48));
    const entryTime = now - (count - i) * 6 * 60 * 60 * 1000;
    const exitTime = entryTime + barsHeld * 15 * 60 * 1000;

    const [trade] = await db.insert(liveTradeRecords).values({
      symbol,
      side,
      entryTime,
      entryPrice,
      exitTime,
      exitPrice: outcome === "TP" ? tp : outcome === "SL" ? sl : entryPrice + rand(-atr, atr),
      stopLoss: sl,
      takeProfit: tp,
      sizePct: rand(1, 5),
      pEnter: rand(0.55, 0.95),
      costsBps: rand(8, 15),
      outcome,
      grossR,
      netR,
      sizedR: netR * rand(1, 2),
      status: "closed",
      reasons: [outcome === "TP" ? "Take profit hit" : outcome === "SL" ? "Stop loss hit" : "Expired after max bars"],
      createdAt: entryTime,
    }).returning();

    await db.insert(tradeEvents).values([
      { tradeId: trade.id, ts: entryTime, eventType: "ENTRY", payloadJson: { price: entryPrice, side } },
      { tradeId: trade.id, ts: exitTime, eventType: "EXIT", payloadJson: { price: trade.exitPrice, outcome } },
    ]);
  }
  console.log(`Seeded ${count} trades with events`);
}

async function seedLearningRuns(count = 10) {
  const now = Date.now();
  const rows = [];
  for (let i = 0; i < count; i++) {
    const symbol = pick(SYMBOLS);
    const startAt = now - (count - i) * 24 * 60 * 60 * 1000;
    const promoted = Math.random() > 0.4;
    rows.push({
      symbol,
      startAt,
      endAt: Math.floor(startAt + rand(1800000, 7200000)),
      dataFrom: Math.floor(startAt - 30 * 24 * 60 * 60 * 1000),
      dataTo: startAt,
      newBars: Math.floor(rand(200, 2000)),
      newTrades: Math.floor(rand(5, 50)),
      epochs: Math.floor(rand(50, 300)),
      bestValLoss: rand(0.2, 0.6),
      prAuc: rand(0.4, 0.85),
      pfNet: rand(0.8, 2.5),
      eNet: rand(-0.1, 0.5),
      profitableRegimes: Math.floor(rand(1, 5)),
      totalRegimes: 5,
      promoted,
      reason: promoted ? "Meets all safety gates" : "PF_net below threshold",
      modelVersion: `v3.5.${i}`,
      metricsJson: { precision: rand(0.4, 0.8), recall: rand(0.3, 0.7) },
      status: "completed",
    });
  }
  await db.insert(learningRuns).values(rows);
  console.log(`Seeded ${count} learning runs`);
}

async function seedHealth() {
  const components = ["data_feed", "model", "websocket", "gpu_trainer"];
  const rows = components.map((component) => ({
    ts: Date.now(),
    component,
    status: Math.random() > 0.2 ? "ok" : "degraded",
    message: Math.random() > 0.2 ? "Operating normally" : "Minor latency detected",
  }));
  await db.insert(healthStatus).values(rows);
  console.log(`Seeded ${components.length} health entries`);
}

async function seedModelStats() {
  const rows = SYMBOLS.map((symbol, i) => ({
    symbol,
    modelVersion: `v3.5.${i}`,
    trainedUntilTs: Date.now() - 3600000,
    trainingSamples: Math.floor(rand(5000, 20000)),
    valPrAuc: rand(0.5, 0.85),
    valPrecision: rand(0.45, 0.75),
    valRecall: rand(0.35, 0.65),
    valF1: rand(0.4, 0.7),
    bestPolicyThreshold: 0.55,
    bestPolicyCooldown: 6,
    bestPolicyTpMult: 3.0,
    bestPolicySlMult: 1.5,
    pfNet: rand(1.0, 2.2),
    eNet: rand(0.05, 0.4),
    tradesPerDay: rand(1.5, 4.0),
    profitableRegimes: Math.floor(rand(3, 5)),
    totalRegimes: 5,
    promoted: true,
    promotionReason: "Meets all safety gates",
    trend7d: "improving",
    prevPfNet: rand(0.9, 1.5),
    prevENet: rand(0.01, 0.2),
    prevTradesPerDay: rand(1.0, 3.0),
    createdAt: Date.now(),
  }));
  await db.insert(modelLearningStats).values(rows);
  console.log(`Seeded ${rows.length} model learning stats`);
}

async function main() {
  console.log("Seeding demo data...");
  await seedCycles();
  await seedTrades();
  await seedLearningRuns();
  await seedHealth();
  await seedModelStats();
  console.log("Demo seed complete!");
  process.exit(0);
}

main().catch((err) => {
  console.error("Seed error:", err);
  process.exit(1);
});
