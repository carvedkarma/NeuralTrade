import { Router } from "express";
import { db } from "./db";
import { broadcast } from "./ws";
import {
  ingestEventPayloadSchema,
  ingestedEvents,
  liveCycleLogs,
  liveTradeRecords,
  tradeEvents,
  learningRuns,
  healthStatus,
  modelLearningStats,
  settings,
} from "@shared/schema";
import type { MoneyConfig } from "@shared/schema";
import { eq } from "drizzle-orm";

async function getMoneyConfig(): Promise<MoneyConfig> {
  const row = await db.select().from(settings).where(eq(settings.key, "money_config")).limit(1);
  if (row.length === 0) return { account_equity_usd: 1500, risk_per_trade_pct: 1.0, base_currency: "USD" };
  return row[0].valueJson as MoneyConfig;
}

const router = Router();

router.post("/ingest/event", async (req, res) => {
  try {
    const parsed = ingestEventPayloadSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({
        error: "Invalid event payload",
        details: parsed.error.flatten(),
      });
    }

    const { event_id, type, payload, ts } = parsed.data;

    const existing = await db
      .select({ id: ingestedEvents.id })
      .from(ingestedEvents)
      .where(eq(ingestedEvents.eventId, event_id))
      .limit(1);

    if (existing.length > 0) {
      return res.status(200).json({ status: "duplicate", event_id });
    }

    await db.insert(ingestedEvents).values({
      eventId: event_id,
      eventType: type,
      ts,
      payloadJson: payload,
      processedAt: Date.now(),
    });

    await processEvent(type, payload, ts);

    broadcast(type, payload);
    console.log(`[Ingest] Event accepted type=${type}, event_id=${event_id}`);

    return res.status(200).json({ status: "accepted", event_id, type });
  } catch (err: any) {
    console.error("[Ingest] Error processing event:", err.message);
    return res.status(500).json({ error: "Internal error", message: err.message });
  }
});

async function processEvent(
  type: string,
  payload: Record<string, unknown>,
  ts: number
): Promise<void> {
  switch (type) {
    case "CYCLE_UPDATE": {
      const p = payload as any;
      await db.insert(liveCycleLogs).values({
        symbol: p.symbol ?? "BTCUSDT",
        cycleTs: p.cycle_ts ?? ts,
        price: p.price ?? null,
        pEnter: p.p_enter ?? null,
        htfH1Trend: p.htf_h1_trend ?? null,
        htfH4Trend: p.htf_h4_trend ?? null,
        slopeOk: p.slope_ok ?? null,
        rangeOk: p.range_ok ?? null,
        direction: p.direction ?? null,
        thresholdUsed: p.threshold_used ?? null,
        decision: p.decision ?? "UNKNOWN",
        reasons: p.reasons ?? [],
        policy: p.policy ?? null,
        coreThr: p.core_thr ?? null,
        flowThr: p.flow_thr ?? null,
        quotaStep: p.quota_step ?? null,
        flowPctUsed: p.flow_pct_used ?? null,
        tradesTodayTotal: p.trades_today_total ?? null,
        tradesTodayTarget: p.trades_today_target ?? null,
        tradesTodayMax: p.trades_today_max ?? null,
        quotaFlowRiskMult: p.quota_flow_risk_mult ?? null,
        laneSelected: p.lane_selected ?? null,
        htfScore: p.htf_score ?? null,
        scalpThr: p.scalp_thr ?? null,
        laneSizeMult: p.lane_size_mult ?? null,
        laneBudgetRemainingR: p.lane_budget_remaining_r ?? null,
        holdReason: p.hold_reason ?? null,
        createdAt: Date.now(),
      });
      break;
    }

    case "TRADE_OPEN": {
      const p = payload as any;
      const moneyConfig = await getMoneyConfig();
      const riskUsd = moneyConfig.account_equity_usd * (moneyConfig.risk_per_trade_pct / 100);
      await db.insert(liveTradeRecords).values({
        symbol: p.symbol ?? "BTCUSDT",
        side: p.side ?? "LONG",
        entryTime: p.entry_time ?? ts,
        entryPrice: p.entry_price ?? 0,
        stopLoss: p.stop_loss ?? null,
        takeProfit: p.take_profit ?? null,
        pEnter: p.p_enter ?? null,
        sizePct: p.size_pct ?? null,
        costsBps: p.costs_bps ?? null,
        leverage: p.leverage ?? null,
        modelVersion: p.model_version ?? null,
        riskUsdUsed: riskUsd,
        equitySnapshotUsd: moneyConfig.account_equity_usd,
        policy: p.policy ?? null,
        flowRiskMult: p.flow_risk_mult ?? null,
        lane: p.lane ?? null,
        htfScore: p.htf_score ?? null,
        laneThresholdUsed: p.lane_threshold_used ?? null,
        laneSizeMult: p.lane_size_mult ?? null,
        laneHorizon: p.lane_horizon ?? null,
        status: "open",
        createdAt: Date.now(),
      });
      break;
    }

    case "TRADE_UPDATE":
    case "TRADE_CLOSE": {
      const p = payload as any;
      if (p.trade_id) {
        const updates: Record<string, any> = {};
        if (p.exit_time) updates.exitTime = p.exit_time;
        if (p.exit_price) updates.exitPrice = p.exit_price;
        if (p.outcome) updates.outcome = p.outcome;
        if (p.gross_r !== undefined) updates.grossR = p.gross_r;
        if (p.cost_r !== undefined) updates.costR = p.cost_r;
        if (p.net_r !== undefined) updates.netR = p.net_r;
        if (p.sized_r !== undefined) updates.sizedR = p.sized_r;
        if (p.bars_held !== undefined) updates.barsHeld = p.bars_held;
        if (p.model_version) updates.modelVersion = p.model_version;
        if (p.leverage !== undefined) updates.leverage = p.leverage;
        if (p.exit_reason) updates.exitReason = p.exit_reason;
        if (type === "TRADE_CLOSE") updates.status = "closed";
        if (p.status) updates.status = p.status;

        if (type === "TRADE_CLOSE") {
          const [existing] = await db.select().from(liveTradeRecords).where(eq(liveTradeRecords.id, p.trade_id)).limit(1);
          if (existing) {
            const riskUsd = existing.riskUsdUsed ?? (await getMoneyConfig().then(c => c.account_equity_usd * (c.risk_per_trade_pct / 100)));
            const netR = p.net_r ?? existing.netR ?? 0;
            const grossR = p.gross_r ?? existing.grossR ?? 0;
            const costR = p.cost_r ?? (grossR - netR);
            updates.costR = costR;
            updates.pnlUsd = netR * riskUsd;
            updates.pnlUsdGross = grossR * riskUsd;
            updates.pnlUsdCost = costR * riskUsd;
            if (!existing.riskUsdUsed) updates.riskUsdUsed = riskUsd;
          }
        }

        if (Object.keys(updates).length > 0) {
          await db
            .update(liveTradeRecords)
            .set(updates)
            .where(eq(liveTradeRecords.id, p.trade_id));
        }

        await db.insert(tradeEvents).values({
          tradeId: p.trade_id,
          ts,
          eventType: type,
          payloadJson: payload,
        });
      }
      break;
    }

    case "LEARNING_PROGRESS": {
      const p = payload as any;
      if (p.run_id) {
        const updates: Record<string, any> = {};
        if (p.status) updates.status = p.status;
        if (p.end_at) updates.endAt = p.end_at;
        if (p.pr_auc !== undefined) updates.prAuc = p.pr_auc;
        if (p.best_val_loss !== undefined) updates.bestValLoss = p.best_val_loss;
        if (p.pf_net !== undefined) updates.pfNet = p.pf_net;
        if (p.e_net !== undefined) updates.eNet = p.e_net;
        if (p.promoted !== undefined) updates.promoted = p.promoted;
        if (p.reason) updates.reason = p.reason;
        if (p.metrics_json) updates.metricsJson = p.metrics_json;

        if (Object.keys(updates).length > 0) {
          await db
            .update(learningRuns)
            .set(updates)
            .where(eq(learningRuns.id, p.run_id));
        }
      } else {
        await db.insert(learningRuns).values({
          symbol: p.symbol ?? "BTCUSDT",
          startAt: p.start_at ?? ts,
          endAt: p.end_at ?? null,
          dataFrom: p.data_from ?? null,
          dataTo: p.data_to ?? null,
          newBars: p.new_bars ?? null,
          newTrades: p.new_trades ?? null,
          epochs: p.epochs ?? null,
          bestValLoss: p.best_val_loss ?? null,
          prAuc: p.pr_auc ?? null,
          pfNet: p.pf_net ?? null,
          eNet: p.e_net ?? null,
          profitableRegimes: p.profitable_regimes ?? null,
          totalRegimes: p.total_regimes ?? null,
          promoted: p.promoted ?? false,
          reason: p.reason ?? null,
          modelVersion: p.model_version ?? null,
          metricsJson: p.metrics_json ?? null,
          status: p.status ?? "running",
        });
      }
      break;
    }

    case "MODEL_PROMOTED": {
      const p = payload as any;
      await db.insert(modelLearningStats).values({
        symbol: p.symbol ?? "BTCUSDT",
        modelVersion: p.model_version ?? "unknown",
        trainedUntilTs: p.trained_until_ts ?? null,
        trainingSamples: p.training_samples ?? null,
        valPrAuc: p.val_pr_auc ?? null,
        valPrecision: p.val_precision ?? null,
        valRecall: p.val_recall ?? null,
        valF1: p.val_f1 ?? null,
        bestPolicyThreshold: p.best_policy_threshold ?? null,
        bestPolicyCooldown: p.best_policy_cooldown ?? null,
        bestPolicyTpMult: p.best_policy_tp_mult ?? null,
        bestPolicySlMult: p.best_policy_sl_mult ?? null,
        pfNet: p.pf_net ?? null,
        eNet: p.e_net ?? null,
        tradesPerDay: p.trades_per_day ?? null,
        profitableRegimes: p.profitable_regimes ?? null,
        totalRegimes: p.total_regimes ?? null,
        promoted: true,
        promotionReason: p.reason ?? null,
        trend7d: p.trend_7d ?? null,
        prevPfNet: p.prev_pf_net ?? null,
        prevENet: p.prev_e_net ?? null,
        prevTradesPerDay: p.prev_trades_per_day ?? null,
        createdAt: Date.now(),
      });
      break;
    }

    case "HEALTH_STATUS": {
      const p = payload as any;
      await db.insert(healthStatus).values({
        ts,
        component: p.component ?? "unknown",
        status: p.status ?? "unknown",
        message: p.message ?? null,
      });
      break;
    }

    default:
      break;
  }
}

export default router;
