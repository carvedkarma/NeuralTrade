/**
 * Step 1 — Verify real Bybit round-trip cost on the V7 Path A whitelist.
 *
 * Pulls:
 *   1. Per-symbol fee tier (taker + maker) for ADA / XRP / AVAX / SOL
 *   2. Last ~100 executions (if any) and computes realised round-trip
 *   3. Top-of-book spread for each symbol → slippage estimate at $15k notional
 *
 * Output: console table + .local/reports/bybit_cost_verification.md
 *
 * Run:  npx tsx scripts/verify_bybit_cost.ts
 */

import { writeFileSync, mkdirSync } from "fs";
import { dirname } from "path";
import {
  getFeeRate,
  getExecutions,
  getTicker,
  getConnectionMode,
  isConfigured,
} from "../server/bybit/client";

const SYMBOLS = ["ADAUSDT", "XRPUSDT", "AVAXUSDT", "SOLUSDT"];
const TARGET_NOTIONAL_USD = 15000;
const OUT_PATH = ".local/reports/bybit_cost_verification.md";

interface CostRow {
  symbol: string;
  takerBps: number | null;
  makerBps: number | null;
  bidPx: number | null;
  askPx: number | null;
  midPx: number | null;
  spreadBps: number | null;
  // Half-spread paid on a $15k market order (assuming take-the-book at top level)
  slippageBpsEstimate: number | null;
  // Round-trip costs
  rtTakerBps: number | null;
  rtMakerBps: number | null;
  rtMixedBps: number | null;     // maker entry + taker exit
}

interface ExecRow {
  symbol: string;
  side: string;
  execPrice: number;
  execQty: number;
  execFeeBps: number;
  isMaker: boolean;
  ts: number;
}

async function fetchOne(symbol: string): Promise<CostRow> {
  const row: CostRow = {
    symbol,
    takerBps: null, makerBps: null,
    bidPx: null, askPx: null, midPx: null,
    spreadBps: null, slippageBpsEstimate: null,
    rtTakerBps: null, rtMakerBps: null, rtMixedBps: null,
  };

  try {
    const fr = await getFeeRate(symbol);
    if (fr.retCode === 0) {
      const f = fr.result?.list?.find(x => x.symbol === symbol) ?? fr.result?.list?.[0];
      if (f) {
        row.takerBps = parseFloat(f.takerFeeRate) * 1e4;
        row.makerBps = parseFloat(f.makerFeeRate) * 1e4;
      }
    } else {
      console.warn(`  ${symbol} fee-rate retCode=${fr.retCode} msg=${fr.retMsg}`);
    }
  } catch (e: any) {
    console.warn(`  ${symbol} fee-rate failed: ${e.message}`);
  }

  try {
    const t = await getTicker(symbol);
    if (t.retCode === 0) {
      const tk = t.result?.list?.[0];
      if (tk) {
        row.bidPx = parseFloat(tk.bid1Price);
        row.askPx = parseFloat(tk.ask1Price);
        row.midPx = (row.bidPx + row.askPx) / 2;
        row.spreadBps = ((row.askPx - row.bidPx) / row.midPx) * 1e4;
        // Conservative single-leg slippage = half the top-of-book spread
        row.slippageBpsEstimate = row.spreadBps / 2;
      }
    }
  } catch (e: any) {
    console.warn(`  ${symbol} ticker failed: ${e.message}`);
  }

  // Round-trip cost models
  if (row.takerBps !== null && row.makerBps !== null && row.slippageBpsEstimate !== null) {
    // Worst case: take both legs + half-spread slippage on each
    row.rtTakerBps = 2 * row.takerBps + 2 * row.slippageBpsEstimate;
    // Best case (theoretical): make both legs (no slippage paid on rebate)
    row.rtMakerBps = 2 * row.makerBps;
    // Realistic: maker entry (limit at mid for first 5min), taker exit at 120m
    row.rtMixedBps = row.makerBps + row.takerBps + row.slippageBpsEstimate;
  }

  return row;
}

async function fetchExecs(): Promise<ExecRow[]> {
  const rows: ExecRow[] = [];
  for (const sym of SYMBOLS) {
    try {
      const r = await getExecutions(sym, 100);
      if (r.retCode !== 0) continue;
      for (const x of (r.result?.list ?? [])) {
        const px = parseFloat(x.execPrice);
        const qty = parseFloat(x.execQty);
        const fee = parseFloat(x.execFee);
        if (!isFinite(px) || !isFinite(qty) || !isFinite(fee) || px <= 0 || qty <= 0) continue;
        const notional = px * qty;
        const feeBps = (fee / notional) * 1e4;
        rows.push({
          symbol: sym,
          side: x.side,
          execPrice: px,
          execQty: qty,
          execFeeBps: feeBps,
          isMaker: x.isMaker === true || x.isMaker === "true" || x.isMaker === 1,
          ts: parseInt(x.execTime || "0", 10),
        });
      }
    } catch {
      // ignore — no exec history is fine
    }
  }
  return rows;
}

function fmt(x: number | null, digits = 2, suffix = ""): string {
  if (x === null || !isFinite(x)) return "—";
  return x.toFixed(digits) + suffix;
}

async function main() {
  if (!isConfigured()) {
    console.error("BYBIT_API_KEY / BYBIT_API_SECRET not set");
    process.exit(1);
  }

  console.log(`Connection mode: ${getConnectionMode()}`);
  console.log(`Pulling fee rates + tickers for: ${SYMBOLS.join(", ")}`);

  const rows = await Promise.all(SYMBOLS.map(fetchOne));
  console.log("\nPer-symbol fee + spread:");
  console.table(rows.map(r => ({
    symbol: r.symbol,
    "taker (bps)": fmt(r.takerBps),
    "maker (bps)": fmt(r.makerBps),
    "spread (bps)": fmt(r.spreadBps),
    "half-spread slip": fmt(r.slippageBpsEstimate),
    "RT @ all-taker": fmt(r.rtTakerBps),
    "RT @ all-maker": fmt(r.rtMakerBps),
    "RT @ maker-entry+taker-exit": fmt(r.rtMixedBps),
  })));

  console.log("\nFetching last 100 executions per symbol (if any):");
  const execs = await fetchExecs();
  if (execs.length === 0) {
    console.log("  no recent executions (account has no fills on these symbols).");
  } else {
    const byMaker: Record<string, number[]> = { maker: [], taker: [] };
    for (const e of execs) {
      const k = e.isMaker ? "maker" : "taker";
      byMaker[k].push(e.execFeeBps);
    }
    console.table([
      { type: "maker fills", n: byMaker.maker.length, "mean fee bps": byMaker.maker.length ? fmt(byMaker.maker.reduce((a,b)=>a+b,0)/byMaker.maker.length) : "—" },
      { type: "taker fills", n: byMaker.taker.length, "mean fee bps": byMaker.taker.length ? fmt(byMaker.taker.reduce((a,b)=>a+b,0)/byMaker.taker.length) : "—" },
    ]);
  }

  // Verdict vs target net @ 6 bps
  const lines: string[] = [];
  lines.push("# Bybit round-trip cost verification");
  lines.push("");
  lines.push(`_Generated: ${new Date().toISOString()}_`);
  lines.push(`Connection mode: \`${getConnectionMode()}\``);
  lines.push(`Target round-trip cost for V7 Path A paper-trade: **≤ 6 bps**`);
  lines.push(`Target notional per trade: $${TARGET_NOTIONAL_USD.toLocaleString()}`);
  lines.push("");
  lines.push("## Per-symbol fee tier + top-of-book spread");
  lines.push("");
  lines.push("| symbol | taker (bps) | maker (bps) | top spread (bps) | half-spread slip | RT all-taker | RT all-maker | RT maker-entry/taker-exit |");
  lines.push("|---|---:|---:|---:|---:|---:|---:|---:|");
  for (const r of rows) {
    lines.push(`| ${r.symbol} | ${fmt(r.takerBps)} | ${fmt(r.makerBps)} | ${fmt(r.spreadBps)} | ${fmt(r.slippageBpsEstimate)} | ${fmt(r.rtTakerBps)} | ${fmt(r.rtMakerBps)} | ${fmt(r.rtMixedBps)} |`);
  }
  lines.push("");

  // Realised costs (if any)
  if (execs.length > 0) {
    lines.push("## Recent realised executions");
    lines.push("");
    const makerFees = execs.filter(e => e.isMaker).map(e => e.execFeeBps);
    const takerFees = execs.filter(e => !e.isMaker).map(e => e.execFeeBps);
    const meanMaker = makerFees.length ? makerFees.reduce((a,b)=>a+b,0)/makerFees.length : null;
    const meanTaker = takerFees.length ? takerFees.reduce((a,b)=>a+b,0)/takerFees.length : null;
    lines.push(`- maker fills: **${makerFees.length}** (mean fee ${fmt(meanMaker)} bps)`);
    lines.push(`- taker fills: **${takerFees.length}** (mean fee ${fmt(meanTaker)} bps)`);
    lines.push("");
    if (meanMaker !== null && meanTaker !== null) {
      lines.push(`**Implied round-trip @ realised mix (maker entry + taker exit): ${fmt(meanMaker + meanTaker)} bps** (excludes slippage)`);
    } else if (meanTaker !== null) {
      lines.push(`**Implied round-trip @ all-taker realised: ${fmt(2*meanTaker)} bps** (excludes slippage)`);
    }
    lines.push("");
  } else {
    lines.push("## Recent realised executions");
    lines.push("");
    lines.push("_No recent executions on the whitelist symbols. Verdict uses fee tier + ticker spread only._");
    lines.push("");
  }

  // Verdict
  lines.push("## Verdict vs V7 Path A paper-trade plan");
  lines.push("");
  const validRows = rows.filter(r => r.rtMixedBps !== null);
  if (validRows.length === 0) {
    lines.push("⚠️ Could not pull live fee tier or ticker data — verify direct API access (the Replit egress IP may be geo-blocked by Bybit).");
  } else {
    const maxMixed = Math.max(...validRows.map(r => r.rtMixedBps!));
    const meanMixed = validRows.reduce((a, r) => a + r.rtMixedBps!, 0) / validRows.length;
    const maxTaker = Math.max(...validRows.map(r => r.rtTakerBps!));
    lines.push(`- Worst-symbol round-trip @ maker-entry + taker-exit: **${fmt(maxMixed)} bps**`);
    lines.push(`- Mean round-trip @ maker-entry + taker-exit: ${fmt(meanMixed)} bps`);
    lines.push(`- Worst-symbol round-trip @ all-taker (worst case): ${fmt(maxTaker)} bps`);
    lines.push("");
    if (maxMixed <= 6) {
      lines.push("✅ **PASS** — every symbol fits under the 6 bps target with maker-entry + taker-exit.");
      lines.push("Back-test net edge at 6 bps = +24.01 bps/trade (rolling-OOS). **Cleared to wire paper engine.**");
    } else if (meanMixed <= 6) {
      lines.push("🟡 **MARGINAL** — mean fits 6 bps but worst symbol exceeds. Consider downsize on the wide-spread symbol.");
    } else {
      lines.push("❌ **FAIL** — even the maker/taker mix exceeds 6 bps. Edge thins. Either:");
      lines.push("  1. Switch to limit exits on the 120m mark (maker on both legs)");
      lines.push("  2. Reduce per-trade notional (smaller orders → less slippage)");
      lines.push("  3. Re-run the cost grid in the back-test report at the realised cost level");
    }
  }
  lines.push("");

  mkdirSync(dirname(OUT_PATH), { recursive: true });
  writeFileSync(OUT_PATH, lines.join("\n"));
  console.log(`\nWrote ${OUT_PATH}`);
}

main().catch(e => { console.error(e); process.exit(1); });
