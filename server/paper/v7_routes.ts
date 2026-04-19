import { Router } from "express";
import {
  getV7State,
  setV7Enabled,
  setV7Notional,
  manualResume,
  getV7Performance,
} from "./v7_path_a";

const router = Router();

router.get("/state", (_req, res) => {
  try {
    res.json(getV7State());
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

router.get("/performance", async (req, res) => {
  try {
    const lookback = parseInt((req.query.lookback as string) || "200", 10);
    res.json(await getV7Performance(lookback));
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

router.post("/enable", (_req, res) => {
  try {
    setV7Enabled(true);
    res.json({ ok: true, ...getV7State() });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

router.post("/disable", (_req, res) => {
  try {
    setV7Enabled(false);
    res.json({ ok: true, ...getV7State() });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

router.post("/notional", (req, res) => {
  try {
    const usd = parseFloat((req.body?.usd ?? "").toString());
    if (!isFinite(usd) || usd <= 0) {
      return res.status(400).json({ error: "usd must be a positive number" });
    }
    setV7Notional(usd);
    res.json({ ok: true, ...getV7State() });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

router.post("/resume/:symbol", async (req, res) => {
  try {
    const r = await manualResume(req.params.symbol.toUpperCase());
    if (!r.resumed) return res.status(400).json(r);
    res.json({ ok: true, ...getV7State() });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

export default router;
