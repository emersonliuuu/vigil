"""Liquid-cooling leak early warning: simulate telemetry, run zero-shot forecasts, score anomalies.

Writes data.js (window.DATA = {...}) for index.html. No training happens anywhere in this file.
Usage: .venv/bin/python pipeline.py [--baseline]
"""
import json
import sys
import time

import numpy as np

STEP_MIN = 5
PER_DAY = 24 * 60 // STEP_MIN          # 288
DAYS = 6
N = DAYS * PER_DAY                      # 1728 points per signal
RACKS = [f"R{i:02d}" for i in range(1, 13)]
LEAK_RACK = "R07"
BURST_RACK = "R03"                      # heavy compute job, NOT a fault
LEAK_START = int(4.5 * PER_DAY)         # day 4.5
LEAK_HOURS = 20.0                       # onset -> severe
CONTEXT = 2 * PER_DAY                   # each rack's own last 2 days
HORIZON = 48                            # 4 h ahead
EVAL_EVERY = 6                          # score every 30 min
CALIB_END = 4 * PER_DAY                 # normal-operation period used to set the alert threshold
PRESSURE_ALARM = 225.0                  # classic static low-pressure alarm (kPa)
SIGNALS = [
    {"key": "pressure", "name": "Supply pressure", "unit": "kPa"},
    {"key": "resist", "name": "Hydraulic resistance index (pump speed ÷ flow)", "unit": "index"},
    {"key": "flow", "name": "Coolant flow", "unit": "L/min"},
    {"key": "dT", "name": "Supply/return ΔT", "unit": "°C"},
]
# Only load-independent channels are scored: a compute job moves flow, dT and pump together,
# but leaves pressure and pump-effort-per-flow alone. A leak moves exactly those two.
SCORE_KEYS = ["pressure", "resist"]
QUANTILES = [0.1, 0.5, 0.9]


def simulate(seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    hour = (t % PER_DAY) / PER_DAY * 24
    out = {}
    for r in RACKS:
        phase = rng.uniform(-1.5, 1.5)
        base_load = rng.uniform(0.55, 0.75)
        load = base_load + 0.12 * np.sin((hour - 14 + phase) / 24 * 2 * np.pi)
        # random training jobs: step up for 1-4 h
        for _ in range(rng.integers(5, 9)):
            s = rng.integers(0, N - 60)
            load[s:s + rng.integers(12, 48)] += rng.uniform(0.04, 0.10)
        if r == BURST_RACK:
            s = int(4.6 * PER_DAY)
            load[s:s + 60] += 0.22      # big job during the leak window, on a healthy rack
        load = np.clip(load + rng.normal(0, 0.01, N), 0.3, 1.05)

        p0 = rng.uniform(246, 254)
        f0 = rng.uniform(57, 63)
        leak = np.zeros(N)
        if r == LEAK_RACK:
            tau = np.clip((t - LEAK_START) * STEP_MIN / 60 / LEAK_HOURS, 0, None)
            leak = tau ** 1.5           # slow at first, then accelerating

        # pump affinity law: speed tracks the flow setpoint; a leak forces extra speed for the same flow
        pump = 0.9 * f0 * (0.78 + 0.3 * load) * (1 + 0.25 * np.minimum(leak, 1.2)) + rng.normal(0, 0.6, N)
        pressure = p0 + 2.0 * (load - base_load) - 45 * leak + rng.normal(0, 0.9, N)
        flow = f0 * (0.78 + 0.3 * load) * (1 - 0.22 * np.clip(leak - 0.15, 0, None)) + rng.normal(0, 0.5, N)
        dT = 4.5 + 9.0 * load * (f0 * (0.78 + 0.3 * load)) / np.maximum(flow, 1) + rng.normal(0, 0.15, N)
        pump = np.clip(pump, 0, 100)
        out[r] = {"pressure": pressure, "resist": 100 * pump / np.maximum(flow, 1), "flow": flow, "dT": dT}
    return out


class ChronosForecaster:
    def __init__(self):
        import torch
        from chronos import BaseChronosPipeline
        self.torch = torch
        self.model_id = "amazon/chronos-bolt-small"
        self.pipe = BaseChronosPipeline.from_pretrained(self.model_id, device_map="cpu")
        inner = getattr(self.pipe, "model", None) or getattr(self.pipe, "inner_model", None)
        self.params = int(sum(p.numel() for p in inner.parameters())) if inner is not None else None
        self.label = "Chronos-Bolt-Small (zero-shot)"

    def predict(self, ctx):               # ctx: [B, C] -> [B, H, 3]
        outs = []
        for i in range(0, len(ctx), 256):
            batch = self.torch.tensor(ctx[i:i + 256], dtype=self.torch.float32)
            q, _ = self.pipe.predict_quantiles(batch, prediction_length=HORIZON, quantile_levels=QUANTILES)
            outs.append(q.numpy())
        return np.concatenate(outs)


class BaselineForecaster:
    """Seasonal-naive fallback: yesterday's shape, re-levelled, band from yesterday's day-over-day error."""
    model_id = "seasonal-naive"
    label = "Seasonal-naive baseline (no ML)"
    params = 0

    def predict(self, ctx):
        last_day = ctx[:, -PER_DAY:]
        prev_day = ctx[:, -2 * PER_DAY:-PER_DAY]
        shift = (ctx[:, -12:].mean(1) - prev_day[:, -12:].mean(1))[:, None]   # level now vs same time yesterday
        med = last_day[:, :HORIZON] + shift
        resid = last_day - prev_day
        lo = np.quantile(resid, 0.1, axis=1)[:, None]
        hi = np.quantile(resid, 0.9, axis=1)[:, None]
        return np.stack([med + lo, med, med + hi], axis=2)


def main():
    data = simulate()
    try:
        if "--baseline" in sys.argv:
            raise RuntimeError("baseline requested")
        fc = ChronosForecaster()
    except Exception as e:                # model unavailable -> still produce a working demo
        print("Falling back to baseline:", repr(e)[:200])
        fc = BaselineForecaster()

    eval_times = list(range(CONTEXT + HORIZON, N + 1, EVAL_EVERY))
    keys = [s["key"] for s in SIGNALS]

    # 1) rolling evaluation: forecast made 4 h ago vs what actually happened since
    ctx, index = [], []
    for te in eval_times:
        for r in RACKS:
            for k in SCORE_KEYS:
                ctx.append(data[r][k][te - HORIZON - CONTEXT: te - HORIZON])
                index.append((te, r, k))
    ctx = np.asarray(ctx)
    t0 = time.time()
    q = fc.predict(ctx)
    infer_s = time.time() - t0
    n_fc = len(ctx)

    scores = {r: {k: [] for k in SCORE_KEYS} for r in RACKS}
    for (te, r, k), qq in zip(index, q):
        actual = data[r][k][te - HORIZON: te]
        width = np.maximum(qq[:, 2] - qq[:, 0], 1e-6)
        dev = np.maximum(np.maximum(actual - qq[:, 2], qq[:, 0] - actual), 0) / width
        scores[r][k].append(float(np.minimum(dev, 3).mean()))
    rack_score = {}
    for r in RACKS:
        rack_score[r] = np.array([scores[r][k] for k in SCORE_KEYS]).mean(0)

    # 2) alert threshold from normal operation only (first 4 days, all racks)
    calib_idx = [i for i, te in enumerate(eval_times) if te <= CALIB_END]
    calib = np.concatenate([rack_score[r][calib_idx] for r in RACKS])
    threshold = float(np.quantile(calib, 0.995) * 1.5)

    def first_alert(r):
        s = rack_score[r]
        for i in range(len(calib_idx), len(s) - 1):
            if s[i] > threshold and s[i + 1] > threshold:      # two consecutive evaluations
                return eval_times[i + 1]
        return None

    alerts = {r: first_alert(r) for r in RACKS}
    p = data[LEAK_RACK]["pressure"]
    smooth = np.convolve(p, np.ones(6) / 6, mode="same")
    below = np.where((np.arange(N) > LEAK_START) & (smooth < PRESSURE_ALARM))[0]
    static_alarm = int(below[0]) if len(below) else None
    false_alerts = [r for r in RACKS if r != LEAK_RACK and alerts[r] is not None]

    # 3) stitched forecast bands for display (same model, same 4 h horizon, non-overlapping windows)
    starts = list(range(CONTEXT, N, HORIZON))
    ctx2 = np.asarray([data[r][k][s - CONTEXT: s] for s in starts for r in RACKS for k in keys])
    q2 = fc.predict(ctx2)
    bands = {r: {k: {"lo": [None] * N, "hi": [None] * N} for k in keys} for r in RACKS}
    i = 0
    for s in starts:
        for r in RACKS:
            for k in keys:
                for j in range(min(HORIZON, N - s)):
                    bands[r][k]["lo"][s + j] = round(float(q2[i][j, 0]), 2)
                    bands[r][k]["hi"][s + j] = round(float(q2[i][j, 2]), 2)
                i += 1

    to_h = lambda steps: None if steps is None else round(steps * STEP_MIN / 60, 1)
    leak_alert = alerts[LEAK_RACK]
    meta = {
        "model": fc.label, "model_id": fc.model_id, "params": fc.params,
        "step_min": STEP_MIN, "n": N, "context_h": CONTEXT * STEP_MIN // 60, "horizon_h": HORIZON * STEP_MIN // 60,
        "leak_rack": LEAK_RACK, "burst_rack": BURST_RACK, "leak_start": LEAK_START,
        "alert_t": leak_alert, "static_alarm_t": static_alarm, "pressure_alarm": PRESSURE_ALARM,
        "detect_after_h": to_h(leak_alert - LEAK_START) if leak_alert else None,
        "lead_h": to_h(static_alarm - leak_alert) if leak_alert and static_alarm else None,
        "false_alert_racks": false_alerts, "threshold": round(threshold, 4),
        "n_forecasts": n_fc, "infer_s": round(infer_s, 2), "ms_per_forecast": round(infer_s / n_fc * 1000, 2),
        "calib_days": CALIB_END // PER_DAY,
    }
    payload = {
        "meta": meta, "signals": SIGNALS, "racks": RACKS, "eval_times": eval_times,
        "scores": {r: [round(float(x), 4) for x in rack_score[r]] for r in RACKS},
        "alerts": alerts,
        "series": {r: {k: [round(float(x), 2) for x in data[r][k]] for k in keys} for r in RACKS},
        "bands": bands,
    }
    with open("data.js", "w") as f:
        f.write("window.DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print("healthy max score:", round(max(float(rack_score[r].max()) for r in RACKS if r != LEAK_RACK), 4),
          "| leak max:", round(float(rack_score[LEAK_RACK].max()), 4))


if __name__ == "__main__":
    main()
