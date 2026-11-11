# Vigil

Leak early warning for liquid-cooled data centers that never leaves the data hall.

Vigil watches the cooling telemetry of every rack and raises an alert when a cooling loop starts drifting away from its own forecast, hours before a fixed low-pressure alarm would fire. It runs a small open-source time-series foundation model on a CPU, on-prem, with no per-site training.

Built at a hackathon in response to the Y Combinator Winter 2027 Request for Startups, "Vertical Foundation Models".

## Why

AI data centers are moving to liquid cooling. When a loop leaks, the whole rack has to come out for repair, and in container data halls the service aisles are tight, so downtime is expensive. Today most operators find out when pressure falls below a fixed threshold or a leak sensor gets wet. By then the options are limited.

Two things make this a good fit for a pretrained model instead of one model per site:

- Every data hall is configured differently and failures are rare, so there is little data to train on.
- Operators do not send telemetry off-site. The model has to come to the data.

## What the demo shows

A simulated container data hall with 12 liquid-cooled racks over 6 days.

- One rack (R07) develops a slow leak on day 4.5.
- Another rack (R03) runs a heavy compute job at the same time. It is healthy and should not be flagged.
- Vigil alerts on R07 **4.0 hours** after the leak starts, **9.4 hours before** the static low-pressure alarm (225 kPa).
- **0 of 11** healthy racks are flagged.
- Each forecast takes about 5 ms on a laptop CPU.

These numbers come from `pipeline.py` on simulated data. See [Status and limitations](#status-and-limitations).

## Run it

The dashboard is a static page with precomputed results. No install is needed.

```bash
python3 -m http.server 8765
```

Then open http://localhost:8765. Opening `index.html` directly in a browser also works.

Controls: **Play** or the space bar to run the timeline, **Before leak** to jump to just before the fault, arrow keys to step, and click any rack to inspect it.

### Regenerate the results

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python pipeline.py
```

This simulates the telemetry, runs the forecasts and rewrites `data.js`. The first run downloads the model weights from Hugging Face (about 190 MB). It takes roughly 30 seconds on an M3 laptop.

To run without the model, using a seasonal-naive forecast instead:

```bash
.venv/bin/python pipeline.py --baseline
```

## How it works

1. **Context, not training.** For each rack and signal, the model reads that rack's own last 48 hours and forecasts the next 4 hours as a 10–90% range.
2. **Compare forecast with reality.** Every 30 minutes, the range forecast 4 hours earlier is compared with what was measured since. The further the measurements fall outside the range, the higher the anomaly score.
3. **Score only load-independent signals.** A compute job moves flow, temperature and pump speed together. A leak shows up in supply pressure and in the hydraulic resistance index (pump speed divided by flow), which a compute job leaves alone. Only these two are scored.
4. **Calibrate on normal operation.** The alert threshold is set from the scores of the first 4 days, when nothing is wrong. An alert requires two consecutive evaluations above it.

## Model

Vigil uses [Chronos-Bolt-Small](https://huggingface.co/amazon/chronos-bolt-small), an open-source pretrained time-series model with about 48M parameters, through the `chronos-forecasting` package.

The model is used as-is. No model was trained or fine-tuned for this project. Fine-tuning on real cooling telemetry is planned future work.

## Status and limitations

- **The telemetry is simulated.** The leak signature (slow pressure decay, pump compensation) is an assumption and has not been validated against real data-center data. The lead time above holds only for this simulation.
- **One scenario.** One site, one fault type, one leak. Sensor drift, pump wear and different cooling hardware are untested.
- **Not yet a vertical foundation model.** This is a general open-source model with domain-specific signal selection on top.

## Roadmap

- Obtain telemetry from real liquid-cooled deployments, including at least one real leak incident.
- Benchmark against simple statistical baselines and newer time-series models on that data.
- Fine-tune on cooling telemetry and add more fault types.
- Package for on-prem deployment next to the cooling control system.

## Files

| File | Purpose |
|---|---|
| `index.html` | Offline dashboard |
| `data.js` | Precomputed telemetry, forecasts and scores |
| `pipeline.py` | Simulation, zero-shot forecasting and anomaly scoring |
| `requirements.txt` | Python dependencies for `pipeline.py` |
