#!/usr/bin/env python3
"""static/mesh/sample-flips.csv: the gate page's sample upload (/gate/, "Fit Your Own Data"), synthetic.  20,000 trials at
random positions, x in [0, 40] and y in [100, 300], each a coin whose odds come from a smooth made-up surface (a ridge
plus a bump, about a quarter successes).  Seeded, so the file is reproducible:  python3 tools/make_sample_flips.py"""
import math, random
from pathlib import Path

rng = random.Random(20260926)
out = Path(__file__).resolve().parents[1] / "static" / "mesh" / "sample-flips.csv"
rows = ["x,y,outcome"]
for _ in range(20000):
    x, y = rng.uniform(0, 40), rng.uniform(100, 300)
    u, v = x / 40, (y - 100) / 200
    logit = -0.75 + 1.1 * math.exp(-((u - 0.3) ** 2 + (v - 0.6) ** 2) / 0.02) + 0.8 * math.tanh(6 * (u - 0.7)) - 0.4 * v
    rows.append(f"{x:.2f},{y:.1f},{int(rng.random() < 1 / (1 + math.exp(-logit)))}")
out.write_text("\n".join(rows) + "\n")
print(f"wrote {out}: {len(rows) - 1} trials, {sum(r.endswith(',1') for r in rows) / (len(rows) - 1):.1%} successes")
