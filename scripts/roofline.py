#!/usr/bin/env python3
"""
Roofline model for BitFilter — plots hardware ceilings and measured variants.

Units: Ops/Byte (bitwise operations per byte transferred), not FLOP/Byte,
because this workload does zero floating-point math.

Usage:
    python3 scripts/roofline.py            # writes site/img/roofline.svg + .png
"""

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Hardware ceilings ──────────────────────────────────────────────────────────
# DDR4-2660 dual-channel
THEORETICAL_DRAM_BW = 42.6   # GB/s
PRACTICAL_DRAM_BW_LO = 30.0  # GB/s  (practical band, low)
PRACTICAL_DRAM_BW_HI = 35.0  # GB/s  (practical band, high)

# Peak compute: 256-bit AVX2 @ ~4.5 GHz, 1 bitwise op/cycle per port × 2 ports
# = 2 × 32 bytes × 4.5 GHz ≈ 288 GB/s of bitwise throughput (ops ≈ bytes here)
# Expressed as ops/s for the roofline diagonal: ~288 Gops/s
PEAK_COMPUTE_OPS_PER_SEC = 288  # Gops/s (bitwise ops, not FLOP)

# ── Measured data points ───────────────────────────────────────────────────────
# Each entry: (label, ops_per_byte, throughput_GBs, marker, color)
# Operational intensity: for eval, ~1 op (AND/ANDN) per 8 bytes read ≈ 0.03 ops/byte
#                        for popcount, ~1 popcnt per 8 bytes ≈ 0.01 ops/byte (lighter)
variants = [
    # Eval variants — actual DRAM throughput (including write-allocate)
    ("Eval scalar (auto-vec)",   0.03, 29.0, "o", "#2563eb"),  # blue
    ("Eval AVX2+prefetch",       0.03, 29.0, "D", "#1d4ed8"),  # dark blue
    # Popcount variants
    ("Popcount ST",              0.01, 17.1, "s", "#dc2626"),  # red
    ("Popcount MT (8T)",         0.01, 31.3, "^", "#991b1b"),  # dark red
]

# App-level point for write-allocate annotation
EVAL_APP_THROUGHPUT = 23.0  # GB/s (what SetBytesProcessed reports)
EVAL_DRAM_THROUGHPUT = 29.0  # GB/s (actual bus traffic)
EVAL_OI = 0.03

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))

# X-axis range (log scale)
oi_range = np.logspace(-3, 2, 500)  # 0.001 to 100 ops/byte

# Roofline: min(peak_compute, peak_bw × OI)
# Throughput (GB/s) = min(PEAK_COMPUTE / OI_to_ops_ratio, BW)
# But since our Y-axis is GB/s (bandwidth achieved), the memory-bound roof
# is horizontal at BW, and the compute-bound roof rises with OI.
# compute_roof: throughput = PEAK_COMPUTE_OPS_PER_SEC / OI ... but we need
# consistent units.  Simpler: the roofline in BW terms is
#   achievable_BW = min( peak_BW, peak_compute / OI )
# where peak_compute is in Gops/s and OI is in ops/byte → BW in GB/s.

compute_roof = PEAK_COMPUTE_OPS_PER_SEC / oi_range  # GB/s if compute-limited
memory_roof = np.full_like(oi_range, THEORETICAL_DRAM_BW)
roofline = np.minimum(compute_roof, memory_roof)

# Ridge point: where memory and compute ceilings meet
ridge_oi = PEAK_COMPUTE_OPS_PER_SEC / THEORETICAL_DRAM_BW  # ~6.8 ops/byte

# Draw roofline
ax.loglog(oi_range, roofline, color="#111827", linewidth=2.5, zorder=5)

# Practical DRAM band (shaded)
ax.axhspan(PRACTICAL_DRAM_BW_LO, PRACTICAL_DRAM_BW_HI,
           color="#22c55e", alpha=0.15, zorder=1)
ax.axhline(PRACTICAL_DRAM_BW_LO, color="#22c55e", linewidth=0.8,
           linestyle="--", alpha=0.5)
ax.axhline(PRACTICAL_DRAM_BW_HI, color="#22c55e", linewidth=0.8,
           linestyle="--", alpha=0.5)
ax.text(50, 32.2, "Practical DRAM\n30–35 GB/s", fontsize=8,
        color="#15803d", ha="right", va="center")

# Theoretical ceiling label
ax.text(0.0015, THEORETICAL_DRAM_BW * 1.08, f"Theoretical DRAM: {THEORETICAL_DRAM_BW} GB/s",
        fontsize=8.5, color="#111827", va="bottom", fontweight="bold")

# Compute ceiling label (on the diagonal)
ax.text(40, PEAK_COMPUTE_OPS_PER_SEC / 40 * 1.15,
        f"Compute: ~{PEAK_COMPUTE_OPS_PER_SEC} Gops/s",
        fontsize=8, color="#111827", rotation=-32, va="bottom")

# Plot data points
for label, oi, bw, marker, color in variants:
    ax.plot(oi, bw, marker=marker, markersize=11, color=color,
            markeredgecolor="white", markeredgewidth=1.2, zorder=10)
    # Use leader lines to place labels in open space, away from the dense cluster
    leader = dict(arrowstyle="-", color=color, linewidth=0.8, alpha=0.6)
    if "Popcount ST" in label:
        ax.annotate(label, (oi, bw), textcoords="offset points",
                    xytext=(80, -40), fontsize=8.5, color=color,
                    fontweight="semibold", arrowprops=leader)
    elif "Popcount MT" in label:
        ax.annotate(label, (oi, bw), textcoords="offset points",
                    xytext=(80, 30), fontsize=8.5, color=color,
                    fontweight="semibold", arrowprops=leader)
    elif "scalar" in label:
        ax.annotate(label, (oi, bw), textcoords="offset points",
                    xytext=(80, -20), fontsize=8.5, color=color,
                    fontweight="semibold", arrowprops=leader)
    elif "AVX2" in label:
        ax.annotate(label, (oi, bw), textcoords="offset points",
                    xytext=(80, 10), fontsize=8.5, color=color,
                    fontweight="semibold", arrowprops=leader)

# ── Write-allocate annotation ──────────────────────────────────────────────────
# Arrow from app-level (23 GB/s) up to actual DRAM (29 GB/s), placed far left
WA_OI = EVAL_OI * 0.35
ax.annotate("",
            xy=(WA_OI, EVAL_DRAM_THROUGHPUT),
            xytext=(WA_OI, EVAL_APP_THROUGHPUT),
            arrowprops=dict(arrowstyle="->", color="#6b7280",
                            linestyle="--", linewidth=1.5))
ax.plot(WA_OI, EVAL_APP_THROUGHPUT, marker="_", markersize=8,
        color="#6b7280", zorder=9)
ax.text(WA_OI * 0.55, 26.0, "Write-allocate\n+26%",
        fontsize=8, color="#6b7280", ha="right", va="center",
        style="italic")
ax.text(WA_OI * 0.55, EVAL_APP_THROUGHPUT - 1.8, "App: 23 GB/s",
        fontsize=7.5, color="#9ca3af", ha="right")

# ── Formatting ─────────────────────────────────────────────────────────────────
ax.set_xlabel("Operational Intensity  (ops / byte)", fontsize=11, labelpad=8)
ax.set_ylabel("Throughput  (GB/s)", fontsize=11, labelpad=8)
ax.set_title("BitFilter Roofline — DDR4-2660 Dual-Channel + AVX2",
             fontsize=13, fontweight="bold", pad=14)

ax.set_xlim(0.001, 100)
ax.set_ylim(1, 500)
ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
ax.tick_params(labelsize=9)

# Legend — manual for clarity
legend_elements = [
    mpatches.Patch(facecolor="#2563eb", label="Eval (3-bitmap boolean)"),
    mpatches.Patch(facecolor="#dc2626", label="Popcount (read-only scan)"),
    mpatches.Patch(facecolor="#22c55e", alpha=0.3, label="Practical DRAM band"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8.5,
          framealpha=0.9)

fig.tight_layout()

# ── Export ─────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent.parent / "site" / "img"
out_dir.mkdir(parents=True, exist_ok=True)

svg_path = out_dir / "roofline.svg"
png_path = out_dir / "roofline.png"

fig.savefig(svg_path, format="svg", bbox_inches="tight")
fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")

print(f"Saved: {svg_path}")
print(f"Saved: {png_path}")
