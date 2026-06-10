"""
plot.py  —  Generate all paper figures from real CSV results
============================================================
Run after training completes. Reads results/*.csv
Produces all figures required by the paper.

Changes vs. original:
  • fig3  — now 3-subplot figure (a) throughput, (b) loss, (c) epsilon
            Adds grey/brown dashed reference lines for Static and AHP-TOPSIS
  • fig4  — latency subplot uses log scale (extreme Random outlier distorted axis)
"""

import glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGS    = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGS, exist_ok=True)

# ── Baseline reference values (N=30, 3 seeds) ─────────────────────────────────
STATIC_THROUGHPUT    = 24.41   # Mbps  (from Table 4)
AHPTOPSIS_THROUGHPUT = 30.61   # Mbps  (from Table 4)

COLORS = {
    "APO-DRL":    "#D32F2F",
    "D3QN+PER":   "#E65100",
    "DQN":        "#7B1FA2",
    "Static":     "#616161",
    "AHP-TOPSIS": "#5D4037",
    "Random":     "#BDBDBD",
}
NAMES = {
    "apo_drl":    "APO-DRL",
    "d3qn_per":   "D3QN+PER",
    "dqn":        "DQN",
    "static":     "Static",
    "ahp_topsis": "AHP-TOPSIS",
    "random":     "Random",
}

plt.rcParams.update({
    "font.size":       10,
    "axes.labelsize":  11,
    "lines.linewidth": 1.5,
    "grid.alpha":      0.3,
})


# ── Data loading ───────────────────────────────────────────────────────────────
def load_all():
    out = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.csv"))):
        if os.path.getsize(f) == 0:
            print(f"  [skip] empty file: {os.path.basename(f)}")
            continue
        base  = os.path.basename(f).replace(".csv", "")
        parts = base.split("_seed")
        mkey  = parts[0]
        rest  = parts[1].split("_n")
        seed  = int(rest[0])
        n     = int(rest[1])
        out.setdefault(mkey, {}).setdefault(n, []).append(pd.read_csv(f))
    # Average across seeds
    avg = {}
    for mkey, nmap in out.items():
        avg[mkey] = {}
        for n, dfs in nmap.items():
            L       = min(len(d) for d in dfs)
            stacked = np.stack([d.values[:L] for d in dfs])
            cols    = dfs[0].columns.tolist()
            avg[mkey][n] = {
                "mean":    pd.DataFrame(stacked.mean(0), columns=cols),
                "std":     pd.DataFrame(stacked.std(0),  columns=cols),
                "n_seeds": len(dfs),
            }
    return avg


def smooth(x, w=30):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


# ── Figure 3: combined 3-panel convergence figure ─────────────────────────────
# Panel (a): throughput convergence with dashed reference lines
# Panel (b): TD loss (log scale)
# Panel (c): exploration rate ε
def fig3(data):
    drl_methods = [k for k in data
                   if NAMES.get(k, k) not in ("Static", "AHP-TOPSIS", "Random")]

    fig, (ax_tp, ax_loss, ax_eps) = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── (a) Throughput convergence ──────────────────────────────────────────
    for mkey in drl_methods:
        nmap = data[mkey]
        n    = sorted(nmap.keys())[0]          # use smallest N (= training N=30)
        mn   = nmap[n]["mean"]
        sd   = nmap[n]["std"]
        lbl  = NAMES.get(mkey, mkey)
        col  = COLORS.get(lbl, "#000")

        x = mn["episode"].values
        y = smooth(mn["throughput_mbps"].values)
        s = smooth(sd["throughput_mbps"].values)

        ax_tp.plot(x, y, color=col, label=lbl)
        ax_tp.fill_between(x, y - s, y + s, color=col, alpha=0.12)

    # Dashed reference lines for baselines
    x_max = ax_tp.get_xlim()[1] if ax_tp.get_xlim()[1] > 1 else 500
    ax_tp.axhline(STATIC_THROUGHPUT,
                  ls="--", color="#616161", linewidth=1.4,
                  label=f"Static Allocation ({STATIC_THROUGHPUT} Mbps)")
    ax_tp.axhline(AHPTOPSIS_THROUGHPUT,
                  ls="--", color="#5D4037", linewidth=1.4,
                  label=f"AHP-TOPSIS ({AHPTOPSIS_THROUGHPUT} Mbps)")

    ax_tp.set_xlabel("Training Episode")
    ax_tp.set_ylabel("Throughput (Mbps)")
    ax_tp.set_title("(a) Throughput Convergence (±1 std, 3 seeds)")
    ax_tp.legend(loc="lower right", fontsize=8)
    ax_tp.grid(True)

    # ── (b) TD Loss ─────────────────────────────────────────────────────────
    for mkey in drl_methods:
        lbl  = NAMES.get(mkey, mkey)
        nmap = data[mkey]
        n    = sorted(nmap.keys())[0]
        mn   = nmap[n]["mean"]

        if "loss" not in mn.columns:
            continue
        loss = mn["loss"].values
        if (loss > 0).sum() == 0:
            continue

        col = COLORS.get(lbl, "#000")
        ax_loss.plot(mn["episode"], smooth(np.maximum(loss, 1e-6)),
                     color=col, label=lbl)

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Training Episode")
    ax_loss.set_ylabel("TD Loss (log scale)")
    ax_loss.set_title("(b) TD Loss (gradient clipping L2=10)")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, which="both")

    # ── (c) Exploration rate ε ───────────────────────────────────────────────
    for mkey in drl_methods:
        lbl  = NAMES.get(mkey, mkey)
        nmap = data[mkey]
        n    = sorted(nmap.keys())[0]
        mn   = nmap[n]["mean"]

        if "epsilon" not in mn.columns:
            continue

        col = COLORS.get(lbl, "#000")
        ax_eps.plot(mn["episode"], mn["epsilon"], color=col, label=lbl)

    ax_eps.set_xlabel("Training Episode")
    ax_eps.set_ylabel("ε (epsilon)")
    ax_eps.set_title("(c) Exploration Rate\n(identical schedule by design, min ε=0.05)")
    ax_eps.legend(fontsize=8)
    ax_eps.grid(True)

    plt.tight_layout()
    p = os.path.join(FIGS, "fig3_convergence.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ── Figure 4: metric bar charts ───────────────────────────────────────────────
# Latency uses LOG scale (Random Selection outlier >2,000,000 ms distorts axis)
def fig4(data):
    # 6 metrics in 2-row × 3-col layout
    metrics = ["throughput_mbps", "latency_ms", "energy_mj",
               "qos_sat_pct",     "packet_loss_pct", "switch_pct"]
    labels  = ["Throughput (Mbps)", "Latency (ms) — log scale", "Energy (mJ)",
               "QoS Satisfaction (%)", "Packet Loss (%)", "Switch Rate (%)"]

    methods = sorted(data.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()

    for i, (m, lbl) in enumerate(zip(metrics, labels)):
        vals = []
        disp = []
        cols = []

        for mkey in methods:
            n    = sorted(data[mkey].keys())[0]
            mn   = data[mkey][n]["mean"]
            tail = mn.tail(max(10, len(mn) // 10))

            if m not in tail.columns:
                continue

            vals.append(tail[m].mean())
            d = NAMES.get(mkey, mkey)
            disp.append(d)
            cols.append(COLORS.get(d, "#444"))

        bars = axes[i].bar(disp, vals, color=cols, edgecolor="white", linewidth=0.5)
        axes[i].set_title(lbl, fontsize=11, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=35, labelsize=9)
        axes[i].grid(True, axis="y", alpha=0.3)

        # Annotate best bar with value
        best_idx = vals.index(min(vals) if m in ["latency_ms","packet_loss_pct","switch_pct"] else max(vals))
        axes[i].bar(disp[best_idx], vals[best_idx],
                    color=cols[best_idx], edgecolor="#FFD700", linewidth=2.5)
        axes[i].annotate(f"{vals[best_idx]:.1f}",
                         xy=(best_idx, vals[best_idx]),
                         ha="center", va="bottom", fontsize=8, fontweight="bold")

        if m == "latency_ms":
            axes[i].set_yscale("log")
            axes[i].set_ylabel("ms (log scale)")
            axes[i].annotate("Log scale: Random Selection\nexceeds 2,000,000 ms",
                             xy=(0.02, 0.98), xycoords="axes fraction",
                             va="top", fontsize=7, color="#777")

    plt.suptitle(
        "Figure 4. Performance Comparison — All Methods at N=30 (Fair Comparison, 3 Seeds)\n"
        "Gold border = best result per metric. Scalability analysis in Table 5.",
        y=1.01, fontsize=11
    )
    plt.tight_layout()
    p = os.path.join(FIGS, "fig4_bars.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ── Figure 5: scalability (previously fig6) ───────────────────────────────────
def fig5(data):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for mkey in data:
        lbl = NAMES.get(mkey, mkey)
        ns  = sorted(data[mkey].keys())
        if len(ns) < 2:
            continue
        tp = []; qos = []
        for n in ns:
            tail = data[mkey][n]["mean"].tail(max(10, len(data[mkey][n]["mean"]) // 10))
            tp.append(tail["throughput_mbps"].mean())
            qos.append(tail["qos_sat_pct"].mean())
        col = COLORS.get(lbl, "#000")
        a1.plot(ns, tp, "o-", color=col, label=lbl)
        a2.plot(ns, qos, "o-", color=col, label=lbl)

    a2.axhline(90, ls="--", color="red", alpha=0.5, label="90% QoS target")

    a1.set_xlabel("Number of IoT Devices (N)")
    a1.set_ylabel("Throughput (Mbps)")
    a1.set_title("(a) Throughput vs. Device Density\n(DRL: N=30–200; baselines: N=30–2000)")
    a1.legend(fontsize=8); a1.grid(True)

    a2.set_xlabel("Number of IoT Devices (N)")
    a2.set_ylabel("QoS Satisfaction (%)")
    a2.set_title("(b) QoS Satisfaction vs. Device Density")
    a2.legend(fontsize=8); a2.grid(True)

    plt.tight_layout()
    p = os.path.join(FIGS, "fig5_scalability.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading CSVs from:", RESULTS)
    data = load_all()
    if not data:
        print("No CSV files found in", RESULTS)
        print("Run train.py first.")
        return
    print(f"Found methods: {list(data.keys())}\n")

    fig3(data)   # 3-panel: throughput + loss + epsilon (with dashed baselines)
    fig4(data)   # 4-panel bars: throughput, latency(log), energy, QoS
    fig5(data)   # scalability curves

    print(f"\nAll figures saved to {FIGS}")


if __name__ == "__main__":
    main()
