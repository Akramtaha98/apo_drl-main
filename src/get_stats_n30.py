"""
Run from apo_drl/src/ directory:
  python3 get_stats_n30.py

Reads all N=30 CSVs from ../results/ and prints the full Table 4.
"""
import csv, numpy as np, os, sys

RES = os.path.join(os.path.dirname(__file__), "../results")
METRICS = ["throughput_mbps","latency_ms","energy_mj","qos_sat_pct","packet_loss_pct","switch_pct"]
SEEDS   = [0, 42, 123]
N = 30
TAIL = 300  # last episodes to average

METHODS = {
    "Static (SA)":    "static",
    "Random (RS)":    "random",
    "AHP-TOPSIS":     "ahp_topsis",
    "DQN":            "dqn",
    "D3QN+PER":       "d3qn_per",
    "APO-DRL ★":      "apo_drl",
}

results = {}
for label, key in METHODS.items():
    seed_means = []
    for s in SEEDS:
        fname = os.path.join(RES, f"{key}_seed{s}_n{N}.csv")
        if not os.path.exists(fname):
            print(f"  MISSING: {fname}", file=sys.stderr)
            continue
        rows = list(csv.DictReader(open(fname)))
        if len(rows) == 0:
            print(f"  EMPTY: {fname}", file=sys.stderr)
            continue
        tail = rows[-TAIL:]
        seed_means.append({m: np.mean([float(r[m]) for r in tail]) for m in METRICS})
    if seed_means:
        results[label] = {
            m: (np.mean([s[m] for s in seed_means]),
                np.std([s[m] for s in seed_means], ddof=1) if len(seed_means)>1 else 0.0)
            for m in METRICS
        }
    else:
        results[label] = None
        print(f"  NO DATA for {label}", file=sys.stderr)

print("\n" + "="*100)
print("TABLE 4 — N=30 devices, 3 seeds {0,42,123}, last 300 of 3000 episodes")
print("="*100)
LABELS = list(METHODS.keys())
header = f"{'Metric':<24}" + "".join(f"  {l:<18}" for l in LABELS)
print(header)
print("-"*100)
for m in METRICS:
    row = f"{m:<24}"
    for l in LABELS:
        if results[l] is None:
            row += f"  {'N/A':<18}"
        else:
            mn, sd = results[l][m]
            val = f"{mn:.2f} ±{sd:.2f}" if sd > 0.005 else f"{mn:.4f}"
            row += f"  {val:<18}"
    print(row)
print("="*100)
