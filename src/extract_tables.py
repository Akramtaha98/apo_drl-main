"""
extract_tables.py  —  Read CSVs and print Table 5 & Table 6 numbers
====================================================================
Run AFTER training is complete:
    python extract_tables.py

Prints:
  * Table 5 (N=1000, all methods, mean ± std across seeds)
  * Table 6 (Scalability, key methods across N=500/1000/1500/2000)
  * Improvement percentages for the abstract and conclusion
"""

import glob, os, sys
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
LAST_FRAC = 0.10   # use last 10% of episodes as "converged" performance

def load(method, n_devices):
    # Only use seeds 0, 42, 123 for fair comparison
    files = []
    for seed in [0, 42, 123]:
        f = os.path.join(RESULTS, f"{method}_seed{seed}_n{n_devices}.csv")
        if os.path.exists(f):
            files.append(f)
    if not files:
        return None, None
    dfs = []
    for f in files:
        df   = pd.read_csv(f)
        tail = df.tail(max(10, int(len(df)*LAST_FRAC)))
        dfs.append(tail.mean(numeric_only=True))
    stacked = pd.DataFrame(dfs)
    return stacked.mean(), stacked.std()


def fmt(mean, std, key, is_det):
    if is_det:
        return f"{mean[key]:.2f}"
    return f"{mean[key]:.2f}±{std[key]:.2f}"


def table5():
    print("\n" + "="*80)
    print("TABLE 5  —  Performance comparison, N = 30 IoT devices")
    print("           (mean ± std over seeds, last 10% of training episodes)")
    print("="*80)

    methods = [
        ("Static",     "static",     True),
        ("Random",     "random",     True),
        ("AHP-TOPSIS", "ahp_topsis", True),
        ("DQN",        "dqn",        False),
        ("D3QN+PER",   "d3qn_per",   False),
        ("APO-DRL",    "apo_drl",    False),
    ]

    header = f"{'Method':<14} {'Throughput':>12} {'Latency':>12} {'Energy':>10} {'QoS%':>10} {'Loss%':>10} {'Switch%':>10}"
    print(header)
    print("-"*80)

    results = {}
    for display, key, is_det in methods:
        mn, sd = load(key, 30)
        if mn is None:
            print(f"  {display:<14}  [NOT FOUND — run train.py --method {key} first]")
            continue
        results[key] = (mn, sd)
        print(f"  {display:<14}"
              f"  {fmt(mn,sd,'throughput_mbps',is_det):>12}"
              f"  {fmt(mn,sd,'latency_ms',is_det):>12}"
              f"  {fmt(mn,sd,'energy_mj',is_det):>10}"
              f"  {fmt(mn,sd,'qos_sat_pct',is_det):>10}"
              f"  {fmt(mn,sd,'packet_loss_pct',is_det):>10}"
              f"  {fmt(mn,sd,'switch_pct',is_det):>10}")

    # Improvement percentages
    if "static" in results and "apo_drl" in results:
        sa  = results["static"][0]
        apo = results["apo_drl"][0]
        print("\n── Improvement: APO-DRL vs Static Allocation ──")
        tp_gain  = 100*(apo["throughput_mbps"] - sa["throughput_mbps"]) / max(sa["throughput_mbps"],1e-9)
        lat_red  = 100*(sa["latency_ms"]  - apo["latency_ms"])  / max(sa["latency_ms"],1e-9)
        eng_diff = 100*(sa["energy_mj"]   - apo["energy_mj"])   / max(sa["energy_mj"],1e-9)
        qos_gain = 100*(apo["qos_sat_pct"]- sa["qos_sat_pct"])  / max(sa["qos_sat_pct"],1e-9)
        print(f"  Throughput improvement : {tp_gain:+.1f}%")
        print(f"  Latency reduction      : {lat_red:+.1f}%")
        print(f"  Energy change          : {eng_diff:+.1f}%  (negative = less energy used = better)")
        print(f"  QoS improvement        : {qos_gain:+.1f}%")
        print()
        print("  ► Copy THESE numbers into your Abstract and Conclusion.")
        print("  ► Do NOT use the old fabricated numbers (27.3%, 34.8%, etc.).")


def table6():
    print("\n" + "="*80)
    print("TABLE 6  —  Scalability (APO-DRL, D3QN+PER, Static across N)")
    print("="*80)

    methods = [("APO-DRL","apo_drl"), ("D3QN+PER","d3qn_per"), ("Static","static")]
    scales  = [500, 1000, 1500, 2000]

    print(f"  {'N':>6}  {'Method':<14} {'Throughput':>12} {'Latency':>12} {'Energy':>10} {'QoS%':>10}")
    print("  " + "-"*68)

    for n in scales:
        for display, key in methods:
            mn, sd = load(key, n)
            if mn is None:
                print(f"  {n:>6}  {display:<14}  [NOT FOUND — run train.py --method {key} --devices {n}]")
            else:
                is_det = key in ("static",)
                print(f"  {n:>6}  {display:<14}"
                      f"  {fmt(mn,sd,'throughput_mbps',is_det):>12}"
                      f"  {fmt(mn,sd,'latency_ms',is_det):>12}"
                      f"  {fmt(mn,sd,'energy_mj',is_det):>10}"
                      f"  {fmt(mn,sd,'qos_sat_pct',is_det):>10}")
        print()


if __name__ == "__main__":
    table5()
    table6()
    print("\nDone. Copy these numbers directly into your paper tables.")
