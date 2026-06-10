"""Run this in your terminal to extract all N=30 stats.
   cd ~/Library/Mobile\ Documents/com\~apple\~CloudDocs/Papers/Saddam\ Project/apo_drl/src
   python3 ../../outputs/get_stats.py
"""
import csv, numpy as np, os, sys

base = os.path.join(os.path.dirname(__file__), '..', 'results')

methods = {
    'dqn':        ['dqn_seed0_n30.csv','dqn_seed42_n30.csv','dqn_seed123_n30.csv'],
    'd3qn_per':   ['d3qn_per_seed0_n30.csv','d3qn_per_seed42_n30.csv','d3qn_per_seed123_n30.csv'],
    'static':     ['static_seed0_n30.csv','static_seed42_n30.csv','static_seed123_n30.csv'],
    'ahp_topsis': ['ahp_topsis_seed0_n30.csv','ahp_topsis_seed42_n30.csv','ahp_topsis_seed123_n30.csv'],
    'random':     ['random_seed0_n30.csv','random_seed42_n30.csv','random_seed123_n30.csv'],
}

METRICS = ['throughput_mbps','latency_ms','energy_mj','qos_sat_pct','packet_loss_pct','switch_pct']

def tail(path, n=300):
    rows = list(csv.DictReader(open(path)))
    return {k: [float(r[k]) for r in rows[-n:]] for k in METRICS}

print(f"{'Method':<12} {'Throughput':>12} {'Latency':>10} {'QoS%':>8} {'PktLoss%':>10} {'Switch%':>10}")
for m, files in methods.items():
    seeds = []
    for f in files:
        p = os.path.join(base, f)
        if os.path.exists(p):
            s = tail(p)
            seeds.append({k: np.mean(v) for k,v in s.items()})
        else:
            print(f"  MISSING: {f}", file=sys.stderr)
    if seeds:
        r = {k: (np.mean([s[k] for s in seeds]), np.std([s[k] for s in seeds])) for k in METRICS}
        print(f"{m:<12} {r['throughput_mbps'][0]:>7.2f}±{r['throughput_mbps'][1]:.2f}  "
              f"{r['latency_ms'][0]:>8.1f}  {r['qos_sat_pct'][0]:>6.2f}  "
              f"{r['packet_loss_pct'][0]:>8.2f}±{r['packet_loss_pct'][1]:.2f}  "
              f"{r['switch_pct'][0]:>8.2f}±{r['switch_pct'][1]:.2f}")
