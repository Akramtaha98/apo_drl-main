"""
per_class_eval.py — Per-class QoS / P99-latency breakdown for ALL methods,
                     using the SAME frozen-policy / direct-evaluation
                     protocol described in the manuscript (Section 5.1
                     and 5.4.2), on the SAME held-out seeds {200,201,202}.

This closes the reviewer comment: "the per class P99 latency for APO-DRL
itself was not separately reported."

It does NOT modify env.py, agent.py, or evaluate.py (the files referenced
by the Data Availability commit hash). It only reads the public env
state (env._classes, env._cur) from the outside to label each logged
step by device class, after calling the existing, unmodified step().

USAGE
-----
1) APO-DRL (needs a saved checkpoint for N=30; train.py saves one
   automatically to <out_dir>/checkpoints/apo_drl_seed<seed>_n30.pt
   the next time you run, e.g.:

     python train.py --method apo_drl --seeds 0 42 123 --episodes 3000 \
                      --devices 30 --out_dir ../results --device cpu

   If you still have the ORIGINAL n30 training run's checkpoints folder
   from when apo_drl_seed{0,42,123}_n30.csv were produced, point
   --checkpoint_dir at that folder instead of retraining — retraining
   is only needed if the checkpoints were not kept.)

   Then:
     python per_class_eval.py --method apo_drl \
            --checkpoint_dir ../results/checkpoints \
            --seeds 0 42 123 --eval_seeds 200 201 202 \
            --episodes 50 --devices 30

2) Static Allocation / AHP-TOPSIS (no training, no checkpoint needed):
     python per_class_eval.py --method static     --eval_seeds 200 201 202 --episodes 50 --devices 30
     python per_class_eval.py --method ahp_topsis  --eval_seeds 200 201 202 --episodes 50 --devices 30

OUTPUT
------
Prints a per-class table (QoS satisfaction %, mean latency ms, P99
latency ms) and writes per_class_<method>.csv to --out_dir.
Send the printed table (or the CSV) back and it will be inserted into
Section 5.4.1 / the per-class limitation paragraph in the manuscript.
"""

import argparse, os, csv, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env import APODRLEnv, CLASS_SPECS

CLASS_NAMES = ["Environmental", "Infrastructure", "Public Safety", "Transportation/URLLC"]


def build_agent(method, checkpoint_dir, seed, n_devices, device="cpu"):
    if method in ("static", "ahp_topsis"):
        from baselines import StaticAllocation, AHPTopsis
        return StaticAllocation() if method == "static" else AHPTopsis()
    elif method == "apo_drl":
        import torch
        from agent import D3QNAgent
        env_tmp = APODRLEnv(n_devices=n_devices, seed=0)
        agent = D3QNAgent(env_tmp.observation_space.shape[0], env_tmp.action_space.n,
                           torch.device(device), qos_aware=True, qos_lambda=1.5, buf_size=1)
        ckpt_path = os.path.join(checkpoint_dir, f"apo_drl_seed{seed}_n{n_devices}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run train.py --method apo_drl --seeds {seed} --devices {n_devices} "
                f"--episodes 3000 first (saves a checkpoint automatically), "
                f"or point --checkpoint_dir at the folder that already has it."
            )
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.online.load_state_dict(ckpt["online_state_dict"])
        agent.target.load_state_dict(ckpt["target_state_dict"])
        agent.online.eval(); agent.target.eval()
        agent._eps = 0.0  # frozen policy: fully greedy, epsilon = 0 exactly
        return agent
    else:
        raise ValueError(method)


def run_eval(method, agent, seed, episodes, n_devices, max_steps):
    """Run `episodes` episodes and return a list of (class_id, latency_ms, qos_sat) tuples,
    one per device-decision-step, labelled with the device class read from the
    environment's own state (no modification to env.py)."""
    env = APODRLEnv(n_devices=n_devices, seed=seed, max_steps=max_steps)
    rows = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 10000 + ep)
        done = False
        while not done:
            i_dev = env._cur                      # device about to be served
            cls = int(env._classes[i_dev])        # its class, read from public env state
            a = agent.act(s)
            s2, r, term, trunc, info = env.step(a)
            done = term or trunc
            rows.append((cls, info["latency_ms"], info["qos_violation"] == 0))
            s = s2
    return rows


def summarize(rows):
    by_class = {c: [] for c in range(4)}
    for cls, lat, ok in rows:
        by_class[cls].append((lat, ok))
    out = {}
    for c in range(4):
        data = by_class[c]
        if not data:
            continue
        lats = np.array([d[0] for d in data])
        oks  = np.array([d[1] for d in data])
        out[c] = {
            "n": len(data),
            "qos_sat_pct": 100.0 * oks.mean(),
            "mean_latency_ms": float(lats.mean()),
            "p99_latency_ms": float(np.percentile(lats, 99)),
            "max_latency_ms": float(lats.max()),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["apo_drl", "static", "ahp_topsis"])
    p.add_argument("--checkpoint_dir", default="../results/checkpoints")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 123],
                   help="training seeds (only used to locate APO-DRL checkpoints)")
    p.add_argument("--eval_seeds", nargs="+", type=int, default=[200, 201, 202])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--devices", type=int, default=30)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--out_dir", default="../results/per_class")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_summaries = []

    if args.method == "apo_drl":
        # one frozen agent per training seed, each evaluated on all eval_seeds,
        # mirroring the held-out protocol already used in Section 5.4.2
        for tr_seed in args.seeds:
            agent = build_agent(args.method, args.checkpoint_dir, tr_seed, args.devices)
            for ev_seed in args.eval_seeds:
                rows = run_eval(args.method, agent, ev_seed, args.episodes, args.devices, args.max_steps)
                all_summaries.append(summarize(rows))
    else:
        agent = build_agent(args.method, args.checkpoint_dir, None, args.devices)
        for ev_seed in args.eval_seeds:
            rows = run_eval(args.method, agent, ev_seed, args.episodes, args.devices, args.max_steps)
            all_summaries.append(summarize(rows))

    # average across the per-seed summaries
    print(f"\n=== Per-class results: {args.method} "
          f"(frozen policy, eval seeds {args.eval_seeds}) ===\n")
    print(f"{'Class':<28}{'QoS Sat (%)':>12}{'Mean Lat (ms)':>15}{'P99 Lat (ms)':>14}")
    csv_rows = []
    for c in range(4):
        vals = [s[c] for s in all_summaries if c in s]
        if not vals:
            continue
        qos = np.mean([v["qos_sat_pct"] for v in vals])
        mlat = np.mean([v["mean_latency_ms"] for v in vals])
        p99 = np.mean([v["p99_latency_ms"] for v in vals])
        label = f"Class {c} ({CLASS_NAMES[c]}, <= {CLASS_SPECS[c][1]} ms)"
        print(f"{label:<28}{qos:>12.1f}{mlat:>15.2f}{p99:>14.2f}")
        csv_rows.append({"class": c, "name": CLASS_NAMES[c], "max_latency_ms": CLASS_SPECS[c][1],
                          "qos_sat_pct": round(qos, 2), "mean_latency_ms": round(mlat, 2),
                          "p99_latency_ms": round(p99, 2)})

    out_path = os.path.join(args.out_dir, f"per_class_{args.method}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
