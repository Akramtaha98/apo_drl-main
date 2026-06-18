"""
evaluate.py  —  Held-out evaluation with frozen APO-DRL policy
==============================================================
Usage (after training with train.py --save_checkpoint):
  python evaluate.py --checkpoint ../results/checkpoints/apo_drl_seed0_n30.pt \
                     --eval_seeds 200 201 202 --episodes 100 --devices 30

This script:
  1. Loads a frozen D3QN checkpoint (no gradient updates)
  2. Runs N evaluation episodes with epsilon=0 (fully greedy)
  3. Reports throughput, QoS satisfaction, latency, energy, packet loss
  4. Saves per-episode CSV and summary statistics

This implements the train/freeze/evaluate protocol for Comment 23.
"""

import argparse, os, csv, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from env import APODRLEnv
from agent import D3QNAgent


def load_agent(checkpoint_path, state_dim, action_dim, device):
    agent = D3QNAgent(state_dim, action_dim, device, qos_aware=True, qos_lambda=1.5, buf_size=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.online.load_state_dict(ckpt["online_state_dict"])
    agent.target.load_state_dict(ckpt["target_state_dict"])
    agent.online.eval()
    agent.target.eval()
    agent._eps = 0.0   # force greedy
    return agent


def eval_seed(agent, seed, episodes, n_devices, max_steps):
    env = APODRLEnv(n_devices=n_devices, seed=seed, max_steps=max_steps)
    rows = []
    with torch.no_grad():
        for ep in range(episodes):
            s, _ = env.reset(seed=seed * 10000 + ep)
            done = False; ep_r = 0
            while not done:
                a = agent.act(s)
                s2, r, term, trunc, info = env.step(a)
                done = term or trunc
                s = s2; ep_r += r
            st = env.stats()
            rows.append({
                "episode": ep, "seed": seed, "reward": round(ep_r, 4),
                "throughput_mbps":  round(st.get("throughput_mbps", 0), 4),
                "latency_ms":       round(st.get("latency_ms", 0), 2),
                "energy_mj":        round(st.get("energy_mj", 0), 6),
                "qos_sat_pct":      round(st.get("qos_sat_pct", 0), 2),
                "packet_loss_pct":  round(st.get("packet_loss_pct", 0), 2),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_seeds", nargs="+", type=int, default=[200, 201, 202])
    p.add_argument("--episodes",   type=int, default=100)
    p.add_argument("--devices",    type=int, default=30)
    p.add_argument("--max_steps",  type=int, default=200)
    p.add_argument("--out_dir",    type=str, default="../results/eval")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_tmp = APODRLEnv(n_devices=args.devices, seed=0)
    agent = load_agent(args.checkpoint, env_tmp.observation_space.shape[0], env_tmp.action_space.n, device)
    print("Checkpoint loaded. Evaluating with epsilon=0 (frozen policy)...\n")

    all_rows = []
    for seed in args.eval_seeds:
        rows = eval_seed(agent, seed, args.episodes, args.devices, args.max_steps)
        all_rows.extend(rows)
        thr = np.mean([r["throughput_mbps"] for r in rows])
        qos = np.mean([r["qos_sat_pct"]      for r in rows])
        print(f"  seed={seed}: {thr:.2f} Mbps, {qos:.2f}% QoS")

    thrs = [r["throughput_mbps"] for r in all_rows]
    qoss = [r["qos_sat_pct"]      for r in all_rows]
    print(f"\nAggregate: {np.mean(thrs):.2f}+/-{np.std(thrs):.2f} Mbps, "
          f"{np.mean(qoss):.2f}+/-{np.std(qoss):.2f}% QoS")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "apo_drl_heldout_eval.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
