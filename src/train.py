"""
train.py  —  Training loop for all methods
==========================================
Usage:
  python train.py --method apo_drl  --seeds 0 42 123 256 999 --episodes 3000 --devices 1000
  python train.py --method dqn      --seeds 0 42 123 256 999 --episodes 3000 --devices 1000
  python train.py --method d3qn_per --seeds 0 42 123 256 999 --episodes 3000 --devices 1000
  python train.py --method static   --seeds 0 42 123 256 999 --episodes 100  --devices 1000
  python train.py --method ahp_topsis --seeds 0 42 123 256 999 --episodes 100 --devices 1000
  python train.py --method random   --seeds 0 42 123 256 999 --episodes 100  --devices 1000

For scalability study (Table 7):
  python3 train.py --method apo_drl --seeds 123 --episodes 3000 --devices 30
  python train.py --method apo_drl --seeds 0 42 123 --episodes 3000 --devices 1500
  python train.py --method apo_drl --seeds 0 42 123 --episodes 3000 --devices 2000
  (repeat for static and ahp_topsis at each scale)
"""

import argparse, os, csv, time, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from env import APODRLEnv, ACTION_DIM
from agent import D3QNAgent
from baselines import StaticAllocation, RandomSelection, AHPTopsis, VanillaDQN


def build(name, state_dim, action_dim, device):
    name = name.lower()
    if name == "apo_drl":
        a = D3QNAgent(state_dim, action_dim, device,
                      qos_aware=True, qos_lambda=1.5,
                      buf_size=100_000)   # match DQN buffer size for fair comparison
        a.name = "APO-DRL"; return a, True
    if name == "d3qn_per":
        a = D3QNAgent(state_dim, action_dim, device,
                      qos_aware=False,
                      buf_size=100_000)   # match DQN/APO-DRL buffer size for fair comparison
        a.name = "D3QN+PER"; return a, True
    if name == "dqn":
        a = VanillaDQN(state_dim, action_dim, device)
        return a, True
    if name == "static":    return StaticAllocation(), False
    if name == "random":    return RandomSelection(),  False
    if name == "ahp_topsis":return AHPTopsis(),        False
    raise ValueError(f"Unknown method: {name}")


def run_seed(method_name, seed, episodes, n_devices, max_steps, out_dir, device):
    torch.manual_seed(seed); np.random.seed(seed)
    env   = APODRLEnv(n_devices=n_devices, seed=seed, max_steps=max_steps)
    s, _  = env.reset(seed=seed)
    sd    = env.observation_space.shape[0]
    ad    = env.action_space.n
    agent, is_drl = build(method_name, sd, ad, device)

    rows = []; t0 = time.time(); global_step = 0
    for ep in range(episodes):
        s, _ = env.reset(seed=seed*10000+ep)
        done = False
        ep_r = 0; ep_loss = []; ep_q = []

        while not done:
            a  = agent.act(s)
            s2, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.observe(s, a, r, s2, done, info.get("qos_violation",0))
            global_step += 1
            if is_drl and global_step % 4 == 0:
                lg = agent.learn()
                if lg:
                    ep_loss.append(lg["loss"])
                    ep_q.append(lg.get("mean_q", 0))
            s = s2; ep_r += r

        st = env.stats()
        rows.append({
            "episode":          ep,
            "reward":           round(ep_r, 4),
            "loss":             round(float(np.mean(ep_loss)) if ep_loss else 0, 6),
            "mean_q":           round(float(np.mean(ep_q))    if ep_q    else 0, 4),
            "epsilon":          round(agent.eps() if hasattr(agent,"eps") and callable(agent.eps) else 0, 4),
            "throughput_mbps":  round(st.get("throughput_mbps", 0), 4),
            "latency_ms":       round(st.get("latency_ms", 0), 2),
            "energy_mj":        round(st.get("energy_mj", 0), 6),
            "qos_sat_pct":      round(st.get("qos_sat_pct", 0), 2),
            "packet_loss_pct":  round(st.get("packet_loss_pct", 0), 2),
            "switch_pct":       round(st.get("switch_pct", 0), 2),
        })

        if (ep+1) % max(1, episodes//10) == 0 or ep == 0:
            elapsed = time.time()-t0
            print(f"  [{method_name} seed={seed}] ep {ep+1}/{episodes} "
                  f"r={ep_r:.1f} qos={st.get('qos_sat_pct',0):.1f}% "
                  f"thr={st.get('throughput_mbps',0):.2f}Mbps "
                  f"lat={st.get('latency_ms',0):.0f}ms "
                  f"elapsed={elapsed:.0f}s")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{method_name}_seed{seed}_n{n_devices}.csv")
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved -> {path}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method",    required=True,
                   choices=["apo_drl","d3qn_per","dqn","static","random","ahp_topsis"])
    p.add_argument("--seeds",     nargs="+", type=int, default=[0])
    p.add_argument("--episodes",  type=int,  default=3000)
    p.add_argument("--devices",   type=int,  default=1000)
    p.add_argument("--max_steps", type=int,  default=200)
    p.add_argument("--out_dir",   type=str,  default="../results")
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")
    print(f"\n{'='*60}")
    print(f"  Method: {args.method}  |  Devices: {args.devices}")
    print(f"  Episodes: {args.episodes}  |  Seeds: {args.seeds}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    for seed in args.seeds:
        run_seed(args.method, seed, args.episodes,
                 args.devices, args.max_steps, args.out_dir, device)


if __name__ == "__main__":
    main()
