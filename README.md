# APO-DRL: Adaptive Protocol Optimization with Deep Reinforcement Learning for Heterogeneous IoT Networks in 5G-Enabled Smart Cities

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code repository for the paper:

> **Deep Reinforcement Learning-Based Adaptive Protocol Optimization for Heterogeneous IoT Networks in 5G-Enabled Smart Cities**
> *MDPI IoT Journal* — [doi link]

---

## Overview

This repository implements **APO-DRL**, a Dueling Double DQN (D3QN) agent with a novel **QoS-Aware Prioritized Experience Replay (QA-PER)** mechanism for intelligent radio resource management in 5G IoT networks. The agent jointly optimizes protocol selection, modulation and coding scheme (MCS), and transmission power for four heterogeneous IoT device classes.

---

## Project Structure

```
├── src/
│   ├── env.py              # 5G IoT simulation environment (Gymnasium)
│   ├── agent.py            # Dueling D3QN agent + QA-PER replay buffer
│   ├── baselines.py        # Comparison baselines (Static, Random, AHP-TOPSIS, DQN, QoS-Greedy)
│   ├── train.py            # Training loop for all methods (saves checkpoints)
│   ├── per_class_eval.py   # Per-class QoS / P99-latency evaluation (frozen policy)
│   ├── plot.py             # Figure generation from training results
│   ├── get_stats_n30.py    # Tail-window statistics for Table 4 (N=30)
│   └── extract_tables.py   # Paper table generation (Tables 5 & ablation)
├── results/                # CSV training logs (auto-created during training)
│   └── checkpoints/        # Saved model weights (auto-created by train.py)
└── figures/                # Generated plots (auto-created by plot.py)
```

---

## Environment (`env.py`)

### Network Topology

- **4 cells**: 1 macro cell + 3 small cells on a 2×2 km grid
- **4 protocols**: NB-IoT, LTE-M, LTE Cat-1, 5G NR
- **Channel model**: 3GPP TR 38.901 Urban Macro path loss + log-normal shadowing (σ=8 dB) + Rayleigh fading
- **Carrier frequency**: 3.5 GHz

### Device Classes

| Class | Type | Min Rate | Max Latency | Traffic Share |
|-------|------|----------|-------------|---------------|
| 0 | Environmental Monitoring | 0.01 Mbps | 10,000 ms | 40% |
| 1 | Infrastructure | 0.50 Mbps | 1,000 ms | 30% |
| 2 | Public Safety | 5.00 Mbps | 10 ms | 20% |
| 3 | Transportation / URLLC | 10.0 Mbps | 1 ms | 10% |

### State Space (16-dimensional)

```
[class_onehot(4), normalized_position(2), pending_bits(1),
 per_cell_SINR(4), per_protocol_load(4), step_fraction(1)]
```

### Action Space

`protocol(4) × MCS(16) × power_level(8) = 512 discrete actions`

### Reward Function

```
R = α·U_rate + β·U_latency + δ·U_energy + η·U_qos − switch_penalty
```

Where α=0.3, β=0.3, δ=0.2, η=0.2, and each U is a normalized utility in [0, 1].

---

## Agent (`agent.py`)

### Architecture: Dueling Double DQN

```
Input (|S|=16)
    └── Shared FC: 512 → 256 → 128  (ReLU + BatchNorm)
            ├── Value Stream:     FC(128) → V(s)      [scalar]
            └── Advantage Stream: FC(128) → A(s,a)   [|A|=512]
                        Q(s,a) = V(s) + A(s,a) − mean(A)
```

### Novel Contribution: QoS-Aware PER (QA-PER)

Standard PER priorities are augmented by QoS violation flags:

```
p_i = (|δ_i| + ε) × (1 + λ · v_i)
```

Where:
- `δ_i` = TD error
- `v_i` = 1 if the transition caused a QoS violation, else 0
- `λ` = QoS penalty weight (default: 1.5)

This forces the agent to revisit QoS-violating experiences more frequently, accelerating convergence on hard constraints. The exponent α (PER prioritization strength) is applied at **sampling time** (`P(i) = p_i^α / Σ p_k^α`), not when the priority is computed.

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Discount factor (γ) | 0.99 |
| Replay buffer size | 100,000 |
| Batch size | 128 |
| Target network update | every 500 steps |
| ε-greedy (start → end) | 1.0 → 0.05 over 50,000 steps |
| PER α | 0.6 |
| PER β₀ | 0.4 |
| QoS λ | 1.5 |

---

## Baselines (`baselines.py`)

| Method | Description |
|--------|-------------|
| `static` | Rule-based: maps device class to fixed protocol/MCS/power |
| `random` | Uniformly random action selection |
| `ahp_topsis` | Multi-criteria decision making using AHP preference matrices |
| `qos_greedy` | Greedy heuristic selecting the action with highest predicted QoS utility |
| `dqn` | Standard DQN with uniform experience replay |
| `d3qn_per` | Dueling D3QN with standard PER (no QoS-aware weighting) |

---

## Training (`train.py`)

### Quick Start

```bash
# Train APO-DRL (3 seeds, 3000 episodes, 30 devices)
python src/train.py --method apo_drl --seeds 0 42 123 --episodes 3000 --devices 30

# Train all baselines
python src/train.py --method dqn        --seeds 0 42 123 --episodes 3000 --devices 30
python src/train.py --method d3qn_per   --seeds 0 42 123 --episodes 3000 --devices 30
python src/train.py --method static     --seeds 0 42 123 --episodes 100  --devices 30
python src/train.py --method ahp_topsis --seeds 0 42 123 --episodes 100  --devices 30
python src/train.py --method random     --seeds 0 42 123 --episodes 100  --devices 30
python src/train.py --method qos_greedy --seeds 0 42 123 --episodes 100  --devices 30

# Scalability study (Table 5)
for N in 50 100 200; do
  python src/train.py --method apo_drl    --seeds 0 42 123 --episodes 3000 --devices $N
  python src/train.py --method static     --seeds 0 42 123 --episodes 100  --devices $N
  python src/train.py --method ahp_topsis --seeds 0 42 123 --episodes 100  --devices $N
done
```

### Output

Each run saves a CSV to `results/{method}_seed{seed}_n{devices}.csv` with columns:

```
episode, reward, loss, mean_q, epsilon, throughput_mbps, latency_ms,
energy_mj, qos_sat_pct, packet_loss_pct, switch_pct
```

A model checkpoint is also saved to `results/checkpoints/{method}_seed{seed}_n{devices}.pt` after training completes.

### Device Selection

Training auto-selects the best available compute device:

- **CUDA** (NVIDIA GPU) → highest priority
- **MPS** (Apple Silicon) → if CUDA unavailable
- **CPU** → fallback

Override with `--device cpu` / `--device mps` / `--device cuda`.

---

## Per-Class Evaluation (`per_class_eval.py`)

Computes per-class QoS satisfaction and P99 latency using a **frozen policy** (ε=0) evaluated on held-out seeds {200, 201, 202}, consistent with the cross-seed generalization protocol.

```bash
# APO-DRL per-class breakdown (requires trained checkpoint)
python src/per_class_eval.py --method apo_drl \
    --checkpoint_dir results/checkpoints \
    --seeds 0 42 123 --eval_seeds 200 201 202 \
    --episodes 50 --devices 30

# Static Allocation and AHP-TOPSIS (no checkpoint needed)
python src/per_class_eval.py --method static     --eval_seeds 200 201 202 --episodes 50 --devices 30
python src/per_class_eval.py --method ahp_topsis --eval_seeds 200 201 202 --episodes 50 --devices 30
```

Output: QoS satisfaction (%), mean latency (ms), P99 latency (ms) per device class, saved to `results/per_class/per_class_{method}.csv`.

---

## Figures (`plot.py`)

```bash
python src/plot.py
```

Produces in `figures/`:

| File | Content |
|------|---------|
| `fig3_convergence.png` | Episode reward curves (mean ±1 std, all methods) |
| `fig3b_loss.png` | TD loss curves (log scale) |
| `fig3c_epsilon_q.png` | Epsilon decay + Q-value evolution |
| `fig4_bars.png` | Performance bar chart at N=30 |
| `fig5_scalability.png` | Throughput & QoS vs. device density (N=30–200) |

---

## Paper Tables

### Table 4 — Main Comparison (N=30)

```bash
python src/get_stats_n30.py
```

Computes mean ± std over the last 300 of 3,000 episodes, across seeds {0, 42, 123}, for all seven methods at N=30 devices.

### Table 5 — Scalability

```bash
python src/extract_tables.py
```

Prints scalability results across N = 30, 50, 100, 200 devices, and APO-DRL vs. Static improvement percentages.

---

## Dependencies

```bash
pip install numpy torch gymnasium pandas matplotlib
```

Python ≥ 3.9 required. Tested on Python 3.13 (macOS, Apple MPS) and Python 3.10 (Linux, CUDA).

---

## Citation

If you use this code, please cite:

```bibtex
@article{taha2025apodrl,
  title     = {Deep Reinforcement Learning-Based Adaptive Protocol Optimization
               for Heterogeneous IoT Networks in 5G-Enabled Smart Cities},
  author    = {Taha, Akram and others},
  journal   = {IoT},
  publisher = {MDPI},
  year      = {2025},
  doi       = {}
}
```
