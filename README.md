# APO-DRL: Adaptive Protocol Optimization with Deep Reinforcement Learning for 5G IoT Networks

## Overview

This project implements a Dueling Double DQN (D3QN) agent with a novel QoS-Aware Prioritized Experience Replay (QA-PER) mechanism for intelligent radio resource management in multi-cell 5G IoT networks. The agent learns to jointly optimize protocol selection, modulation and coding scheme (MCS), and transmission power for heterogeneous IoT device classes.

## Project Structure

```
├── src/
│   ├── env.py              # 5G IoT simulation environment (Gymnasium)
│   ├── agent.py            # Dueling D3QN agent + QA-PER replay buffer
│   ├── baselines.py        # Comparison baselines (Static, Random, AHP-TOPSIS, DQN)
│   ├── train.py            # Training loop for all methods
│   ├── plot.py             # Figure generation from training results
│   ├── get_stats_n30.py    # Tail-window (last 300/3000 episodes) stats for Table 4 (N=30)
│   ├── get_stats.py        # General results-statistics helper
│   └── extract_tables.py   # Paper table generation (Tables 5 & 6)
├── results/                # CSV training logs (auto-created during training)
└── figures/                # Generated plots (auto-created by plot.py)
```

## Environment (env.py)

### Network Topology
- 4 cells: 1 macro cell + 3 small cells on a 2×2 km grid
- 4 protocols: NB-IoT, LTE-M, LTE Cat-1, 5G NR
- Channel model: 3GPP TR 38.901 Urban Macro path loss + log-normal shadowing (σ=8 dB) + Rayleigh fading
- Carrier frequency: 3.5 GHz

### Device Classes

| Class | Type | Min Rate | Max Latency | Traffic Share |
|---|---|---|---|---|
| 0 | Environmental Monitoring | 0.01 Mbps | 10,000 ms | 40% |
| 1 | Infrastructure | 0.50 Mbps | 1,000 ms | 30% |
| 2 | Public Safety | 5.00 Mbps | 10 ms | 20% |
| 3 | Transportation | 10.0 Mbps | 1 ms | 10% |

### State Space (16-dimensional)
```
[class_onehot(4), normalized_position(2), pending_bits(1),
 per_cell_SINR(4), per_protocol_load(4), step_fraction(1)]
```

### Action Space
protocol(4) × MCS(16) × power_level(8) = 512 discrete actions

### Reward Function
```
R = 0.3·U_rate + 0.3·U_latency + 0.2·U_energy + 0.2·U_qos − 0.1·switch_penalty
```
Where each U is a normalized utility in [0, 1].

## Agent (agent.py)

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
- δ_i = TD error
- v_i = 1 if the transition caused a QoS violation, else 0
- λ = QoS penalty weight (default: 1.5)

This forces the agent to revisit QoS-violating experiences more frequently, accelerating convergence on hard constraints. The exponent α (PER prioritization strength) is applied at **sampling time** (P(i) = p_i^α / Σ p_k^α), not when the priority itself is computed.

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Discount factor (γ) | 0.99 |
| Replay buffer size | 100,000 (all methods, including D3QN+PER baseline) |
| Batch size | 128 |
| Target network update | every 500 steps |
| ε-greedy (start → end) | 1.0 → 0.05 over 50,000 steps |
| PER α | 0.6 |
| PER β₀ | 0.4 |
| QoS λ | 1.5 |

## Baselines (baselines.py)

| Method | Description |
|---|---|
| Static | Rule-based: maps device class to fixed protocol/MCS/power |
| Random | Uniformly random action selection |
| AHP-TOPSIS | Multi-criteria decision making using preference matrices |
| DQN | Standard DQN with uniform experience replay |

## Training (train.py)

### Quick Start

```bash
# Train the full APO-DRL agent (3 seeds, 3000 episodes, 30 devices)
python src/train.py --method apo_drl --seeds 0 42 123 --episodes 3000 --devices 30

# Train all baselines
python src/train.py --method dqn        --seeds 0 42 123 --episodes 3000 --devices 30
python src/train.py --method d3qn_per   --seeds 0 42 123 --episodes 3000 --devices 30
python src/train.py --method static     --seeds 0 42 123 --episodes 100  --devices 30
python src/train.py --method ahp_topsis --seeds 0 42 123 --episodes 100  --devices 30
python src/train.py --method random     --seeds 0 42 123 --episodes 100  --devices 30

# Scalability study (Table 5)
# APO-DRL across N = 50, 100, 150, 200 devices
for N in 50 100 150 200; do
  python src/train.py --method apo_drl --seeds 0 42 123 --episodes 3000 --devices $N
done

# Static / AHP-TOPSIS baselines across N = 500-2000 devices
for N in 500 1000 1500 2000; do
  python src/train.py --method static     --seeds 0 42 123 --episodes 100 --devices $N
  python src/train.py --method ahp_topsis --seeds 0 42 123 --episodes 100 --devices $N
done
```

### Output

Each run saves a CSV to `results/{method}_seed{seed}_n{devices}.csv` with columns:
```
episode, reward, loss, mean_q, epsilon, throughput_mbps, latency_ms, energy_mj, qos_sat_pct, packet_loss_pct, switch_pct
```

### Device Selection

Training auto-selects the best available device:
1. CUDA (NVIDIA GPU) → highest priority
2. MPS (Apple Silicon) → if CUDA unavailable
3. CPU → fallback

## Figures (plot.py)

Run after training to generate all paper figures:

```bash
python src/plot.py
```

Produces in `figures/`:

| File | Content |
|---|---|
| fig3_convergence.png | Episode reward curves (mean ±1 std, all methods) |
| fig4_bars.png | Performance bar chart at N=30 |
| fig5_scalability.png | Throughput & QoS vs. device density (APO-DRL: N=50-200; Static/AHP-TOPSIS: N=500-2000) |

## Paper Tables

### Table 4 — Main Comparison (N=30)

```bash
python src/get_stats_n30.py
```

Computes mean ± std over the last 300 of 3000 episodes, across seeds {0, 42, 123}, for each method at N=30 devices.

### Tables 5 & 6 — Scalability and Ablation (extract_tables.py)

```bash
python src/extract_tables.py
```

Prints:
- Table 5: Scalability across device densities (APO-DRL: N=50-200; Static/AHP-TOPSIS: N=500-2000)
- Table 6: Ablation study (e.g., D3QN+PER vs. APO-DRL)
- Improvement % of APO-DRL vs. Static (for Abstract/Conclusion)

⚠️ Always use these computed numbers in the paper — do not use manually estimated values.

## Dependencies

```bash
pip install numpy torch gymnasium pandas matplotlib
```

Python ≥ 3.9 required. Tested on Python 3.13 (macOS MPS) and Python 3.10 (CUDA).
