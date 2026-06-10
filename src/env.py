"""
env.py  —  APO-DRL 5G IoT Environment
=======================================
Multi-cell 5G IoT network with real physics:
  * 4 cells (1 macro + 3 small) on 2x2 km grid
  * 4 protocols: NB-IoT, LTE-M, LTE Cat-1, 5G NR
  * 4 device classes with real QoS requirements
  * 3GPP TR 38.901 Urban Macro path loss
  * Log-normal shadowing + Rayleigh fading
  * Poisson traffic per class
  * Inter-cell interference

Action space: protocol(4) x MCS(16) x power_level(8) = 512
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Protocol constants ──────────────────────────────────────────────────
PROTOCOLS     = ["NB-IoT", "LTE-M", "LTE Cat-1", "5G NR"]
BW_HZ         = {0: 180e3, 1: 1.4e6, 2: 20e6, 3: 100e6}
MAX_TX_DBM    = {0: 23,    1: 23,    2: 23,   3: 26}
NOISE_PSD_DBM = -174.0
CIRCUIT_MW    = 50.0
CARRIER_GHZ   = 3.5

# ── Device class specs: (min_rate_mbps, max_lat_ms, share) ─────────────
CLASS_SPECS = [
    (0.01, 10000, 0.40),   # Class 0 — Environmental
    (0.50,  1000, 0.30),   # Class 1 — Infrastructure
    (5.00,    10, 0.20),   # Class 2 — Public Safety
    (10.0,     1, 0.10),   # Class 3 — Transportation
]
POISSON_RATES = [0.01, 0.1, 1.0, 10.0]   # packets/step
PKT_BITS      = [200*8, 1000*8, 10000*8, 50000*8]

# ── Cell layout (metres) ────────────────────────────────────────────────
CELL_POS = np.array([[1000,1000],[400,400],[1600,400],[1000,1600]], dtype=float)

N_PROT = 4
N_MCS  = 16
N_POW  = 8
POWER_DBM = np.array([-10,-5,0,5,10,15,20,23], dtype=float)
ACTION_DIM = N_PROT * N_MCS * N_POW   # 512

def dbm2mw(x): return 10**(x/10)
def mw2dbm(x): return 10*np.log10(np.maximum(x, 1e-12))

def path_loss(dist_m):
    d = np.maximum(dist_m, 1.0)
    return 28.0 + 22*np.log10(d) + 20*np.log10(CARRIER_GHZ)


class APODRLEnv(gym.Env):
    """One step = one device served (sequential per-device decisions)."""

    def __init__(self, n_devices=1000, seed=0, max_steps=200):
        super().__init__()
        self.n_devices = n_devices
        self.max_steps = max_steps
        self._seed     = seed
        self.rng       = np.random.default_rng(seed)

        # State: [class_onehot(4), norm_pos(2), pending(1),
        #         sinr_per_cell(4), load_per_protocol(4), step_frac(1)] = 16
        self.observation_space = spaces.Box(-5, 5, shape=(16,), dtype=np.float32)
        self.action_space      = spaces.Discrete(ACTION_DIM)

        self._devices   = []
        self._protocols = None
        self._pending   = None
        self._prev_prot = None
        self._t         = 0
        self._cur       = 0
        self._log       = []

    # ── Reset ────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            self.rng   = np.random.default_rng(seed)
        self._t   = 0
        self._cur = 0
        self._log = []
        self._pending  = np.zeros(self.n_devices)
        # Spawn devices
        classes = []
        for c, (_, _, sh) in enumerate(CLASS_SPECS):
            classes += [c] * int(self.n_devices * sh)
        diff = self.n_devices - len(classes)
        classes += [0] * diff
        self.rng.shuffle(classes)

        self._classes  = np.array(classes)
        self._pos      = self.rng.uniform(0, 2000, size=(self.n_devices, 2))
        self._protocols= np.zeros(self.n_devices, dtype=int)
        self._mcs      = np.full(self.n_devices, 7)
        self._power    = np.full(self.n_devices, 10.0)
        self._pending  = np.zeros(self.n_devices)
        self._prev_prot= np.zeros(self.n_devices, dtype=int)
        self._switches = 0

        return self._obs(), {}

    # ── Observation ──────────────────────────────────────────────────────
    def _obs(self):
        i   = self._cur
        cls = self._classes[i]
        oh  = np.zeros(4, dtype=np.float32); oh[cls] = 1.0
        pos = (self._pos[i] / 2000 - 0.5).astype(np.float32)
        pnd = np.array([np.clip(self._pending[i]/1e6, 0, 5)], dtype=np.float32)

        sinr = np.zeros(4, dtype=np.float32)
        for c in range(4):
            d   = np.linalg.norm(self._pos[i] - CELL_POS[c])
            pl  = path_loss(d)
            rx  = self._power[i] - pl
            nz  = NOISE_PSD_DBM + 10*np.log10(BW_HZ[3])
            sinr[c] = np.clip((rx - nz)/30, -2, 2)

        load = np.zeros(4, dtype=np.float32)
        for p in self._protocols:
            load[p] += 1
        load /= max(self.n_devices, 1)

        sf = np.array([self._t / self.max_steps], dtype=np.float32)
        return np.concatenate([oh, pos, pnd, sinr, load, sf])

    # ── Decode action ────────────────────────────────────────────────────
    @staticmethod
    def decode(action):
        p = action // (N_MCS * N_POW)
        r = action  % (N_MCS * N_POW)
        m = r // N_POW
        l = r  % N_POW
        return p, m, l

    # ── Step ─────────────────────────────────────────────────────────────
    def step(self, action):
        i  = self._cur
        cls= self._classes[i]
        p, m, lv = self.decode(int(action))

        tx = float(min(POWER_DBM[lv], MAX_TX_DBM[p]))
        bw = BW_HZ[p]

        # Track switches
        if self._prev_prot[i] != p and self._t > 0:
            self._switches += 1
        self._prev_prot[i] = p
        self._protocols[i] = p
        self._mcs[i]       = m
        self._power[i]     = tx

        # Path loss + fading
        cell = int(np.argmin(np.linalg.norm(CELL_POS - self._pos[i], axis=1)))
        dist = np.linalg.norm(self._pos[i] - CELL_POS[cell])
        pl   = path_loss(dist) + self.rng.normal(0, 8)      # shadowing
        fade = -np.log(max(self.rng.random(), 1e-9))        # Rayleigh

        rx_mw = dbm2mw(tx - pl) * fade

        # Interference (simplified: sample one device per other cell)
        intf_mw = 0.0
        for oc in range(4):
            if oc == cell: continue
            others = np.where(
                (self._protocols == p) & (np.arange(self.n_devices) != i)
            )[0]
            if len(others):
                j    = self.rng.choice(others)
                idst = np.linalg.norm(self._pos[i] - CELL_POS[oc])
                ipl  = path_loss(idst)
                intf_mw += dbm2mw(self._power[j] - ipl) * 0.3

        noise_mw = dbm2mw(NOISE_PSD_DBM + 10*np.log10(bw))
        sinr     = rx_mw / (intf_mw + noise_mw + 1e-15)

        # Rate (Shannon + MCS cap)
        mcs_eff  = 0.1 + (m/15)*5.4
        shannon  = bw * np.log2(1 + sinr)
        rate_bps = min(mcs_eff * bw, shannon)
        rate_mbps= rate_bps / 1e6

        # Traffic
        pkts  = self.rng.poisson(POISSON_RATES[cls])
        bits  = pkts * PKT_BITS[cls]
        dt    = 0.01
        served= min(self._pending[i] + bits, rate_bps * dt)
        self._pending[i] = max(0, self._pending[i] + bits - served)

        # Latency
        tx_ms = (bits / max(rate_bps, 1)) * 1000
        q_ms  = (self._pending[i] / max(rate_bps, 1)) * 1000
        lat_ms = min(tx_ms + q_ms + 1.0, CLASS_SPECS[cls][1] * 10)

        # Energy
        eng_mj = (dbm2mw(tx) + CIRCUIT_MW) * dt

        # QoS
        spec   = CLASS_SPECS[cls]
        qos_ok = (rate_mbps >= spec[0]) and (lat_ms <= spec[1])
        pkt_loss = (bits > 0) and (served < 0.5*bits)

        # Reward
        u_r = np.clip(rate_mbps / max(spec[0]*2, 0.1), 0, 1)
        u_l = np.clip(1 - lat_ms / max(spec[1], 1), 0, 1)
        u_e = np.clip(1 - dbm2mw(tx)/dbm2mw(26), 0, 1)
        u_q = 1.0 if qos_ok else 0.0
        pen = 0.1 if (self._prev_prot[i] != p) else 0.0
        reward = 0.3*u_r + 0.3*u_l + 0.2*u_e + 0.2*u_q - pen

        info = {
            "qos_violation":   int(not qos_ok),
            "throughput_mbps": rate_mbps,
            "latency_ms":      lat_ms,
            "energy_mj":       eng_mj,
            "packet_loss":     int(pkt_loss),
        }
        self._log.append({**info, "qos_sat": float(qos_ok)})

        # Advance
        self._cur = (self._cur + 1) % self.n_devices
        if self._cur == 0:
            self._t += 1
        done = self._t >= self.max_steps

        return self._obs(), float(reward), done, False, info

    # ── Episode stats ────────────────────────────────────────────────────
    def stats(self):
        if not self._log: return {}
        n = len(self._log)
        return {
            "throughput_mbps": float(np.mean([r["throughput_mbps"] for r in self._log])),
            "latency_ms":      float(np.mean([r["latency_ms"]      for r in self._log])),
            "energy_mj":       float(np.mean([r["energy_mj"]       for r in self._log])),
            "qos_sat_pct":     float(100*np.mean([r["qos_sat"]     for r in self._log])),
            "packet_loss_pct": float(100*np.mean([r["packet_loss"] for r in self._log])),
            "switch_pct":      float(100*self._switches / max(n,1)),
        }
