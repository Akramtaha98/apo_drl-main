"""
baselines.py  —  Non-DRL and simple-DRL baselines
MPS-compatible: all tensors forced to float32 at creation time.
"""
import numpy as np
import torch
import torch.nn as nn

N_PROT = 4; N_MCS = 16; N_POW = 8
ACTION_DIM = N_PROT * N_MCS * N_POW


class StaticAllocation:
    name = "Static"
    def act(self, s):
        cls = int(np.argmax(s[:4]))
        p, m, l = cls, 7, 5
        return p*(N_MCS*N_POW) + m*N_POW + l
    def observe(self, *a, **k): pass
    def learn(self): return None
    def eps(self): return 0.0


class RandomSelection:
    name = "Random"
    def act(self, s): return np.random.randint(ACTION_DIM)
    def observe(self, *a, **k): pass
    def learn(self): return None
    def eps(self): return 1.0


class AHPTopsis:
    name = "AHP-TOPSIS"
    PREF = [[0,1,2,3],[1,0,2,3],[2,3,1,0],[3,2,1,0]]
    def act(self, s):
        cls  = int(np.argmax(s[:4]))
        load = s[10:14]
        p    = next((x for x in self.PREF[cls] if load[x] < 0.6), self.PREF[cls][0])
        m    = 10 if cls >= 2 else 5
        l    = 6  if cls >= 2 else 4
        return p*(N_MCS*N_POW) + m*N_POW + l
    def observe(self, *a, **k): pass
    def learn(self): return None
    def eps(self): return 0.0


class VanillaDQN:
    name = "DQN"

    def __init__(self, sd, ad, device, lr=1e-4, gamma=0.99,
                 buf=100_000, batch=128, t_upd=500,
                 eps_start=1.0, eps_end=0.05, eps_steps=50_000):
        self.device    = device
        self.ad        = ad
        self.gamma     = gamma
        self.batch     = batch
        self.t_upd     = t_upd
        self.eps_end   = eps_end
        self.eps_steps = eps_steps
        self.eps_start = eps_start
        self.step_n    = 0

        def make_net():
            return nn.Sequential(
                nn.Linear(sd, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, ad),
            ).to(device)

        self.net = make_net()
        self.tgt = make_net()
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        # numpy circular buffer — faster sampling than deque
        self._buf_s  = np.zeros((buf, sd), dtype=np.float32)
        self._buf_a  = np.zeros(buf, dtype=np.int64)
        self._buf_r  = np.zeros(buf, dtype=np.float32)
        self._buf_s2 = np.zeros((buf, sd), dtype=np.float32)
        self._buf_d  = np.zeros(buf, dtype=np.float32)
        self._buf_cap = buf
        self._buf_ptr = 0
        self._buf_len = 0

    def eps(self):
        return self.eps_start + (self.eps_end - self.eps_start) * min(1, self.step_n / self.eps_steps)

    def _obs_tensor(self, s):
        """Convert numpy state to float32 tensor — MPS requires float32."""
        return torch.tensor(np.array(s, dtype=np.float32), device=self.device)

    def act(self, s):
        if np.random.rand() < self.eps():
            return np.random.randint(self.ad)
        with torch.no_grad():
            return int(self.net(self._obs_tensor(s[None])).argmax())

    def greedy(self, s):
        with torch.no_grad():
            return int(self.net(self._obs_tensor(s[None])).argmax())

    def observe(self, s, a, r, s2, done, qv=0):
        i = self._buf_ptr
        self._buf_s[i]  = s
        self._buf_a[i]  = int(a)
        self._buf_r[i]  = float(r)
        self._buf_s2[i] = s2
        self._buf_d[i]  = float(done)
        self._buf_ptr = (i + 1) % self._buf_cap
        self._buf_len = min(self._buf_len + 1, self._buf_cap)
        self.step_n += 1

    def learn(self):
        if self._buf_len < max(1000, self.batch):
            return None

        idx = np.random.randint(0, self._buf_len, size=self.batch)

        s_t  = torch.as_tensor(self._buf_s[idx],  device=self.device)
        a_t  = torch.as_tensor(self._buf_a[idx],  device=self.device)
        r_t  = torch.as_tensor(self._buf_r[idx],  device=self.device)
        s2_t = torch.as_tensor(self._buf_s2[idx], device=self.device)
        d_t  = torch.as_tensor(self._buf_d[idx],  device=self.device)

        with torch.no_grad():
            nq  = self.tgt(s2_t).max(1).values
            tgt = r_t + self.gamma * (1.0 - d_t) * nq

        cur  = self.net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        loss = ((cur - tgt) ** 2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.step_n % self.t_upd == 0:
            self.tgt.load_state_dict(self.net.state_dict())

        return {
            "loss":   float(loss.item()),
            "mean_q": float(cur.mean().item()),
            "eps":    self.eps(),
        }
