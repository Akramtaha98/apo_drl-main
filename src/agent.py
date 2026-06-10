"""
agent.py  —  Dueling Double DQN with QoS-Aware PER
====================================================
Novel contribution: QoS-Aware Prioritized Experience Replay (QA-PER)
    p_i = (|delta_i| + eps) * (1 + lambda * v_i)
where v_i=1 when transition caused a QoS violation.
"""

import numpy as np
import torch
import torch.nn as nn

# ── Neural network ────────────────────────────────────────────────────────
class DuelingNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256, 256, 128)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], hidden[2]), nn.ReLU(),
        )
        self.val = nn.Linear(hidden[2], 1)
        self.adv = nn.Linear(hidden[2], action_dim)

    def forward(self, x):
        h = self.shared(x)
        v = self.val(h)
        a = self.adv(h)
        return v + (a - a.mean(dim=1, keepdim=True))


# ── Sum-tree for PER ──────────────────────────────────────────────────────
class SumTree:
    def __init__(self, cap):
        self.cap  = cap
        self.tree = np.zeros(2*cap - 1, dtype=np.float64)
        self.ptr  = 0
        self.size = 0

    def _prop(self, idx, delta):
        p = (idx - 1) // 2
        self.tree[p] += delta
        if p: self._prop(p, delta)

    def update(self, idx, p):
        ti = idx + self.cap - 1
        self.tree[ti] += p - self.tree[ti]; self._prop(ti, p - self.tree[ti+0] - (p - self.tree[ti]))

    def set(self, idx, p):
        ti = idx + self.cap - 1
        d  = p - self.tree[ti]
        self.tree[ti] = p
        i = ti
        while i:
            i = (i-1)//2
            self.tree[i] += d

    def add(self, p):
        self.set(self.ptr, p)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def get(self, s):
        i = 0
        while True:
            l, r = 2*i+1, 2*i+2
            if l >= len(self.tree): break
            if s <= self.tree[l]:
                i = l
            else:
                s -= self.tree[l]
                i = r
        di = i - (self.cap - 1)
        return di, self.tree[i]

    @property
    def total(self): return self.tree[0]


# ── Replay buffer ─────────────────────────────────────────────────────────
class PERBuffer:
    def __init__(self, cap, state_dim,
                 alpha=0.6, beta0=0.4, beta_frames=100_000,
                 qos_aware=False, qos_lambda=1.5):
        self.cap         = cap
        self.alpha       = alpha
        self.beta0       = beta0
        self.beta_frames = beta_frames
        self.qos_aware   = qos_aware
        self.qos_lambda  = qos_lambda
        self.eps         = 1e-5
        self.frame       = 0
        self.max_p       = 1.0
        self.tree        = SumTree(cap)

        self.s  = np.zeros((cap, state_dim), np.float32)
        self.a  = np.zeros(cap, np.int64)
        self.r  = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, state_dim), np.float32)
        self.d  = np.zeros(cap, np.float32)
        self.v  = np.zeros(cap, np.float32)   # QoS violation flag
        self.wi = 0

    def add(self, s, a, r, s2, done, qos_viol=0):
        self.s[self.wi]  = s
        self.a[self.wi]  = a
        self.r[self.wi]  = r
        self.s2[self.wi] = s2
        self.d[self.wi]  = float(done)
        self.v[self.wi]  = float(qos_viol)
        p = self.max_p
        if self.qos_aware and qos_viol:
            p *= (1.0 + self.qos_lambda)
        self.tree.add(p ** self.alpha)
        self.wi = (self.wi + 1) % self.cap

    def sample(self, k, device):
        self.frame += 1
        beta = min(1.0, self.beta0 + (1-self.beta0)*self.frame/self.beta_frames)
        idxs, prios = np.zeros(k, np.int64), np.zeros(k)
        seg = self.tree.total / k
        for i in range(k):
            s = np.random.uniform(seg*i, seg*(i+1))
            idx, p = self.tree.get(s)
            idxs[i]  = idx
            prios[i] = max(p, 1e-12)
        probs = prios / (self.tree.total + 1e-12)
        w = (self.tree.size * probs) ** (-beta)
        w /= w.max() + 1e-12
        def t(x): return torch.tensor(x, device=device)
        return (t(self.s[idxs]), t(self.a[idxs]), t(self.r[idxs]),
                t(self.s2[idxs]), t(self.d[idxs]),
                t(w.astype(np.float32)), idxs)

    def update_prios(self, idxs, td_errs):
        for idx, err in zip(idxs, td_errs):
            p = float(abs(err)) + self.eps
            if self.qos_aware:
                p *= (1.0 + self.qos_lambda * float(self.v[idx]))
            self.max_p = max(self.max_p, p)
            self.tree.set(int(idx), p ** self.alpha)

    def __len__(self): return self.tree.size


# ── D3QN Agent ────────────────────────────────────────────────────────────
class D3QNAgent:
    def __init__(self, state_dim, action_dim, device,
                 lr=1e-4, gamma=0.99,
                 buf_size=20_000, batch=128,
                 target_update=500,
                 eps_start=1.0, eps_end=0.05, eps_steps=50_000,
                 qos_aware=False, qos_lambda=1.5):
        self.device  = device
        self.action_dim = action_dim
        self.gamma   = gamma
        self.batch   = batch
        self.t_upd   = target_update
        self.eps_end = eps_end
        self.eps_rng = eps_steps
        self.eps_st  = eps_start
        self.step_n  = 0

        self.online = DuelingNet(state_dim, action_dim).to(device)
        self.target = DuelingNet(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters(): p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buf = PERBuffer(buf_size, state_dim,
                             qos_aware=qos_aware, qos_lambda=qos_lambda)

    def eps(self):
        return self.eps_st + (self.eps_end-self.eps_st)*min(1, self.step_n/self.eps_rng)

    def act(self, s):
        if np.random.rand() < self.eps():
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q = self.online(torch.tensor(s[None], device=self.device))
            return int(q.argmax())

    def greedy(self, s):
        with torch.no_grad():
            q = self.online(torch.tensor(s[None], device=self.device))
            return int(q.argmax())

    def observe(self, s, a, r, s2, done, qv=0):
        self.buf.add(s, a, r, s2, done, qv)
        self.step_n += 1

    def learn(self):
        if len(self.buf) < max(1000, self.batch): return None
        s,a,r,s2,d,w,idx = self.buf.sample(self.batch, self.device)
        with torch.no_grad():
            na  = self.online(s2).argmax(1, keepdim=True)
            nq  = self.target(s2).gather(1, na).squeeze(1)
            tgt = r + self.gamma*(1-d)*nq
        cur = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        td  = tgt - cur
        loss= (w * td.pow(2)).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.opt.step()
        self.buf.update_prios(idx, td.detach().cpu().numpy())
        if self.step_n % self.t_upd == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss), "mean_q": float(cur.mean()), "eps": self.eps()}
