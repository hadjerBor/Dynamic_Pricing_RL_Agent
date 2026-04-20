"""
agents/dqn_agent.py  — v3  (updated for 10-dim state + normalized rewards)
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Linear(64, n_actions))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=20_000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, ns, done):
        self.buf.append((np.array(s,dtype=np.float32), int(a), float(r),
                         np.array(ns,dtype=np.float32), float(done)))
    def sample(self, n):
        b = random.sample(self.buf, n)
        s,a,r,ns,d = zip(*b)
        return (torch.tensor(np.array(s),dtype=torch.float32),
                torch.tensor(a,dtype=torch.long),
                torch.tensor(r,dtype=torch.float32),
                torch.tensor(np.array(ns),dtype=torch.float32),
                torch.tensor(d,dtype=torch.float32))
    def __len__(self): return len(self.buf)

class DQNAgent:
    def __init__(self, state_dim=10, n_actions=11, hidden_size=256,
                 gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.995, batch_size=128, target_update=10,
                 buffer_capacity=20_000, device="cpu"):
        self.n_actions=n_actions; self.gamma=gamma; self.epsilon=epsilon_start
        self.epsilon_end=epsilon_end; self.epsilon_decay=epsilon_decay
        self.batch_size=batch_size; self.target_update=target_update
        self.device=torch.device(device); self.step_count=0; self.episode_count=0

        self.q_net      = QNetwork(state_dim, n_actions, hidden_size).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict()); self.target_net.eval()
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler  = optim.lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=0.5)
        self.buffer     = ReplayBuffer(buffer_capacity)
        self.loss_fn    = nn.SmoothL1Loss()
        self.losses=[]; self.epsilons=[]

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        st = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad(): return int(self.q_net(st).argmax(dim=1).item())

    def store_transition(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self):
        if len(self.buffer) < self.batch_size: return 0.0
        s,a,r,ns,d = [x.to(self.device) for x in self.buffer.sample(self.batch_size)]
        cur_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            tgt = r + self.gamma * self.target_net(ns).max(1).values * (1-d)
            tgt = tgt.clamp(-50.0, 50.0)
        loss = self.loss_fn(cur_q, tgt)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.step_count += 1; self.losses.append(loss.item())
        return loss.item()

    def end_episode(self):
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilons.append(self.epsilon)
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        if self.episode_count % 300 == 0:
            self.scheduler.step()

    def train(self, env, n_episodes=600):
        ep_rewards=[]; ep_revenues=[]
        print(f"\n[DQN] Training: {env.product_id} | Episodes: {n_episodes}")
        for ep in range(n_episodes):
            state,_ = env.reset(); total_r=0.0
            while True:
                a = self.select_action(state, training=True)
                ns,r,done,_,_ = env.step(a)
                self.store_transition(state,a,r,ns,done)
                self.update()
                state=ns; total_r+=r
                if done: break
            self.end_episode()
            ep_rewards.append(total_r); ep_revenues.append(env.episode_revenue)
            if (ep+1)%100==0:
                print(f"  Ep {ep+1:4d}/{n_episodes} | Avg(100):{np.mean(ep_rewards[-100:]):8.3f} | ε:{self.epsilon:.3f}")
        return {"rewards": ep_rewards, "revenues": ep_revenues}

    def save(self, path):
        torch.save({"q_net":self.q_net.state_dict(),"target_net":self.target_net.state_dict(),
                    "optimizer":self.optimizer.state_dict(),"epsilon":self.epsilon,
                    "episode_count":self.episode_count}, path)
        print(f"[DQN] Saved → {path}")

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ck["q_net"]); self.target_net.load_state_dict(ck["target_net"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.epsilon=ck["epsilon"]; self.episode_count=ck["episode_count"]
        print(f"[DQN] Loaded ← {path}")
