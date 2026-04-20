"""agents/baselines.py — v3 (works with 10-dim state)"""
import numpy as np

class FixedPriceAgent:
    def __init__(self, price_levels, base_price):
        self.action = int(np.argmin(np.abs(price_levels - base_price)))
        self.price  = price_levels[self.action]
    def select_action(self, state, **kw): return self.action
    def __str__(self): return f"Fixed(${self.price:.2f})"

class RandomAgent:
    def __init__(self, n_actions, seed=42):
        self.n=n_actions; self.rng=np.random.RandomState(seed)
    def select_action(self, state, **kw): return self.rng.randint(0, self.n)
    def __str__(self): return "Random"

class RuleBasedAgent:
    """Raise price when demand trend up (state[5]), lower when down."""
    def __init__(self, n_actions, threshold=0.1):
        self.n=n_actions; self.thr=threshold
        self.cur=n_actions//2
    def select_action(self, state, **kw):
        demand_trend = float(state[5])  # state[5] = demand_trend feature
        if demand_trend > self.thr:   self.cur = min(self.cur+1, self.n-1)
        elif demand_trend < -self.thr: self.cur = max(self.cur-1, 0)
        return self.cur
    def reset(self): self.cur = self.n//2
    def __str__(self): return "RuleBased"

class OptimalAgent:
    def __init__(self, price_levels, base_price, elasticity):
        eps = abs(elasticity)
        opt = base_price * eps/(eps-1) if eps > 1 else base_price*1.5
        opt = np.clip(opt, price_levels[0], price_levels[-1])
        self.action = int(np.argmin(np.abs(price_levels - opt)))
        self.price  = price_levels[self.action]
    def select_action(self, state, **kw): return self.action
    def __str__(self): return f"Optimal(${self.price:.2f})"

def evaluate_agent(agent, env, n_episodes=50):
    ep_rewards=[]; price_choices=[]
    for _ in range(n_episodes):
        if hasattr(agent,'reset'): agent.reset()
        state,_ = env.reset(); er=0.0
        while True:
            a = agent.select_action(state)
            ns,r,done,_,info = env.step(a)
            er+=r; price_choices.append(info.get("price",0.0))
            state=ns
            if done: break
        ep_rewards.append(er)
    return {"mean_reward":float(np.mean(ep_rewards)),"std_reward":float(np.std(ep_rewards)),
            "mean_revenue":float(np.mean(ep_rewards)),"avg_price":float(np.mean(price_choices)),
            "n_episodes":n_episodes}
