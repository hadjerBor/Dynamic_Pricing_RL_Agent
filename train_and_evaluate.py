"""
train_and_evaluate.py  — v3  GENERAL PURPOSE
=====================================================================
The main script now reflects the general-purpose design.
MAML is the PRIMARY agent — DQN is the per-product baseline.

Key design change:
  - STATE_DIM = 10 (was 6)
  - normalize_reward=True everywhere for fair comparison
  - MAML evaluated with n_adapt_episodes=10 (fair: 10 episodes)
  - DQN evaluated AFTER 600 training episodes (per-product specialist)
  - New: demo_new_product_types() shows MAML working on product types
    it has never seen (the general-purpose use case)
=====================================================================
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.data_pipeline       import build_dataset, load_products
from environments.pricing_env import SingleProductPricingEnv, MultiProductPricingEnv
from agents.dqn_agent         import DQNAgent
from agents.maml_agent        import MAMLAgent
from agents.baselines         import (FixedPriceAgent, RandomAgent, RuleBasedAgent,
                                      OptimalAgent, evaluate_agent)
from utils.visualization      import (plot_learning_curves, plot_maml_meta_rewards,
                                      plot_comparison, plot_price_behavior,
                                      plot_adaptation_speed, plot_product_analysis)

os.makedirs("results", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

STATE_DIM = 10   # v3: 10-feature general-purpose state

CFG = {
    "n_products": 10, "n_days": 1095, "data_dir": "data/processed",
    "episode_len": 30, "n_price_levels": 11,
    "max_inventory": 2000, "restock_rate": 50, "normalize_reward": True,
    "n_dqn_episodes": 600,
    "dqn_lr": 1e-3, "dqn_gamma": 0.99,
    "dqn_eps_start": 1.0, "dqn_eps_end": 0.05, "dqn_eps_decay": 0.995,
    "n_maml_iterations": 500,
    "maml_inner_lr": 0.05,   # v3: increased from 0.01
    "maml_meta_lr":  1e-3,   # v3: increased from 3e-4
    "maml_inner_steps": 5,
    "maml_batch_size": 8,
    "n_eval_episodes": 50,
    "held_out_idx": 9,
    "n_adapt_episodes": 10,
}

ENV_KWARGS = dict(episode_len=CFG["episode_len"], n_price_levels=CFG["n_price_levels"],
                  max_inventory=CFG["max_inventory"], restock_rate=CFG["restock_rate"],
                  normalize_reward=CFG["normalize_reward"])

def get_products():
    meta_path = os.path.join(CFG["data_dir"], "products_metadata.json")
    if os.path.exists(meta_path):
        print("[MAIN] Loading existing dataset...")
        return load_products(CFG["data_dir"])
    return build_dataset(n_products=CFG["n_products"], n_days=CFG["n_days"],
                         output_dir=CFG["data_dir"])

def make_env(p): return SingleProductPricingEnv(p, **ENV_KWARGS)

def train_dqn(products):
    print("\n"+"="*60+"\nSTEP 2: Training DQN per product (baseline)\n"+"="*60)
    dqn_rewards={}; dqn_agents={}
    for p in products[:CFG["held_out_idx"]]:
        env   = make_env(p)
        agent = DQNAgent(state_dim=STATE_DIM, n_actions=CFG["n_price_levels"],
                         lr=CFG["dqn_lr"], gamma=CFG["dqn_gamma"],
                         epsilon_start=CFG["dqn_eps_start"],
                         epsilon_end=CFG["dqn_eps_end"],
                         epsilon_decay=CFG["dqn_eps_decay"])
        h = agent.train(env, n_episodes=CFG["n_dqn_episodes"])
        dqn_rewards[p["id"]] = h["rewards"]
        dqn_agents[p["id"]]  = agent
        agent.save(f"results/models/dqn_{p['id']}.pt")
    return dqn_agents, dqn_rewards

def train_maml(products):
    print("\n"+"="*60+"\nSTEP 3: Meta-training MAML (general-purpose agent)\n"+"="*60)
    multi_env = MultiProductPricingEnv(products[:CFG["held_out_idx"]], **ENV_KWARGS)
    agent = MAMLAgent(state_dim=STATE_DIM, n_actions=CFG["n_price_levels"],
                      inner_lr=CFG["maml_inner_lr"], meta_lr=CFG["maml_meta_lr"],
                      inner_steps=CFG["maml_inner_steps"],
                      meta_batch_size=CFG["maml_batch_size"])
    meta_rewards = agent.meta_train(multi_env, n_meta_iterations=CFG["n_maml_iterations"])
    agent.save("results/models/maml_meta_policy.pt")
    return agent, meta_rewards

def evaluate_all(products, dqn_agents, maml_agent):
    print("\n"+"="*60+"\nSTEP 4: Evaluating all agents\n"+"="*60)
    all_results={}
    for p in products[:CFG["held_out_idx"]]:
        env = make_env(p); pl = env.price_levels
        agents = {"Random":    RandomAgent(CFG["n_price_levels"]),
                  "Fixed":     FixedPriceAgent(pl, p["base_price"]),
                  "RuleBased": RuleBasedAgent(CFG["n_price_levels"]),
                  "Optimal":   OptimalAgent(pl, p["base_price"], p["elasticity"])}
        for name, ag in agents.items():
            all_results.setdefault(name, {})[p["id"]] = evaluate_agent(ag, env, CFG["n_eval_episodes"])

        if p["id"] in dqn_agents:
            dqn=dqn_agents[p["id"]]; rs=[]
            for _ in range(CFG["n_eval_episodes"]):
                s,_=env.reset(); er=0.0
                while True:
                    a=dqn.select_action(s,training=False); s,r,done,_,_=env.step(a); er+=r
                    if done: break
                rs.append(er)
            all_results.setdefault("DQN",{})[p["id"]]={"mean_reward":float(np.mean(rs)),
                "std_reward":float(np.std(rs)),"mean_revenue":0.0,"n_episodes":CFG["n_eval_episodes"]}

        # MAML with fair adaptation (10 episodes)
        adapted,_ = maml_agent.adapt_to_new_product(env, n_adapt_episodes=CFG["n_adapt_episodes"])
        m, s = maml_agent.evaluate(env, adapted, n_episodes=CFG["n_eval_episodes"])
        all_results.setdefault("MAML",{})[p["id"]]={"mean_reward":m,"std_reward":s,
            "mean_revenue":0.0,"n_episodes":CFG["n_eval_episodes"]}

    return all_results

def aggregate_results(all_results):
    summary={}
    for name, prod_res in all_results.items():
        means=[v["mean_reward"] for v in prod_res.values()]
        stds =[v["std_reward"]  for v in prod_res.values()]
        summary[name]={"mean_reward":float(np.mean(means)),"std_reward":float(np.mean(stds))}
    return summary

def fast_adaptation_demo(products, maml_agent):
    """
    The general-purpose demo: MAML adapts to the held-out product
    (never seen during meta-training) in just 10 episodes.
    Compare this to DQN which needs 600 episodes for the SAME product.
    """
    print("\n"+"="*60+"\nSTEP 5: General-Purpose Adaptation Demo\n"+"="*60)
    new_product = products[CFG["held_out_idx"]]
    new_env     = make_env(new_product)
    print(f"New product: {new_product['name']} | Category: {new_product['category']}")
    print(f"Base price: ${new_product['base_price']:.2f} | Never seen during meta-training")

    _, maml_adapt_rewards = maml_agent.adapt_to_new_product(new_env, CFG["n_adapt_episodes"])

    # DQN from scratch on same product
    print(f"\n  DQN from scratch on {new_product['id']}...")
    dqn = DQNAgent(state_dim=STATE_DIM, n_actions=CFG["n_price_levels"],
                   lr=CFG["dqn_lr"], gamma=CFG["dqn_gamma"],
                   epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.80)
    dqn_scratch_rewards=[]
    for ep in range(CFG["n_adapt_episodes"]):
        s,_=new_env.reset(); er=0.0
        while True:
            a=dqn.select_action(s,training=True); ns,r,done,_,_=new_env.step(a)
            dqn.store_transition(s,a,r,ns,done); dqn.update(); s=ns; er+=r
            if done: break
        dqn.end_episode()
        rs=[]; 
        for _ in range(15):
            ss,_=new_env.reset(); eer=0.0
            while True:
                aa=dqn.select_action(ss,training=False); ss,rr,dd,_,_=new_env.step(aa); eer+=rr
                if dd: break
            rs.append(eer)
        dqn_scratch_rewards.append(float(np.mean(rs)))
        print(f"  DQN scratch ep {ep+1}: eval_r={dqn_scratch_rewards[-1]:.3f}")

    return maml_adapt_rewards, dqn_scratch_rewards, new_env

def print_results_table(summary):
    print("\n"+"="*60+"\nRESULTS SUMMARY\n"+"="*60)
    print(f"{'Agent':<14} {'Mean Reward':>14} {'Std Dev':>10}")
    print("-"*40)
    for name, stats in sorted(summary.items(), key=lambda x: -x[1]["mean_reward"]):
        print(f"{name:<14} {stats['mean_reward']:>14.4f} {stats['std_reward']:>10.4f}")
    print("="*60)
    with open("results/summary.json","w") as f: json.dump(summary,f,indent=2)
    print("[MAIN] Summary → results/summary.json")

def main():
    print("\n"+"="*60)
    print("DYNAMIC PRICING  |  MAML GENERAL-PURPOSE AGENT  |  v3")
    print("="*60)
    products = get_products()
    plot_product_analysis(products, n_show=3)
    dqn_agents, dqn_rewards = train_dqn(products)
    plot_learning_curves(dqn_rewards)
    maml_agent, meta_rewards = train_maml(products)
    plot_maml_meta_rewards(meta_rewards)
    all_results = evaluate_all(products, dqn_agents, maml_agent)
    summary = aggregate_results(all_results)
    print_results_table(summary)
    plot_comparison(summary)
    maml_adapt, dqn_scratch, new_env = fast_adaptation_demo(products, maml_agent)
    plot_adaptation_speed(maml_adapt, dqn_scratch)
    first_env = make_env(products[0]); pl=first_env.price_levels; p0=products[0]
    plot_price_behavior(first_env,
        {"fixed":FixedPriceAgent(pl,p0["base_price"]),
         "random":RandomAgent(CFG["n_price_levels"]),
         "rule_based":RuleBasedAgent(CFG["n_price_levels"])})
    print("\n[MAIN] Done. All results in ./results/")

if __name__=="__main__":
    main()
