"""
backend.py — Dynamic Pricing RL Agent API
Flask REST API that wires the RL models to the frontend.

USAGE: Place this file inside the Dynamic_Pricing_RL_Agent/ folder and run:
    python backend.py
"""

import os, sys, json, copy, threading
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# ── Path setup (works on Windows, Linux, Mac) ─────────────────────────────────
# backend.py lives inside Dynamic_Pricing_RL_Agent/ — that IS the project root.
PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from flask import Flask, jsonify, request
from flask_cors import CORS

from environments.pricing_env import SingleProductPricingEnv
from agents.dqn_agent         import DQNAgent
from agents.maml_agent        import MAMLAgent, collect_episode, compute_td_loss
from agents.baselines         import (FixedPriceAgent, RandomAgent,
                                      RuleBasedAgent, OptimalAgent, evaluate_agent)

app = Flask(__name__)
CORS(app)

# ── Constants ─────────────────────────────────────────────────────────────────
STATE_DIM  = 10
N_ACTIONS  = 11
DATA_DIR   = os.path.join(PROJECT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT, "results", "models")
META_PATH  = os.path.join(MODELS_DIR, "maml_meta.pt")

ENV_KWARGS = dict(
    episode_len=30, n_price_levels=N_ACTIONS,
    max_inventory=2000, restock_rate=50, normalize_reward=True
)

# ── In-memory state ───────────────────────────────────────────────────────────
_products   = []
_maml_agent = None
_dqn_agents = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_products():
    global _products
    meta_path = os.path.join(DATA_DIR, "products_metadata.json")
    with open(meta_path) as f:
        metas = json.load(f)
    products = []
    for m in metas:
        # timeseries_file is relative to PROJECT root
        ts_path = os.path.join(PROJECT, m["timeseries_file"].replace("/", os.sep))
        if os.path.exists(ts_path):
            df = pd.read_csv(ts_path)
            products.append({**m, "timeseries": df})
        else:
            print(f"[backend] WARNING: timeseries not found: {ts_path}")
    _products = products
    print(f"[backend] Loaded {len(_products)} products from {DATA_DIR}")
    return products


def get_product(pid):
    for p in _products:
        if str(p["id"]) == str(pid):
            return p
    return None


def make_env(product):
    return SingleProductPricingEnv(product, **ENV_KWARGS)


def _build_maml():
    agent = MAMLAgent(
        state_dim=STATE_DIM, n_actions=N_ACTIONS,
        inner_lr=0.05, meta_lr=1e-3, inner_steps=5, gamma=0.99
    )
    if os.path.exists(META_PATH):
        try:
            agent.load(META_PATH)
            print(f"[backend] MAML weights loaded from {META_PATH}")
        except Exception as e:
            print(f"[backend] Could not load MAML weights: {e}")
    else:
        print(f"[backend] No MAML checkpoint found at {META_PATH} — using random init")
    return agent


def _run_episode_greedy(policy_or_agent, env, agent_type="maml"):
    """Run one greedy episode, return step-by-step info."""
    state, _ = env.reset()
    steps = []
    total_reward = 0.0
    done = False
    day = 0
    while not done:
        if agent_type == "maml":
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = int(policy_or_agent(st).argmax(dim=1).item())
        elif agent_type == "dqn":
            action = policy_or_agent.select_action(state, training=False)
        else:
            action = policy_or_agent.select_action(state)

        next_state, reward, done, _, info = env.step(action)
        steps.append({
            "day": day + 1,
            "price": round(float(info["price"]), 2),
            "demand": round(float(info["demand"]), 1),
            "units_sold": int(info["units_sold"]),
            "revenue": round(float(info["revenue"]), 2),
            "inventory": int(info["inventory"]),
            "reward": round(float(reward), 4),
        })
        total_reward += reward
        state = next_state
        day += 1
    return steps, round(float(total_reward), 4)


# ── Initialise on first request ───────────────────────────────────────────────
_initialised = False

@app.before_request
def _init():
    global _products, _maml_agent, _initialised
    if not _initialised:
        _initialised = True
        load_products()
        _maml_agent = _build_maml()


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Dynamic Pricing RL backend running"})


# ── 1. Products list ──────────────────────────────────────────────────────────
@app.route("/api/products", methods=["GET"])
def api_products():
    out = []
    for p in _products:
        out.append({
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "base_price": p["base_price"],
            "price_min": p["price_min"],
            "price_max": p["price_max"],
            "elasticity": p["elasticity"],
            "base_demand": p["base_demand"],
            "is_held_out": p["id"] == 21931,
        })
    return jsonify(out)


# ── 2. Product time-series ────────────────────────────────────────────────────
@app.route("/api/products/<pid>/timeseries", methods=["GET"])
def api_timeseries(pid):
    p = get_product(pid)
    if not p:
        return jsonify({"error": "Not found"}), 404
    df = p["timeseries"].head(365)
    records = df[["date", "price", "demand", "revenue", "seasonal"]].to_dict(orient="records")
    return jsonify(records)


# ── 3. Single simulation episode ─────────────────────────────────────────────
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json()
    pid        = str(data.get("product_id"))
    agent_type = data.get("agent_type", "maml")

    p = get_product(pid)
    if not p:
        return jsonify({"error": "Product not found"}), 404

    env = make_env(p)

    if agent_type == "maml":
        policy = copy.deepcopy(_maml_agent.meta_policy)
        policy.eval()
        steps, total_reward = _run_episode_greedy(policy, env, "maml")

    elif agent_type == "dqn":
        pid_int = p["id"]
        if pid_int not in _dqn_agents:
            model_path = os.path.join(MODELS_DIR, f"dqn_{pid_int}.pt")
            agent = DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)
            if os.path.exists(model_path):
                agent.load(model_path)
            else:
                print(f"[backend] No DQN model for {pid_int}, using random weights")
            _dqn_agents[pid_int] = agent
        steps, total_reward = _run_episode_greedy(_dqn_agents[pid_int], env, "dqn")

    elif agent_type == "fixed":
        agent = FixedPriceAgent(env.price_levels, p["base_price"])
        steps, total_reward = _run_episode_greedy(agent, env, "baseline")

    elif agent_type == "random":
        agent = RandomAgent(N_ACTIONS)
        steps, total_reward = _run_episode_greedy(agent, env, "baseline")

    elif agent_type == "rulebased":
        agent = RuleBasedAgent(N_ACTIONS)
        steps, total_reward = _run_episode_greedy(agent, env, "baseline")

    elif agent_type == "optimal":
        agent = OptimalAgent(env.price_levels, p["base_price"], p["elasticity"])
        steps, total_reward = _run_episode_greedy(agent, env, "baseline")

    else:
        return jsonify({"error": f"Unknown agent type: {agent_type}"}), 400

    return jsonify({
        "product": {"id": p["id"], "name": p["name"],
                    "base_price": p["base_price"], "category": p["category"]},
        "agent": agent_type,
        "total_reward": total_reward,
        "total_revenue": round(sum(s["revenue"] for s in steps), 2),
        "steps": steps,
    })


# ── 4. Benchmark: compare all agents on a product ────────────────────────────
@app.route("/api/benchmark/<pid>", methods=["GET"])
def api_benchmark(pid):
    p = get_product(pid)
    if not p:
        return jsonify({"error": "Not found"}), 404

    N_EVAL = int(request.args.get("n", 20))
    env = make_env(p)
    results = {}

    # Baselines
    for name, agent in [
        ("fixed",     FixedPriceAgent(env.price_levels, p["base_price"])),
        ("random",    RandomAgent(N_ACTIONS)),
        ("rulebased", RuleBasedAgent(N_ACTIONS)),
        ("optimal",   OptimalAgent(env.price_levels, p["base_price"], p["elasticity"])),
    ]:
        r = evaluate_agent(agent, env, n_episodes=N_EVAL)
        results[name] = {"mean": round(r["mean_reward"], 3), "std": round(r["std_reward"], 3)}

    # MAML
    policy = copy.deepcopy(_maml_agent.meta_policy)
    policy.eval()
    rewards = []
    for _ in range(N_EVAL):
        state, _ = env.reset()
        er = 0.0
        done = False
        while not done:
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = int(policy(st).argmax(dim=1).item())
            state, r, done, _, _ = env.step(a)
            er += r
        rewards.append(er)
    results["maml"] = {"mean": round(float(np.mean(rewards)), 3),
                       "std":  round(float(np.std(rewards)),  3)}

    # DQN (if trained weights exist)
    pid_int = p["id"]
    model_path = os.path.join(MODELS_DIR, f"dqn_{pid_int}.pt")
    if os.path.exists(model_path):
        if pid_int not in _dqn_agents:
            agent = DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)
            agent.load(model_path)
            _dqn_agents[pid_int] = agent
        dqn_rewards = []
        for _ in range(N_EVAL):
            state, _ = env.reset()
            er = 0.0
            done = False
            while not done:
                a = _dqn_agents[pid_int].select_action(state, training=False)
                state, r, done, _, _ = env.step(a)
                er += r
            dqn_rewards.append(er)
        results["dqn"] = {"mean": round(float(np.mean(dqn_rewards)), 3),
                          "std":  round(float(np.std(dqn_rewards)),  3)}

    return jsonify({"product_id": pid, "n_eval": N_EVAL, "results": results})


# ── 5. MAML fast-adaptation ───────────────────────────────────────────────────
@app.route("/api/adapt", methods=["POST"])
def api_adapt():
    data    = request.get_json()
    pid     = str(data.get("product_id"))
    n_adapt = int(data.get("n_adapt", 10))

    p = get_product(pid)
    if not p:
        return jsonify({"error": "Product not found"}), 404

    env     = make_env(p)
    adapted = copy.deepcopy(_maml_agent.meta_policy)
    optimizer = optim.SGD(adapted.parameters(), lr=0.05, momentum=0.9)

    rewards_per_ep = []
    for ep in range(n_adapt):
        epsilon = max(0.05, 0.3 - ep * 0.025)
        transitions, _ = collect_episode(env, adapted, epsilon=epsilon)
        loss = compute_td_loss(transitions, adapted, gamma=0.99)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), 5.0)
        optimizer.step()

        adapted.eval()
        ep_rewards = []
        for _ in range(5):
            state, _ = env.reset()
            er = 0.0
            done = False
            while not done:
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    a = int(adapted(st).argmax(dim=1).item())
                state, r, done, _, _ = env.step(a)
                er += r
            ep_rewards.append(er)
        adapted.train()
        rewards_per_ep.append(round(float(np.mean(ep_rewards)), 3))

    return jsonify({"product_id": pid, "n_adapt": n_adapt, "rewards": rewards_per_ep})


# ── 6. Register a custom product ──────────────────────────────────────────────
@app.route("/api/products/custom", methods=["POST"])
def api_custom_product():
    data         = request.get_json()
    base_price   = float(data.get("base_price", 10.0))
    base_demand  = float(data.get("base_demand", 50.0))
    elasticity   = float(data.get("elasticity", -0.5))
    category     = data.get("category", "physical_good")
    name         = data.get("name", "Custom Product")

    n_days   = 365
    dates    = pd.date_range("2023-01-01", periods=n_days)
    doy      = np.arange(1, n_days + 1)
    seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * doy / 365)
    dow      = np.array([d.weekday() for d in dates])
    price    = np.full(n_days, base_price)
    demand   = np.maximum(0, base_demand * seasonal * (1 + 0.05 * np.random.randn(n_days)))
    revenue  = demand * price

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "day_of_week": dow, "day_of_year": doy,
        "price": price, "demand": demand,
        "revenue": revenue, "seasonal": seasonal,
    })

    custom_id = 99000 + len([p for p in _products if p["id"] >= 99000])
    product = {
        "id": custom_id, "name": name, "category": category,
        "base_price": base_price, "price_min": base_price * 0.5,
        "price_max": base_price * 2.0, "elasticity": elasticity,
        "base_demand": base_demand, "noise_std": 0.10,
        "timeseries": df, "is_custom": True,
    }
    _products.append(product)
    return jsonify({"id": custom_id, "name": name, "message": "Custom product registered"})


# ── 7. Model / server status ──────────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def api_status():
    dqn_files = []
    if os.path.exists(MODELS_DIR):
        dqn_files = [f for f in os.listdir(MODELS_DIR)
                     if f.startswith("dqn_") and f.endswith(".pt")]
    return jsonify({
        "maml_loaded": os.path.exists(META_PATH),
        "dqn_models": [f.replace("dqn_", "").replace(".pt", "") for f in dqn_files],
        "n_products": len(_products),
        "project_root": PROJECT,
    })


# ── 8. Summary stats ──────────────────────────────────────────────────────────
@app.route("/api/summary", methods=["GET"])
def api_summary():
    summary_path = os.path.join(PROJECT, "results", "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return jsonify(json.load(f))
    # Fallback: paper results
    return jsonify({
        "agents": {
            "MAML":      {"mean_reward": 40.57, "std_reward": 7.49},
            "Optimal":   {"mean_reward": 36.07, "std_reward": 6.88},
            "DQN":       {"mean_reward": 31.87, "std_reward": 6.23},
            "Random":    {"mean_reward": 26.40, "std_reward": 8.09},
            "Fixed":     {"mean_reward": 21.62, "std_reward": 7.10},
            "RuleBased": {"mean_reward": 21.22, "std_reward": 12.61},
        },
        "source": "paper"
    })


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[backend] Project root: {PROJECT}")
    print(f"[backend] Data dir:     {DATA_DIR}")
    print(f"[backend] Models dir:   {MODELS_DIR}")
    app.run(host="0.0.0.0", port=5050, debug=False)
