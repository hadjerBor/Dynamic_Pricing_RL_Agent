"""
backend.py — Dynamic Pricing RL Agent API
Place this file inside Dynamic_Pricing_RL_Agent/ and run: python backend.py

HOW MODELS ARE USED:
  - MAML: loads results/models/maml_meta_policy.pt  (saved by train_and_evaluate.py)
          Falls back to random init if not yet trained — results will be poor.
  - DQN:  loads results/models/dqn_<product_id>.pt  per product
          Falls back to random init if not trained.
  Run train_and_evaluate.py first to produce real weights!
"""

import os, sys, json, copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# ── Path setup (Windows / Linux / Mac) ───────────────────────────────────────
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

# ── Constants (must match train_and_evaluate.py CFG exactly) ─────────────────
STATE_DIM  = 10
N_ACTIONS  = 11
DATA_DIR   = os.path.join(PROJECT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT, "results", "models")

# BUG FIX: train_and_evaluate.py saves as "maml_meta_policy.pt", not "maml_meta.pt"
META_PATH  = os.path.join(MODELS_DIR, "maml_meta_policy.pt")

ENV_KWARGS = dict(
    episode_len=30, n_price_levels=N_ACTIONS,
    max_inventory=2000, restock_rate=50, normalize_reward=True
)

# ── In-memory state ───────────────────────────────────────────────────────────
_products    = []
_maml_agent  = None
_dqn_agents  = {}
_initialised = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_products():
    global _products
    meta_path = os.path.join(DATA_DIR, "products_metadata.json")
    with open(meta_path) as f:
        metas = json.load(f)
    products = []
    for m in metas:
        ts_path = os.path.join(PROJECT, m["timeseries_file"].replace("/", os.sep))
        if os.path.exists(ts_path):
            df = pd.read_csv(ts_path)
            products.append({**m, "timeseries": df})
        else:
            print(f"[backend] WARNING: timeseries not found: {ts_path}")
    _products = products
    print(f"[backend] Loaded {len(_products)} products")
    return products


def get_product(pid):
    for p in _products:
        if str(p["id"]) == str(pid):
            return p
    return None


def make_env(product):
    return SingleProductPricingEnv(product, **ENV_KWARGS)


def _build_maml():
    """
    Build MAMLAgent with EXACT same hyperparams as train_and_evaluate.py CFG.
    Then load weights if they exist.
    """
    agent = MAMLAgent(
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        inner_lr=0.05,          # CFG["maml_inner_lr"]
        meta_lr=1e-3,           # CFG["maml_meta_lr"]
        inner_steps=5,          # CFG["maml_inner_steps"]
        meta_batch_size=8,      # CFG["maml_batch_size"]
        gamma=0.99,
    )
    if os.path.exists(META_PATH):
        try:
            agent.load(META_PATH)
            print(f"[backend] ✓ MAML trained weights loaded from {META_PATH}")
        except Exception as e:
            print(f"[backend] ✗ Could not load MAML weights: {e}")
            print(f"[backend]   → Using UNTRAINED random init. Run train_and_evaluate.py first!")
    else:
        print(f"[backend] ✗ No MAML checkpoint at {META_PATH}")
        print(f"[backend]   → Using UNTRAINED random init. Run train_and_evaluate.py first!")
    return agent


def _load_dqn(pid_int):
    """Load a DQN agent for a specific product id, or return random-init agent."""
    agent = DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS,
                     lr=1e-3, gamma=0.99,
                     epsilon_start=0.0, epsilon_end=0.0)  # no exploration at eval
    model_path = os.path.join(MODELS_DIR, f"dqn_{pid_int}.pt")
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"[backend] ✓ DQN weights loaded for product {pid_int}")
    else:
        print(f"[backend] ✗ No DQN weights for product {pid_int} at {model_path}")
        print(f"[backend]   → Using UNTRAINED random init.")
    return agent


def _run_episode_greedy(agent_obj, env, agent_type):
    """
    Run one full greedy episode (no exploration).
    agent_type: "maml_policy" | "dqn" | "baseline"
    """
    state, _ = env.reset()
    steps = []
    total_reward = 0.0
    done = False
    day = 0
    while not done:
        if agent_type == "maml_policy":
            # agent_obj is a PolicyNetwork (torch.nn.Module)
            agent_obj.eval()
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = int(agent_obj(st).argmax(dim=1).item())
        elif agent_type == "dqn":
            # agent_obj is a DQNAgent — epsilon=0 so fully greedy
            action = agent_obj.select_action(state, training=False)
        else:
            # agent_obj is a baseline (Fixed/Random/RuleBased/Optimal)
            action = agent_obj.select_action(state)

        next_state, reward, done, _, info = env.step(action)
        steps.append({
            "day":        day + 1,
            "price":      round(float(info["price"]), 4),
            "demand":     round(float(info["demand"]), 2),
            "units_sold": int(info["units_sold"]),
            "revenue":    round(float(info["revenue"]), 2),
            "inventory":  int(info["inventory"]),
            "reward":     round(float(reward), 4),
        })
        total_reward += reward
        state = next_state
        day += 1
    return steps, round(float(total_reward), 4)


# ── First-request initialisation ─────────────────────────────────────────────
@app.before_request
def _init():
    global _initialised, _maml_agent
    if not _initialised:
        _initialised = True
        load_products()
        _maml_agent = _build_maml()


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Dynamic Pricing RL backend"})


# ── 1. Products list ──────────────────────────────────────────────────────────
@app.route("/api/products", methods=["GET"])
def api_products():
    return jsonify([{
        "id":          p["id"],
        "name":        p["name"],
        "category":    p["category"],
        "base_price":  p["base_price"],
        "price_min":   p["price_min"],
        "price_max":   p["price_max"],
        "elasticity":  p["elasticity"],
        "base_demand": p["base_demand"],
        "is_held_out": p["id"] == 21931,
    } for p in _products])


# ── 2. Model status (honest about what is loaded) ────────────────────────────
@app.route("/api/status", methods=["GET"])
def api_status():
    dqn_files = []
    if os.path.exists(MODELS_DIR):
        dqn_files = [f for f in os.listdir(MODELS_DIR)
                     if f.startswith("dqn_") and f.endswith(".pt")]

    maml_trained = os.path.exists(META_PATH)
    return jsonify({
        "maml_trained":    maml_trained,
        "maml_path":       META_PATH,
        "dqn_trained_ids": [f.replace("dqn_","").replace(".pt","") for f in dqn_files],
        "n_dqn_trained":   len(dqn_files),
        "n_products":      len(_products),
        "project_root":    PROJECT,
        "warning": None if maml_trained else
            "MAML weights not found — run train_and_evaluate.py first for real results"
    })


# ── 3. Summary (real results.json or paper fallback) ─────────────────────────
@app.route("/api/summary", methods=["GET"])
def api_summary():
    summary_path = os.path.join(PROJECT, "results", "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        data["source"] = "trained"   # real trained results
        return jsonify(data)
    # Fallback: paper-reported numbers
    return jsonify({
        "source": "paper_fallback",
        "warning": "No summary.json found — showing paper results. Run train_and_evaluate.py to get your own.",
        "agents": {
            "MAML":      {"mean_reward": 40.57, "std_reward": 7.49},
            "Optimal":   {"mean_reward": 36.07, "std_reward": 6.88},
            "DQN":       {"mean_reward": 31.87, "std_reward": 6.23},
            "Random":    {"mean_reward": 26.40, "std_reward": 8.09},
            "Fixed":     {"mean_reward": 21.62, "std_reward": 7.10},
            "RuleBased": {"mean_reward": 21.22, "std_reward": 12.61},
        },
    })


# ── 4. Product time-series ────────────────────────────────────────────────────
@app.route("/api/products/<pid>/timeseries", methods=["GET"])
def api_timeseries(pid):
    p = get_product(pid)
    if not p:
        return jsonify({"error": "Not found"}), 404
    df = p["timeseries"].head(365)
    return jsonify(df[["date","price","demand","revenue","seasonal"]].to_dict(orient="records"))


# ── 5. Simulate a single episode ─────────────────────────────────────────────
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data       = request.get_json()
    pid        = str(data.get("product_id"))
    agent_type = data.get("agent_type", "maml")

    p = get_product(pid)
    if not p:
        return jsonify({"error": "Product not found"}), 404

    env = make_env(p)

    if agent_type == "maml":
        # Use a fresh copy of the meta-policy (no adaptation — raw meta-init)
        policy = copy.deepcopy(_maml_agent.meta_policy)
        steps, total_reward = _run_episode_greedy(policy, env, "maml_policy")
        model_note = "Trained MAML meta-policy" if os.path.exists(META_PATH) else "UNTRAINED random init"

    elif agent_type == "maml_adapted":
        # Run the proper adapt_to_new_product() method — this is the real MAML use case
        adapted_policy, _ = _maml_agent.adapt_to_new_product(env, n_adapt_episodes=10)
        steps, total_reward = _run_episode_greedy(adapted_policy, env, "maml_policy")
        model_note = "MAML after 10-episode fast adaptation"

    elif agent_type == "dqn":
        pid_int = p["id"]
        if pid_int not in _dqn_agents:
            _dqn_agents[pid_int] = _load_dqn(pid_int)
        steps, total_reward = _run_episode_greedy(_dqn_agents[pid_int], env, "dqn")
        model_path = os.path.join(MODELS_DIR, f"dqn_{pid_int}.pt")
        model_note = "Trained DQN (600 eps)" if os.path.exists(model_path) else "UNTRAINED random init"

    elif agent_type == "fixed":
        steps, total_reward = _run_episode_greedy(
            FixedPriceAgent(env.price_levels, p["base_price"]), env, "baseline")
        model_note = "Always charges base price"

    elif agent_type == "random":
        steps, total_reward = _run_episode_greedy(RandomAgent(N_ACTIONS), env, "baseline")
        model_note = "Uniformly random pricing"

    elif agent_type == "rulebased":
        steps, total_reward = _run_episode_greedy(RuleBasedAgent(N_ACTIONS), env, "baseline")
        model_note = "Demand-trend following heuristic"

    elif agent_type == "optimal":
        steps, total_reward = _run_episode_greedy(
            OptimalAgent(env.price_levels, p["base_price"], p["elasticity"]), env, "baseline")
        model_note = "Closed-form revenue-maximising price (knows elasticity)"

    else:
        return jsonify({"error": f"Unknown agent type: {agent_type}"}), 400

    return jsonify({
        "product":      {"id": p["id"], "name": p["name"],
                         "base_price": p["base_price"], "category": p["category"]},
        "agent":        agent_type,
        "model_note":   model_note,
        "total_reward": total_reward,
        "total_revenue":round(sum(s["revenue"] for s in steps), 2),
        "steps":        steps,
    })


# ── 6. Benchmark all agents on one product ────────────────────────────────────
@app.route("/api/benchmark/<pid>", methods=["GET"])
def api_benchmark(pid):
    p = get_product(pid)
    if not p:
        return jsonify({"error": "Not found"}), 404

    N_EVAL = int(request.args.get("n", 20))
    env    = make_env(p)
    results = {}

    # ── Baselines ──
    for name, agent in [
        ("fixed",     FixedPriceAgent(env.price_levels, p["base_price"])),
        ("random",    RandomAgent(N_ACTIONS)),
        ("rulebased", RuleBasedAgent(N_ACTIONS)),
        ("optimal",   OptimalAgent(env.price_levels, p["base_price"], p["elasticity"])),
    ]:
        r = evaluate_agent(agent, env, n_episodes=N_EVAL)
        results[name] = {
            "mean": round(r["mean_reward"], 3),
            "std":  round(r["std_reward"],  3),
            "trained": True,
        }

    # ── MAML (adapted — the real use case) ──
    adapted_policy, _ = _maml_agent.adapt_to_new_product(env, n_adapt_episodes=10)
    adapted_policy.eval()
    maml_rewards = []
    for _ in range(N_EVAL):
        state, _ = env.reset(); er = 0.0; done = False
        while not done:
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = int(adapted_policy(st).argmax(dim=1).item())
            state, r, done, _, _ = env.step(a); er += r
        maml_rewards.append(er)
    results["maml"] = {
        "mean":    round(float(np.mean(maml_rewards)), 3),
        "std":     round(float(np.std(maml_rewards)),  3),
        "trained": os.path.exists(META_PATH),
    }

    # ── DQN (if trained) ──
    pid_int    = p["id"]
    model_path = os.path.join(MODELS_DIR, f"dqn_{pid_int}.pt")
    if pid_int not in _dqn_agents:
        _dqn_agents[pid_int] = _load_dqn(pid_int)
    dqn_rewards = []
    for _ in range(N_EVAL):
        state, _ = env.reset(); er = 0.0; done = False
        while not done:
            a = _dqn_agents[pid_int].select_action(state, training=False)
            state, r, done, _, _ = env.step(a); er += r
        dqn_rewards.append(er)
    results["dqn"] = {
        "mean":    round(float(np.mean(dqn_rewards)), 3),
        "std":     round(float(np.std(dqn_rewards)),  3),
        "trained": os.path.exists(model_path),
    }

    return jsonify({
        "product_id": pid,
        "n_eval":     N_EVAL,
        "results":    results,
    })


# ── 7. MAML fast-adaptation curve ────────────────────────────────────────────
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

        # Evaluate adapted policy (5 greedy episodes)
        adapted.eval()
        ep_r = []
        for _ in range(5):
            state, _ = env.reset(); er = 0.0; done = False
            while not done:
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad(): a = int(adapted(st).argmax(1).item())
                state, r, done, _, _ = env.step(a); er += r
            ep_r.append(er)
        adapted.train()
        rewards_per_ep.append(round(float(np.mean(ep_r)), 3))

    return jsonify({
        "product_id":  pid,
        "n_adapt":     n_adapt,
        "rewards":     rewards_per_ep,
        "maml_trained": os.path.exists(META_PATH),
    })


# ── 8. Register a custom product ──────────────────────────────────────────────
@app.route("/api/products/custom", methods=["POST"])
def api_custom_product():
    data        = request.get_json()
    base_price  = float(data.get("base_price",  10.0))
    base_demand = float(data.get("base_demand", 50.0))
    elasticity  = float(data.get("elasticity",  -0.5))
    category    = data.get("category", "physical_good")
    name        = data.get("name", "Custom Product")

    n_days   = 365
    dates    = pd.date_range("2023-01-01", periods=n_days)
    doy      = np.arange(1, n_days + 1)
    seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * doy / 365)
    dow      = np.array([d.weekday() for d in dates])
    demand   = np.maximum(0, base_demand * seasonal * (1 + 0.05 * np.random.randn(n_days)))

    df = pd.DataFrame({
        "date":        dates.strftime("%Y-%m-%d"),
        "day_of_week": dow, "day_of_year": doy,
        "price":       np.full(n_days, base_price),
        "demand":      demand,
        "revenue":     demand * base_price,
        "seasonal":    seasonal,
    })

    custom_id = 99000 + len([p for p in _products if p["id"] >= 99000])
    _products.append({
        "id":         custom_id, "name": name, "category": category,
        "base_price": base_price, "price_min": base_price * 0.5,
        "price_max":  base_price * 2.0, "elasticity": elasticity,
        "base_demand":base_demand, "noise_std": 0.10,
        "timeseries": df, "is_custom": True,
    })
    return jsonify({"id": custom_id, "name": name, "message": "Custom product registered"})


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Dynamic Pricing RL — Backend")
    print(f"{'='*60}")
    print(f"Project root : {PROJECT}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Models dir   : {MODELS_DIR}")
    print(f"MAML weights : {'FOUND ✓' if os.path.exists(META_PATH) else 'NOT FOUND ✗  (run train_and_evaluate.py)'}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=5050, debug=False)