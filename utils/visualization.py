"""
=====================================================================
FILE: utils/visualization.py
PURPOSE:
  All plotting functions for the project report and analysis.
  Produces publication-quality figures saved as PNG files.

FIGURES PRODUCED:
  1. learning_curves.png     — DQN reward over training episodes
  2. maml_meta_rewards.png   — MAML meta-training rewards over iterations
  3. comparison_bar.png      — All agents compared (mean reward + std dev)
  4. price_behavior.png      — How agents set prices over a sample episode
  5. adaptation_speed.png    — MAML vs DQN adaptation to a NEW product
  6. product_analysis.png    — Demand/price/revenue analysis per product
=====================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Consistent color palette across all figures
COLORS = {
    "dqn":        "#2196F3",   # blue
    "maml":       "#4CAF50",   # green
    "fixed":      "#F44336",   # red
    "random":     "#9E9E9E",   # gray
    "rule_based": "#FF9800",   # orange
    "optimal":    "#9C27B0",   # purple
}

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "legend.fontsize":  10,
}

plt.rcParams.update(STYLE)
os.makedirs("results", exist_ok=True)


def smooth(values, window: int = 20):
    """Smooths a curve using a rolling average (for cleaner learning curve plots)."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ─── FIGURE 1: DQN LEARNING CURVE ───────────────────────────────────────────

def plot_learning_curves(dqn_rewards_per_product: dict, save_path="results/learning_curves.png"):
    """
    Plots per-product DQN learning curves showing how reward improves
    over training episodes.

    dqn_rewards_per_product: {product_id: [ep_reward_1, ep_reward_2, ...]}
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    products = list(dqn_rewards_per_product.keys())[:6]

    for i, pid in enumerate(products):
        rewards = dqn_rewards_per_product[pid]
        raw_x   = np.arange(len(rewards))
        raw_y   = np.array(rewards)

        # Smoothed curve
        sm_y    = smooth(raw_y, window=30)
        sm_x    = np.arange(len(sm_y)) + 15  # center offset

        axes[i].fill_between(raw_x, raw_y, alpha=0.15, color=COLORS["dqn"])
        axes[i].plot(sm_x, sm_y, color=COLORS["dqn"], linewidth=2, label="DQN (smoothed)")
        axes[i].axhline(np.max(sm_y), color="gray", linestyle="--", alpha=0.5, label="Best avg")

        axes[i].set_title(f"Product {pid}")
        axes[i].set_xlabel("Episode")
        axes[i].set_ylabel("Total Reward")
        axes[i].legend(loc="lower right")

    # Hide unused axes
    for j in range(len(products), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("DQN Learning Curves — Per Product", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {save_path}")


# ─── FIGURE 2: MAML META-TRAINING REWARDS ───────────────────────────────────

def plot_maml_meta_rewards(meta_rewards: list, save_path="results/maml_meta_rewards.png"):
    """
    Plots the MAML meta-training reward curve.
    Shows average reward across the task batch at each meta-iteration.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    raw_x = np.arange(len(meta_rewards))
    raw_y = np.array(meta_rewards)
    sm_y  = smooth(raw_y, window=20)
    sm_x  = np.arange(len(sm_y)) + 10

    ax.fill_between(raw_x, raw_y, alpha=0.15, color=COLORS["maml"])
    ax.plot(sm_x, sm_y, color=COLORS["maml"], linewidth=2.5, label="MAML meta reward (smoothed)")
    ax.plot(raw_x, raw_y, alpha=0.3, color=COLORS["maml"], linewidth=0.8)

    ax.set_xlabel("Meta-iteration")
    ax.set_ylabel("Avg Task Reward")
    ax.set_title("MAML Meta-Training Progress", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {save_path}")


# ─── FIGURE 3: AGENT COMPARISON BAR CHART ───────────────────────────────────

def plot_comparison(results: dict, save_path="results/comparison_bar.png"):
    """
    Bar chart comparing all agents by mean reward across all products.

    results = {
      "DQN":       {"mean_reward": X, "std_reward": Y},
      "MAML":      {...},
      "Fixed":     {...},
      "Random":    {...},
      "RuleBased": {...},
      "Optimal":   {...},
    }
    """
    agents     = list(results.keys())
    means      = [results[a]["mean_reward"] for a in agents]
    stds       = [results[a]["std_reward"]  for a in agents]
    color_keys = ["dqn", "maml", "fixed", "random", "rule_based", "optimal"]
    colors     = [COLORS.get(color_keys[i], "#333") for i in range(len(agents))]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(agents, means, yerr=stds, capsize=6, color=colors,
                  alpha=0.85, edgecolor="white", linewidth=0.8)

    # Annotate bars with value
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
            f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_xlabel("Agent")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Agent Performance Comparison (Mean ± Std)", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {save_path}")


# ─── FIGURE 4: PRICE BEHAVIOR OVER AN EPISODE ───────────────────────────────

def plot_price_behavior(env, agents: dict, n_days: int = 30,
                        save_path="results/price_behavior.png"):
    """
    Shows how different agents set prices over a single test episode.
    Lets you visually confirm that DQN/MAML learn sensible pricing patterns.
    """
    import copy

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    price_traces   = {}
    revenue_traces = {}

    for agent_name, agent in agents.items():
        color  = COLORS.get(agent_name.lower().replace(" ", "_"), "#333")
        prices  = []
        revenues= []

        if hasattr(agent, "reset"):
            agent.reset()
        state, _ = env.reset()

        for _ in range(n_days):
            if hasattr(agent, "select_action"):
                action = agent.select_action(state)
            else:
                import torch
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = int(agent(state_t).argmax(dim=1).item())

            next_state, reward, done, _, info = env.step(action)
            prices.append(info.get("price", env.current_price))
            revenues.append(info.get("revenue", 0))
            state = next_state
            if done:
                break

        price_traces[agent_name]   = prices
        revenue_traces[agent_name] = revenues

    days_x = np.arange(1, n_days + 1)

    for agent_name, prices in price_traces.items():
        color = COLORS.get(agent_name.lower().replace(" ", "_"), "#333")
        ax1.plot(days_x[:len(prices)], prices, label=agent_name, color=color,
                 linewidth=2, marker="o", markersize=3)

    ax1.axhline(env.base_price, color="black", linestyle=":", alpha=0.5, label="Base price")
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"Price Behavior — {env.product_id}", fontweight="bold")
    ax1.legend(loc="upper right")

    for agent_name, revenues in revenue_traces.items():
        color = COLORS.get(agent_name.lower().replace(" ", "_"), "#333")
        ax2.plot(days_x[:len(revenues)], revenues, label=agent_name, color=color,
                 linewidth=2, alpha=0.8)

    ax2.set_xlabel("Day in Episode")
    ax2.set_ylabel("Daily Revenue ($)")
    ax2.set_title("Daily Revenue per Agent", fontweight="bold")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {save_path}")


# ─── FIGURE 5: ADAPTATION SPEED ─────────────────────────────────────────────

def plot_adaptation_speed(maml_adapt_rewards: list, dqn_scratch_rewards: list,
                          save_path="results/adaptation_speed.png"):
    """
    MAML's key advantage: it adapts to a new product MUCH faster than
    training DQN from scratch.
    This plot shows reward vs. number of adaptation episodes.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ep_x = np.arange(1, len(maml_adapt_rewards) + 1)
    ax.plot(ep_x, maml_adapt_rewards, color=COLORS["maml"], linewidth=2.5,
            marker="o", markersize=6, label="MAML (adapt from meta-init)")

    # DQN from scratch only has the first N points of its full training
    n_compare = len(maml_adapt_rewards)
    dqn_x = np.arange(1, n_compare + 1)
    ax.plot(dqn_x, dqn_scratch_rewards[:n_compare], color=COLORS["dqn"], linewidth=2.5,
            marker="s", markersize=6, label="DQN (train from scratch)")

    ax.set_xlabel("Number of Adaptation Episodes")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Adaptation Speed: MAML vs DQN (New Product)", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {save_path}")


# ─── FIGURE 6: PRODUCT DEMAND ANALYSIS ──────────────────────────────────────

def plot_product_analysis(products: list, n_show: int = 3,
                          save_path="results/product_analysis.png"):
    """
    Shows price / demand / revenue time series for a few products.
    Helps readers understand the simulated environment in the report.
    """
    n_show  = min(n_show, len(products))
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, p in enumerate(products[:n_show]):
        ts   = p["timeseries"].head(365)   # first year
        days = np.arange(len(ts))

        # Price
        axes[row, 0].plot(days, ts["price"], color=COLORS["dqn"], linewidth=1)
        axes[row, 0].set_title(f"{p['name']} — Price")
        axes[row, 0].set_ylabel("Price ($)")

        # Demand
        axes[row, 1].plot(days, ts["demand"], color=COLORS["maml"], linewidth=1, alpha=0.7)
        axes[row, 1].set_title(f"{p['name']} — Demand")
        axes[row, 1].set_ylabel("Units/day")

        # Revenue
        axes[row, 2].plot(days, ts["revenue"], color=COLORS["rule_based"], linewidth=1)
        axes[row, 2].set_title(f"{p['name']} — Revenue")
        axes[row, 2].set_ylabel("Revenue ($)")

        for col in range(3):
            axes[row, col].set_xlabel("Day")

    fig.suptitle("Product Time-Series Analysis (Year 1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {save_path}")
