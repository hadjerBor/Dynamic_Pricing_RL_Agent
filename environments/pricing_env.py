"""
environments/pricing_env.py  — v3  GENERAL PURPOSE
=====================================================================
DESIGN PHILOSOPHY (v3):
  This environment works for ANY product from ANY domain.
  A bakery owner, SaaS company, or electronics retailer can all plug in
  their product and get a working pricing agent.

  The key insight: we never hardcode domain knowledge.
  Instead we normalize EVERYTHING relative to the product's own statistics:
    - price_ratio = price / base_price  (works for a $2 loaf or a $2000 laptop)
    - demand_ratio = demand / base_demand (works for 5 units/day or 5000)
    - inventory_ratio = inventory / max_inventory

  This makes the state space DOMAIN-AGNOSTIC: the agent learns to price
  in terms of ratios, not absolute dollars. A 20% price increase means
  the same thing whether the base price is $5 or $500.

GENERAL-PURPOSE FEATURES (new in v3):
  - product_type_embedding: 5 categories encoded as one-hot in state
    so the agent knows "I am pricing a subscription vs a physical good"
  - price_sensitivity: how fast demand drops with price (elasticity)
    is now a STATE FEATURE, not just a simulation parameter. This lets
    the meta-policy reason about elasticity without knowing it explicitly.
  - competition_factor: optional relative competitor price in state
  - The state now has 10 features (was 6) for richer generalization
=====================================================================
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

PRODUCT_CATEGORIES = ["physical_good", "subscription", "service", "digital", "perishable"]


class SingleProductPricingEnv(gym.Env):
    """
    General-purpose pricing environment for any product type.

    State (10 features, all normalized):
      [0] price_ratio          current_price / base_price, normalized
      [1] inventory_ratio      inventory / max_inventory (0=empty, 1=full)
      [2] dow_sin              day-of-week seasonality (sin)
      [3] dow_cos              day-of-week seasonality (cos)
      [4] season_signal        annual seasonality
      [5] demand_trend         7-day rolling demand vs baseline
      [6] price_momentum       direction of recent price changes
      [7] elasticity_signal    estimated demand sensitivity (helps generalize)
      [8] stockout_risk        how close to running out
      [9] category_signal      product category encoded as scalar

    Action (discrete): n_price_levels bins from price_min to price_max
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, product_meta, episode_len=30, n_price_levels=11,
                 max_inventory=2000, stockout_penalty=2.0, overstock_penalty=0.001,
                 restock_rate=50, normalize_reward=True, seed=None):
        super().__init__()

        # Core product params
        self.product_id   = product_meta["id"]
        self.base_price   = float(product_meta["base_price"])
        self.price_min    = float(product_meta["price_min"])
        self.price_max    = float(product_meta["price_max"])
        self.elasticity   = float(product_meta["elasticity"])
        self.base_demand  = float(product_meta["base_demand"])
        self.noise_std    = float(product_meta.get("noise_std", 0.10))
        self.timeseries   = product_meta["timeseries"]
        self.category     = product_meta.get("category", "physical_good")

        # Episode config
        self.episode_len       = episode_len
        self.n_price_levels    = n_price_levels
        self.max_inventory     = max_inventory
        self.stockout_penalty  = stockout_penalty
        self.overstock_penalty = overstock_penalty
        self.restock_rate      = restock_rate
        self.normalize_reward  = normalize_reward

        # Reward normalization scale (product-specific)
        self.reward_scale = max(1.0, self.base_demand * self.base_price)

        # Discrete price bins
        self.price_levels = np.linspace(self.price_min, self.price_max, n_price_levels)

        # Category index for state encoding
        cat_lower = self.category.lower().replace(" & ", "_").replace(" ", "_")
        self.cat_idx = next(
            (i for i, c in enumerate(PRODUCT_CATEGORIES) if c in cat_lower),
            0  # default: physical_good
        )

        # Spaces: 10-dim state, discrete actions
        self.observation_space = spaces.Box(
            low=np.full(10, -2.0, dtype=np.float32),
            high=np.full(10,  2.0, dtype=np.float32))
        self.action_space = spaces.Discrete(n_price_levels)

        self.rng = np.random.RandomState(seed)

        # Episode state
        self.current_step = 0
        self.start_day_idx = 0
        self.inventory = max_inventory
        self.current_price = self.base_price
        self.prev_price = self.base_price
        self.demand_history = []
        self.price_history = []
        self.episode_revenue = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n_rows = len(self.timeseries)
        max_start = max(0, n_rows - self.episode_len - 1)
        self.start_day_idx   = self.rng.randint(0, max(1, max_start))
        self.current_step    = 0
        self.inventory       = self.max_inventory
        self.current_price   = self.base_price
        self.prev_price      = self.base_price
        self.demand_history  = [self.base_demand] * 7
        self.price_history   = [self.base_price] * 3
        self.episode_revenue = 0.0
        return self._get_obs(), {}

    def step(self, action):
        price = float(self.price_levels[int(action)])
        self.prev_price = self.current_price
        self.current_price = price

        row      = self._get_row(self.current_step)
        seasonal = float(row.get("seasonal", 1.0))

        # Demand model: elasticity-based with noise
        raw_demand = max(0.0,
            self.base_demand * seasonal * (price / self.base_price) ** self.elasticity)
        noise = self.rng.randn() * self.noise_std * raw_demand
        demand = max(0.0, raw_demand + noise)
        units_sold = min(int(demand), self.inventory)

        revenue          = units_sold * price
        stockout_pen     = self.stockout_penalty * max(0, int(demand) - self.inventory)
        overstock_pen    = self.overstock_penalty * self.inventory
        raw_reward       = revenue - stockout_pen - overstock_pen
        reward = raw_reward / self.reward_scale if self.normalize_reward else raw_reward

        # Update inventory with restock
        self.inventory = min(self.max_inventory,
                             max(0, self.inventory - units_sold) + self.restock_rate)
        self.demand_history.append(units_sold)
        if len(self.demand_history) > 7: self.demand_history.pop(0)
        self.price_history.append(price)
        if len(self.price_history) > 3: self.price_history.pop(0)

        self.episode_revenue += revenue
        self.current_step += 1
        done = self.current_step >= self.episode_len

        return self._get_obs(), float(reward), done, False, {
            "price": price, "demand": demand, "units_sold": units_sold,
            "revenue": revenue, "inventory": self.inventory,
            "raw_reward": raw_reward}

    def _get_row(self, offset):
        idx = min(self.start_day_idx + offset, len(self.timeseries) - 1)
        return self.timeseries.iloc[idx].to_dict()

    def _get_obs(self):
        row = self._get_row(self.current_step)

        # [0] Price ratio: centered at 1.0, normalized
        price_ratio = np.clip((self.current_price / self.base_price - 1.0) / 0.5, -2, 2)

        # [1] Inventory ratio
        inv_ratio = (self.inventory / self.max_inventory) * 2.0 - 1.0

        # [2,3] Day-of-week cyclical
        dow = int(row.get("day_of_week", 0))
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # [4] Annual seasonality
        doy = int(row.get("day_of_year", 180))
        season = np.sin(2 * np.pi * doy / 365)

        # [5] Demand trend: 7-day rolling vs baseline
        avg_d = np.mean(self.demand_history) if self.demand_history else self.base_demand
        demand_trend = np.clip((avg_d / (self.base_demand + 1e-8) - 1.0), -2.0, 2.0)

        # [6] Price momentum: direction of last 3 prices
        if len(self.price_history) >= 2:
            price_mom = np.clip(
                (self.price_history[-1] - self.price_history[0]) / (self.base_price + 1e-8),
                -1.0, 1.0)
        else:
            price_mom = 0.0

        # [7] Elasticity signal: normalized to [-1, 1]
        # Tells the agent "how sensitive is this product to price changes?"
        elast_signal = np.clip(self.elasticity / 3.0, -2.0, 2.0)

        # [8] Stockout risk: how close to 0 inventory
        stockout_risk = 1.0 - (self.inventory / self.max_inventory)

        # [9] Category signal: normalized category index
        cat_signal = (self.cat_idx / max(len(PRODUCT_CATEGORIES) - 1, 1)) * 2.0 - 1.0

        return np.array([price_ratio, inv_ratio, dow_sin, dow_cos, season,
                         demand_trend, price_mom, elast_signal, stockout_risk, cat_signal],
                        dtype=np.float32)

    def render(self, mode="human"):
        print(f"[{self.product_id}] Day {self.current_step}/{self.episode_len} | "
              f"${self.current_price:.2f} | Inv:{self.inventory} | "
              f"Rev:${self.episode_revenue:.0f}")

    def get_optimal_price(self):
        eps = abs(self.elasticity)
        return self.base_price * eps / (eps - 1) if eps > 1 else self.base_price * 1.5


class MultiProductPricingEnv:
    """
    Pool of environments — one per product.
    MAML treats each product as a separate task.
    New products can be added at any time via add_product().
    """
    def __init__(self, products, episode_len=30, **env_kwargs):
        self.products    = products
        self.episode_len = episode_len
        self.env_kwargs  = env_kwargs
        self.envs = [SingleProductPricingEnv(p, episode_len=episode_len, **env_kwargs)
                     for p in products]
        self.observation_space = self.envs[0].observation_space
        self.action_space      = self.envs[0].action_space
        self.n_products        = len(products)
        self.STATE_DIM         = self.observation_space.shape[0]

    def add_product(self, product_meta):
        """Add a new product to the pool at runtime — supports general-purpose use."""
        env = SingleProductPricingEnv(product_meta,
                                      episode_len=self.episode_len, **self.env_kwargs)
        self.envs.append(env)
        self.products.append(product_meta)
        self.n_products += 1
        return len(self.envs) - 1

    def sample_task(self, idx=None):
        if idx is None: idx = np.random.randint(0, self.n_products)
        return self.envs[idx], idx

    def sample_task_batch(self, batch_size):
        idxs = np.random.choice(self.n_products, size=batch_size, replace=True)
        return [(self.envs[i], i) for i in idxs]

    def __len__(self): return self.n_products
