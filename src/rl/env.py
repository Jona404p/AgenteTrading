from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config.rl_settings import RLSettings
from src.data.features import FeatureEngineer
from src.rl.costs import CostModel
from src.rl.splits import FeatureScaler

_N_EXTRA = 3   # posición actual, PnL no realizado, drawdown


class TradingEnv(gym.Env):
    """Entorno de trading de un activo con exposición objetivo continua.

    Temporalidad (sin look-ahead):
      · El agente observa la vela cerrada `t` (features ya escaladas + estado de cartera).
      · Su acción es la exposición objetivo w ∈ [-1, 1] (fracción del equity; long-only si no hay short).
      · La orden se ejecuta al PRECIO DE APERTURA de la vela t+1 con comisión taker y deslizamiento.
      · El equity se valora al cierre de t+1; la recompensa es el log-retorno neto de ese paso.

    Recompensa = scale·[ ln(E_{t+1}/E_t) − λ_dd·Δdrawdown⁺ − λ_to·|Δw| ]. Las comisiones y el
    deslizamiento ya están dentro del equity, por lo que el agente aprende a evitar sobreoperar.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        scaler: FeatureScaler,
        cfg: RLSettings,
        random_start: bool = True,
        feature_columns: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.cost = CostModel(cfg.costs)
        self.feature_columns = feature_columns or FeatureEngineer.FEATURE_COLUMNS
        missing = [c for c in ["open", "close", "volume", *self.feature_columns] if c not in data.columns]
        if missing:
            raise ValueError(f"Faltan columnas en los datos: {missing}")
        if len(data) < cfg.window + 3:
            raise ValueError("Serie demasiado corta para la ventana configurada")

        self.window = cfg.window
        self.random_start = random_start
        self.index = data.index
        self.open = data["open"].to_numpy(dtype=np.float64)
        self.close = data["close"].to_numpy(dtype=np.float64)
        self.qvol = data["volume"].to_numpy(dtype=np.float64) * self.close   # volumen en USDT
        self.x = scaler.transform(data)
        self.n = len(data)
        self.n_features = len(self.feature_columns)
        self.w_min = -cfg.max_leverage if cfg.allow_short else 0.0
        self.w_max = cfg.max_leverage

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.window * self.n_features + _N_EXTRA,), dtype=np.float32)
        self._reset_state(self.window - 1, self.n - 1)

    # ------------------------------------------------------------------ estado
    def _reset_state(self, start: int, end: int) -> None:
        self.start, self.end, self.t = start, end, start
        self.cash = self.cfg.initial_cash
        self.qty = 0.0
        self.avg_entry = 0.0
        self.equity = self.cfg.initial_cash
        self.peak = self.equity
        self.fees_paid = 0.0
        self.slippage_paid = 0.0
        self.turnover = 0.0
        self.n_trades = 0
        self.equity_curve = [self.equity]
        self.weight_curve = [0.0]
        # Benchmark comprar-y-mantener: entra en la apertura de start+1 pagando los mismos costes.
        entry = self.cost.execute(1.0, 1.0, 1e18)
        self._bh_units = self.cfg.initial_cash / (self.close[start] * (1 + (entry.price - 1) + entry.fee))
        self.bench_curve = [self.cfg.initial_cash]

    @property
    def weight(self) -> float:
        return self.qty * self.close[self.t] / self.equity if self.equity > 0 else 0.0

    def _obs(self) -> np.ndarray:
        t = self.t
        feats = self.x[t - self.window + 1: t + 1].reshape(-1)
        unreal = 0.0
        if self.qty != 0.0 and self.avg_entry > 0:
            unreal = math.copysign(1.0, self.qty) * (self.close[t] / self.avg_entry - 1.0)
        dd = 1.0 - self.equity / self.peak
        extra = np.array([self.weight, np.clip(unreal / 0.05, -5, 5), np.clip(dd / 0.05, 0, 5)], dtype=np.float32)
        return np.concatenate([feats, extra]).astype(np.float32)

    # ------------------------------------------------------------------ API gym
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        first, last = self.window - 1, self.n - 1
        length = self.cfg.episode_len
        if "start" in options:
            start = int(options["start"])
        elif self.random_start and 0 < length < last - first:
            start = int(self.np_random.integers(first, last - length))
        else:
            start = first
        end = min(start + length, last) if (self.random_start and length > 0) else last
        self._reset_state(start, end)
        return self._obs(), self._info(0.0)

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        target_w = float(np.clip(a * self.cfg.max_leverage, self.w_min, self.w_max))
        t1 = self.t + 1
        px, qv = self.open[t1], self.qvol[t1]

        # 1) Ejecución a la apertura de t+1
        equity_open = self.cash + self.qty * px
        cur_w = self.qty * px / equity_open if equity_open > 0 else 0.0
        fee = slip = 0.0
        if abs(target_w - cur_w) >= self.cfg.min_rebalance and equity_open > 0:
            dq = target_w * equity_open / px - self.qty
            if dq > 0:   # compra: respeta el efectivo disponible (y el apalancamiento máximo)
                spendable = self.cash + (self.cfg.max_leverage - 1.0) * equity_open
                dq = min(dq, self.cost.max_affordable_buy(spendable, px, qv))
            if abs(dq) * px > 1e-8:
                fill = self.cost.execute(dq, px, qv)
                self._update_entry(dq, fill.price)
                self.cash -= fill.qty * fill.price + fill.fee
                self.qty += dq
                if abs(self.qty) * px < 1e-9:
                    self.qty, self.avg_entry = 0.0, 0.0
                fee, slip = fill.fee, fill.slippage_cost
                self.fees_paid += fee
                self.slippage_paid += slip
                self.turnover += fill.notional / max(equity_open, 1e-9)
                self.n_trades += 1

        # 2) Valoración al cierre de t+1
        prev_equity, prev_dd = self.equity, 1.0 - self.equity / self.peak
        self.t = t1
        self.equity = self.cash + self.qty * self.close[t1]
        self.peak = max(self.peak, self.equity)
        dd = 1.0 - self.equity / self.peak
        new_w = self.weight

        # 3) Recompensa
        if self.equity > 0:
            ret = math.log(self.equity / prev_equity)
        else:
            ret = -1.0
        reward = self.cfg.reward_scale * (
            ret
            - self.cfg.drawdown_penalty * max(0.0, dd - prev_dd)
            - self.cfg.turnover_penalty * abs(new_w - cur_w)
        )

        self.equity_curve.append(self.equity)
        self.weight_curve.append(new_w)
        self.bench_curve.append(self._bh_units * self.close[t1])

        terminated = self.equity <= self.cfg.initial_cash * self.cfg.ruin_equity_frac
        truncated = (not terminated) and self.t >= self.end
        return self._obs(), float(reward), bool(terminated), bool(truncated), self._info(fee)

    # ------------------------------------------------------------------ utilidades
    def _update_entry(self, dq: float, price: float) -> None:
        new_qty = self.qty + dq
        if self.qty == 0 or math.copysign(1, new_qty) != math.copysign(1, self.qty):
            self.avg_entry = price                      # posición nueva o inversión
        elif abs(new_qty) > abs(self.qty):              # aumenta en el mismo sentido
            self.avg_entry = (self.avg_entry * abs(self.qty) + price * abs(dq)) / abs(new_qty)

    def _info(self, fee: float) -> dict:
        return {
            "equity": self.equity, "weight": self.weight, "drawdown": 1 - self.equity / self.peak,
            "fees_paid": self.fees_paid, "n_trades": self.n_trades, "step_fee": fee,
            "timestamp": self.index[self.t],
        }

    def episode_stats(self) -> dict:
        return {
            "equity": np.array(self.equity_curve), "bench": np.array(self.bench_curve),
            "weights": np.array(self.weight_curve), "fees_paid": self.fees_paid,
            "slippage_paid": self.slippage_paid, "n_trades": self.n_trades, "turnover": self.turnover,
        }
