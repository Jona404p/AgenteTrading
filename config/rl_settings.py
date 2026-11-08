from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config.settings import ROOT_DIR, MarketType, Settings


def _f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _b(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class CostSettings:
    """Comisiones y deslizamiento. Valores en puntos base (1 bps = 0.01 %)."""
    taker_fee_bps: float = 10.0       # Spot Binance regular: 0.10 %
    maker_fee_bps: float = 10.0       # Spot Binance regular: 0.10 %
    fee_discount: float = 0.0         # p.ej. 0.25 si pagas comisiones con BNB (Spot)
    base_slippage_bps: float = 1.0    # deslizamiento fijo por ejecución (medio spread + latencia)
    impact_coef_bps: float = 10.0     # impacto: coef * sqrt(notional / volumen_en_USDT_de_la_vela)

    @property
    def taker_rate(self) -> float:
        return self.taker_fee_bps / 1e4 * (1.0 - self.fee_discount)

    @property
    def maker_rate(self) -> float:
        return self.maker_fee_bps / 1e4 * (1.0 - self.fee_discount)


@dataclass(frozen=True)
class RLSettings:
    costs: CostSettings
    allow_short: bool
    max_leverage: float
    initial_cash: float
    window: int                  # velas de historia que ve el agente
    min_rebalance: float         # banda muerta: cambios de exposición menores se ignoran
    reward_scale: float
    drawdown_penalty: float      # penalización por aumento del drawdown
    turnover_penalty: float      # penalización extra por rotación (además de las comisiones)
    ruin_equity_frac: float      # termina el episodio si el equity cae por debajo de esta fracción
    episode_len: int             # longitud de episodio en entrenamiento (0 = serie completa)
    train_frac: float
    val_frac: float
    algo: str                    # ppo | sac | a2c
    total_timesteps: int
    seed: int
    n_envs: int
    models_dir: Path

    @classmethod
    def from_env(cls, market_type: MarketType | None = None) -> "RLSettings":
        market = market_type or Settings.from_env().market_type
        futures = market is MarketType.FUTURE
        # Defaults de comisión según mercado: Futures USDT-M 0.05 % taker / 0.02 % maker.
        costs = CostSettings(
            taker_fee_bps=_f("RL_TAKER_FEE_BPS", "5" if futures else "10"),
            maker_fee_bps=_f("RL_MAKER_FEE_BPS", "2" if futures else "10"),
            fee_discount=_f("RL_FEE_DISCOUNT", "0"),
            base_slippage_bps=_f("RL_BASE_SLIPPAGE_BPS", "1"),
            impact_coef_bps=_f("RL_IMPACT_COEF_BPS", "10"),
        )
        models_dir = ROOT_DIR / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            costs=costs,
            allow_short=_b("RL_ALLOW_SHORT", "true" if futures else "false") and futures,
            max_leverage=_f("RL_MAX_LEVERAGE", "1.0"),
            initial_cash=_f("RL_INITIAL_CASH", "10000"),
            window=_i("RL_WINDOW", "8"),
            min_rebalance=_f("RL_MIN_REBALANCE", "0.05"),
            reward_scale=_f("RL_REWARD_SCALE", "100"),
            drawdown_penalty=_f("RL_DRAWDOWN_PENALTY", "0.5"),
            turnover_penalty=_f("RL_TURNOVER_PENALTY", "0.0"),
            ruin_equity_frac=_f("RL_RUIN_EQUITY_FRAC", "0.5"),
            episode_len=_i("RL_EPISODE_LEN", "720"),
            train_frac=_f("RL_TRAIN_FRAC", "0.70"),
            val_frac=_f("RL_VAL_FRAC", "0.15"),
            algo=os.getenv("RL_ALGO", "ppo").strip().lower(),
            total_timesteps=_i("RL_TOTAL_TIMESTEPS", "300000"),
            seed=_i("RL_SEED", "42"),
            n_envs=_i("RL_N_ENVS", "4"),
            models_dir=models_dir,
        )
