from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.rl_settings import RLSettings
from src.data.features import FeatureEngineer
from src.rl.env import TradingEnv
from src.rl.metrics import compute_metrics
from src.rl.splits import DataSplit, FeatureScaler, temporal_split

log = logging.getLogger(__name__)


def load_features(symbol: str, timeframe: str, processed_dir: Path, raw_dir: Path) -> pd.DataFrame:
    """Carga features de F1; si no existen, las calcula desde el OHLCV crudo."""
    tag = symbol.replace("/", "")
    fpath = processed_dir / f"{tag}_{timeframe}_features.parquet"
    if fpath.exists():
        features = pd.read_parquet(fpath)
        _validate_training_data(features, symbol)
        return features
    rpath = raw_dir / f"{tag}_{timeframe}.parquet"
    if not rpath.exists():
        raise FileNotFoundError(f"No hay datos para {symbol}: ejecuta scripts/download_data.py ({rpath})")
    log.info("Features no encontradas; calculando desde %s", rpath)
    features = FeatureEngineer().transform(pd.read_parquet(rpath))
    _validate_training_data(features, symbol)
    return features


def _validate_training_data(data: pd.DataFrame, symbol: str) -> None:
    required = {"open", "high", "low", "close", "volume", *FeatureEngineer.FEATURE_COLUMNS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Datos de entrenamiento incompletos para {symbol}; faltan columnas: {missing}.")
    if len(data) < 200:
        raise ValueError(f"Datos insuficientes para {symbol}: {len(data)} filas; se requieren al menos 200.")
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError(f"El índice de entrenamiento de {symbol} debe ser DatetimeIndex con zona horaria.")
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError(f"Los datos de entrenamiento de {symbol} deben estar ordenados y sin timestamps duplicados.")
    values = data[list(required)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Los datos de entrenamiento de {symbol} contienen NaN o valores no finitos.")
    if (data[["open", "high", "low", "close"]] <= 0).any().any() or (data["volume"] < 0).any():
        raise ValueError(f"Los datos OHLCV de {symbol} contienen precios inválidos o volumen negativo.")


def evaluate_policy(model, env: TradingEnv, timeframe: str, deterministic: bool = True) -> dict:
    """Recorre la serie completa una vez (sin inicio aleatorio) y devuelve métricas netas de costes."""
    obs, _ = env.reset(options={"start": env.window - 1})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    s = env.episode_stats()
    return compute_metrics(s["equity"], s["bench"], timeframe, s["fees_paid"], s["n_trades"], s["turnover"])


def _make_algo(name: str):
    from stable_baselines3 import A2C, PPO, SAC
    return {"ppo": PPO, "a2c": A2C, "sac": SAC}[name]


@dataclass
class TrainResult:
    run_dir: Path
    val_metrics: dict
    test_metrics: dict
    split_summary: str


def train(df: pd.DataFrame, cfg: RLSettings, timeframe: str, symbol: str,
          total_timesteps: int | None = None, eval_freq: int | None = None) -> TrainResult:
    """Entrena un agente con selección de modelo en VALIDACIÓN; el TEST se evalúa una sola vez al final."""
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv

    _validate_training_data(df, symbol)
    steps = total_timesteps or cfg.total_timesteps
    set_random_seed(cfg.seed)
    split: DataSplit = temporal_split(df, cfg.train_frac, cfg.val_frac, embargo=cfg.window)
    scaler = FeatureScaler(FeatureEngineer.FEATURE_COLUMNS).fit(split.train)   # SOLO train
    reference = scaler.transform(split.train)
    if len(reference) > 5000:
        reference = reference[np.linspace(0, len(reference) - 1, 5000, dtype=int)]
    log.info("Split: %s", split.summary())
    log.info("Entrenamiento: %d barras; validación: %d; test fuera de muestra: %d",
             len(split.train), len(split.val), len(split.test))

    run_dir = cfg.models_dir / f"{symbol.replace('/', '')}_{timeframe}_{cfg.algo}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(run_dir / "scaler.json")
    np.save(run_dir / "drift_reference.npy", reference)

    def make_train_env(rank: int):
        def _f():
            e = TradingEnv(split.train, scaler, cfg, random_start=True)
            e = Monitor(e)
            e.reset(seed=cfg.seed + rank)
            return e
        return _f

    n_envs = 1 if cfg.algo == "sac" else cfg.n_envs
    train_env = DummyVecEnv([make_train_env(i) for i in range(n_envs)])
    val_env = TradingEnv(split.val, scaler, cfg, random_start=False)
    test_env = TradingEnv(split.test, scaler, cfg, random_start=False)

    Algo = _make_algo(cfg.algo)
    model = Algo("MlpPolicy", train_env, seed=cfg.seed, verbose=0, **_hyperparams(cfg.algo, n_envs))

    best = {"score": -np.inf, "step": 0}
    history: list[dict] = []
    freq = eval_freq or max(steps // 10, 1)

    class ValidationCallback(BaseCallback):
        """Cada `freq` pasos evalúa en validación y guarda el mejor por Sharpe neto de costes."""
        def _on_step(self) -> bool:
            if self.n_calls % max(freq // n_envs, 1) == 0:
                m = evaluate_policy(self.model, val_env, timeframe)
                history.append({"timesteps": self.num_timesteps, **m})
                log.info("[val] t=%d sharpe=%.2f ret=%.2f%% dd=%.2f%% trades=%d",
                         self.num_timesteps, m["sharpe"], 100 * m["total_return"],
                         100 * m["max_drawdown"], m["n_trades"])
                if m["sharpe"] > best["score"]:
                    best.update(score=m["sharpe"], step=self.num_timesteps)
                    self.model.save(run_dir / "best_model")
            return True

    model.learn(total_timesteps=steps, callback=ValidationCallback())
    model.save(run_dir / "last_model")
    if not (run_dir / "best_model.zip").exists():
        model.save(run_dir / "best_model")

    best_model = Algo.load(run_dir / "best_model")
    val_m = evaluate_policy(best_model, val_env, timeframe)
    test_m = evaluate_policy(best_model, test_env, timeframe)       # única evaluación sobre test

    (run_dir / "report.json").write_text(json.dumps({
        "symbol": symbol, "timeframe": timeframe, "algo": cfg.algo, "timesteps": steps,
        "best_val_step": best["step"], "split": split.summary(),
        "data": {
            "rows": len(df),
            "train_rows": len(split.train),
            "validation_rows": len(split.val),
            "test_rows": len(split.test),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "drift_reference_rows": len(reference),
            "features": list(scaler.columns),
        },
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in _cfg_dict(cfg).items()},
        "validation_history": history, "val": val_m, "test": test_m,
    }, indent=2, default=float))
    return TrainResult(run_dir, val_m, test_m, split.summary())


def _hyperparams(algo: str, n_envs: int) -> dict:
    if algo == "ppo":
        return dict(learning_rate=3e-4, n_steps=1024, batch_size=256, gamma=0.995,
                    gae_lambda=0.95, ent_coef=0.005, clip_range=0.2, n_epochs=10,
                    policy_kwargs=dict(net_arch=[64, 64]))
    if algo == "a2c":
        return dict(learning_rate=7e-4, n_steps=64, gamma=0.995, ent_coef=0.005,
                    policy_kwargs=dict(net_arch=[64, 64]))
    return dict(learning_rate=3e-4, buffer_size=200_000, batch_size=256, gamma=0.995,
                learning_starts=2000, policy_kwargs=dict(net_arch=[64, 64]))


def _cfg_dict(cfg: RLSettings) -> dict:
    from dataclasses import asdict
    return asdict(cfg)
