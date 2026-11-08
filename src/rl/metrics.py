from __future__ import annotations

import numpy as np

_TF_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def bars_per_year(timeframe: str) -> float:
    unit, n = timeframe[-1], int(timeframe[:-1])
    return 365 * 86400 / (n * _TF_SECONDS[unit])      # cripto opera 24/7


def compute_metrics(equity: np.ndarray, bench_equity: np.ndarray | None, timeframe: str,
                    fees_paid: float = 0.0, n_trades: int = 0, turnover: float = 0.0) -> dict:
    equity = np.asarray(equity, dtype=float)
    ppy = bars_per_year(timeframe)
    rets = np.diff(equity) / equity[:-1]
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    neg = rets[rets < 0]
    dstd = np.sqrt((neg ** 2).mean()) if len(neg) else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min())
    total = equity[-1] / equity[0] - 1
    years = len(rets) / ppy
    out = {
        "total_return": float(total),
        "cagr": float((equity[-1] / equity[0]) ** (1 / years) - 1) if years > 0 and equity[-1] > 0 else float("nan"),
        "sharpe": float(rets.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0,
        "sortino": float(rets.mean() / dstd * np.sqrt(ppy)) if dstd > 0 else 0.0,
        "max_drawdown": max_dd,
        "calmar": float(total / abs(max_dd)) if max_dd < 0 else 0.0,
        "fees_paid": float(fees_paid),
        "n_trades": int(n_trades),
        "turnover": float(turnover),
        "bars": int(len(equity)),
    }
    if bench_equity is not None and len(bench_equity) == len(equity):
        b = np.asarray(bench_equity, dtype=float)
        out["buy_hold_return"] = float(b[-1] / b[0] - 1)
        out["excess_return"] = float(total - out["buy_hold_return"])
    return out
