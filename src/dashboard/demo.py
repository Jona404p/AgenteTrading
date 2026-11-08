"""Datos de demostración con los MISMOS esquemas que escribe el bot (para ver el panel sin operar)."""
from __future__ import annotations

import json
import math
import random
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.dashboard.data import DashboardPaths

SYM = "BTC/USDT"


def _w(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def generate_demo(root: Path, bars: int = 720, seed: int = 1, mode: str = "DEMO") -> DashboardPaths:
    rnd = random.Random(seed)
    paths = DashboardPaths(root / "live", root / "risk", root / "sentiment", root / "models")
    for d in (paths.live, paths.risk, paths.sentiment, paths.models):
        d.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t0 = end - timedelta(hours=bars)

    price, equity, qty, entry, stop, tp, extreme = 62000.0, 10_000.0, 0.0, 0.0, 0.0, 0.0, 0.0
    peak, trend = equity, 0.0004
    eq_rows, fills, closed, decisions, kelly, drift, logs = [], [], [], [], [], [], []
    cooldown_until = -1
    regime = 0.0
    for i in range(bars):
        ts = t0 + timedelta(hours=i + 1)
        if i % 120 == 0:
            regime = rnd.choice([0.0009, -0.0004, 0.0004])
        ret = regime + rnd.gauss(0, 0.006)
        new_price = price * math.exp(ret)
        hi, lo = max(price, new_price) * 1.002, min(price, new_price) * 0.998
        w_before = qty * price / equity
        equity += qty * (new_price - price)
        price = new_price
        bar = ts.isoformat()
        codes: list[str] = []
        action = "ALLOW"
        requested = round(min(1.0, max(0.0, 0.5 + 6 * ret + rnd.gauss(0, 0.3))), 3)
        approved = requested

        if qty > 0:                                           # stops
            extreme = max(extreme, hi)
            stop = max(stop, extreme - 2.5 * entry * 0.012)
            hit = "stop_loss" if lo <= stop else ("take_profit" if hi >= tp else None)
            if hit:
                exit_px = stop if hit == "stop_loss" else tp
                pnl = exit_px / entry - 1 - 0.002
                closed.append({"bar": bar, "symbol": SYM, "side": 1, "entry": entry, "exit": exit_px, "qty": qty,
                               "pnl_pct": pnl, "reason": "trailing_stop" if hit == "stop_loss" and stop > entry else hit})
                kelly.append(pnl)
                fills.append({"bar": bar, "symbol": SYM, "side": "sell", "qty": qty, "price": exit_px, "fee_quote": qty * exit_px * 0.001,
                              "fee_base": 0, "order_id": f"d{i}", "method": "exchange_stop", "status": "filled", "codes": [hit.upper()]})
                equity += qty * (exit_px - price) - qty * exit_px * 0.001
                qty = 0.0
                codes = [hit.upper()]
                action, approved = "FLATTEN", 0.0
                cooldown_until = i + 2
            else:                                             # posición mantenida: el gestor la recorta a Kelly
                approved = min(requested, 0.5)
                codes = ["KELLY_CAP"] if requested > 0.5 else []
                action = "REDUCE" if codes else "ALLOW"
                logs.append((ts, "WARNING", "src.live.orchestrator", f"[stop] {SYM} ejecutado en el exchange ({hit})"))
        elif i > cooldown_until and requested > 0.45 and rnd.random() < 0.35:
            approved = min(requested, 0.5)
            codes = ["KELLY_CAP"] if requested > 0.5 else []
            action = "REDUCE" if codes else "ALLOW"
            qty = approved * equity / price
            entry, stop, tp, extreme = price * 1.0003, price * (1 - 2 * 0.012), price * (1 + 4 * 0.012), price
            fills.append({"bar": bar, "symbol": SYM, "side": "buy", "qty": qty, "price": entry, "fee_quote": qty * entry * 0.001, "fee_base": 0,
                          "order_id": f"d{i}", "method": "twap", "status": "filled", "requested_weight": requested,
                          "approved_weight": approved, "codes": codes})
            equity -= qty * entry * 0.001
            logs.append((ts, "INFO", "src.live.orchestrator", f"[exec] twap buy {SYM} {qty:.6f} @ {entry:.2f} (filled, 4 órdenes)"))
        else:
            if i <= cooldown_until and requested > 0.2:
                codes, action, approved = ["REENTRY_COOLDOWN"], "BLOCK", 0.0
            elif rnd.random() < 0.06:
                codes, action, approved = ["SPREAD"], "BLOCK", 0.0
            elif qty > 0 and requested > 0.5:
                codes, action, approved = ["KELLY_CAP"], "BLOCK", w_before
            elif qty == 0:
                approved = 0.0 if requested < 0.45 else requested
        peak = max(peak, equity)
        weight = qty * price / equity
        eq_rows.append({"bar": bar, "symbol": SYM, "equity": equity, "weight": weight, "requested": requested,
                        "approved": approved if action != "FLATTEN" else 0.0, "action": action, "codes": codes,
                        "breaker": "OK", "price": price})
        if codes:
            decisions.append({"timestamp": bar, "symbol": SYM, "action": action, "requested": requested,
                              "approved": approved, "current": w_before,
                              "reasons": [{"code": c, "msg": "demo", "before": requested, "after": approved} for c in codes],
                              "stop": stop or None, "take_profit": tp or None, "breaker": "OK",
                              "kelly_cap": 0.5, "var_cap": 0.62, "var_unit": 0.048})
        if i % 6 == 5:
            lvl = "OK" if (i // 6) % 9 not in (4, 5) else "WARN"
            nd = 1 if lvl == "OK" else 5
            drift.append({"bar": bar, "symbol": SYM, "level": lvl, "n_drifted": nd, "n_features": 14,
                          "drifted": ["volatility", "atr_pct", "bb_bandwidth", "hl_range_pct", "volume_zscore"][:nd],
                          "reasons": ["features desplazadas %d/14" % nd] if nd > 3 else [], "tstat": rnd.gauss(0.5, 1), "drawdown": 0.02})
        if i % 24 == 0:
            logs.append((ts, "INFO", "src.live.orchestrator", f"[bar] {SYM} close={price:.2f} a={requested:+.2f} req={requested:+.2f} → {approved:+.2f} {action} {','.join(codes)} | equity={equity:.2f} w={weight:+.2f}"))
    logs.append((end, "WARNING", "src.data.streams", "WebSocket caído (ConnectionClosed). Reintento en 2s"))
    logs.append((end, "ERROR", "src.live.orchestrator", "[drift] BTC/USDT WARN feats 5/14 features desplazadas 5/14"))

    _w(paths.live / "equity.jsonl", eq_rows)
    _w(paths.live / "fills.jsonl", fills)
    _w(paths.live / "closed_trades.jsonl", closed)
    _w(paths.live / "drift.jsonl", drift)
    _w(paths.risk / "decisions.jsonl", decisions)
    (paths.risk / "trades.json").write_text(json.dumps(kelly))
    (paths.risk / "breaker.json").write_text(json.dumps({"state": "OK", "peak_equity": peak, "day": end.strftime("%Y-%m-%d"),
                                                           "day_start_equity": eq_rows[-24]["equity"], "halted_at": None, "cooldown_until": None}))
    (paths.live / "bot.log").write_text("".join(
        f"{t:%Y-%m-%d %H:%M:%S} | {lv:<8} | {lg} | {m}\n" for t, lv, lg, m in sorted(logs, key=lambda x: x[0])[-300:]))
    _w(paths.sentiment / "BTC.jsonl", [{
        "timestamp": (end - timedelta(minutes=15 * k)).isoformat(), "asset": "BTC",
        "score": 50 + 22 * math.sin(k / 7) + rnd.gauss(0, 4), "confidence": min(1, 0.4 + rnd.random() * 0.6),
        "n_headlines": 14, "n_scored": 12, "fear_greed": 58, "degraded": False,
        "drivers": ["ETF inflows hit weekly high as institutions return", "Fed minutes signal patience on rate path"]}
        for k in range(120, -1, -1)])
    run = paths.models / "BTCUSDT_1h_ppo_20260401_120000"
    run.mkdir(exist_ok=True)
    (run / "report.json").write_text(json.dumps({
        "symbol": SYM, "timeframe": "1h", "algo": "ppo", "timesteps": 300000, "best_val_step": 210000,
        "split": "train: 6132 velas | val: 1314 velas | test: 1314 velas",
        "val": {"sharpe": 1.42, "total_return": 0.113, "max_drawdown": -0.061, "n_trades": 74, "fees_paid": 96.4},
        "test": {"sharpe": 0.87, "total_return": 0.058, "max_drawdown": -0.083, "n_trades": 69, "fees_paid": 88.1},
        "validation_history": [{"timesteps": 30000 * k, "sharpe": 0.2 + 0.14 * k - 0.01 * k * k / 2} for k in range(1, 11)]}))

    last = eq_rows[-1]
    now = datetime.now(timezone.utc)
    (paths.live / "status.json").write_text(json.dumps({
        "mode": mode, "timeframe": "1h", "tf_seconds": 3600, "errors": 0, "queue": 0, "pid": 0,
        "started_at": (now - timedelta(hours=bars)).isoformat(), "updated_at": now.isoformat(), "stopped_at": None,
        "halt_file": False, "open_positions": {SYM: qty} if qty else {},
        "sentiment": {"BTC": {"score": 63.2, "confidence": 0.74, "degraded": False, "n_scored": 12, "age_min": 6.0}},
        "symbols": {SYM: {
            "bar": last["bar"], "price": last["price"], "atr_pct": 0.0118, "agent_action": last["requested"], "requested": last["requested"],
            "approved": last["approved"], "action": last["action"], "codes": last["codes"],
            "reasons": [{"code": c, "message": "Kelly fraccional (bootstrap) → máx 0.50" if c == "KELLY_CAP" else "demo",
                         "before": last["requested"], "after": last["approved"]} for c in last["codes"]],
            "equity": last["equity"], "weight_before": last["weight"], "weight_after": last["weight"], "qty": qty,
            "entry": entry or None, "stop": stop or None, "take_profit": tp or None, "kelly_cap": 0.5, "var_cap": 0.62,
            "var_unit": 0.048, "spread_bps": 1.4, "breaker": "OK", "stale": False,
            "protection": {"id": "demo-1", "stop": stop, "initial": stop, "qty": qty} if qty else None,
            "last_exec": {"method": "twap", "side": "buy", "status": "filled", "filled": qty, "avg_price": entry,
                          "orders": 4, "fee_quote": 1.2, "note": "", "bar": last["bar"]},
            "entries_paused": False,
            "drift": {"level": drift[-1]["level"], "n_drifted": drift[-1]["n_drifted"], "n_features": 14,
                      "drifted": drift[-1]["drifted"], "reasons": drift[-1]["reasons"], "tstat": 0.4, "drawdown": 0.02},
            "model": {"run": run.name, "window": 8, "n_features": 14, "val": {"sharpe": 1.42, "max_drawdown": -0.061}}}}}))
    return paths


class DemoHeartbeat(threading.Thread):
    """Mantiene `updated_at` fresco para que el panel muestre el bot de demo como ACTIVO."""

    def __init__(self, paths: DashboardPaths) -> None:
        super().__init__(daemon=True, name="demo-heartbeat")
        self.path, self._stop_ev = paths.live / "status.json", threading.Event()

    def run(self) -> None:
        while not self._stop_ev.wait(5):
            try:
                st = json.loads(self.path.read_text())
                st["updated_at"] = datetime.now(timezone.utc).isoformat()
                st["halt_file"] = (self.path.parent / "HALT").exists()
                self.path.write_text(json.dumps(st))
            except (OSError, ValueError):
                pass

    def stop(self) -> None:
        self._stop_ev.set()
