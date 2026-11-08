from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RISK_CODES = {
    "STOP_LOSS": ("Stop-loss", "El precio tocó el stop inicial (ATR). Se cierra la posición.", "hard"),
    "TRAILING_STOP": ("Trailing stop", "El precio retrocedió desde su máximo más de la distancia de trailing.", "hard"),
    "TAKE_PROFIT": ("Take-profit", "Se alcanzó el objetivo de beneficio (ejecutado al cierre de vela).", "hard"),
    "BREAKER_DAILY": ("Pérdida diaria", "Circuit breaker: pérdida del día ≥ límite; se cierra y se detiene.", "hard"),
    "BREAKER_DRAWDOWN": ("Drawdown máximo", "Circuit breaker: caída desde máximos ≥ límite; requiere reinicio manual.", "hard"),
    "BREAKER_DAILY_HALT": ("Pausa diaria", "Solo se permite reducir hasta el siguiente día UTC.", "hard"),
    "BREAKER_DRAWDOWN_HALT": ("Pausa por drawdown", "Solo se permite reducir hasta reinicio/cooldown.", "hard"),
    "BREAKER_STREAK_COOLDOWN": ("Racha de pérdidas", "Varias pérdidas seguidas: pausa temporal de entradas.", "hard"),
    "NO_SHORT": ("Sin cortos", "El mercado/config no permite posiciones cortas.", "hard"),
    "MAX_EXPOSURE": ("Exposición máxima", "Tope por símbolo, apalancamiento o cartera.", "hard"),
    "VAR_LIMIT": ("Límite de VaR", "El VaR/CVaR de la posición supera el límite; se recorta el tamaño.", "hard"),
    "KELLY_CAP": ("Tope Kelly", "Tamaño limitado por Kelly fraccional (o tope de arranque con pocas operaciones).", "entry"),
    "REENTRY_COOLDOWN": ("Espera tras stop", "No se reentra durante unos minutos tras un stop-loss.", "entry"),
    "SPREAD": ("Spread alto", "El spread del libro supera el máximo; no se aumenta riesgo.", "entry"),
    "PARTICIPATION": ("Participación", "La orden sería demasiado grande vs. el volumen de la vela.", "entry"),
    "MIN_NOTIONAL": ("Mínimo del exchange", "Orden por debajo del notional mínimo: se mantiene la posición.", "entry"),
    "SENTIMENT": ("Sentimiento", "Sentimiento contrario a la posición: se recorta el objetivo.", "overlay"),
}

_SEV = {"critical": 0, "warn": 1, "info": 2, "ok": 3}


@dataclass(frozen=True)
class DashboardPaths:
    live: Path
    risk: Path
    sentiment: Path
    models: Path

    @classmethod
    def default(cls) -> "DashboardPaths":
        from config.settings import ROOT_DIR
        d = ROOT_DIR / "data"
        return cls(d / "live", d / "risk", d / "sentiment", d / "models")


# ───────────────────────────────────────────────────────────── utilidades de lectura
def tail_lines(path: Path, n: int, chunk: int = 65536) -> list[str]:
    if not path.exists() or n <= 0:
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        pos, data = f.tell(), b""
        while pos > 0 and data.count(b"\n") <= n:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    return [ln.decode("utf-8", "replace") for ln in data.splitlines()[-n:]]


def read_jsonl(path: Path, n: int = 5000) -> list[dict]:
    out = []
    for ln in tail_lines(path, n):
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _num(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _parse_ts(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:
        return None


def max_drawdown(values: list[float]) -> tuple[float, float]:
    """(drawdown máximo, drawdown actual) como fracciones positivas."""
    peak, mdd = -math.inf, 0.0
    cur = 0.0
    for v in values:
        peak = max(peak, v)
        cur = 1.0 - v / peak if peak > 0 else 0.0
        mdd = max(mdd, cur)
    return mdd, cur


def trade_stats(trades: list[dict]) -> dict:
    pnl = [t["pnl_pct"] for t in trades if _num(t.get("pnl_pct")) is not None]
    if not pnl:
        return {"n": 0}
    wins, losses = [p for p in pnl if p > 0], [p for p in pnl if p <= 0]
    gross_loss = -sum(losses)
    return {"n": len(pnl), "win_rate": len(wins) / len(pnl), "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else None,
            "expectancy": sum(pnl) / len(pnl), "best": max(pnl), "worst": min(pnl), "total": sum(pnl)}


# ───────────────────────────────────────────────────────────── fachada
class DashboardData:
    """Lee los archivos que escribe el bot (solo lectura) y los convierte en JSON para el panel."""

    def __init__(self, paths: DashboardPaths) -> None:
        self.p = paths

    # ------------------------------------------------------------ estado general
    def status(self) -> dict | None:
        return read_json(self.p.live / "status.json")

    def health(self, st: dict | None, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        if not st:
            return {"state": "SIN_DATOS", "label": "Sin datos", "heartbeat_age": None}
        upd = _parse_ts(st.get("updated_at"))
        age = (now - upd).total_seconds() if upd else None
        if st.get("stopped_at"):
            return {"state": "DETENIDO", "label": "Detenido", "heartbeat_age": age}
        if age is None or age > 120:
            return {"state": "SIN_LATIDO", "label": "Sin latido (¿caído?)", "heartbeat_age": age}
        return {"state": "ACTIVO", "label": "Activo", "heartbeat_age": age}

    def overview(self, symbol: str | None = None) -> dict:
        st = self.status()
        now = datetime.now(timezone.utc)
        eq_rows = read_jsonl(self.p.live / "equity.jsonl", 20000)
        symbols = sorted({r.get("symbol") for r in eq_rows if r.get("symbol")} | set((st or {}).get("symbols", {})))
        sym = symbol if symbol in symbols else (symbols[0] if symbols else None)
        rows = [r for r in eq_rows if r.get("symbol") == sym]
        eq = [r["equity"] for r in rows if _num(r.get("equity")) is not None]
        kpi: dict = {}
        if eq:
            mdd, cur = max_drawdown(eq)
            today = now.strftime("%Y-%m-%d")
            day_eq = next((r["equity"] for r in rows if str(r.get("bar", ""))[:10] == today), eq[0])
            kpi = {"equity": eq[-1], "start_equity": eq[0], "return": eq[-1] / eq[0] - 1, "max_dd": mdd,
                   "cur_dd": cur, "pnl_today": eq[-1] / day_eq - 1 if day_eq else 0.0, "bars": len(eq)}
        trades = read_jsonl(self.p.live / "closed_trades.jsonl", 2000)
        breaker = read_json(self.p.risk / "breaker.json", {})
        kelly = read_json(self.p.risk / "trades.json", [])
        n_open = len((st or {}).get("open_positions", {}))
        return {
            "now": now.isoformat(), "health": self.health(st, now), "status": st, "symbols": symbols, "symbol": sym,
            "kpi": kpi, "trades": trade_stats(trades), "breaker": breaker, "kelly_trades": len(kelly),
            "open_positions": n_open, "halt": (self.p.live / "HALT").exists(),
            "diagnosis": self.diagnose(st, breaker, now),
        }

    # ------------------------------------------------------------ diagnóstico
    def diagnose(self, st: dict | None, breaker: dict | None, now: datetime) -> list[dict]:
        out: list[dict] = []

        def add(level: str, title: str, detail: str) -> None:
            out.append({"level": level, "title": title, "detail": detail})

        if (self.p.live / "HALT").exists():
            add("critical", "Kill-switch HALT activo", "El bot cerrará todo y se detendrá (o ya lo hizo). Elimina HALT antes de reiniciar.")
        if not st:
            add("info", "Aún no hay datos", "Arranca el bot (python main.py --paper) o usa --demo para ver el panel con datos de ejemplo.")
            return out
        h = self.health(st, now)
        if h["state"] == "SIN_LATIDO":
            add("critical", "El bot no da señales de vida", f"Último latido hace {h['heartbeat_age']:.0f} s. ¿Proceso caído? Revisa la pestaña Logs.")
        if h["state"] == "DETENIDO":
            add("warn", "Bot detenido", f"Se detuvo en {st.get('stopped_at')}. Los stops del exchange siguen activos si había posición.")
        if st.get("mode") == "DEMO":
            add("info", "Datos de demostración", "Todo lo que ves son datos sintéticos generados por --demo; no hay un bot real conectado.")
        if st.get("mode") == "LIVE":
            add("warn", "Modo LIVE", "Operando con dinero real.")
        if st.get("errors"):
            add("warn", f"{st['errors']} errores consecutivos", "Revisa Logs. Con demasiados errores el bot se detiene.")
        tf = st.get("tf_seconds") or 3600
        for sym, s in (st.get("symbols") or {}).items():
            bar = _parse_ts(s.get("bar"))
            if bar and h["state"] == "ACTIVO" and (now - bar).total_seconds() > tf * 1.5 + 120:
                add("warn", f"{sym}: sin velas nuevas", f"Última vela procesada hace {(now - bar).total_seconds() / 60:.0f} min: ¿stream caído?")
            if s.get("breaker") not in (None, "OK"):
                add("critical", f"{sym}: circuit breaker {s['breaker']}", "Solo se permite reducir riesgo. Reinicio manual: scripts/reset_breaker.py (con el bot detenido).")
            if s.get("entries_paused"):
                add("warn", f"{sym}: entradas pausadas por drift", "El monitor detectó un cambio de régimen; solo se permite reducir.")
            if s.get("stale"):
                add("warn", f"{sym}: señal tardía", "La última señal llegó fuera de plazo; solo se permitió reducir.")
            ex = s.get("last_exec") or {}
            if ex.get("status") in ("failed", "aborted", "partial"):
                add("warn", f"{sym}: ejecución {ex['status']}", ex.get("note") or "Revisa fills y logs.")
            if abs(s.get("qty") or 0) > 1e-9 and not s.get("protection"):
                add("warn", f"{sym}: posición sin stop en el exchange", "Solo protege el stop por cierre de vela (o los stops están desactivados).")
            for r in s.get("reasons") or []:
                code = r.get("code")
                if code in ("KELLY_CAP", "VAR_LIMIT", "SPREAD", "PARTICIPATION", "REENTRY_COOLDOWN", "SENTIMENT", "MIN_NOTIONAL", "NO_SHORT", "MAX_EXPOSURE"):
                    name = RISK_CODES.get(code, (code,))[0]
                    add("info", f"{sym}: {name}", f"Pedido {r['before']:+.2f} → {r['after']:+.2f}. {r.get('message', '')}")
        if not out:
            add("ok", "Todo en orden", "El bot está activo y sin alertas.")
        out.sort(key=lambda d: _SEV[d["level"]])
        return out

    # ------------------------------------------------------------ series
    def equity(self, symbol: str | None = None, limit: int = 4000) -> dict:
        rows = read_jsonl(self.p.live / "equity.jsonl", 40000)
        syms = sorted({r.get("symbol") for r in rows if r.get("symbol")})
        sym = symbol if symbol in syms else (syms[0] if syms else None)
        rows = [r for r in rows if r.get("symbol") == sym][-limit:]
        if not rows:
            return {"symbol": sym, "rows": []}
        peak, p0, e0 = -math.inf, rows[0].get("price") or 1.0, rows[0]["equity"]
        out = []
        for r in rows:
            peak = max(peak, r["equity"])
            out.append({"t": r["bar"], "equity": r["equity"], "dd": 1 - r["equity"] / peak, "weight": r.get("weight", 0.0),
                        "requested": r.get("requested", 0.0), "approved": r.get("approved", 0.0),
                        "price": r.get("price"), "bench": e0 * (r.get("price") or p0) / p0, "action": r.get("action"),
                        "codes": r.get("codes", []), "breaker": r.get("breaker", "OK")})
        return {"symbol": sym, "rows": out}

    def fills(self, limit: int = 300) -> list[dict]:
        return read_jsonl(self.p.live / "fills.jsonl", limit)[::-1]

    def closed_trades(self, limit: int = 300) -> list[dict]:
        return read_jsonl(self.p.live / "closed_trades.jsonl", limit)[::-1]

    def risk(self, limit: int = 300) -> dict:
        dec = read_jsonl(self.p.risk / "decisions.jsonl", limit)[::-1]
        rows = read_jsonl(self.p.live / "equity.jsonl", 20000)
        counts: dict[str, int] = {}
        for r in rows:
            for c in r.get("codes", []):
                counts[c] = counts.get(c, 0) + 1
        return {"decisions": dec, "counts": counts, "bars": len(rows), "kelly": read_json(self.p.risk / "trades.json", []),
                "breaker": read_json(self.p.risk / "breaker.json", {})}

    def drift(self, limit: int = 500) -> list[dict]:
        return read_jsonl(self.p.live / "drift.jsonl", limit)

    def sentiment(self, limit: int = 300) -> dict:
        out = {}
        if self.p.sentiment.exists():
            for f in sorted(self.p.sentiment.glob("*.jsonl")):
                out[f.stem] = read_jsonl(f, limit)
        return out

    def logs(self, n: int = 400, level: str | None = None, q: str | None = None) -> list[dict]:
        pat = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| (\w+)\s*\| ([^|]*)\| (.*)$")
        out: list[dict] = []
        for ln in tail_lines(self.p.live / "bot.log", n * 3 if (level or q) else n):
            m = pat.match(ln)
            rec = ({"t": m[1], "level": m[2], "logger": m[3].strip(), "msg": m[4]} if m
                   else {"t": "", "level": "CONT", "logger": "", "msg": ln})
            if level and rec["level"] not in level.split(","):
                if not (rec["level"] == "CONT" and out):          # las líneas de traceback siguen al registro anterior
                    continue
            if q and q.lower() not in ln.lower():
                continue
            out.append(rec)
        return out[-n:]

    def models(self) -> list[dict]:
        out = []
        if self.p.models.exists():
            for d in sorted(self.p.models.iterdir(), reverse=True):
                rep = read_json(d / "report.json") if d.is_dir() else None
                if rep:
                    out.append({"run": d.name, "symbol": rep.get("symbol"), "algo": rep.get("algo"),
                                "timesteps": rep.get("timesteps"), "split": rep.get("split"),
                                "val": rep.get("val", {}), "test": rep.get("test", {}),
                                "best_val_step": rep.get("best_val_step"),
                                "history": [{"t": h.get("timesteps"), "sharpe": h.get("sharpe")}
                                            for h in rep.get("validation_history", [])]})
        return out[:30]

    def config(self) -> dict:
        """Configuración EFECTIVA (defaults + .env). No incluye credenciales."""
        groups: dict[str, dict] = {}

        def add(name: str, fn) -> None:
            try:
                obj = fn()
                groups[name] = {k: (str(v) if isinstance(v, Path) else v)
                                for k, v in dataclasses.asdict(obj).items() if k not in ("api_key", "api_secret")}
            except Exception as exc:
                groups[name] = {"error": f"{type(exc).__name__}: {exc}"}
        from config.live_settings import DriftSettings, ExecutionSettings, LiveSettings
        from config.risk_settings import RiskSettings
        from config.rl_settings import RLSettings
        from config.settings import Settings
        add("Mercado", Settings.from_env)
        add("Riesgo (F4)", RiskSettings.from_env)
        add("RL (F3)", RLSettings.from_env)
        add("Ejecución (F5)", ExecutionSettings.from_env)
        add("Drift (F5)", DriftSettings.from_env)
        add("Live (F5)", LiveSettings.from_env)
        return groups

    def meta(self) -> dict:
        return {"risk_codes": {k: {"name": v[0], "desc": v[1], "kind": v[2]} for k, v in RISK_CODES.items()}}
