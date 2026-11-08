"""Agente de trading en bucle cerrado (Fase 5).

    python main.py --paper --dashboard    # datos públicos, órdenes simuladas, panel local
    python main.py --check --paper        # valida config, modelo y warmup; no opera
    touch data/live/HALT                  # kill-switch: en la próxima vela cierra todo y se detiene
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from config.live_settings import LiveSettings
from config.risk_settings import RiskSettings, bars_per_day
from config.rl_settings import RLSettings
from config.settings import BinanceEnv, Settings
from src.core.events import EventBus
from src.core.logger import setup_logging
from src.dashboard import DashboardPaths, DashboardServer
from src.data.buffer import MarketDataBuffer
from src.data.exchange_factory import ExchangeFactory
from src.data.features import FeatureEngineer
from src.data.historical import HistoricalDataDownloader
from src.data.streams import BinanceWebSocketClient
from src.execution import OrderExecutor, PaperGateway
from src.live import (ModelBundle, ModelDriftMonitor, StateStore, TradeLog, TradingOrchestrator,
                      book_or_last_close, find_model_dir)
from src.risk import RiskManager
from src.rl.training import load_features

log = logging.getLogger("main")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper", action="store_true", help="órdenes simuladas con datos reales (no necesita API keys)")
    p.add_argument("--check", action="store_true", help="valida todo y sale sin operar")
    p.add_argument("--no-sentiment", action="store_true", help="no arranca el SentimentService (LLM)")
    p.add_argument("--symbols", default=None, help="subconjunto de SYMBOLS, separado por comas")
    p.add_argument("--dashboard", action="store_true", help="levanta el panel web en este mismo proceso (http://localhost:8765)")
    p.add_argument("--manual-approval", action="store_true",
                   help="en PAPER, exige aprobar cada orden desde --dashboard")
    p.add_argument("--dashboard-port", type=int, default=None, help="puerto del panel (por defecto DASHBOARD_PORT o 8765)")
    p.add_argument("--flatten-on-exit", action="store_true", help="cierra posiciones al apagar (por defecto quedan con su stop)")
    return p.parse_args()


async def _fetch_closed(loader: HistoricalDataDownloader, symbol: str, timeframe: str, since_ms: int,
                        tf: pd.Timedelta) -> pd.DataFrame:
    df = await loader.fetch_range(symbol, timeframe, since_ms)
    if not df.empty and df.index[-1] + tf > pd.Timestamp.now(tz="UTC"):
        df = df.iloc[:-1]                                   # la última puede estar abierta
    return df


def _build_drift(sym: str, bundle: ModelBundle, settings: Settings, rl: RLSettings, live: LiveSettings):
    ref = bundle.drift_reference
    if ref is None:
        try:
            feats = load_features(sym, settings.timeframe, settings.processed_dir, settings.raw_dir)
            ref = bundle.scaler.transform(feats.iloc[: int(len(feats) * rl.train_frac)])
            log.warning("%s: el modelo no incluye referencia de drift; se reconstruyó del histórico local", sym)
        except FileNotFoundError:
            log.warning("%s: no hay referencia para detectar drift", sym)
    val_dd = ((bundle.report.get("val") or {}).get("max_drawdown"))
    return ModelDriftMonitor(live.drift, bundle.scaler.columns, ref, val_dd)


async def amain(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    live = LiveSettings.from_env()
    setup_logging(settings.log_level, live.state_dir / "bot.log")
    if args.flatten_on_exit:
        live = replace(live, flatten_on_exit=True)
    rl = RLSettings.from_env(settings.market_type)
    risk_cfg = RiskSettings.from_env(settings.market_type, settings.timeframe)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else list(settings.symbols)
    if risk_cfg.allow_short != rl.allow_short:
        log.warning("RISK_ALLOW_SHORT (%s) ≠ RL_ALLOW_SHORT (%s): el RiskManager recortará lo que el agente no debería pedir",
                    risk_cfg.allow_short, rl.allow_short)
    mode = "PAPER" if args.paper else settings.env.value.upper()
    banner = f"MODO {mode} | {settings.market_type.value} | {symbols} | {settings.timeframe}"
    (log.critical if mode == "LIVE" else log.warning)("⚠️  %s%s", banner, " — DINERO REAL" if mode == "LIVE" else "")
    if not args.paper:
        raise RuntimeError(
            "Este checkout solo dispone de ejecución simulada (--paper); "
            "no hay gateway que envíe órdenes reales. No se creó un cliente privado."
        )
    if args.manual_approval and not args.dashboard:
        raise ValueError("--manual-approval requiere --dashboard para aprobar o rechazar propuestas.")
    # --- modelos (uno por símbolo) ---
    models: dict[str, ModelBundle] = {}
    for sym in symbols:
        run_dir = find_model_dir(sym, settings.timeframe, live.models_root)
        models[sym] = ModelBundle.load(sym, run_dir)
        log.info("Modelo %s → %s (ventana=%d, %d features)", sym, run_dir.name, models[sym].window, models[sym].n_features)

    # --- datos ---
    bus = EventBus()
    buffer = MarketDataBuffer(bus, max_len=max(1500, live.warmup_candles + 200))   # ANTES del orquestador
    tf = pd.Timedelta(seconds=86400 / bars_per_day(settings.timeframe))
    data_settings = replace(settings, env=BinanceEnv.LIVE) if args.paper else settings
    use_public = args.paper or settings.historical_from_live
    trading_client = None if args.paper else ExchangeFactory.create_trading_client(settings)
    data_client = ExchangeFactory.create_public_data_client(settings) if use_public else trading_client
    loader = HistoricalDataDownloader(data_client, settings)
    since = int((datetime.now(timezone.utc) - tf * live.warmup_candles).timestamp() * 1000)
    for sym in symbols:
        df = await _fetch_closed(loader, sym, settings.timeframe, since, tf)
        if df.empty:
            raise RuntimeError(f"Sin velas de warmup para {sym}")
        buffer.upsert(sym, df)
        log.info("Warmup %s: %d velas hasta %s", sym, len(df), df.index[-1])

    async def backfill(sym: str, since_ms: int) -> pd.DataFrame:
        return await _fetch_closed(loader, sym, settings.timeframe, since_ms, tf)

    # --- ejecución ---
    gateway = PaperGateway(lambda s: book_or_last_close(buffer, s), live.paper_initial_cash, rl.costs,
                            allow_short=rl.allow_short)
    if args.manual_approval:
        log.warning("Aprobación manual activa: no se simularán órdenes sin autorización desde el panel.")
    executor = OrderExecutor(gateway, live.execution)

    risk = RiskManager(risk_cfg, bus=bus, persist=True, require_exchange_state=not args.paper)
    drift = {s: _build_drift(s, models[s], settings, rl, live) for s in symbols}
    orch = TradingOrchestrator(
        symbols=symbols, timeframe=settings.timeframe, live=live, rl=rl, risk=risk, buffer=buffer,
        engineer=FeatureEngineer(), models=models, gateway=gateway, executor=executor, bus=bus,
        trade_log=TradeLog(live.state_dir), store=StateStore(live.state_dir / ("state_paper.json" if args.paper else "state.json")),
        drift=drift, backfill=backfill, mode=mode, sentiment_enabled=not args.no_sentiment,
        manual_approval=args.manual_approval)
    orch.start()
    await orch.initialize()

    if args.check:
        eq, _ = await orch._account(symbols[0], orch._last_price(symbols[0]))
        print(f"\nCHECK OK | {banner} | equity={eq:.2f} | breaker={risk.breaker.state.value} | "
              f"kelly_trades={len(risk.stats.to_list())}")
        await _close_all(data_client, trading_client)
        return

    # --- sentimiento en paralelo (opcional) ---
    service = None
    if not args.no_sentiment:
        try:
            from config.llm_settings import LLMSettings, SentimentSettings
            from src.sentiment.analyzer import SentimentEngine
            from src.sentiment.llm_factory import LLMFactory
            from src.sentiment.news_sources import NewsAggregator
            from src.sentiment.service import SentimentService
            from src.sentiment.store import SentimentStore
            s_cfg = SentimentSettings.from_env()
            engine = SentimentEngine(LLMFactory.create(LLMSettings.from_env()), NewsAggregator(s_cfg), s_cfg,
                                     SentimentStore(s_cfg.store_dir))
            service = SentimentService(engine, bus, sorted({s.split("/")[0] for s in symbols}), s_cfg.refresh_minutes)
        except Exception:
            log.exception("SentimentService no disponible: se opera sin overlay de sentimiento")

    ws = BinanceWebSocketClient(data_settings, bus, symbols=symbols, timeframe=settings.timeframe, klines=True, depth=True)
    dash = None
    loop = asyncio.get_running_loop()
    if args.dashboard:
        try:
            def approval_handler(proposal_id: str, approve: bool):
                if not args.manual_approval:
                    raise ValueError("Aprobación manual no está habilitada en este proceso.")
                future = asyncio.run_coroutine_threadsafe(
                    orch.handle_manual_approval(proposal_id, approve), loop
                )
                return future.result()

            dash = DashboardServer(DashboardPaths.default(), os.getenv("DASHBOARD_HOST", "127.0.0.1"),
                                   args.dashboard_port or int(os.getenv("DASHBOARD_PORT", "8765")),
                                   os.getenv("DASHBOARD_TOKEN"),
                                   approval_handler=approval_handler if args.manual_approval else None).start()
            log.warning("Panel web: %s", dash.url)
        except Exception:
            log.exception("No se pudo iniciar el panel web; el bot continúa sin él")
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):         # Windows: lo cubre CancelledError/KeyboardInterrupt
            pass

    tasks = {"ws": asyncio.create_task(ws.run(), name="ws"),
             "orquestador": asyncio.create_task(orch.run(), name="orquestador")}
    if service:
        tasks["sentimiento"] = asyncio.create_task(service.run(), name="sentimiento")
    stopper = asyncio.create_task(stop.wait())
    log.info("Bot en marcha. Ctrl+C para apagar limpiamente; kill-switch: crear %s", orch.halt_file)
    try:
        done, _ = await asyncio.wait([*tasks.values(), stopper], return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t is not stopper and t.exception():
                log.error("Tarea '%s' terminó con error: %r", t.get_name(), t.exception())
    except asyncio.CancelledError:
        pass
    finally:
        await _shutdown(live, orch, ws, service, tasks, stopper, data_client, trading_client, dash)


async def _close_all(data_client, trading_client) -> None:
    for c in {id(x): x for x in (data_client, trading_client) if x is not None}.values():
        try:
            await c.close()
        except Exception:
            pass


async def _shutdown(live, orch, ws, service, tasks, stopper, data_client, trading_client, dash=None) -> None:
    log.warning("Apagando…")
    stopper.cancel()
    await ws.stop()                                        # no entran más velas
    if service:
        service.stop()
    orch.request_stop()                                    # procesa lo pendiente y termina
    try:
        await asyncio.wait_for(asyncio.shield(tasks["orquestador"]), live.shutdown_timeout_seconds)
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
        log.error("El orquestador no terminó a tiempo; se cancela")
    if live.flatten_on_exit:
        try:
            await orch.flatten_all("apagado")
        except Exception:
            log.exception("Error cerrando posiciones al apagar: REVISA el exchange manualmente")
    for t in tasks.values():
        t.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)
    orch._save()
    orch._write_status()
    if dash:
        dash.stop()
    await _close_all(data_client, trading_client)
    log.warning("Apagado completo. Posiciones abiertas: %s",
                {s: orch.book.qty(s) for s in orch.symbols if abs(orch.book.qty(s)) > 1e-9} or "ninguna")


if __name__ == "__main__":
    try:
        asyncio.run(amain(_parse()))
    except KeyboardInterrupt:
        pass
