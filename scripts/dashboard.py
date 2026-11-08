"""Panel web del agente (solo lectura + kill-switch).

    python scripts/dashboard.py                  # lee data/live, data/risk, data/sentiment, data/models
    python scripts/dashboard.py --demo           # datos de ejemplo en una carpeta temporal
    python main.py --paper --dashboard           # el bot y el panel en el mismo proceso
"""
import argparse
import os
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dashboard import DashboardPaths, DashboardServer
from src.dashboard.demo import DemoHeartbeat, generate_demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("DASHBOARD_PORT", "8765")))
    ap.add_argument("--host", default=os.getenv("DASHBOARD_HOST", "127.0.0.1"))
    ap.add_argument("--demo", action="store_true", help="genera datos sintéticos y los muestra")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    paths, hb = None, None
    if a.demo:
        paths = generate_demo(Path(tempfile.mkdtemp(prefix="trading_demo_")))
        hb = DemoHeartbeat(paths)
        hb.start()
        print(f"Datos de demo en {paths.live.parent}")
    srv = DashboardServer(paths or DashboardPaths.default(), a.host, a.port, os.getenv("DASHBOARD_TOKEN")).start()
    print(f"Panel: {srv.url}   (Ctrl+C para salir)")
    if not a.no_browser:
        webbrowser.open(srv.url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
        if hb:
            hb.stop()


if __name__ == "__main__":
    main()
