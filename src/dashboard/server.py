from __future__ import annotations

import hmac
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from src.dashboard.data import DashboardData, DashboardPaths

log = logging.getLogger(__name__)

_STATIC = Path(__file__).parent / "static"
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _int(q: dict, key: str, default: int, lo: int = 1, hi: int = 50000) -> int:
    try:
        return max(lo, min(hi, int(q.get(key, [default])[0])))
    except (ValueError, TypeError):
        return default


class DashboardServer:
    """Panel web local (solo stdlib). Lectura de los archivos del bot + un único control: el kill-switch HALT.

    Seguridad:
      · escucha solo en 127.0.0.1 por defecto; fuera de loopback exige `token` y valida el encabezado Host;
      · los POST exigen la cabecera `X-Dashboard: 1` (un sitio externo no puede enviarla sin preflight CORS,
        que este servidor nunca concede) y, si hay token, `X-Dashboard-Token`;
      · no sirve credenciales ni el .env; todo texto externo (titulares, logs) se escapa en el cliente.
    """

    def __init__(self, paths: DashboardPaths | None = None, host: str = "127.0.0.1", port: int = 8765,
                 token: str | None = None,
                 approval_handler: Callable[[str, bool], object] | None = None) -> None:
        self.paths = paths or DashboardPaths.default()
        self.data = DashboardData(self.paths)
        self.host, self.port, self.token = host, port, token or None
        self.approval_handler = approval_handler
        self.control_enabled = host in _LOOPBACK or bool(self.token)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ ciclo de vida
    def start(self) -> "DashboardServer":
        if self.host not in _LOOPBACK and not self.token:
            raise RuntimeError("DASHBOARD_HOST fuera de loopback exige DASHBOARD_TOKEN")
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        self.port = self._httpd.server_address[1]                   # por si se pidió el puerto 0
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="dashboard", daemon=True)
        self._thread.start()
        log.info("Panel en http://%s:%d", self.host if self.host != "0.0.0.0" else "localhost", self.port)
        return self

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    @property
    def url(self) -> str:
        return f"http://{'localhost' if self.host in _LOOPBACK else self.host}:{self.port}"

    # ------------------------------------------------------------------ enrutado
    def _get(self, path: str, q: dict) -> tuple[int, object]:
        d = self.data
        sym = q.get("symbol", [None])[0]
        routes = {
            "/api/overview": lambda: d.overview(sym),
            "/api/equity": lambda: d.equity(sym, _int(q, "limit", 4000)),
            "/api/fills": lambda: d.fills(_int(q, "limit", 300)),
            "/api/trades": lambda: d.closed_trades(_int(q, "limit", 300)),
            "/api/risk": lambda: d.risk(_int(q, "limit", 300)),
            "/api/drift": lambda: d.drift(_int(q, "limit", 500)),
            "/api/sentiment": lambda: d.sentiment(_int(q, "limit", 300)),
            "/api/logs": lambda: d.logs(_int(q, "n", 400, 10, 5000), q.get("level", [None])[0], q.get("q", [None])[0]),
            "/api/models": d.models,
            "/api/config": d.config,
            "/api/meta": lambda: {**d.meta(), "control": self.control_enabled, "token_required": bool(self.token)},
        }
        fn = routes.get(path)
        return (200, fn()) if fn else (404, {"error": "no encontrado"})

    def _post(self, path: str, body: dict) -> tuple[int, object]:
        if path == "/api/approval":
            proposal_id = body.get("proposal_id")
            approve = body.get("approve")
            if not isinstance(proposal_id, str) or not proposal_id or not isinstance(approve, bool):
                return 400, {"error": "proposal_id y approve son obligatorios"}
            if self.approval_handler is None:
                return 409, {"error": "aprobación manual no habilitada"}
            try:
                result = self.approval_handler(proposal_id, approve)
            except (ValueError, RuntimeError) as exc:
                return 409, {"error": str(exc)}
            return 200, result
        if path != "/api/halt":
            return 404, {"error": "no encontrado"}
        halt = self.paths.live / "HALT"
        if body.get("on"):
            halt.parent.mkdir(parents=True, exist_ok=True)
            halt.write_text("halt desde el panel web\n")
            log.critical("HALT activado desde el panel")
        else:
            halt.unlink(missing_ok=True)
            log.warning("HALT retirado desde el panel")
        return 200, {"halt": halt.exists()}

    def _handler(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            server_version = "TradingDash/1"

            def log_message(self, *a):          # silencia el log por petición
                pass

            def _send(self, code: int, payload, ctype: str = "application/json") -> None:
                raw = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", f"{ctype}; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                                 "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(raw)

            def _host_ok(self) -> bool:
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0] if not (self.headers.get("Host") or "").startswith("[") \
                    else (self.headers.get("Host") or "").split("]")[0] + "]"
                allowed = _LOOPBACK | {srv.host}
                return host in allowed                                  # defensa contra DNS rebinding

            def do_GET(self):
                if not self._host_ok():
                    return self._send(403, {"error": "host no permitido"})
                u = urlparse(self.path)
                if u.path in ("/", "/index.html"):
                    return self._send(200, (_STATIC / "index.html").read_bytes(), "text/html")
                try:
                    code, payload = srv._get(u.path, parse_qs(u.query))
                except Exception as exc:                                # un archivo corrupto no tumba el panel
                    log.exception("Error en %s", u.path)
                    code, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
                self._send(code, payload)

            def do_POST(self):
                if not self._host_ok():
                    return self._send(403, {"error": "host no permitido"})
                if not srv.control_enabled:
                    return self._send(403, {"error": "controles deshabilitados"})
                if self.headers.get("X-Dashboard") != "1":
                    return self._send(403, {"error": "falta X-Dashboard"})
                if srv.token and not hmac.compare_digest(self.headers.get("X-Dashboard-Token", ""), srv.token):
                    return self._send(401, {"error": "token inválido"})
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
                    code, payload = srv._post(urlparse(self.path).path, body)
                except Exception as exc:
                    code, payload = 400, {"error": f"{type(exc).__name__}: {exc}"}
                self._send(code, payload)

        return H
