from __future__ import annotations

import math
from dataclasses import dataclass

from config.rl_settings import CostSettings


@dataclass(frozen=True)
class Fill:
    """Resultado de ejecutar una orden simulada."""
    qty: float          # cantidad con signo (+ compra / - venta)
    price: float        # precio de ejecución (ya con deslizamiento)
    fee: float          # comisión en moneda de cotización
    slippage_cost: float  # coste del deslizamiento vs. precio de referencia
    notional: float


class CostModel:
    """Modelo de costes de Binance: comisión taker + deslizamiento (fijo + impacto raíz cuadrada).

    Las órdenes del agente se asumen "market" (taker), el caso conservador. TWAP/Limit
    en F5 reducirán costes reales; aquí no se premia por ello.
    """

    def __init__(self, cfg: CostSettings) -> None:
        self.cfg = cfg

    def slippage_bps(self, notional: float, bar_quote_volume: float) -> float:
        participation = notional / max(bar_quote_volume, 1e-9)
        return self.cfg.base_slippage_bps + self.cfg.impact_coef_bps * math.sqrt(max(participation, 0.0))

    def execute(self, qty: float, ref_price: float, bar_quote_volume: float) -> Fill:
        """Ejecuta `qty` (con signo) al precio de referencia con deslizamiento adverso y comisión taker."""
        if qty == 0.0:
            return Fill(0.0, ref_price, 0.0, 0.0, 0.0)
        notional_ref = abs(qty) * ref_price
        slip = self.slippage_bps(notional_ref, bar_quote_volume) / 1e4
        price = ref_price * (1.0 + slip) if qty > 0 else ref_price * (1.0 - slip)
        notional = abs(qty) * price
        return Fill(
            qty=qty, price=price, fee=notional * self.cfg.taker_rate,
            slippage_cost=abs(qty) * ref_price * slip, notional=notional,
        )

    def max_affordable_buy(self, cash: float, ref_price: float, bar_quote_volume: float) -> float:
        """Mayor cantidad comprable con `cash` incluyendo deslizamiento y comisión (bisección corta)."""
        if cash <= 0:
            return 0.0
        lo, hi = 0.0, cash / ref_price
        for _ in range(30):
            mid = (lo + hi) / 2
            f = self.execute(mid, ref_price, bar_quote_volume)
            if f.notional + f.fee <= cash:
                lo = mid
            else:
                hi = mid
        return lo
