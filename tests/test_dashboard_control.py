from __future__ import annotations

from pathlib import Path

from src.dashboard.data import DashboardPaths
from src.dashboard.server import DashboardServer


def test_manual_approval_route_forwards_only_valid_decisions(tmp_path: Path):
    received = []
    paths = DashboardPaths(
        live=tmp_path / "live",
        risk=tmp_path / "risk",
        sentiment=tmp_path / "sentiment",
        models=tmp_path / "models",
    )
    server = DashboardServer(
        paths,
        approval_handler=lambda proposal_id, approve: received.append((proposal_id, approve)) or {
            "status": "executed" if approve else "rejected"
        },
    )

    status, result = server._post(
        "/api/approval", {"proposal_id": "proposal-1", "approve": True}
    )

    assert status == 200
    assert result == {"status": "executed"}
    assert received == [("proposal-1", True)]
    assert server._post("/api/approval", {"proposal_id": "", "approve": True})[0] == 400


def test_manual_approval_is_rejected_when_callback_is_not_configured(tmp_path: Path):
    server = DashboardServer(DashboardPaths(
        live=tmp_path / "live",
        risk=tmp_path / "risk",
        sentiment=tmp_path / "sentiment",
        models=tmp_path / "models",
    ))

    status, result = server._post(
        "/api/approval", {"proposal_id": "proposal-1", "approve": True}
    )

    assert status == 409
    assert "no habilitada" in result["error"]
