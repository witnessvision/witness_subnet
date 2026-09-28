from fastapi.testclient import TestClient

from witness_web.app import create_app


def test_public_routes_do_not_expose_local_files_or_retired_report_apis(tmp_path, monkeypatch):
    marker = "LOCAL_FILE_MUST_NOT_BE_SERVED"
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/report.json").write_text(marker)
    (tmp_path / ".env").write_text(marker)
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(sources=[]))
    for path in ("/.env", "/reports/report.json", "/api/evidence", "/api/readiness",
                 "/api/scenes", "/api/scenes/example", "/api/scenes/example/video",
                 "/documents/unknown", "/static/../.env", "/static/scenes.py",
                 "/static/app.js", "/static/inspector.js"):
        response = client.get(path)
        assert response.status_code == 404
        assert marker not in response.text
    for path in ("/", "/dashboard", "/api/subnet", "/api/health", "/documents/launch"):
        response = client.get(path)
        assert response.status_code == 200
        assert marker not in response.text
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Content-Security-Policy"].startswith("default-src 'self'")
    assert client.get("/api/health").json() == {"status": "ok", "mode": "read-only"}
