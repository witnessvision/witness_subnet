"""Read-only public website and subnet display API."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .subnet import add_subnet_routes

STATIC = Path(__file__).parent / "static"
DOCS = Path(__file__).resolve().parents[2] / "docs"


def create_app(*, sources=None, telemetry_root=None) -> FastAPI:
    app = FastAPI(title="Witness subnet", docs_url=None, redoc_url=None, openapi_url=None)
    add_subnet_routes(app, sources, telemetry_root=telemetry_root)

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        return response

    @app.get("/api/health")
    def health():
        return {"status": "ok", "mode": "read-only"}

    @app.get("/documents/{name}")
    def document(name: str):
        names = {"launch": "launch.md"}
        if name not in names or not (DOCS / names[name]).is_file():
            raise HTTPException(404, "Document unavailable")
        return FileResponse(DOCS / names[name], media_type="text/plain")

    @app.get("/")
    def landing():
        return FileResponse(STATIC / "index.html")

    @app.get("/dashboard")
    def dashboard():
        return FileResponse(STATIC / "dashboard.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
