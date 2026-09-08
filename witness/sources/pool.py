"""Collect YouTube candidates with declared CC licenses and provenance evidence.

Uploader declarations do not certify original footage ownership or independence.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable
from urllib.parse import quote_plus


FFPROBE = Path(
    os.environ.get("WITNESS_FFPROBE")
    or (str(Path.home() / "bin" / "ffprobe") if (Path.home() / "bin" / "ffprobe").is_file() else "")
    or shutil.which("ffprobe")
    or "/usr/bin/ffprobe"
)
CC_FILTER = "EgIwAQ%3D%3D"
CC_LICENSE = "Creative Commons Attribution license (reuse allowed)"
DEFAULT_QUERIES = (
    "cooking vlog",
    "home tour",
    "podcast interview",
    "city walk",
    "workshop tutorial",
    "sports highlights",
    "news segment",
    "lecture",
    "gaming",
    "DIY repair",
    "street food",
)
MIN_DURATION = 4 * 60
MAX_DURATION = 15 * 60


def _yt_dlp_command() -> list[str]:
    return [sys.executable, "-m", "yt_dlp", "--no-update"]


def _run_json(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        [*_yt_dlp_command(), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"yt-dlp exited {result.returncode}")
    return json.loads(result.stdout)


def _search(query: str, candidates: int) -> list[dict[str, Any]]:
    url = (
        "https://www.youtube.com/results?search_query="
        f"{quote_plus(query)}&sp={CC_FILTER}"
    )
    result = _run_json(
        ["--flat-playlist", "--playlist-end", str(candidates), "--dump-single-json", url]
    )
    return [entry for entry in result.get("entries", []) if isinstance(entry, dict)]


def _metadata(video_id: str) -> dict[str, Any]:
    return _run_json(
        [
            "--skip-download",
            "--no-playlist",
            "--dump-single-json",
            f"https://www.youtube.com/watch?v={video_id}",
        ]
    )


def _download(video_id: str, videos_dir: Path) -> Path:
    template = videos_dir / "%(id)s.%(ext)s"
    result = subprocess.run(
        [
            *_yt_dlp_command(),
            "--no-playlist",
            "--format",
            "bv*[height<=480]+ba/b[height<=480]",
            "--merge-output-format",
            "mp4",
            "--remux-video",
            "mp4",
            "--output",
            str(template),
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"yt-dlp exited {result.returncode}")
    matches = sorted(videos_dir.glob(f"{video_id}.*"))
    mp4 = next((path for path in matches if path.suffix.casefold() == ".mp4"), None)
    if mp4 is None:
        raise RuntimeError("yt-dlp did not produce an mp4")
    probe = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(mp4)],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        raise RuntimeError("ffprobe could not inspect downloaded media")
    streams = json.loads(probe.stdout).get("streams", [])
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    if len(video_streams) != 1 or not audio_streams:
        raise RuntimeError("downloaded mp4 must contain one video stream and audio")
    if int(video_streams[0].get("height", 10_000)) > 480:
        raise RuntimeError("downloaded video exceeds 480p")
    return mp4


def _valid_metadata(metadata: dict[str, Any]) -> bool:
    duration = metadata.get("duration")
    return (
        metadata.get("license") == CC_LICENSE
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and MIN_DURATION <= float(duration) <= MAX_DURATION
    )


def build_pool(
    output: Path,
    *,
    queries: Iterable[str] = DEFAULT_QUERIES,
    formats_by_query: dict[str, str] | None = None,
    excluded_ids: Iterable[str] = (),
    excluded_uploaders: Iterable[str] = (),
    max_per_uploader: int = 3,
    limit: int = 5,
    candidates_per_query: int = 12,
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    """Search, download, and manifest a declared-CC candidate source pool.

    Individual search, metadata, and download failures are returned as concise
    diagnostics and never abort the remaining candidates.
    """

    if limit < 1:
        raise ValueError("limit must be at least 1")
    if candidates_per_query < 1:
        raise ValueError("candidates_per_query must be at least 1")
    if max_per_uploader < 1:
        raise ValueError("max_per_uploader must be at least 1")
    output = output.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("source manifest already exists; use a new collection directory")
    videos_dir = output / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    failures: list[str] = []
    seen = {str(video_id) for video_id in excluded_ids}
    blocked_uploaders = {str(value).strip().casefold() for value in excluded_uploaders if value}
    uploader_counts: dict[str, int] = {}
    used_queries: list[str] = []

    searches: list[tuple[str, list[dict[str, Any]]]] = []
    for query in queries:
        query = query.strip()
        if not query:
            continue
        used_queries.append(query)
        try:
            candidates = _search(query, candidates_per_query)
        except Exception as exc:
            failures.append(f"search {query!r}: {exc}")
            continue
        searches.append((query, candidates))

    # Round-robin preserves category diversity without making search order or
    # intermittent failures part of the license/duration acceptance rule.
    for rank in range(candidates_per_query):
        for _query, candidates in searches:
            if len(selected) >= limit:
                break
            if rank >= len(candidates):
                continue
            candidate = candidates[rank]
            video_id = str(candidate.get("id") or "")
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)
            try:
                metadata = _metadata(video_id)
                if not _valid_metadata(metadata):
                    continue
                uploader = str(metadata.get("uploader") or "")
                channel_id = str(metadata.get("channel_id") or "").strip()
                if not channel_id:
                    failures.append(f"video {video_id}: missing channel identity")
                    continue
                uploader_key = channel_id.casefold()
                if uploader_key in blocked_uploaders or uploader.strip().casefold() in blocked_uploaders:
                    continue
                if uploader_counts.get(uploader_key, 0) >= max_per_uploader:
                    continue
                path = _download(video_id, videos_dir)
                provenance = {
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "status": "UPLOADER_DECLARATION_NOT_CERTIFIED",
                    **{key: metadata.get(key) for key in (
                        "id", "webpage_url", "title", "uploader", "channel_id",
                        "channel_url", "upload_date", "license", "description",
                    )},
                }
                provenance_bytes = (json.dumps(provenance, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
                provenance_path = output / "provenance" / f"{video_id}.json"
                provenance_path.parent.mkdir(exist_ok=True)
                with provenance_path.open("xb") as handle:
                    handle.write(provenance_bytes)
                uploader_counts[uploader_key] = uploader_counts.get(uploader_key, 0) + 1
                selected.append(
                    {
                        "id": video_id,
                        "url": str(metadata.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"),
                        "title": str(metadata.get("title") or ""),
                        "uploader": uploader,
                        "uploader_id": str(metadata.get("channel_id") or ""),
                        "license": str(metadata["license"]),
                        "duration": float(metadata["duration"]),
                        "path": path.relative_to(output).as_posix(),
                        "query": _query,
                        "format": (formats_by_query or {}).get(_query, "other"),
                        "provenance_path": provenance_path.relative_to(output).as_posix(),
                        "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
                    }
                )
            except Exception as exc:
                failures.append(f"video {video_id}: {exc}")
        if len(selected) >= limit:
            break

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "license_required": CC_LICENSE,
        "duration_range_seconds": [MIN_DURATION, MAX_DURATION],
        "queries": used_queries,
        "videos": selected,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path, selected, failures


def audit_licenses(manifest_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Re-check current YouTube metadata and annotate every manifest entry in place."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise ValueError("manifest must contain a videos array")
    audited_at = datetime.now(timezone.utc).isoformat()
    failures: list[str] = []
    for item in videos:
        video_id = str(item.get("id") or "")
        try:
            metadata = _metadata(video_id)
            current_license = metadata.get("license")
            revoked = current_license != CC_LICENSE
            item["license_audit"] = {
                "audited_at": audited_at,
                "status": "revoked" if revoked else "verified",
                "revoked": revoked,
                "current_license": current_license,
            }
        except Exception as exc:
            failures.append(f"video {video_id}: {exc}")
            item["license_audit"] = {
                "audited_at": audited_at,
                "status": "unavailable",
                "revoked": None,
                "current_license": None,
            }
    manifest["license_audit"] = {
        "audited_at": audited_at,
        "verified": sum(item["license_audit"]["status"] == "verified" for item in videos),
        "revoked": sum(item["license_audit"]["status"] == "revoked" for item in videos),
        "unavailable": sum(item["license_audit"]["status"] == "unavailable" for item in videos),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest, failures


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build a verified CC YouTube source pool")
    result.add_argument("--out", type=Path, default=Path("data/pool"))
    result.add_argument("--limit", type=int, default=5)
    result.add_argument("--candidates-per-query", type=int, default=12)
    result.add_argument("--max-per-uploader", type=int, default=3)
    result.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=[],
        help="repeat to skip source IDs already present in another pool",
    )
    result.add_argument("--audit", type=Path, metavar="MANIFEST", help="audit licenses in an existing manifest instead of building")
    result.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="repeat to replace the default query set",
    )
    result.add_argument(
        "--format-query",
        action="append",
        default=[],
        metavar="FORMAT::QUERY",
        help="repeat to add a query with the format tag stored on selected videos",
    )
    return result


def _format_queries(values: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    queries: list[str] = []
    formats: dict[str, str] = {}
    for value in values:
        format_name, separator, query = value.partition("::")
        format_name, query = format_name.strip(), query.strip()
        if not separator or not format_name or not query:
            raise ValueError(f"invalid --format-query {value!r}; expected FORMAT::QUERY")
        if query in formats and formats[query] != format_name:
            raise ValueError(f"query {query!r} has conflicting format tags")
        queries.append(query)
        formats[query] = format_name
    return queries, formats


def _manifest_ids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        videos = manifest.get("videos")
        if not isinstance(videos, list):
            raise ValueError(f"manifest must contain a videos array: {path}")
        result.update(str(item["id"]) for item in videos if isinstance(item, dict) and item.get("id"))
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.audit:
        try:
            manifest, failures = audit_licenses(args.audit)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        audit = manifest["license_audit"]
        print(f"manifest: {args.audit}")
        print(f"verified: {audit['verified']}")
        print(f"revoked: {audit['revoked']}")
        print(f"unavailable: {audit['unavailable']}")
        for failure in failures:
            print(f"warning: {failure}")
        return
    try:
        tagged_queries, formats_by_query = _format_queries(args.format_query)
        queries = tagged_queries or args.queries or DEFAULT_QUERIES
        manifest, videos, failures = build_pool(
            args.out,
            queries=queries,
            formats_by_query=formats_by_query,
            excluded_ids=_manifest_ids(args.exclude_manifest),
            max_per_uploader=args.max_per_uploader,
            limit=args.limit,
            candidates_per_query=args.candidates_per_query,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"manifest: {manifest}")
    print(f"downloaded: {len(videos)}")
    print(f"skipped_failures: {len(failures)}")
    for failure in failures:
        print(f"warning: {failure}")
    if not videos:
        raise SystemExit("no eligible videos were downloaded")


if __name__ == "__main__":
    main()
