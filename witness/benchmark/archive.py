"""Licensed Internet Archive originals: catalogue snapshot and exact-byte acquisition.

Only public archive.org URLs are touched; no source captions or annotations
are used. Acquisition never labels, infers or charges an API.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote, urlparse

import httpx

from witness.events import content_hash
from witness.storage import sha256_file, store_immutable, write_private

DATASET = "ia-community-video-unlabelled-v1"
LICENSES = {"creativecommons.org/licenses/by/4.0", "creativecommons.org/licenses/by/3.0",
            "creativecommons.org/publicdomain/zero/1.0"}
QUERY = ('collection:opensource_movies AND mediatype:movies AND '
         '(licenseurl:"https://creativecommons.org/licenses/by/4.0/" OR '
         'licenseurl:"http://creativecommons.org/licenses/by/3.0/" OR '
         'licenseurl:"https://creativecommons.org/publicdomain/zero/1.0/")')
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024


def license_ok(value) -> bool:
    values = value if isinstance(value, list) else [value]
    return bool(values) and all(
        isinstance(url, str) and (urlparse(url).netloc + urlparse(url).path).rstrip("/") in LICENSES
        for url in values)


def request(client: httpx.Client, url, *, params=None, output: Path | None = None, expected_size: int | None = None):
    """GET from archive.org only, following at most five same-site redirects.

    With ``output`` the body is streamed to that file and must have exactly
    ``expected_size`` bytes; otherwise a bounded JSON body is returned.
    """
    for _ in range(6):
        parsed = urlparse(str(url))
        host = parsed.hostname or ""
        if (parsed.scheme != "https" or not (host == "archive.org" or host.endswith(".archive.org"))
                or parsed.username or parsed.password):
            raise ValueError("archive_unsafe_url")
        with client.stream("GET", url, params=params, follow_redirects=False) as response:
            if response.is_redirect:
                url, params = response.url.join(response.headers["location"]), None
                continue
            response.raise_for_status()
            limit = expected_size if output else MAX_JSON_BYTES
            size, chunks = 0, []
            target = output.open("wb") if output else None
            try:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("archive_response_too_large")
                    if target:
                        target.write(chunk)
                    else:
                        chunks.append(chunk)
            finally:
                if target:
                    target.close()
            if output:
                if size != expected_size:
                    raise ValueError("archive_source_size_changed")
                return None
            return json.loads(b"".join(chunks))
    raise ValueError("archive_redirect_limit")


def index(path: Path, *, client: httpx.Client | None = None, page_size: int = 1000) -> dict:
    """Load or create the hash-sealed snapshot of licensed candidate identifiers."""
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    owned = client is None
    client = client or httpx.Client(timeout=45, headers={"User-Agent": "WitnessVideoResearch/1.0"})
    try:
        items, cursor = {}, None
        for _ in range(1000):
            params = {"q": QUERY, "fields": "identifier,licenseurl,creator", "count": page_size}
            if cursor:
                params["cursor"] = cursor
            data = request(client, "https://archive.org/services/search/v1/scrape", params=params)
            for row in data["items"]:
                if IDENTIFIER.fullmatch(row.get("identifier", "")) and license_ok(row.get("licenseurl")):
                    items[row["identifier"]] = row
            new_cursor = data.get("cursor")
            if not data["items"] or not new_cursor:
                break
            if new_cursor == cursor:
                raise ValueError("archive_pagination_failed")
            cursor = new_cursor
        else:
            raise ValueError("archive_pagination_failed")
        body = {"dataset": DATASET, "query": QUERY, "created_unix": time.time(), "candidate_count": len(items),
                "items": sorted(items.values(), key=lambda row: row["identifier"]),
                "label_policy": "no_source_annotations", "availability": "unverified_until_acquisition"}
        body["snapshot_hash"] = content_hash(body)
        write_private(path, body)
        return body
    finally:
        if owned:
            client.close()


def media_files(metadata: dict) -> list[tuple[int, str, float]]:
    """Eligible MP4 files of one item as (size, name, length), smallest first."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get("metadata"), dict):
        raise ValueError("archive_invalid_metadata")
    row = metadata["metadata"]
    if row.get("mediatype") != "movies" or not license_ok(row.get("licenseurl")):
        raise ValueError("archive_ineligible_license_or_type")
    collections = row.get("collection", [])
    if "opensource_movies" not in ([collections] if isinstance(collections, str) else collections):
        raise ValueError("archive_wrong_collection")
    files = metadata.get("files", [])
    if not isinstance(files, list):
        raise ValueError("archive_invalid_file_list")
    result = []
    for file in files:
        if not isinstance(file, dict):
            continue
        try:
            size, length = int(file.get("size", 0)), float(file.get("length", 0))
        except (ValueError, TypeError):
            continue
        name = file.get("name", "")
        if (isinstance(name, str) and name.lower().endswith(".mp4") and "/" not in name and "\\" not in name
                and 0 < size <= MAX_SOURCE_BYTES and 60 <= length <= 7200 and not file.get("private")):
            result.append((size, name, length))
    return sorted(result)


def probe(path: Path) -> float:
    """Duration of a downloaded original with audio and video at a sane resolution."""
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError("archive_unreadable_media")
    value = json.loads(result.stdout)
    try:
        streams = value["streams"]
        videos = [stream for stream in streams if stream.get("codec_type") == "video"]
        if not videos or not any(stream.get("codec_type") == "audio" for stream in streams):
            raise ValueError("archive_audio_video_required")
        width, height = int(videos[0]["width"]), int(videos[0]["height"])
        if min(width, height) < 180:
            raise ValueError("archive_resolution_too_low")
        if max(width, height) > 4096 or width * height > 3840 * 2160:
            raise ValueError("archive_resolution_too_large")
        duration = float(value["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("archive_invalid_duration")
        return duration
    except (KeyError, TypeError, AttributeError):
        raise ValueError("archive_invalid_probe_metadata") from None


def acquire_archive(catalogue: Path, root: Path, requested: dict[str, str]) -> dict:
    """Download exact CC originals once; caller chooses train/dev/eval before draw.

    This touches only public Archive URLs and a private local directory. It does
    not label, infer, charge an API or submit anything on chain.
    """
    if not requested or any(not IDENTIFIER.fullmatch(identifier) or split not in
                            {"train", "dev", "eval"} for identifier, split in requested.items()):
        raise ValueError("invalid_archive_source_request")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_mode & 0o077:
        raise ValueError("archive_source_root_not_private")
    listing = index(catalogue)
    if content_hash({key: value for key, value in listing.items() if key != "snapshot_hash"}) != listing["snapshot_hash"]:
        raise ValueError("archive_catalogue_changed")
    candidates = {row["identifier"]: row for row in listing["items"]}
    if not set(requested) <= candidates.keys():
        raise ValueError("archive_identifier_not_in_snapshot")
    manifest = []
    with httpx.Client(timeout=60, headers={"User-Agent": "WitnessVideoResearch/2.0"}) as client:
        for identifier, split in sorted(requested.items()):
            target = root / (identifier + ".mp4")
            receipt_path = root / (identifier + ".source.json")
            marker = root / (identifier + ".started.json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (receipt.get("identifier") != identifier or receipt.get("split") != split
                        or receipt.get("catalogue_hash") != listing["snapshot_hash"]
                        or sha256_file(target) != receipt.get("sha256")):
                    raise ValueError("saved_archive_source_changed")
            else:
                if marker.exists():
                    raise ValueError("archive_download_outcome_requires_reconciliation")
                metadata = request(client, "https://archive.org/metadata/" + quote(identifier, safe=""))
                files = media_files(metadata)
                if not files:
                    raise ValueError("archive_no_eligible_media")
                size, name, _ = files[0]
                licenses = metadata["metadata"]["licenseurl"]
                if not license_ok(licenses):
                    raise ValueError("archive_license_changed")
                license_url = licenses[0] if isinstance(licenses, list) else licenses
                store_immutable(marker, {"identifier": identifier, "catalogue_hash": listing["snapshot_hash"],
                                          "file": name, "size": size})
                url = "https://archive.org/download/" + quote(identifier, safe="") + "/" + quote(name, safe="")
                target.touch(mode=0o600, exist_ok=False)
                request(client, url, output=target, expected_size=size)
                duration = probe(target)
                if not 60 <= duration <= 7200:
                    raise ValueError("archive_original_duration_changed")
                target.chmod(0o600)
                creator = metadata["metadata"].get("creator") or identifier
                if isinstance(creator, list):
                    creator = " | ".join(sorted(str(value).strip() for value in creator if str(value).strip()))
                if not isinstance(creator, str) or not creator.strip():
                    creator = identifier
                receipt = {"identifier": identifier, "catalogue_hash": listing["snapshot_hash"],
                           "sha256": sha256_file(target), "duration": duration,
                           "file": name, "size": size, "license_url": license_url,
                           "creator": creator,
                           "split": split}
                store_immutable(receipt_path, receipt)
            manifest.append({"id": identifier, "path": str(target),
                             "creator": receipt["creator"], "parents": [],
                             "license_url": receipt["license_url"], "split": split})
    store_immutable(root / "manifest.json", manifest)
    return {"originals": len(manifest), "catalogue_hash": listing["snapshot_hash"],
            "manifest": str(root / "manifest.json")}
