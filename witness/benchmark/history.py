"""Immutable source reservations shared across benchmark challenges."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, model_validator

from witness.events import StrictModel, content_hash
from witness.storage import store_immutable
from .lineage import Catalog, Draw, Source, related


class Reservation(StrictModel):
    challenge_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    draw_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_groups: list[str] = Field(min_length=1)
    selected_sources: list[Source] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        if len(self.selected_groups) != len(set(self.selected_groups)) or len(
                {source.id for source in self.selected_sources}) != len(self.selected_sources):
            raise ValueError("duplicate_source_reservation")
        return self


def load_history(directory: Path) -> list[Reservation]:
    if not directory.exists():
        return []
    if directory.is_symlink() or directory.stat().st_mode & 0o077:
        raise ValueError("unsafe_benchmark_history_directory")
    reservations = []
    for path in sorted(directory.glob("*.reservation.json")):
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise ValueError("unsafe_benchmark_history_receipt")
        receipt = Reservation.model_validate(json.loads(path.read_text()))
        if path.name != receipt.challenge_id + ".reservation.json":
            raise ValueError("history_filename_binding_failure")
        reservations.append(receipt)
    if len({row.challenge_id for row in reservations}) != len(reservations):
        raise ValueError("duplicate_historical_challenge")
    return reservations


def used_groups(catalog: Catalog, reservations: list[Reservation]) -> set[str]:
    prior_groups = {group for row in reservations for group in row.selected_groups}
    historical = [source for row in reservations for source in row.selected_sources]
    sources = {source.id: source for source in catalog.sources}
    blocked = set(prior_groups)
    for group_id, member_ids in catalog.eval_groups():
        if any(related(sources[source_id], prior) for source_id in member_ids for prior in historical):
            blocked.add(group_id)
    return blocked


def reserve(directory: Path, draw: Draw, catalog: Catalog) -> Reservation:
    groups = dict(catalog.eval_groups())
    lookup = {source.id: source for source in catalog.sources}
    if any(group not in groups for group in draw.groups):
        raise ValueError("reservation_group_not_in_catalog")
    selected = sorted({source_id for group in draw.groups for source_id in groups[group]})
    receipt = Reservation(challenge_id=draw.commit.challenge_id, catalog_hash=catalog.identity,
                          draw_hash=content_hash(draw.model_dump()), selected_groups=draw.groups,
                          selected_sources=[lookup[source_id] for source_id in selected])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.stat().st_mode & 0o077:
        raise ValueError("unsafe_benchmark_history_directory")
    store_immutable(directory / (receipt.challenge_id + ".reservation.json"), receipt.model_dump())
    return receipt
