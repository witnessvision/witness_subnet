"""Deterministic selection of licensed video sources from a catalogue partition.

Rule ``witness-selection-v1``, applied to one split in this order:

1. Only creator groups of that split; identifier-only groups (no named creator)
   are skipped.
2. Only CC BY 4.0 / 3.0 items (an explicit attribution licence). CC0 is excluded
   because mislicensed re-uploads concentrate there.
3. Creator round-robin: every creator's first video comes before any creator's
   second, so the slice spreads over as many independent sources as possible.
4. Within and across creators: archive.org download count, highest first, ties
   by identifier. The counts are a frozen snapshot passed in by the caller.
5. Technical eligibility, walked in that order by the caller's ``eligible``
   check (the benchmark media rule). Every rejection is kept with its reason.

A slice of size N is the first N accepted videos, so larger slices are
prefixes of smaller ones. The caller chooses the catalogue partition.
"""
from __future__ import annotations

from collections.abc import Callable

RULE = "witness-selection-v1"
BY_LICENCES = ("creativecommons.org/licenses/by/4.0", "creativecommons.org/licenses/by/3.0")


def ordered(split: dict, licences: dict[str, str], downloads: dict[str, int],
            split_name: str) -> list[tuple[str, str]]:
    """(identifier, creator group) pairs of one split in rule order."""
    groups = {key: [item for item in group["items"] if any(by in str(licences.get(item)) for by in BY_LICENCES)]
              for key, group in split["groups"].items()
              if group["split"] == split_name and not key.startswith("id:")}

    def rank(item: str) -> tuple[int, str]:
        return -downloads.get(item, 0), item
    queues = {key: sorted(items, key=rank) for key, items in groups.items() if items}
    order, depth = [], 0
    while any(len(queue) > depth for queue in queues.values()):
        layer = [(queue[depth], key) for key, queue in queues.items() if len(queue) > depth]
        order += sorted(layer, key=lambda pair: rank(pair[0]))
        depth += 1
    return order


def select(order: list[tuple[str, str]], eligible: Callable[[str], dict], count: int) -> dict:
    """Walk the order until ``count`` videos pass ``eligible`` ({"status": "accepted"|"rejected", ...})."""
    accepted, rejected = [], []
    for identifier, group in order:
        if len(accepted) == count:
            break
        result = eligible(identifier)
        row = {"identifier": identifier, "creator_group": group, **result}
        (accepted if result["status"] == "accepted" else rejected).append(row)
    return {"rule": RULE, "accepted": accepted, "rejected": rejected}
