"""Build a private sampling catalogue from source availability, never miner hits.

The original catalogue and historical campaigns remain immutable. A new source
snapshot removes missing media and confirmed incompatible versions before any
random draw. Transient download failures are not permanent exclusions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from witness.events import content_hash
from witness.storage import write_private

VERSION = 'activitynet-available-v1'
PERMANENT_FAILURES = {'source_duration_changed', 'source_media_too_large',
                      'mirror_source_size_changed'}


def restrict_catalog(catalog, source_ids, failures, source_identity):
    if catalog.get('availability'):
        raise ValueError('use_original_annotation_catalog')
    if (failures['catalog_hash'] != content_hash(catalog)
            or failures['source'] != source_identity):
        raise ValueError('availability_evidence_identity_mismatch')
    rejected = failures['failures']
    if set(rejected) - set(catalog['originals']):
        raise ValueError('availability_failure_outside_catalog')
    if any(reason not in PERMANENT_FAILURES for reason in rejected.values()):
        raise ValueError('nonpermanent_availability_failure')
    selected, excluded = {}, {}
    for original, row in catalog['originals'].items():
        if not row['eligible_annotation_geometry']:
            excluded[original] = row['exclusion']
        elif original not in source_ids:
            excluded[original] = 'source_absent_from_public_mirror'
        elif original in rejected:
            excluded[original] = rejected[original]
        else:
            selected[original] = row
    if not selected:
        raise ValueError('empty_available_catalog')
    return {**catalog, 'originals': selected,
            'counts': {'total': len(selected), 'annotation_eligible': len(selected),
                       'exclusions': {}},
            'availability': {'version': VERSION, 'source': source_identity,
                'parent_catalog_hash': content_hash(catalog),
                'parent_counts': catalog['counts'],
                'source_failures_hash': content_hash(failures),
                'excluded': excluded, 'exclusion_counts': dict(Counter(excluded.values())),
                'policy': 'present_in_pinned_source_without_confirmed_media_failure; independent_of_miner_index'}}


def validate_available_source(catalog, cache):
    availability = catalog.get('availability')
    if not availability or availability.get('version') != VERSION:
        raise ValueError('available_catalog_required')
    if getattr(cache, 'identity', None) != availability['source']:
        raise ValueError('available_catalog_source_mismatch')
    originals = set(catalog['originals'])
    if (originals - set(cache.sources) or originals & set(availability['excluded'])
            or any(not r['eligible_annotation_geometry'] for r in catalog['originals'].values())):
        raise ValueError('unavailable_original_in_catalog')


def main():
    from .activitynet_mirror import MirrorSourceCache
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--mirror-root', type=Path, required=True)
    parser.add_argument('--source-failures', type=Path, required=True,
                        help='Pinned permanent source failures, not miner retrieval outcomes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = MirrorSourceCache(args.output.parent/'.pin-check', args.mirror_root)
    catalog = restrict_catalog(json.loads(args.catalog.read_text()), set(cache.sources),
                               json.loads(args.source_failures.read_text()), cache.identity)
    validate_available_source(catalog, cache)
    if args.output.exists() and json.loads(args.output.read_text()) != catalog:
        raise ValueError('available_catalog_output_exists_use_new_version')
    write_private(args.output, catalog)
    print(json.dumps({'catalog': str(args.output.resolve()), 'catalog_hash': content_hash(catalog),
                      'eligible': len(catalog['originals']),
                      'exclusions': catalog['availability']['exclusion_counts']}))


if __name__ == '__main__':
    main()
