"""Fact/event scoring: repeated claims share credit and temporal coverage.

References remain independent recall targets. Review evidence can repair
precision, but can never change a published baseline's recall targets.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque


def text_key(value):
    return re.sub(r'\W+', ' ', value.casefold()).strip()


def intervals(spans):
    result = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def length(spans):
    return sum(b-a for a, b in intervals(spans))


def intersection(left, right):
    return length([(max(a,c), min(b,d)) for a,b in intervals(left) for c,d in intervals(right)])


def _credit(demands):
    """Maximum fractional matching: each referenced event has capacity one."""
    graph = defaultdict(dict)
    def edge(a, b, cap):
        graph[a][b] = cap
        graph[b].setdefault(a, 0.)
    for index, (facts, amount) in enumerate(demands):
        node = ('prediction', index)
        edge('source', node, amount)
        for fact in facts:
            edge(node, ('fact', fact), 1.)
            edge(('fact', fact), 'sink', 1.)
    total = 0.
    while True:
        parent = {'source': None}
        queue = deque(['source'])
        while queue and 'sink' not in parent:
            node = queue.popleft()
            for target, capacity in graph[node].items():
                if capacity > 1e-12 and target not in parent:
                    parent[target] = node
                    queue.append(target)
        if 'sink' not in parent:
            return total
        node, amount = 'sink', 1.
        while parent[node] is not None:
            previous = parent[node]
            amount = min(amount, graph[previous][node])
            node = previous
        node = 'sink'
        while parent[node] is not None:
            previous = parent[node]
            graph[previous][node] -= amount
            graph[node][previous] += amount
            node = previous
        total += amount


def measure(references, response, assessments, policy, reviews=None):
    from .scoring import validate_assessment
    if not references or len(references) != len(assessments):
        raise ValueError('reference_assessment_count_mismatch')
    for ref, assessment in zip(references, assessments):
        validate_assessment(ref, response, assessment, policy)
    reviews = reviews or {}
    # Exact duplicate labels for an overlapping occurrence have one identity.
    # Cross-reference aliases also arise when the same atomic proposition is
    # independently supported by one fact in each annotation.
    facts = {(i, f.claim.id): f for i, ref in enumerate(references) for f in ref.facts}
    parents = {key: key for key in facts}
    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    def join(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[max(a,b)] = min(a,b)
    grouped = defaultdict(list)
    for key, fact in facts.items():
        c = fact.claim
        grouped[(c.modality,text_key(c.subject),text_key(c.description))].append(key)
    for keys in grouped.values():
        for j, a in enumerate(keys):
            for b in keys[j+1:]:
                x,y = facts[a].claim,facts[b].claim
                if min(x.end,y.end) >= max(x.start,y.start):
                    join(a,b)
    maps = [{d.prediction_id:d for d in a.decisions} for a in assessments]
    for claim in response.claims:
        aliases = defaultdict(list)
        for i,mapping in enumerate(maps):
            for part in mapping[claim.id].parts:
                if part.status == 'supported' and len(part.fact_ids) == 1:
                    aliases[text_key(part.text)].append((i,part.fact_ids[0]))
        for keys in aliases.values():
            for a,b in zip(keys,keys[1:]):
                x, y = facts[a].claim, facts[b].claim
                if a[0] != b[0] and min(x.end,y.end) > max(x.start,y.start):
                    join(a,b)
    fact_spans = defaultdict(list)
    for key,fact in facts.items():
        fact_spans[root(key)].append((fact.claim.start, fact.claim.end))
    predicted = defaultdict(list)
    unsupported, contradicted = set(), set()
    tolerance = policy.temporal_tolerance_s
    coverage = defaultdict(list)
    # Recall is measured against each annotation independently, without adding
    # reviewed propositions or counting duplicate overlapping facts twice.
    for i,mapping in enumerate(maps):
        for claim in response.claims:
            for part in mapping[claim.id].parts:
                if part.status == 'supported':
                    for fid in part.fact_ids:
                        coverage[(i,root((i,fid)))].append((claim.start-tolerance,claim.end+tolerance))
    for claim in response.claims:
        # Preserve the previous best-annotation rule at the claim boundary.
        i = max(range(len(maps)), key=lambda j:(maps[j][claim.id].share('supported'),
                    -maps[j][claim.id].share('contradicted'), -j))
        for part in maps[i][claim.id].parts:
            key = (claim.modality,text_key(claim.subject),text_key(part.text))
            review = reviews.get((claim.id,text_key(part.text)))
            if part.status == 'supported':
                targets = tuple(sorted({root((i,fid)) for fid in part.fact_ids}))
                predicted[targets].append((claim.start,claim.end))
            elif review and review['status'] == 'supported':
                target = ('review',review['event'])
                fact_spans[target].extend(review.get('spans',[(review['start'],review['end'])]))
                predicted[(target,)].append((claim.start,claim.end))
            elif part.status == 'contradicted' or review and review['status'] == 'contradicted':
                contradicted.add(key)
            else:
                unsupported.add(key)
    # Unsupported duplicates never earn credit; contradictory evidence wins.
    unsupported -= contradicted
    demands = []
    for targets, spans in predicted.items():
        supported_spans = [(a-tolerance,b+tolerance) for f in targets for a,b in fact_spans[f]]
        purity = intersection(spans,supported_spans)/length(spans)
        demands.append((targets,min(1.,purity)))
    count = len(predicted)+len(unsupported)+len(contradicted)
    credit = _credit(demands)
    precision = max(0.,credit-policy.contradiction_penalty*len(contradicted))/count if count else 0.
    per_reference = []
    for i,ref in enumerate(references):
        unique = defaultdict(list)
        for fact in ref.facts:
            if fact.origin == 'annotator':
                unique[root((i,fact.claim.id))].append(fact)
        weighted, total = 0., 0.
        for key, own in unique.items():
            weight = max(1. if f.salience == 'core' else policy.detail_weight for f in own)
            spans = [(f.claim.start,f.claim.end) for f in own]
            weighted += weight*intersection(spans,coverage[(i,key)])/length(spans)
            total += weight
        per_reference.append({'recall':weighted/total if total else 0.,'reference_events':len(unique)})
    recall = sum(r['recall'] for r in per_reference)/len(per_reference)
    quality = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return {'quality':quality,'precision':precision,'recall':recall,'per_reference':per_reference,
            'events':count,'supported_credit':credit,'contradicted_events':len(contradicted),
            'unresolved_events':len(unsupported),'scoring':'events-v2'}
