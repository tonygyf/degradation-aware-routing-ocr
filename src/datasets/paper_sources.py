"""Canonical-source audit and development-only source-filter sensitivity (S14)."""

from collections import Counter, defaultdict
from pathlib import PurePosixPath

BASELINE_ROLES = ("Support", "Dev", "Cal", "Test")
SHC_ROLES = ("Exemplar", "Prompt_dev", "Tau_dev", "Test")
CORPUS_ALIASES = {
    "ic13": "icdar2013",
    "icdar2013": "icdar2013",
    "sroie": "sroie",
    "icdar2019sroie": "sroie",
}


def canonical_source(row):
    corpus, image = row["source_corpus"], row["source_image"]
    if not isinstance(corpus, str) or not corpus.strip() or not isinstance(image, str):
        raise ValueError("Require audited source_corpus and parent source_image")
    corpus = corpus.strip().lower()
    compact = "".join(c for c in corpus if c.isalnum())
    corpus = CORPUS_ALIASES.get(compact, corpus)
    name = image.replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in {".", ".."}:
        raise ValueError("Require a parent source filename")
    return f"{corpus}:{PurePosixPath(name).stem.lower()}"


def audit_sources(baseline, shc):
    def index(rows, roles):
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Require nonempty resources with unique sample IDs")
        if any(not isinstance(r["id"], str) or not r["id"] for r in rows):
            raise ValueError("Require nonempty sample IDs")
        result = {role: set() for role in roles}
        for row in rows:
            if row["role"] not in roles:
                raise ValueError("Unknown resource role")
            result[row["role"]].add(canonical_source(row))
        return result

    base_sets, shc_sets = index(baseline, BASELINE_ROLES), index(shc, SHC_ROLES)
    shc_roles = defaultdict(set)
    for role, sources in shc_sets.items():
        for source in sources:
            shc_roles[source].add(role)
    filtered = {}
    for role in SHC_ROLES[:-1]:
        original = [r for r in shc if r["role"] == role]
        cross = [r for r in original if canonical_source(r) not in base_sets["Test"]]
        strict = [r for r in cross if canonical_source(r) not in shc_sets["Test"]]
        filtered[role] = {
            "original_samples": len(original),
            "remove_baseline_test_ids": [r["id"] for r in cross],
            "also_remove_shc_test_ids": [r["id"] for r in strict],
        }
    return {
        "canonicalization": (
            "source_corpus:lowercase parent source_image stem; never infer from derived image_id"
        ),
        "corpus_aliases": CORPUS_ALIASES,
        "baseline_role_counts": dict(Counter(r["role"] for r in baseline)),
        "shc_role_counts": dict(Counter(r["role"] for r in shc)),
        "baseline_to_shc_unique_source_overlap": {
            a: {b: len(base_sets[a] & shc_sets[b]) for b in SHC_ROLES} for a in BASELINE_ROLES
        },
        "shc_unique_parent_sources": len(shc_roles),
        "shc_parents_in_multiple_roles": sum(len(roles) > 1 for roles in shc_roles.values()),
        "shc_development_sensitivity": filtered,
        "baseline_development_without_shc_test": {
            role: [
                r["id"]
                for r in baseline
                if r["role"] == role and canonical_source(r) not in shc_sets["Test"]
            ]
            for role in BASELINE_ROLES[:-1]
        },
        "frozen_test_ids": {
            "baseline": [r["id"] for r in baseline if r["role"] == "Test"],
            "shc": [r["id"] for r in shc if r["role"] == "Test"],
        },
        "note": (
            "Audit/sensitivity only. Original roles and Test inputs are "
            "not rewritten; source reuse is reported, not silently "
            "repaired."
        ),
    }
