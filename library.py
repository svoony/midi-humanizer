"""Indexes the MusicXML corpora pulled from ASAP and KernScores (see
vendor/datasets/) so the web UI can search them and load a piece straight
into the existing render pipeline, without the user needing to manually
download and re-upload a file.
"""
import csv
import os
import re
import time

import piece_naming
from itertools import zip_longest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "vendor", "datasets")
ASAP_DIR = os.path.join(DATASETS_DIR, "asap-dataset")
ASAP_METADATA_CSV = os.path.join(ASAP_DIR, "metadata.csv")
KERNSCORES_DIR = os.path.join(DATASETS_DIR, "kernscores-musicxml")
# piano-midi.de recordings by Bernd Krueger, used under CC BY-SA Germany (see
# piano-midi-de/ATTRIBUTION.txt). One subfolder per composer of .mid files.
PIANO_MIDI_DIR = os.path.join(DATASETS_DIR, "piano-midi-de")

# Piano-only allowlist for the KernScores corpus. The humdrum-data meta-repo
# also pulls in vocal polyphony (Josquin, Tasso, Lassus), folk-song melodies
# (Essen, Densmore), chant, Cantopop, and film themes - all excluded here
# since this tool is a piano performance model. Each allowlisted repo is a
# single-composer piano collection, so the repo name gives the composer.
# (pl-wnifc_humdrum-polish-scores and the Bach chorales are deliberately left
# out: the former is mixed instrumentation, the latter is vocal SATB writing.)
PIANO_KERNSCORES_REPOS = {
    "craigsapp_beethoven-piano-sonatas": "Beethoven",
    "craigsapp_haydn-piano-sonatas": "Haydn",
    "craigsapp_mozart-piano-sonatas": "Mozart",
    "craigsapp_joplin": "Joplin",
    # pl-wnifc_humdrum-chopin-first-editions intentionally excluded: 508 scanned
    # first editions named by cryptic catalog codes (e.g. "001 1 Brz"), heavily
    # duplicative and not reliably nameable. Chopin is covered by piano-midi.de.
}

# Composer (surname, as it appears in ASAP metadata and the piano repo map
# above) -> era, kept consistent with the model's own ERA_BY_COMPOSER scheme
# in note_dataset.py (e.g. Beethoven is classified classical, not romantic).
# This drives both the UI era tag and the era the piece is rendered with.
COMPOSER_ERA = {
    "Bach": "baroque",
    "Haydn": "classical",
    "Mozart": "classical",
    "Beethoven": "classical",
    "Schubert": "romantic",
    "Chopin": "romantic",
    "Schumann": "romantic",
    "Liszt": "romantic",
    "Brahms": "romantic",
    "Glinka": "romantic",
    "Balakirev": "romantic",
    "Debussy": "modern",
    "Scriabin": "modern",
    "Rachmaninoff": "modern",
    "Ravel": "modern",
    "Prokofiev": "modern",
    "Joplin": "modern",
    # additional composers from the piano-midi.de roster
    "Albeniz": "romantic",
    "Borodin": "romantic",
    "Burgmueller": "romantic",
    "Clementi": "classical",
    "Granados": "romantic",
    "Grieg": "romantic",
    "Mendelssohn": "romantic",
    "Moszkowski": "romantic",
    "Mussorgsky": "romantic",
    "Tchaikovsky": "romantic",
    "Sinding": "romantic",
}
DEFAULT_ERA_NAME = "romantic"  # matches note_dataset.DEFAULT_ERA


def _era_for_composer(composer):
    return COMPOSER_ERA.get((composer or "").strip(), DEFAULT_ERA_NAME)

# The kern->MusicXML conversion runs in the background and keeps adding files,
# so the index is rebuilt on a short TTL rather than cached permanently - that
# way newly-converted tracks show up in the panel without a server restart.
# (A rebuild is just an os.walk + CSV read, cheap enough to redo periodically.)
INDEX_TTL_SECONDS = 30.0

_index = None
_index_built_at = 0.0


def _index_asap():
    entries = []
    if not os.path.exists(ASAP_METADATA_CSV):
        return entries
    seen_paths = set()
    with open(ASAP_METADATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel_path = row.get("xml_score")
            if not rel_path or rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            abs_path = os.path.join(ASAP_DIR, rel_path)
            if not os.path.exists(abs_path):
                continue
            composer = row.get("composer", "").strip()
            raw_title = row.get("title", "").replace("_", " ").strip()
            title, dkey = piece_naming.nice_name("ASAP", composer, raw_title, abs_path)
            entries.append({
                "id": f"asap::{rel_path}",
                "composer": composer,
                "title": title,
                "source": "ASAP",
                "era": _era_for_composer(composer),
                "path": abs_path,
                "_dedup_key": dkey,
            })
    return entries


def _index_kernscores():
    entries = []
    if not os.path.exists(KERNSCORES_DIR):
        return entries
    repos_dir = os.path.join(KERNSCORES_DIR, "_repos")
    for dirpath, _, filenames in os.walk(repos_dir):
        for filename in filenames:
            if not filename.endswith(".musicxml"):
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, repos_dir)
            repo_dir_name = rel_path.split(os.sep)[0]
            composer = PIANO_KERNSCORES_REPOS.get(repo_dir_name)
            if composer is None:  # non-piano repo, skip
                continue
            rel_id = os.path.relpath(abs_path, KERNSCORES_DIR).replace(os.sep, "/")
            title, dkey = piece_naming.nice_name("KernScores", composer, filename, abs_path)
            entries.append({
                "id": f"kernscores::{rel_id}",
                "composer": composer,
                "title": title,
                "source": "KernScores",
                "era": _era_for_composer(composer),
                "path": abs_path,
                "_dedup_key": dkey,
            })
    return entries


def _index_piano_midi():
    entries = []
    if not os.path.exists(PIANO_MIDI_DIR):
        return entries
    for composer in sorted(os.listdir(PIANO_MIDI_DIR)):
        comp_dir = os.path.join(PIANO_MIDI_DIR, composer)
        if composer.startswith("_") or not os.path.isdir(comp_dir):  # skips _zips/
            continue
        for filename in sorted(os.listdir(comp_dir)):
            if not filename.lower().endswith(".mid"):
                continue
            abs_path = os.path.join(comp_dir, filename)
            title, dkey = piece_naming.nice_name("piano-midi.de", composer, filename, abs_path)
            entries.append({
                "id": f"pianomidi::{composer}/{filename}",
                "composer": composer,
                "title": title,
                "source": "piano-midi.de",
                "era": _era_for_composer(composer),
                "path": abs_path,
                "_dedup_key": dkey,
            })
    return entries


# Lower number = preferred when the same work appears in multiple sources.
_SOURCE_PRIORITY = {"piano-midi.de": 0, "KernScores": 1, "ASAP": 2}


def _dedup(entries):
    """Collapse cross-source duplicates by canonical key, keeping the highest-
    priority source. Entries without a key (work not confidently identified)
    are always kept."""
    best = {}
    kept = []
    for e in entries:
        key = e.get("_dedup_key")
        if not key:
            kept.append(e)
            continue
        cur = best.get(key)
        if cur is None or _SOURCE_PRIORITY.get(e["source"], 9) < _SOURCE_PRIORITY.get(cur["source"], 9):
            best[key] = e
    return kept + list(best.values())


def build_index(force=False):
    global _index, _index_built_at
    now = time.time()
    if _index is None or force or (now - _index_built_at) > INDEX_TTL_SECONDS:
        _index = _dedup(_index_asap() + _index_kernscores() + _index_piano_midi())
        _index_built_at = now
    return _index


def _interleave_by_source(entries):
    """Round-robin across sources. The index is concatenated source-by-source,
    and some sources have far more (and alphabetically-earlier) titles than
    others - e.g. KernScores' numeric-prefixed chopin-first-editions - so
    picking the result set this way keeps every source represented within the
    limit instead of one crowding the rest out."""
    by_source = {}
    for e in entries:
        by_source.setdefault(e["source"], []).append(e)
    out = []
    for tup in zip_longest(*by_source.values()):
        out.extend(e for e in tup if e is not None)
    return out


def search(query, limit=120):
    index = build_index()
    query = (query or "").strip().lower()
    if not query:
        matched = index
    else:
        terms = query.split()

        def matches(entry):
            haystack = f"{entry['composer']} {entry['title']}".lower()
            return all(term in haystack for term in terms)

        matched = [e for e in index if matches(e)]

    # Pick a source-balanced set up to the limit, then present it alphabetically
    # (composer, then title). The balanced selection keeps every source visible;
    # the sort gives the alphabetic ordering.
    results = _interleave_by_source(matched)[:limit]
    results.sort(key=lambda e: (e["composer"].lower(), e["title"].lower()))
    return [
        {"id": e["id"], "composer": e["composer"], "title": e["title"],
         "source": e["source"], "era": e["era"]}
        for e in results
    ]


def find(library_id):
    """Full entry (path + era + metadata) for a library id, or None."""
    for entry in build_index():
        if entry["id"] == library_id:
            return entry
    return None
