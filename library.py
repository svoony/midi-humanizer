"""Indexes the MusicXML corpora pulled from ASAP and KernScores (see
vendor/datasets/) so the web UI can search them and load a piece straight
into the existing render pipeline, without the user needing to manually
download and re-upload a file.
"""
import csv
import os
import re
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "vendor", "datasets")
ASAP_DIR = os.path.join(DATASETS_DIR, "asap-dataset")
ASAP_METADATA_CSV = os.path.join(ASAP_DIR, "metadata.csv")
KERNSCORES_DIR = os.path.join(DATASETS_DIR, "kernscores-musicxml")

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
    "pl-wnifc_humdrum-chopin-first-editions": "Chopin",
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


def _humanize_title(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r"[-_]+", " ", name)
    return name.strip().title()


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
            entries.append({
                "id": f"asap::{rel_path}",
                "composer": composer,
                "title": row.get("title", "").replace("_", " ").strip(),
                "source": "ASAP",
                "era": _era_for_composer(composer),
                "path": abs_path,
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
            entries.append({
                "id": f"kernscores::{os.path.relpath(abs_path, KERNSCORES_DIR)}",
                "composer": composer,
                "title": _humanize_title(filename),
                "source": "KernScores",
                "era": _era_for_composer(composer),
                "path": abs_path,
            })
    return entries


def build_index(force=False):
    global _index, _index_built_at
    now = time.time()
    if _index is None or force or (now - _index_built_at) > INDEX_TTL_SECONDS:
        _index = _index_asap() + _index_kernscores()
        _index_built_at = now
    return _index


def search(query, limit=50):
    index = build_index()
    query = (query or "").strip().lower()
    if not query:
        results = index[:limit]
    else:
        terms = query.split()

        def matches(entry):
            haystack = f"{entry['composer']} {entry['title']}".lower()
            return all(term in haystack for term in terms)

        results = [e for e in index if matches(e)][:limit]
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
