"""Turns the raw, source-specific titles/filenames in the library into clean
display names, and produces a canonical key for cross-source de-duplication.

Design principle: prefer reformatting tokens that are *already in the data*
(catalog numbers, opus, movement numbers) over recalling them from memory, so
names stay accurate. A small curated table expands famous nicknames that the
sources encode as short codes (appass -> Appassionata). Anything not
confidently decodable falls back to a cleaned, composer-prefix-stripped,
title-cased version rather than a guessed name.
"""
import os
import re

# Catalog abbreviation per composer, for when a bare number in the source is
# that composer's catalogue number (e.g. Mozart 311 -> K. 311).
CATALOG_PREFIX = {
    "Mozart": "K.", "Bach": "BWV", "Schubert": "D.", "Haydn": "Hob. XVI:",
    "Scarlatti": "K.",
}

# Beethoven piano sonata number -> (opus text, nickname or "").  Lets the three
# conventions in the library (piano-midi.de nicknames, KernScores "sonataNN",
# ASAP "Piano Sonatas N-M") canonicalise to the same opus for de-duplication.
BEETHOVEN_SONATAS = {
    1: ("2 No. 1", ""), 2: ("2 No. 2", ""), 3: ("2 No. 3", ""), 4: ("7", ""),
    5: ("10 No. 1", ""), 6: ("10 No. 2", ""), 7: ("10 No. 3", ""), 8: ("13", "Pathétique"),
    9: ("14 No. 1", ""), 10: ("14 No. 2", ""), 11: ("22", ""), 12: ("26", ""),
    13: ("27 No. 1", ""), 14: ("27 No. 2", "Moonlight"), 15: ("28", "Pastoral"),
    16: ("31 No. 1", ""), 17: ("31 No. 2", "Tempest"), 18: ("31 No. 3", ""),
    19: ("49 No. 1", ""), 20: ("49 No. 2", ""), 21: ("53", "Waldstein"), 22: ("54", ""),
    23: ("57", "Appassionata"), 24: ("78", ""), 25: ("79", ""), 26: ("81a", "Les Adieux"),
    27: ("90", ""), 28: ("101", ""), 29: ("106", "Hammerklavier"), 30: ("109", ""),
    31: ("110", ""), 32: ("111", ""),
}
_BEETHOVEN_OPUS_TO_NO = {opus: no for no, (opus, _) in BEETHOVEN_SONATAS.items()}

# Mozart piano sonata number -> Köchel (standard numbering), so piano-midi.de
# (which encodes K. in the filename) and KernScores (which encodes the number)
# canonicalise to the same Köchel for de-duplication.
MOZART_SONATAS = {
    1: 279, 2: 280, 3: 281, 4: 282, 5: 283, 6: 284, 7: 309, 8: 310, 9: 311,
    10: 330, 11: 331, 12: 332, 13: 333, 14: 457, 15: 545, 16: 570, 17: 576, 18: 533,
}
_MOZART_K_TO_NO = {k: no for no, k in MOZART_SONATAS.items()}

# piano-midi.de filename stems -> (display title, canonical work id). Covers the
# nickname/abbreviation-coded pieces that can't be decoded by rule. Systematic
# stems (mz_311_1, chpn_op10_e01, haydn_33_1, schub_d960_1 ...) are handled by
# rules in _piano_midi_title instead.
PIANO_MIDI_WORKS = {
    "elise": ("Bagatelle \"Für Elise\", WoO 59", "beethoven:wungrelise"),
    "appass": ("Sonata No. 23 \"Appassionata\", Op. 57", "beethoven:son23"),
    "mond": ("Sonata No. 14 \"Moonlight\", Op. 27 No. 2", "beethoven:son14"),
    "pathetique": ("Sonata No. 8 \"Pathétique\", Op. 13", "beethoven:son8"),
    "waldstein": ("Sonata No. 21 \"Waldstein\", Op. 53", "beethoven:son21"),
    "beethoven_hammerklavier": ("Sonata No. 29 \"Hammerklavier\", Op. 106", "beethoven:son29"),
    "beethoven_les_adieux": ("Sonata No. 26 \"Les Adieux\", Op. 81a", "beethoven:son26"),
    "islamei": ("Islamey", "balakirev:islamey"),
    "liz_liebestraum": ("Liebestraum No. 3", "liszt:liebestraum3"),
    "liz_donjuan": ("Réminiscences de Don Juan", "liszt:donjuan"),
    "schum_abegg": ("Abegg Variations, Op. 1", "schumann:abegg"),
    "schub_d760": ("Wanderer Fantasy, D. 760", "schubert:d760"),
}

# Named-piece sets keyed by composer where piano-midi.de uses a numeric index:
# muss_N (Pictures at an Exhibition), bor_psN (Petite Suite), etc. We keep these
# as "<Set>, No. N" without inventing the individual movement titles.
PIANO_MIDI_NUMBERED_SETS = {
    "muss": "Pictures at an Exhibition",
    "bor_ps": "Petite Suite",
    "debussy_cc": "Children's Corner",
    "deb_cc": "Children's Corner",
}

_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
          9: "IX", 10: "X", 11: "XI", 12: "XII"}


def _titlecase(text):
    small = {"and", "of", "the", "in", "for", "a", "no", "op", "de", "la", "le"}
    words = re.split(r"[\s_\-]+", text.strip())
    out = []
    for i, w in enumerate(words):
        if not w:
            continue
        if w.isupper() and len(w) <= 4:  # keep acronyms (BWV)
            out.append(w)
        elif i and w.lower() in small:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _clean_fallback(stem, composer):
    """Strip a leading composer-name/abbreviation token, then title-case."""
    s = stem
    comp = composer.lower()
    prefixes = [comp, comp[:5], comp[:4], comp[:3]]
    for p in sorted(set(prefixes), key=len, reverse=True):
        if p and s.lower().startswith(p):
            rest = s[len(p):].lstrip("_-")
            if rest:
                s = rest
                break
    return _titlecase(s)


def _mvt(n):
    return f", Mvt {n}" if n else ""


def _piano_midi_title(stem, composer):
    """Decode a piano-midi.de filename stem -> (display title, work key).
    The work key omits the movement so a sonata's movements share one work
    identity for de-duplication; the movement is folded into the final key by
    nice_name()."""
    s = stem.lower()

    # exact curated works (famous nicknames), possibly with a trailing _movement
    for code, (title, key) in PIANO_MIDI_WORKS.items():
        if s == code or s.startswith(code + "_"):
            tail = s[len(code):].lstrip("_")
            mv = int(tail) if tail.isdigit() else 0
            return (title + _mvt(mv), key)

    def num(x):
        m = re.search(r"\d+", x)
        return int(m.group()) if m else 0

    # Mozart sonatas: mz_311_1 -> Piano Sonata No. 9, K. 311, Mvt 1
    m = re.match(r"mz_(\d+)_?(\d+)?$", s)
    if m:
        k, mv = int(m.group(1)), int(m.group(2) or 0)
        no = _MOZART_K_TO_NO.get(k)
        nopart = f"No. {no}, " if no else ""
        return (f"Piano Sonata {nopart}K. {k}{_mvt(mv)}", f"mozart:k{k}")

    # Haydn sonatas: haydn_33_1 / hay_40_1 -> Sonata Hob. XVI:33, Mvt 1
    m = re.match(r"hay(?:dn)?_(\d+)_?(\d+)?$", s)
    if m:
        h, mv = m.group(1), int(m.group(2) or 0)
        return (f"Piano Sonata, Hob. XVI:{h}{_mvt(mv)}", f"haydn:hob{h}")

    # Schubert D-numbered: schub_d960_1 / schubert_d850_1 -> Sonata D. 960, Mvt 1
    m = re.match(r"schub(?:ert)?_d(\d+)_?(\d+)?$", s)
    if m:
        d, mv = m.group(1), int(m.group(2) or 0)
        return (f"Piano Sonata, D. {d}{_mvt(mv)}", f"schubert:d{d}")
    # Schubert opus-coded sonata: schu_143_1 -> Sonata, Op. 143, Mvt 1
    m = re.match(r"schu_(\d+)_?(\d+)?$", s)
    if m:
        op, mv = m.group(1), int(m.group(2) or 0)
        return (f"Piano Sonata, Op. {op}{_mvt(mv)}", f"schubert:op{op}")
    if re.match(r"schuim-?\d+$", s):
        return (f"Impromptu No. {num(s)}", f"schubert:impromptu{num(s)}")
    if re.match(r"schumm-?\d+$", s):
        return (f"Moment Musical No. {num(s)}", f"schubert:mm{num(s)}")

    # Chopin
    m = re.match(r"chpn-p(\d+)$", s)
    if m:
        return (f"Prelude, Op. 28 No. {m.group(1)}", f"chopin:prelude{m.group(1)}")
    m = re.match(r"chpn_op(\d+)_e(\d+)$", s)
    if m:
        return (f"Étude, Op. {m.group(1)} No. {int(m.group(2))}", f"chopin:op{m.group(1)}e{int(m.group(2))}")
    m = re.match(r"chp?n?_op(\d+)_?(\d+)?$", s)
    if m:
        op, sub = m.group(1), m.group(2)
        suff = f" No. {sub}" if sub else ""
        return (f"Op. {op}{suff}", f"chopin:op{op}{('n'+sub) if sub else ''}")

    # Albéniz España / Suite española
    m = re.match(r"alb_esp(\d+)$", s)
    if m:
        return (f"España, Op. 165, No. {m.group(1)}", f"albeniz:esp{m.group(1)}")
    m = re.match(r"alb_se(\d+)$", s)
    if m:
        return (f"Suite española, Op. 47, No. {m.group(1)}", f"albeniz:se{m.group(1)}")

    # Bach WTC prelude & fugue
    m = re.match(r"bach_(\d+)$", s)
    if m:
        return (f"Prelude and Fugue, BWV {m.group(1)}", f"bach:bwv{m.group(1)}")

    # Beethoven opus-coded sonatas: beethoven_opus22_1 -> Sonata, Op. 22, Mvt 1.
    # Key by sonata number (consistent with the nickname/KernScores/ASAP keys);
    # only opus values that map to a single sonata get a key (so the ambiguous
    # "opus10", which is three sonatas, is left un-deduped).
    m = re.match(r"beethoven_opus(\d+\w?)_(\d+)$", s)
    if m:
        op, mv = m.group(1), int(m.group(2))
        no = _BEETHOVEN_OPUS_TO_NO.get(op)
        key = f"beethoven:son{no}" if no else None
        return (f"Piano Sonata, Op. {op}{_mvt(mv)}", key)

    # Brahms opus: brahms_opus117_1 -> Op. 117 No. 1
    m = re.match(r"brahms_opus(\d+)_(\d+)$", s)
    if m:
        return (f"Op. {m.group(1)} No. {m.group(2)}", f"brahms:op{m.group(1)}n{m.group(2)}")

    # Mendelssohn Songs Without Words: mendel_op19_1
    m = re.match(r"mendel_op(\d+)_(\d+)$", s)
    if m:
        return (f"Song Without Words, Op. {m.group(1)} No. {m.group(2)}", f"mendelssohn:op{m.group(1)}n{m.group(2)}")

    # Schumann numbered sets
    for code, name in (("scn15", "Kinderszenen"), ("scn16", "Kreisleriana"), ("scn68", "Album for the Young")):
        m = re.match(code + r"_(\d+)$", s)
        if m:
            opus = {"scn15": "15", "scn16": "16", "scn68": "68"}[code]
            return (f"{name}, Op. {opus}, No. {m.group(1)}", f"schumann:op{opus}n{m.group(1)}")

    # Liszt
    m = re.match(r"liz_rhap0*(\d+)$", s)
    if m:
        return (f"Hungarian Rhapsody No. {m.group(1)}", f"liszt:rhap{m.group(1)}")
    m = re.match(r"liz_et(\d+)$", s)
    if m:
        return (f"Transcendental Étude No. {m.group(1)}", f"liszt:te{m.group(1)}")

    # Tchaikovsky: The Seasons (German month names)
    months = {"januar": "January", "februar": "February", "maerz": "March", "april": "April",
              "mai": "May", "juni": "June", "juli": "July", "august": "August",
              "september": "September", "oktober": "October", "november": "November", "dezember": "December"}
    m = re.match(r"ty_(\w+)$", s)
    if m and m.group(1) in months:
        mo = months[m.group(1)]
        return (f"The Seasons: {mo}", f"tchaikovsky:seasons:{mo.lower()}")

    # Generic numbered sets (Pictures, Petite Suite, Children's Corner)
    for code, setname in sorted(PIANO_MIDI_NUMBERED_SETS.items(), key=lambda kv: -len(kv[0])):
        m = re.match(re.escape(code) + r"_?(\d+)$", s)
        if m:
            return (f"{setname}, No. {m.group(1)}", f"{composer.lower()}:{code}{m.group(1)}")

    return (_clean_fallback(stem, composer), None)


def _kernscores_title(stem, composer):
    """KernScores: sonataNN-M -> Sonata No. NN, Mvt M (with Beethoven nickname/
    opus where known); Joplin filenames title-cased."""
    m = re.match(r"sonata(\d+)-(\d+)([a-z]?)$", stem.lower())
    if m:
        no, mv = int(m.group(1)), int(m.group(2))
        if composer == "Beethoven" and no in BEETHOVEN_SONATAS:
            opus, nick = BEETHOVEN_SONATAS[no]
            nickpart = f" \"{nick}\"" if nick else ""
            return (f"Piano Sonata No. {no}{nickpart}, Op. {opus}{_mvt(mv)}", f"beethoven:son{no}")
        if composer == "Mozart" and no in MOZART_SONATAS:
            k = MOZART_SONATAS[no]
            return (f"Piano Sonata No. {no}, K. {k}{_mvt(mv)}", f"mozart:k{k}")
        if composer == "Haydn":
            # KernScores Haydn filenames already use Hoboken numbers
            return (f"Piano Sonata, Hob. XVI:{no}{_mvt(mv)}", f"haydn:hob{no}")
        return (f"Piano Sonata No. {no}{_mvt(mv)}", f"{composer.lower()}:son{no}")
    # Joplin and anything else: just clean the filename
    return (_titlecase(stem), None)


def _asap_title(raw, composer):
    """ASAP titles are space-separated already; normalise catalogue tokens and
    'N-M' (sonata No. - movement) patterns."""
    t = raw.strip()
    t = re.sub(r"\bbwv\b", "BWV", t, flags=re.I)
    t = re.sub(r"\bop\.?\s*", "Op. ", t, flags=re.I)
    # "Op. 8 11" (two trailing numbers after opus) -> "Op. 8 No. 11"
    t = re.sub(r"(Op\. \d+)\s+(\d+)\b", r"\1 No. \2", t)
    # "D.899 1" -> "D. 899 No. 1"
    t = re.sub(r"\bD\.?\s*(\d+)\s+(\d+)\b", r"D. \1 No. \2", t)
    # "Sonatas 11-1" -> "Sonata No. 11, Mvt 1"
    m = re.search(r"(Sonatas?)\s+(\d+)-(\d+)", t, flags=re.I)
    key = None
    if m:
        no, mv = int(m.group(2)), m.group(3)
        t = re.sub(r"Sonatas?\s+\d+-\d+", f"Sonata No. {no}, Mvt {mv}", t, flags=re.I)
        # movement-less work key; nice_name() folds in the Mvt from the title
        if composer == "Beethoven" and no in BEETHOVEN_SONATAS:
            key = f"beethoven:son{no}"
        elif composer == "Mozart" and no in MOZART_SONATAS:
            key = f"mozart:k{MOZART_SONATAS[no]}"
        else:
            key = f"{composer.lower()}:son{no}"
    # "Ballades 1" -> "Ballade No. 1" (one plural work-word + a trailing number)
    t = re.sub(r"^([A-Z][a-z]+)s (\d+)$", r"\1 No. \2", t)
    # Mozart catalogues standalone works by Köchel: "Fantasie 475" -> "Fantasia, K. 475"
    if composer == "Mozart":
        t = re.sub(r"^Fantasie (\d+)$", r"Fantasia, K. \1", t)
    return (t, key)


def nice_name(source, composer, raw, path):
    """Returns (display_title, canonical_key). canonical_key is None when we
    can't confidently identify the work across sources (those items are never
    treated as duplicates of anything)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if source == "piano-midi.de":
        title, work = _piano_midi_title(stem, composer)
    elif source == "KernScores":
        title, work = _kernscores_title(stem, composer)
    else:
        title, work = _asap_title(raw, composer)

    # fold the movement number into the key so movements stay distinct items
    key = None
    if work:
        mvm = re.search(r"Mvt (\d+)", title)
        key = f"{work}:m{mvm.group(1)}" if mvm else work
    return title, key
