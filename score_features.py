"""
Score-derived harmonic / tonal-tension / phrase features for the
PerformanceCVAE.

Everything here is computed from the *quantized score* (pitches + grid
steps), so it is reproducible identically at training time (grid recomputed
from original.midi) and at inference (grid estimated from the flat MIDI the
player feeds). Nothing in here may look at the performance - these are
decoder inputs, and a decoder input the player can't reconstruct would cause
train/inference skew.

The full harmonic feature set the design calls for - local key, scale
degree, chord root + quality, harmonic function, a spiral-array tension
ribbon, vertical dissonance, and heuristic phrase structure - is implemented
as a vectorized NumPy pipeline rather than via music21.chordify /
romanNumeralFromChord. The information content is the same, but a literal
music21 pass costs hours over the whole dataset and seconds-per-upload in the
player; this costs neither.

Public entry point: compute_score_features(...). It returns a dict of
per-note arrays whose keys line up with the new categorical / continuous
fields consumed by note_dataset and model.
"""
import numpy as np

# ---- vocab sizes (kept in sync with note_dataset constants) ------------------
KEY_TONIC_VOCAB = 13      # 0..11 pitch class, 12 = unknown (too few notes)
KEY_MODE_VOCAB = 3        # 0 major, 1 minor, 2 unknown
SCALE_DEGREE_VOCAB = 13   # 0..11 chromatic degree from tonic, 12 = unknown
CHORD_ROOT_VOCAB = 13     # 0..11 pitch class, 12 = none (silence)
CHORD_QUALITY_VOCAB = 12  # see QUALITY_* below
ROMAN_DEGREE_VOCAB = 8    # I..VII = 0..6, 7 = chromatic/other

KEY_UNKNOWN = 12
MODE_UNKNOWN = 2
DEGREE_UNKNOWN = 12
ROOT_NONE = 12
ROMAN_OTHER = 7

# chord-quality ids
(Q_NONE, Q_MAJ, Q_MIN, Q_DIM, Q_AUG, Q_DOM7, Q_MAJ7, Q_MIN7,
 Q_HALFDIM7, Q_DIM7, Q_SUS, Q_OTHER) = range(12)

# (intervals-from-root, quality-id) templates, richer chords first so they win
# ties over their triad subsets.
_CHORD_TEMPLATES = [
    ((0, 4, 7, 10), Q_DOM7),
    ((0, 4, 7, 11), Q_MAJ7),
    ((0, 3, 7, 10), Q_MIN7),
    ((0, 3, 6, 10), Q_HALFDIM7),
    ((0, 3, 6, 9), Q_DIM7),
    ((0, 4, 7), Q_MAJ),
    ((0, 3, 7), Q_MIN),
    ((0, 3, 6), Q_DIM),
    ((0, 4, 8), Q_AUG),
    ((0, 5, 7), Q_SUS),
]

# Krumhansl-Kessler key profiles.
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# semitone-from-tonic -> roman degree bucket (major+minor scale tones folded
# onto the same functional degree; b2 and the tritone fall through to "other").
_SEMITONE_TO_ROMAN = {0: 0, 2: 1, 3: 2, 4: 2, 5: 3, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6}

# interval-class -> roughness weight (octave-folded), for vertical dissonance.
_IC_ROUGHNESS = np.array([0.0, 1.0, 0.55, 0.20, 0.12, 0.06, 0.85, 0.0, 0.12, 0.20, 0.55, 1.0])


def _build_profiles():
    profs = np.empty((24, 12))
    for t in range(12):
        profs[t] = np.roll(_KK_MAJOR, t)
        profs[12 + t] = np.roll(_KK_MINOR, t)
    return profs - profs.mean(axis=1, keepdims=True)


_PROFILES_CENTERED = _build_profiles()


def _build_quality_matrix():
    mat = np.zeros((len(_CHORD_TEMPLATES), 12))
    qids = np.empty(len(_CHORD_TEMPLATES), dtype=np.int64)
    for i, (intervals, qid) in enumerate(_CHORD_TEMPLATES):
        for iv in intervals:
            mat[i, iv] = 1.0
        qids[i] = qid
    return mat, qids


_QUALITY_MAT, _QUALITY_IDS = _build_quality_matrix()
_QUALITY_SIZE = _QUALITY_MAT.sum(axis=1)  # tones per template


def _spiral_coords():
    """3D spiral-array point per pitch class (Chew). Neighbours on the line of
    fifths are 90 deg around and one step up the helix."""
    h = np.sqrt(2.0 / 15.0)
    coords = np.zeros((12, 3))
    for pc in range(12):
        k = (pc * 7) % 12          # position on the line of fifths
        coords[pc] = [np.sin(k * np.pi / 2), np.cos(k * np.pi / 2), k * h]
    return coords


_SPIRAL = _spiral_coords()


def _windowed_key(grid_steps, pc, durations, subdivisions_per_beat):
    """Local key per note via a trailing+leading window Krumhansl correlation.
    Returns (tonic[n], mode[n]) with UNKNOWN where the window is too sparse."""
    n = len(grid_steps)
    tonic = np.full(n, KEY_UNKNOWN, dtype=np.int64)
    mode = np.full(n, MODE_UNKNOWN, dtype=np.int64)
    if n == 0:
        return tonic, mode

    # per-pitch-class duration mass, accumulated along the (sorted) onset axis
    pc_dur = np.zeros((n, 12))
    pc_dur[np.arange(n), pc] = durations
    cumsum = np.vstack([np.zeros(12), np.cumsum(pc_dur, axis=0)])  # (n+1, 12)

    half = 8 * subdivisions_per_beat  # ~4 measures of 4/4 either side
    lo = np.searchsorted(grid_steps, grid_steps - half, side="left")
    hi = np.searchsorted(grid_steps, grid_steps + half, side="right")
    hist = cumsum[hi] - cumsum[lo]                                  # (n, 12)

    mass = hist.sum(axis=1)
    valid = mass > 1e-6
    if not valid.any():
        return tonic, mode

    h = hist[valid] - hist[valid].mean(axis=1, keepdims=True)
    corr = h @ _PROFILES_CENTERED.T                                # (m, 24)
    best = corr.argmax(axis=1)
    t = np.where(valid)[0]
    tonic[t] = best % 12
    mode[t] = best // 12
    return tonic, mode


def _slice_harmony(active_counts):
    """Best (root_pc, quality_id) for a sounding pitch-class count vector via
    template matching. Returns (ROOT_NONE, Q_NONE) for silence."""
    total = active_counts.sum()
    if total <= 0:
        return ROOT_NONE, Q_NONE
    # rotations[r, j] = weight of pc (r + j) mod 12  -> align root r to template col 0
    idx = (np.arange(12)[:, None] + np.arange(12)[None, :]) % 12
    rotations = active_counts[idx]                                 # (12 roots, 12)
    matched = rotations @ _QUALITY_MAT.T                           # (12, n_templates)
    present = (rotations > 0).astype(np.float64) @ _QUALITY_MAT.T  # template tones sounding

    # coverage = how much of the sounding mass the chord explains; completeness
    # = fraction of the chord's tones actually present. The completeness term is
    # what lets a full triad beat an incomplete seventh that contains it.
    coverage = matched / total
    completeness = present / _QUALITY_SIZE
    scores = coverage + 0.3 * completeness
    flat = scores.argmax()
    root = flat // scores.shape[1]
    template = flat % scores.shape[1]
    if matched[root, template] <= 0:
        return ROOT_NONE, Q_NONE
    return int(root), int(_QUALITY_IDS[template])


def _phrase_features(onsets, subdivisions_per_beat):
    """Heuristic phrase segmentation over the unique-onset axis: a boundary
    falls where a rest/gap to the next onset is unusually long. Returns per
    *slice* arrays (boundary_strength, position_in_phrase, dist_to_boundary)."""
    s = len(onsets)
    if s == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0))
    gaps = np.diff(onsets).astype(np.float64)
    if len(gaps) == 0:
        return (np.zeros(1), np.zeros(1), np.zeros(1))

    med = max(np.median(gaps), 1.0)
    beat = subdivisions_per_beat
    # boundary where the gap after this onset is both a real rest (>=1 beat)
    # and well above the local typical gap
    is_boundary = (gaps >= beat) & (gaps >= 2.5 * med)
    bnd_idx = np.flatnonzero(is_boundary)

    strength = np.zeros(s)
    strength[:-1] = np.clip(gaps / (4.0 * med), 0.0, 1.0)

    # phrase id = number of boundaries seen so far; boundary closes a phrase
    phrase_id = np.zeros(s, dtype=np.int64)
    if len(bnd_idx):
        phrase_id[bnd_idx + 1] = 1
    phrase_id = np.cumsum(phrase_id)

    pos = np.zeros(s)
    dist = np.zeros(s)
    for pid in range(phrase_id[-1] + 1):
        members = np.flatnonzero(phrase_id == pid)
        m = len(members)
        pos[members] = np.arange(m) / max(m - 1, 1)
        # notes-to-end-of-phrase, normalized into ~[0,1] over the phrase
        dist[members] = (m - 1 - np.arange(m)) / max(m, 1)
    return strength, pos, dist


def compute_score_features(pitches, grid_steps, end_grid_steps,
                           subdivisions_per_beat=4, beats_per_measure=4):
    """Per-note harmonic / tension / phrase features. All inputs are the
    quantized score; all outputs are length-n arrays aligned to `pitches`."""
    n = len(pitches)
    out = {
        "key_tonic": np.full(n, KEY_UNKNOWN, dtype=np.int64),
        "key_mode": np.full(n, MODE_UNKNOWN, dtype=np.int64),
        "scale_degree": np.full(n, DEGREE_UNKNOWN, dtype=np.int64),
        "chord_root": np.full(n, ROOT_NONE, dtype=np.int64),
        "chord_quality": np.zeros(n, dtype=np.int64),
        "roman_degree": np.full(n, ROMAN_OTHER, dtype=np.int64),
        "tension_diameter": np.zeros(n, dtype=np.float32),
        "tension_strain": np.zeros(n, dtype=np.float32),
        "tension_momentum": np.zeros(n, dtype=np.float32),
        "dissonance": np.zeros(n, dtype=np.float32),
        "phrase_strength": np.zeros(n, dtype=np.float32),
        "phrase_pos": np.zeros(n, dtype=np.float32),
        "phrase_dist": np.zeros(n, dtype=np.float32),
    }
    if n == 0:
        return out

    pitches = np.asarray(pitches, dtype=np.int64)
    grid_steps = np.asarray(grid_steps, dtype=np.int64)
    end_grid_steps = np.asarray(end_grid_steps, dtype=np.int64)
    pc = pitches % 12
    durations = np.maximum(end_grid_steps - grid_steps, 1).astype(np.float64)

    # --- local key + melodic scale degree ---
    tonic, mode = _windowed_key(grid_steps, pc, durations, subdivisions_per_beat)
    out["key_tonic"], out["key_mode"] = tonic, mode
    known = tonic != KEY_UNKNOWN
    sd = np.full(n, DEGREE_UNKNOWN, dtype=np.int64)
    sd[known] = (pc[known] - tonic[known]) % 12
    out["scale_degree"] = sd

    # --- per-slice harmony + tension via a sweep over unique onsets ---
    onsets, inv = np.unique(grid_steps, return_inverse=True)
    s = len(onsets)
    on_order = np.argsort(grid_steps, kind="stable")
    off_order = np.argsort(end_grid_steps, kind="stable")
    on_ptr = off_ptr = 0
    active = np.zeros(12)            # pitch-class counts currently sounding

    sl_root = np.full(s, ROOT_NONE, dtype=np.int64)
    sl_qual = np.zeros(s, dtype=np.int64)
    sl_diam = np.zeros(s, dtype=np.float32)
    sl_strain = np.zeros(s, dtype=np.float32)
    sl_moment = np.zeros(s, dtype=np.float32)
    sl_diss = np.zeros(s, dtype=np.float32)

    # representative note per slice (first note at that onset) for local key
    first_note_at = np.zeros(s, dtype=np.int64)
    seen = np.zeros(s, dtype=bool)
    for i in range(n):
        si = inv[i]
        if not seen[si]:
            seen[si] = True
            first_note_at[si] = i

    prev_ce = None
    for si in range(s):
        o = onsets[si]
        while on_ptr < n and grid_steps[on_order[on_ptr]] <= o:
            active[pc[on_order[on_ptr]]] += 1
            on_ptr += 1
        while off_ptr < n and end_grid_steps[off_order[off_ptr]] <= o:
            active[pc[off_order[off_ptr]]] -= 1
            off_ptr += 1
        np.maximum(active, 0, out=active)  # guard against fp/dup drift

        root, qual = _slice_harmony(active)
        sl_root[si] = root
        sl_qual[si] = qual

        present = np.flatnonzero(active > 0)
        if len(present):
            pts = _SPIRAL[present]
            weights = active[present][:, None]
            ce = (pts * weights).sum(axis=0) / weights.sum()
            if len(present) > 1:
                d = pts[:, None, :] - pts[None, :, :]
                sl_diam[si] = np.sqrt((d ** 2).sum(axis=-1)).max()
                # vertical dissonance: roughness-weighted mean over pc pairs
                ics = np.abs(present[:, None] - present[None, :]) % 12
                iu = np.triu_indices(len(present), k=1)
                sl_diss[si] = _IC_ROUGHNESS[ics[iu]].mean()
            t = tonic[first_note_at[si]]
            if t != KEY_UNKNOWN:
                third = 3 if mode[first_note_at[si]] == 1 else 4
                triad = [(t) % 12, (t + third) % 12, (t + 7) % 12]
                key_center = _SPIRAL[triad].mean(axis=0)
                sl_strain[si] = np.linalg.norm(ce - key_center)
            if prev_ce is not None:
                sl_moment[si] = np.linalg.norm(ce - prev_ce)
            prev_ce = ce

    # roman degree from chord root vs local key (at each slice's first note)
    sl_roman = np.full(s, ROMAN_OTHER, dtype=np.int64)
    for si in range(s):
        if sl_root[si] == ROOT_NONE:
            continue
        t = tonic[first_note_at[si]]
        if t == KEY_UNKNOWN:
            continue
        sl_roman[si] = _SEMITONE_TO_ROMAN.get((sl_root[si] - t) % 12, ROMAN_OTHER)

    # --- phrase structure (per slice) ---
    ph_strength, ph_pos, ph_dist = _phrase_features(onsets, subdivisions_per_beat)

    # --- broadcast slice arrays back to notes ---
    out["chord_root"] = sl_root[inv]
    out["chord_quality"] = sl_qual[inv]
    out["roman_degree"] = sl_roman[inv]
    out["tension_diameter"] = sl_diam[inv]
    out["tension_strain"] = sl_strain[inv]
    out["tension_momentum"] = sl_moment[inv]
    out["dissonance"] = sl_diss[inv]
    out["phrase_strength"] = ph_strength[inv]
    out["phrase_pos"] = ph_pos[inv]
    out["phrase_dist"] = ph_dist[inv]
    return out
