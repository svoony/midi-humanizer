"""
Performance-style MIDI <-> token vocabulary (Oore et al. "performance" encoding):
NOTE_ON_<pitch>, NOTE_OFF_<pitch>, TIME_SHIFT_<steps of 10ms, up to 1s>,
VELOCITY_<bin> (emitted only when the velocity bin changes).

This is a single shared vocabulary used for both the quantized/flat source
sequence and the expressive/performed target sequence, so one transformer
vocab covers both sides of the translation task.
"""

PITCH_MIN = 21
PITCH_MAX = 108
N_PITCHES = PITCH_MAX - PITCH_MIN + 1  # 88

TIME_SHIFT_STEPS = 100  # 100 steps * 10ms = up to 1s per token
STEP_MS = 10

N_VELOCITY_BINS = 32

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def _build_vocab():
    tokens = [PAD, BOS, EOS]
    tokens += [f"NOTE_ON_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
    tokens += [f"NOTE_OFF_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
    tokens += [f"TIME_SHIFT_{s}" for s in range(1, TIME_SHIFT_STEPS + 1)]
    tokens += [f"VELOCITY_{v}" for v in range(N_VELOCITY_BINS)]
    stoi = {t: i for i, t in enumerate(tokens)}
    itos = {i: t for i, t in enumerate(tokens)}
    return stoi, itos


STOI, ITOS = _build_vocab()
VOCAB_SIZE = len(STOI)
PAD_ID, BOS_ID, EOS_ID = STOI[PAD], STOI[BOS], STOI[EOS]


def velocity_to_bin(velocity):
    return min(N_VELOCITY_BINS - 1, (velocity * N_VELOCITY_BINS) // 128)


def bin_to_velocity(v_bin):
    # midpoint of the bin's range
    return int((v_bin + 0.5) * 128 / N_VELOCITY_BINS)


def notes_to_events(notes):
    """notes: list of objects with .pitch, .start, .end, .velocity (any order).
    Returns a list of token strings (no BOS/EOS)."""
    OFF, ON = 0, 1
    raw = []
    for n in notes:
        raw.append((n.start, ON, n.pitch, n.velocity))
        raw.append((n.end, OFF, n.pitch, None))
    raw.sort(key=lambda e: (e[0], e[1]))  # note-offs before note-ons at equal time

    tokens = []
    last_time = 0.0
    last_vel_bin = None
    for time, typ, pitch, vel in raw:
        delta = time - last_time
        if delta > 1e-6:
            steps = round(delta * 1000.0 / STEP_MS)
            while steps > 0:
                chunk = min(steps, TIME_SHIFT_STEPS)
                tokens.append(f"TIME_SHIFT_{chunk}")
                steps -= chunk
            last_time = time
        if typ == ON:
            v_bin = velocity_to_bin(vel)
            if v_bin != last_vel_bin:
                tokens.append(f"VELOCITY_{v_bin}")
                last_vel_bin = v_bin
            tokens.append(f"NOTE_ON_{pitch}")
        else:
            tokens.append(f"NOTE_OFF_{pitch}")
    return tokens


def events_to_token_ids(tokens, add_bos_eos=True):
    ids = [STOI[t] for t in tokens]
    if add_bos_eos:
        ids = [BOS_ID] + ids + [EOS_ID]
    return ids


def token_ids_to_notes(ids):
    """Inverse of notes_to_events + events_to_token_ids. Returns a list of
    (pitch, start, end, velocity) tuples. Useful for round-trip sanity checks
    and for rendering model output back to a playable MIDI file."""
    time = 0.0
    vel_bin = N_VELOCITY_BINS // 2
    open_notes = {}
    finished = []
    for i in ids:
        tok = ITOS[i]
        if tok in (PAD, BOS, EOS):
            continue
        if tok.startswith("TIME_SHIFT_"):
            steps = int(tok.split("_")[-1])
            time += steps * STEP_MS / 1000.0
        elif tok.startswith("VELOCITY_"):
            vel_bin = int(tok.split("_")[-1])
        elif tok.startswith("NOTE_ON_"):
            pitch = int(tok.split("_")[-1])
            open_notes[pitch] = (time, bin_to_velocity(vel_bin))
        elif tok.startswith("NOTE_OFF_"):
            pitch = int(tok.split("_")[-1])
            if pitch in open_notes:
                start, velocity = open_notes.pop(pitch)
                finished.append((pitch, start, time, velocity))
    finished.sort(key=lambda n: n[1])
    return finished
