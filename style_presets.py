"""Per-piece starting values for the render levers, so a freshly-loaded
library piece opens with musically-sensible settings instead of flat neutral
defaults.

These are heuristic, not learned: an era profile (the broad stylistic
baseline) refined by composer-specific overrides and a few piece-type signals
read from the title. They're deliberately a starting point - the user can
adjust any of them afterward via the chat box.

Note the model already conditions on era internally, so the model-scaling
levers (rubato/dynamics/pedal/articulation) stay near 1.0 here - the bigger,
more deliberate departures are in the three post-processing levers the model
has no opinion about (melody_emphasis, chord_roll, metric_accent), which would
otherwise always start at 0.
"""

# The seven expressive levers this module sets. tempo_multiplier and
# pedal_boost are intentionally left at their neutral defaults, and era is
# assigned separately (from the composer) by the library.
_LEVER_KEYS = (
    "rubato_intensity", "dynamics_intensity", "pedal_scale",
    "articulation_intensity", "melody_emphasis", "chord_roll", "metric_accent",
)
_LEVER_BOUNDS = (0.0, 2.0)

ERA_PROFILES = {
    # steady pulse, terraced dynamics, dry (harpsichord heritage), voices
    # balanced for counterpoint, crisp articulation
    "baroque":   {"rubato_intensity": 0.8,  "dynamics_intensity": 0.9,  "pedal_scale": 0.4,
                  "articulation_intensity": 1.2, "melody_emphasis": 0.3, "chord_roll": 0.0, "metric_accent": 0.6},
    # clear, melody-led homophony, light pedal, disciplined timing
    "classical": {"rubato_intensity": 0.9,  "dynamics_intensity": 1.0,  "pedal_scale": 0.7,
                  "articulation_intensity": 1.1, "melody_emphasis": 0.5, "chord_roll": 0.0, "metric_accent": 0.5},
    # expressive rubato, lush pedal, singing top line, flowing over metric stress
    "romantic":  {"rubato_intensity": 1.25, "dynamics_intensity": 1.15, "pedal_scale": 1.2,
                  "articulation_intensity": 0.9, "melody_emphasis": 0.8, "chord_roll": 0.35, "metric_accent": 0.3},
    # broad bucket; leans impressionist-lush (color/pedal) as the default
    "modern":    {"rubato_intensity": 1.1,  "dynamics_intensity": 1.15, "pedal_scale": 1.2,
                  "articulation_intensity": 0.95, "melody_emphasis": 0.6, "chord_roll": 0.25, "metric_accent": 0.45},
}

# Composer-specific overrides merged on top of the era profile. Most telling
# for composers whose style departs from their era bucket - above all Joplin,
# who is era-tagged "modern" but whose ragtime wants the opposite of the
# impressionist default: steady, crisp, strong-beat.
COMPOSER_OVERRIDES = {
    "Joplin":       {"rubato_intensity": 0.5, "pedal_scale": 0.5, "articulation_intensity": 1.2,
                     "melody_emphasis": 0.6, "chord_roll": 0.0, "metric_accent": 1.0},
    "Chopin":       {"rubato_intensity": 1.35, "pedal_scale": 1.3, "melody_emphasis": 0.9, "chord_roll": 0.45},
    "Liszt":        {"rubato_intensity": 1.3, "dynamics_intensity": 1.3, "pedal_scale": 1.25,
                     "melody_emphasis": 0.85, "chord_roll": 0.4},
    "Rachmaninoff": {"rubato_intensity": 1.25, "dynamics_intensity": 1.25, "pedal_scale": 1.3,
                     "melody_emphasis": 0.85, "chord_roll": 0.4},
    "Debussy":      {"rubato_intensity": 1.1, "pedal_scale": 1.4, "articulation_intensity": 0.85,
                     "melody_emphasis": 0.5, "chord_roll": 0.3, "metric_accent": 0.2},
    "Ravel":        {"pedal_scale": 1.3, "articulation_intensity": 0.95, "melody_emphasis": 0.55,
                     "metric_accent": 0.3},
}

_CONTRAPUNTAL = ("fugue", "fughetta", "invention", "canon", "contrapunctus", "ricercar")
_DANCE = ("waltz", "valse", "mazurka", "minuet", "menuet", "scherzo", "march",
          "landler", "polonaise", "rag", "tango", "gigue", "gavotte")
_LYRICAL = ("nocturne", "romance", "berceuse", "reverie", "song", "lied", "cantabile", "arabesque")


def _clamp(v):
    lo, hi = _LEVER_BOUNDS
    return max(lo, min(hi, v))


def preset_for(composer, era, title):
    """Returns a dict of starting lever values for a piece. Does not include
    era (set separately) or tempo/pedal_boost (left neutral)."""
    p = dict(ERA_PROFILES.get(era, ERA_PROFILES["romantic"]))
    p.update(COMPOSER_OVERRIDES.get((composer or "").strip(), {}))

    t = (title or "").lower()
    if any(k in t for k in _CONTRAPUNTAL):
        # clarity over wash; balance the voices rather than spotlight one
        p["pedal_scale"] = min(p["pedal_scale"], 0.4)
        p["melody_emphasis"] = min(p["melody_emphasis"], 0.25)
        p["articulation_intensity"] = max(p["articulation_intensity"], 1.2)
        p["chord_roll"] = 0.0
    if any(k in t for k in _DANCE):
        p["metric_accent"] = max(p["metric_accent"], 0.8)
    if any(k in t for k in _LYRICAL):
        p["rubato_intensity"] = max(p["rubato_intensity"], 1.2)
        p["pedal_scale"] = max(p["pedal_scale"], 1.1)
        p["melody_emphasis"] = max(p["melody_emphasis"], 0.8)

    return {k: round(_clamp(p[k]), 2) for k in _LEVER_KEYS}
