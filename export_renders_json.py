"""
Export every renders/<name>/{input_flat,model_rendered,ground_truth}.mid
into a single JSON file for the piano-roll diagnostic dashboard, since the
dashboard runs in a sandboxed environment with no filesystem access and
needs the note data embedded directly.
"""
import glob
import json
import os

import pretty_midi

RENDERS_DIR = "renders"
TRACKS = ["input_flat", "model_rendered", "ground_truth"]


SUSTAIN_CC = 64


def track_to_dict(path):
    if not os.path.exists(path):
        return None
    pm = pretty_midi.PrettyMIDI(path)
    inst = pm.instruments[0]
    notes = sorted(inst.notes, key=lambda n: n.start)
    pedal = sorted(
        (cc for cc in inst.control_changes if cc.number == SUSTAIN_CC),
        key=lambda cc: cc.time,
    )
    return {
        "notes": [[n.pitch, round(n.start, 4), round(n.end, 4), n.velocity] for n in notes],
        "pedal": [[round(cc.time, 4), cc.value] for cc in pedal],
    }


def main():
    out = {}
    for run_dir in sorted(glob.glob(os.path.join(RENDERS_DIR, "*"))):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        out[name] = {}
        for track in TRACKS:
            data = track_to_dict(os.path.join(run_dir, f"{track}.mid"))
            if data is not None:
                out[name][track] = data

    out_path = os.path.join(RENDERS_DIR, "renders_data.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"wrote {out_path} ({len(out)} renders)")
    return out


if __name__ == "__main__":
    main()
