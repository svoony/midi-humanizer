"""
Walk raw_midi/, normalize every performance MIDI, and write each
original/normalized pair into paired_data/<year>/<piece_name>/.
"""
import os
import glob
import shutil
import traceback

from normalize_midi import normalize_midi

RAW_ROOT = "raw_midi/maestro-v3.0.0"
OUT_ROOT = "paired_data"


def main():
    files = sorted(glob.glob(os.path.join(RAW_ROOT, "**", "*.midi"), recursive=True))
    print(f"found {len(files)} midi files")

    failures = []
    for i, src in enumerate(files):
        year = os.path.basename(os.path.dirname(src))
        piece = os.path.splitext(os.path.basename(src))[0]
        dest_dir = os.path.join(OUT_ROOT, year, piece)
        os.makedirs(dest_dir, exist_ok=True)

        orig_dest = os.path.join(dest_dir, "original.midi")
        norm_dest = os.path.join(dest_dir, "normalized.midi")

        try:
            shutil.copy2(src, orig_dest)
            normalize_midi(src, norm_dest)
        except Exception:
            failures.append(src)
            print(f"FAILED: {src}")
            traceback.print_exc()

        if i % 100 == 0:
            print(f"{i}/{len(files)}")

    print(f"done. {len(files) - len(failures)} succeeded, {len(failures)} failed")
    if failures:
        with open("failed_files.txt", "w") as f:
            f.write("\n".join(failures))


if __name__ == "__main__":
    main()
