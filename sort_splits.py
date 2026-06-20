"""
Move each paired_data/<year>/<piece>/ folder into
paired_data/<split>/<year>/<piece>/ according to the official
MAESTRO train/validation/test split in maestro-v3.0.0.csv.
"""
import csv
import os
import shutil

CSV_PATH = "raw_midi/maestro-v3.0.0/maestro-v3.0.0.csv"
PAIRED_ROOT = "paired_data"


def main():
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    moved, missing = 0, []
    for row in rows:
        split = row["split"]
        year, filename = row["midi_filename"].split("/", 1)
        piece = os.path.splitext(filename)[0]

        src = os.path.join(PAIRED_ROOT, year, piece)
        dest = os.path.join(PAIRED_ROOT, split, year, piece)

        if not os.path.isdir(src):
            missing.append(src)
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)
        moved += 1

    print(f"moved {moved} pieces, {len(missing)} missing")
    if missing:
        for m in missing:
            print("MISSING:", m)

    # remove now-empty year folders left at the root
    for entry in os.listdir(PAIRED_ROOT):
        path = os.path.join(PAIRED_ROOT, entry)
        if os.path.isdir(path) and entry not in ("train", "validation", "test"):
            if not os.listdir(path):
                os.rmdir(path)
            else:
                print("non-empty leftover dir:", path)


if __name__ == "__main__":
    main()
