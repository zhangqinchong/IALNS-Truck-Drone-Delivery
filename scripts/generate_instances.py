"""Rebuild archived 102 attributes or nested exposure inputs as solver-ready CSV."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["id", "x", "y", "demand", "can_drone", "ready", "due", "max_outside"]
WINDOWS = {5: (0.5, 5, 1), 10: (1, 6, 1), 15: (1, 7, 1), 20: (1, 8, 1),
           30: (2, 9, 2), 50: (2, 10, 2), 75: (3, 12, 2), 100: (3, 15, 2)}
MASK = 0xFFFFFFFF


def hash_seed(text):
    value = 2166136261
    for byte in text.encode("ascii"):
        value = ((value ^ byte) * 16777619) & MASK
    return value


class Mulberry32:
    def __init__(self, seed):
        self.state = seed & MASK

    def random(self):
        self.state = (self.state + 0x6D2B79F5) & MASK
        value = ((self.state ^ (self.state >> 15)) * (self.state | 1)) & MASK
        value ^= (value + (((value ^ (value >> 7)) * (value | 61)) & MASK)) & MASK
        return ((value ^ (value >> 14)) & MASK) / 4294967296

    def shuffle(self, values):
        values = list(values)
        for i in range(len(values) - 1, 0, -1):
            j = math.floor(self.random() * (i + 1))
            values[i], values[j] = values[j], values[i]
        return values


def round1(value):
    # Match Math.round((value + Number.EPSILON) * 10) / 10 for positive draws.
    return math.floor((value + sys.float_info.epsilon) * 10 + 0.5) / 10


def read_excel(path):
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book.active.values)
        if list(rows[0]) != COLUMNS:
            raise ValueError(f"Unexpected columns: {path}")
        return [[None if cell == "" else cell for cell in row]
                for row in rows[1:] if row and row[0] is not None]
    finally:
        book.close()


def generate(family, size, master_seed=20260929):
    with (ROOT / "instances" / "coordinates" / f"{family}.csv").open(encoding="utf-8", newline="") as stream:
        source = {int(r["id"]): r for r in csv.DictReader(stream)}
    seed = hash_seed(f"{master_seed}:{family}:{size}")
    rng = Mulberry32(seed)
    light, medium = math.floor(0.50 * size), math.floor(0.36 * size)
    categories = rng.shuffle(["light"] * light + ["medium"] * medium + ["heavy"] * (size - light - medium))
    sensitive = set(rng.shuffle(range(1, size + 1))[:size - math.floor(0.90 * size)])
    ready_max, due_max, minimum_width = WINDOWS[size]
    depot = source[0]
    rows = [[0, float(depot["x"]), float(depot["y"]), 0, None,
             float(depot["ready"]), float(depot["due"]), None]]
    for cid, category in enumerate(categories, 1):
        lower, upper = {"light": (0.1, 0.7), "medium": (0.8, 2.5), "heavy": (2.6, 10)}[category]
        demand = round1(lower + rng.random() * (upper - lower))
        ready = round1(rng.random() * ready_max)
        due = round1(ready + minimum_width + rng.random() * (due_max - ready - minimum_width))
        if due - ready < minimum_width - 1e-9:
            due = round1(ready + minimum_width)
        outside = round1(0.3 + rng.random() * 0.7) if cid in sensitive else 100
        rows.append([cid, float(source[cid]["x"]), float(source[cid]["y"]), demand,
                     int(demand <= 2.5 + 1e-9), ready, due, outside])
    return rows, seed


def exposure_rows(design, ratio):
    rows = read_excel(ROOT / "instances" / "base" / design["base_instance"])
    active = math.floor(50 * ratio + 0.5)
    assignments = {item["customer_id"]: item["max_outside"]
                   for item in design["customer_assignments_used"][:active]}
    for row in rows:
        if row[0] != 0:
            row[7] = assignments.get(row[0], 100)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=[f"{f}{s}" for s in (101, 102) for f in ("c", "r", "rc")])
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50, 100], choices=WINDOWS)
    parser.add_argument("--master-seed", type=int)
    parser.add_argument("--exposure-ratio", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    metadata = []
    if args.exposure_ratio:
        if args.family or args.master_seed is not None:
            parser.error("Archived exposure generation does not use a new family or seed")
        settings = json.loads((ROOT / "settings" / "generation_metadata.json").read_text(encoding="utf-8"))
        for design in settings["exposure_ratio"]["designs"]:
            for ratio in settings["exposure_ratio"]["levels"]:
                name = f"{Path(design['base_instance']).stem}_exposure_ratio_{ratio:.1f}_tight.csv"
                write_csv(args.output_dir / name, exposure_rows(design, ratio))
                metadata.append(dict(file=name, ratio=ratio, selection_seed=design["customer_selection_seed"]))
    else:
        if not args.family:
            parser.error("Specify --family or --exposure-ratio")
        if args.family.endswith("101") and args.master_seed is None:
            parser.error("Historical 101 seed is unknown; new 101 instances require explicit --master-seed")
        master = 20260929 if args.master_seed is None else args.master_seed
        for size in args.sizes:
            rows, seed = generate(args.family, size, master)
            name = f"{args.family}改{size}.csv"
            write_csv(args.output_dir / name, rows)
            metadata.append(dict(file=name, master_seed=master, derived_seed=seed,
                                 archived_102_design=args.family.endswith("102") and master == 20260929))
    with (args.output_dir / "generation_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    print(f"Generated {len(metadata)} CSV inputs in {args.output_dir}")


if __name__ == "__main__":
    main()
