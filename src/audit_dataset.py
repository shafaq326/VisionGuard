
from pathlib import Path
from PIL import Image
import hashlib
import csv
import statistics
from collections import Counter, defaultdict

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []
hashes = defaultdict(list)
corrupt_files = []

def calculate_hash(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

for class_dir in sorted(DATA_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for path in class_dir.rglob("*"):
        if not path.is_file():
            continue

        record = {
            "class": label,
            "file": str(path),
            "file_size": path.stat().st_size,
            "width": "",
            "height": "",
            "colour_mode": "",
            "format": "",
            "sha256": "",
            "status": "OK"
        }

        try:
            image_hash = calculate_hash(path)

            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                record.update({
                    "width": img.width,
                    "height": img.height,
                    "colour_mode": img.mode,
                    "format": img.format,
                    "sha256": image_hash
                })

            hashes[image_hash].append(str(path))

        except Exception as error:
            record["status"] = "CORRUPT"
            corrupt_files.append(str(path))

        records.append(record)

# Save complete audit table
csv_path = OUTPUT_DIR / "image_audit.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# Exact duplicates: same SHA-256 hash
duplicates = {
    h: paths for h, paths in hashes.items() if len(paths) > 1
}

valid_records = [r for r in records if r["status"] == "OK"]

print("\nIMAGE COUNTS PER CLASS")
for label, count in Counter(r["class"] for r in records).items():
    print(f"{label}: {count}")

print("\nMINIMUM, MEDIAN AND MAXIMUM")
for label in sorted(set(r["class"] for r in records)):
    class_records = [
        r for r in valid_records if r["class"] == label
    ]

    print(f"\n{label}")
    for field in ["width", "height", "file_size"]:
        values = [int(r[field]) for r in class_records]
        if values:
            print(
                f"{field}: min={min(values)}, "
                f"median={statistics.median(values)}, "
                f"max={max(values)}"
            )

print("\nCORRUPT FILES:", len(corrupt_files))
print("DUPLICATE GROUPS:", len(duplicates))
print("TOTAL IMAGES AUDITED:", len(records))
print("Audit table saved to:", csv_path)

