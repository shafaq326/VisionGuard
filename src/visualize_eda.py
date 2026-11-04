
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

# Project settings
PROJECT_DIR = Path.cwd()
AUDIT_FILE = PROJECT_DIR / "data/processed/image_audit.csv"
OUTPUT_DIR = PROJECT_DIR / "reports/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)

# Read dataset audit CSV
with open(AUDIT_FILE, newline="") as f:
    rows = list(csv.DictReader(f))

# Keep only readable images
valid_rows = [
    row for row in rows
    if row["status"] == "OK"
    and row["file"]
]

classes = sorted(set(row["class"] for row in valid_rows))

# =====================================
# 1. Sample Image Grid
# =====================================
fig, axes = plt.subplots(
    4, len(classes),
    figsize=(10, 12),
    squeeze=False
)

for col, class_name in enumerate(classes):

    class_rows = [
        row for row in valid_rows
        if row["class"] == class_name
    ]

    samples = random.sample(
        class_rows,
        min(4, len(class_rows))
    )

    for row_index in range(4):
        ax = axes[row_index][col]
        ax.axis("off")

        if row_index < len(samples):

            image_path = Path(samples[row_index]["file"])

            if not image_path.is_absolute():
                image_path = PROJECT_DIR / image_path

            with Image.open(image_path) as img:
                ax.imshow(img.convert("RGB"))

            if row_index == 0:
                ax.set_title(class_name, fontsize=12)

fig.suptitle("Sample Images from Each Class", fontsize=16)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "sample_image_grid.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()


# =====================================
# 2. Image Width Distribution
# =====================================
plt.figure(figsize=(9, 5))

for class_name in classes:

    values = [
        int(row["width"])
        for row in valid_rows
        if row["class"] == class_name
    ]

    plt.hist(
        values,
        bins=30,
        alpha=0.5,
        label=class_name
    )

plt.title("Image Width Distribution by Class")
plt.xlabel("Width (pixels)")
plt.ylabel("Number of Images")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "width_distribution.png",
    dpi=150
)
plt.close()


# =====================================
# 3. Image Height Distribution
# =====================================
plt.figure(figsize=(9, 5))

for class_name in classes:

    values = [
        int(row["height"])
        for row in valid_rows
        if row["class"] == class_name
    ]

    plt.hist(
        values,
        bins=30,
        alpha=0.5,
        label=class_name
    )

plt.title("Image Height Distribution by Class")
plt.xlabel("Height (pixels)")
plt.ylabel("Number of Images")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "height_distribution.png",
    dpi=150
)
plt.close()


# =====================================
# 4. Image File Size Distribution
# =====================================
plt.figure(figsize=(9, 5))

for class_name in classes:

    values = [
        int(row["file_size"]) / (1024 * 1024)
        for row in valid_rows
        if row["class"] == class_name
    ]

    plt.hist(
        values,
        bins=30,
        alpha=0.5,
        label=class_name
    )

plt.title("Image File Size Distribution by Class")
plt.xlabel("File Size (MB)")
plt.ylabel("Number of Images")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "file_size_distribution.png",
    dpi=150
)
plt.close()


# =====================================
# 5. Display Generated Figures
# =====================================
print("\nEDA figures generated successfully!")
print("Total valid images:", len(valid_rows))
print("Classes:", classes)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file)
