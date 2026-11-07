
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Load test split information and saved predictions
splits = pd.read_csv("data/processed/splits.csv")
predictions = pd.read_csv("reports/test_predictions.csv")

# Keep only test images
test_data = splits[splits["split"] == "test"].reset_index(drop=True)

# Match predictions with their corresponding image paths.
# The prediction order follows the test dataset order.
if len(test_data) != len(predictions):
    raise ValueError(
        f"Mismatch: {len(test_data)} test images but "
        f"{len(predictions)} saved predictions."
    )

test_data["predicted_label"] = predictions["predicted_label"].astype(int).values
test_data["real_probability"] = predictions["real_probability"].values
test_data["correct"] = predictions["correct"].astype(bool).values

# Select incorrectly classified images
errors = test_data[test_data["correct"] == False].copy()

print(f"Total test images: {len(test_data)}")
print(f"Misclassified images: {len(errors)}")

if len(errors) == 0:
    print("No misclassified images found.")
    raise SystemExit

# Select up to 10 errors
selected = errors.head(10).reset_index(drop=True)

# Create figure
fig, axes = plt.subplots(2, 5, figsize=(18, 9))
axes = axes.flatten()

for i, row in selected.iterrows():
    image_path = Path(row["file"])

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    actual = "AI" if row["label"] == 0 else "Real"
    predicted = "AI" if row["predicted_label"] == 0 else "Real"

    axes[i].imshow(image)
    axes[i].set_title(
        f"Actual: {actual}\nPredicted: {predicted}\n"
        f"Real probability: {row['real_probability']:.2f}",
        fontsize=9
    )
    axes[i].axis("off")

# Hide unused axes if fewer than 10 errors
for i in range(len(selected), len(axes)):
    axes[i].axis("off")

plt.suptitle("Misclassified Test Images - MobileNetV2", fontsize=15)
plt.tight_layout()

output_path = "reports/figures/misclassified_test_images.png"
plt.savefig(output_path, dpi=180, bbox_inches="tight")
plt.show()

# Save a table for writing the error analysis
selected.to_csv(
    "reports/misclassified_test_images.csv",
    index=False
)

print("\nSaved:")
print("- reports/figures/misclassified_test_images.png")
print("- reports/misclassified_test_images.csv")

