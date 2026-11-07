import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from build_model import build_cnn
from data_pipeline import create_dataset

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Limit CPU thread usage
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# Load datasets
train_ds = create_dataset("train", augment=True)
val_ds = create_dataset("validation")

# Build model
model = build_cnn()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "models/baseline_cnn_best.keras",
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv("models/baseline_cnn_history.csv", index=False)

# Find best epoch based on validation loss
best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

print("\nBEST EPOCH:", best_epoch)
print("BEST VALIDATION LOSS:", min(history.history["val_loss"]))
print("BEST VALIDATION ACCURACY:",
      max(history.history["val_accuracy"]))

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="Training Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].axvline(best_epoch - 1, linestyle="--", label="Best Epoch")
axes[0].set_title("CNN Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Training Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[1].axvline(best_epoch - 1, linestyle="--", label="Best Epoch")
axes[1].set_title("CNN Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("reports/figures/baseline_cnn_curves.png", dpi=150)
plt.close()

print("\nSaved best model: models/baseline_cnn_best.keras")
print("Saved history: models/baseline_cnn_history.csv")
print("Saved curves: reports/figures/baseline_cnn_curves.png")