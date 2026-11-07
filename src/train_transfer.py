
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from data_pipeline import create_dataset


# 1. Reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


# 2. Load datasets
train_ds = create_dataset("train", augment=True)
val_ds = create_dataset("validation")


# 3. Load pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)

# Freeze all pretrained layers initially
base_model.trainable = False


# 4. Build classification head
inputs = layers.Input(shape=(160, 160, 3))

# Convert pixel values from [0, 1] to [-1, 1]
x = layers.Lambda(
    lambda image: preprocess_input(image * 255.0)
)(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)


# 5. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

print("\nMOBILENETV2 MODEL SUMMARY")
model.summary()


# 6. Training callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "models/mobilenetv2_head_best.keras",
        monitor="val_loss",
        save_best_only=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )
]


# 7. Train classification head
print("\nTRAINING CLASSIFICATION HEAD")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)


# 8. Save training history
history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "models/mobilenetv2_head_history.csv",
    index=False
)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

print("\nBEST HEAD TRAINING EPOCH:", best_epoch)
print("BEST VALIDATION LOSS:", min(history.history["val_loss"]))
print("BEST VALIDATION ACCURACY:",
      history.history["val_accuracy"][best_epoch - 1])


# 9. Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="Training Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("MobileNetV2 Head: Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Training Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[1].set_title("MobileNetV2 Head: Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig(
    "reports/figures/mobilenetv2_head_curves.png",
    dpi=150
)
plt.close()

print("\nSaved head model and training history.")
print("Saved curves: reports/figures/mobilenetv2_head_curves.png")
