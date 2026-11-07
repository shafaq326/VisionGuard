
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


# 3. Rebuild the same MobileNetV2 architecture
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(160, 160, 3))

x = layers.Lambda(
    lambda image: preprocess_input(image * 255.0)
)(inputs)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)


# 4. Load best Stage 1 checkpoint
model.load_weights("models/mobilenetv2_head_best.keras")

print("\nLoaded best Stage 1 model.")


# 5. Unfreeze only the last 20 MobileNetV2 layers
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

for layer in base_model.layers[-20:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


# 6. Display trainable layers
print("\nTRAINABLE MOBILENETV2 LAYERS:")

for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)

print("\nTotal trainable weights:", len(model.trainable_weights))


# 7. Compile with a smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)


# 8. Fine-tuning callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "models/mobilenetv2_finetuned_best.keras",
        monitor="val_loss",
        save_best_only=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001
    )
]


# 9. Fine-tune
print("\nSTARTING MOBILENETV2 FINE-TUNING")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=callbacks
)


# 10. Save history
history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "models/mobilenetv2_finetune_history.csv",
    index=False
)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

print("\nBEST FINE-TUNING EPOCH:", best_epoch)
print("BEST VALIDATION LOSS:", min(history.history["val_loss"]))
print("BEST VALIDATION ACCURACY:",
      history.history["val_accuracy"][best_epoch - 1])


# 11. Plot curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="Training Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("MobileNetV2 Fine-tuning: Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Training Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[1].set_title("MobileNetV2 Fine-tuning: Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()

plt.savefig(
    "reports/figures/mobilenetv2_finetune_curves.png",
    dpi=150
)

plt.close()

print("\nSaved fine-tuned model and training history.")
print("Saved curves: reports/figures/mobilenetv2_finetune_curves.png")
