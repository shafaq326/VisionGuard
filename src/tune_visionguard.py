
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from data_pipeline import create_dataset

tf.keras.utils.set_random_seed(42)

train_ds = create_dataset("train", augment=True)
val_ds = create_dataset("validation", augment=False)


def build_model(hp):
    learning_rate = hp.Choice(
        "learning_rate", [0.0001, 0.00005, 0.00001]
    )
    dropout_rate = hp.Choice(
        "dropout", [0.3, 0.4, 0.5]
    )
    unfrozen_layers = hp.Choice(
        "unfrozen_layers", [10, 20, 30]
    )

    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(160, 160, 3)
    )

    base_model.trainable = True

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-unfrozen_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    inputs = keras.Input(shape=(160, 160, 3))
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return model


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective("val_auc", direction="max"),
    max_trials=8,
    executions_per_trial=1,
    directory="tuner_results",
    project_name="visionguard",
    overwrite=True,
    seed=42
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

best_model.save("models/visionguard_tuned_best.keras")

with open("reports/visionguard_tuner_results.txt", "w") as f:
    f.write(f"Best learning rate: {best_hp.get('learning_rate')}\n")
    f.write(f"Best dropout: {best_hp.get('dropout')}\n")
    f.write(f"Best unfrozen layers: {best_hp.get('unfrozen_layers')}\n")
    f.write(f"Best validation AUC: {best_trial.score:.4f}\n")

print("Best learning rate:", best_hp.get("learning_rate"))
print("Best dropout:", best_hp.get("dropout"))
print("Best unfrozen layers:", best_hp.get("unfrozen_layers"))
print("Best validation AUC:", round(best_trial.score, 4))
print("Completed trials:", len(tuner.oracle.trials))
