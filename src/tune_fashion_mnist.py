
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    keras.datasets.fashion_mnist.load_data()
)

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Use a smaller subset to keep CPU tuning manageable
x_train = x_train[:12000]
y_train = y_train[:12000]
x_val = x_test[:2000]
y_val = y_test[:2000]

def build_model(hp):
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),

        layers.Dense(
            units=hp.Choice("units", [64, 128, 256]),
            activation="relu"
        ),

        layers.Dense(10, activation="softmax")
    ])

    learning_rate = hp.Choice(
        "learning_rate",
        [0.001, 0.0005, 0.0001]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=6,
    executions_per_trial=1,
    directory="tuner_results",
    project_name="fashion_mnist",
    overwrite=True,
    seed=SEED
)

tuner.search(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=3,
    batch_size=128,
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nBEST FASHION-MNIST SETTINGS")
print("Units:", best_hp.get("units"))
print("Learning rate:", best_hp.get("learning_rate"))

best_model = tuner.get_best_models(num_models=1)[0]

test_loss, test_accuracy = best_model.evaluate(
    x_test, y_test, verbose=0
)

print("Test accuracy:", round(test_accuracy, 4))

os.makedirs("reports", exist_ok=True)

with open("reports/fashion_tuner_results.txt", "w") as f:
    f.write(f"Best units: {best_hp.get('units')}\n")
    f.write(f"Best learning rate: {best_hp.get('learning_rate')}\n")
    f.write(f"Test accuracy: {test_accuracy:.4f}\n")

print("\nResults saved to reports/fashion_tuner_results.txt")
