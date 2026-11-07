import mlflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC

from data_pipeline import create_dataset
from build_model import build_cnn

mlflow.set_tracking_uri(
    "https://dagshub.com/shafaq_hanif/VisionGuard.mlflow"
)
mlflow.set_experiment("VisionGuard")

mlflow.tensorflow.autolog(
    log_models=False,
    log_input_examples=False
)

train_ds = create_dataset("train", augment=True)
val_ds = create_dataset("validation", augment=False)

learning_rates = [0.001, 0.0005, 0.0001]

for lr in learning_rates:
    with mlflow.start_run(run_name=f"CNN Autolog LR {lr}"):
        mlflow.set_tag("logging_method", "autolog")
        mlflow.set_tag("project", "VisionGuard")
        mlflow.set_tag("architecture", "Custom CNN")

        model = build_cnn()

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc")
            ]
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            verbose=1
        )

        print(f"Completed autolog run with learning rate {lr}")

print("\nAll three autologging runs completed.")
