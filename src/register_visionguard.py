import mlflow
import tensorflow as tf

from mlflow.tracking import MlflowClient
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

mlflow.set_tracking_uri(
    "https://dagshub.com/shafaq_hanif/VisionGuard.mlflow"
)
mlflow.set_experiment("VisionGuard")

MODEL_NAME = "VisionGuardClassifier"

client = MlflowClient()

models = [
    {
        "name": "Baseline CNN Registry Version",
        "path": "models/baseline_cnn_best.keras",
        "version_label": "Baseline CNN",
        "custom_objects": {}
    },
    {
        "name": "Fine Tuned MobileNetV2 Registry Version",
        "path": "models/mobilenetv2_finetuned_best.keras",
        "version_label": "MobileNetV2 Fine-Tuned",
        "custom_objects": {"preprocess_input": preprocess_input}
    }
]

registered_versions = []

for item in models:
    model = tf.keras.models.load_model(
        item["path"],
        compile=False,
        safe_mode=False,
        custom_objects=item["custom_objects"]
    )

    with mlflow.start_run(run_name=item["name"]) as run:
        mlflow.set_tag("project", "VisionGuard")
        mlflow.set_tag("registry_version", item["version_label"])

        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        run_id = run.info.run_id

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    matching = [
        version for version in versions
        if version.run_id == run_id
    ]

    if not matching:
        raise RuntimeError(
            f"Could not find registered version for {item['name']}"
        )

    version = matching[0].version
    registered_versions.append(version)

    print(
        f"Registered {item['version_label']} as "
        f"{MODEL_NAME} version {version}"
    )

champion_version = registered_versions[1]

client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    champion_version
)

print("\nRegistry setup completed.")
print("Model:", MODEL_NAME)
print("Baseline version:", registered_versions[0])
print("Fine-tuned version:", registered_versions[1])
print("Champion alias points to version:", champion_version)
