import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri(
    "https://dagshub.com/shafaq_hanif/VisionGuard.mlflow"
)

mlflow.set_experiment("MLflow Local Learning Rate Demo")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for lr in [0.001, 0.01]:
    with mlflow.start_run(run_name=f"dagshub_lr_{lr}"):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=lr,
                max_iter=1000,
                random_state=42
            ))
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions)
        }

        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(5)
        )

        print(f"\nDagsHub run: learning rate {lr}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

print("\nBoth runs logged to DagsHub.")
