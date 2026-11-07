
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from data_pipeline import create_dataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/mobilenetv2_finetuned_best.keras"

# Load the saved fine-tuned model.
# safe_mode=False is used because this is our own trusted model
# and it contains a Lambda preprocessing layer.
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)


# --------------------------------------------------
# Prediction helper
# --------------------------------------------------

def get_predictions(dataset):
    y_true = []
    y_prob = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0).reshape(-1)

        y_true.extend(labels.numpy().astype(int))
        y_prob.extend(probabilities)

    return np.array(y_true), np.array(y_prob)


# --------------------------------------------------
# 1. Select threshold using validation data only
# --------------------------------------------------

print("\nLoading validation data...")

validation_ds = create_dataset("validation", augment=False)
val_true, val_prob = get_predictions(validation_ds)

thresholds = np.linspace(0.05, 0.95, 181)
threshold_results = []

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)

    score = f1_score(
        val_true,
        val_pred,
        zero_division=0
    )

    threshold_results.append((score, threshold))

best_val_f1, best_threshold = max(threshold_results)

print("\nVALIDATION THRESHOLD SELECTION")
print(f"Selected threshold: {best_threshold:.3f}")
print(f"Validation F1 at selected threshold: {best_val_f1:.4f}")


# --------------------------------------------------
# 2. Final test evaluation (once)
# --------------------------------------------------

print("\nLoading test data for final evaluation...")

test_ds = create_dataset("test", augment=False)
test_true, test_prob = get_predictions(test_ds)

test_pred = (test_prob >= best_threshold).astype(int)


# --------------------------------------------------
# 3. Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(test_true, test_pred)

precision = precision_score(
    test_true,
    test_pred,
    zero_division=0
)

recall = recall_score(
    test_true,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    test_true,
    test_pred,
    zero_division=0
)

auc = roc_auc_score(test_true, test_prob)

cm = confusion_matrix(
    test_true,
    test_pred,
    labels=[0, 1]
)


# --------------------------------------------------
# 4. Display results
# --------------------------------------------------

print("\nFINAL TEST RESULTS")
print(f"Threshold: {best_threshold:.3f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")

print("\nConfusion matrix (rows=true, columns=predicted)")
print("Class order: AI (0), Real (1)")
print(cm)


# --------------------------------------------------
# 5. Save metrics and predictions
# --------------------------------------------------

results = {
    "selected_threshold": float(best_threshold),
    "validation_f1_at_threshold": float(best_val_f1),
    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "test_roc_auc": float(auc),
    "confusion_matrix": cm.tolist(),
    "class_order": ["AI (0)", "Real (1)"]
}

with open("reports/test_evaluation.json", "w") as file:
    json.dump(results, file, indent=4)


predictions_df = pd.DataFrame({
    "true_label": test_true,
    "real_probability": test_prob,
    "predicted_label": test_pred,
    "correct": test_true == test_pred
})

predictions_df.to_csv(
    "reports/test_predictions.csv",
    index=False
)


# --------------------------------------------------
# 6. Confusion matrix plot
# --------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["AI", "Real"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("MobileNetV2 Fine-Tuned: Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "reports/figures/test_confusion_matrix.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 7. ROC curve
# --------------------------------------------------

fpr, tpr, _ = roc_curve(test_true, test_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("MobileNetV2 Fine-Tuned: Test ROC Curve")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/figures/test_roc_curve.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 8. Completion message
# --------------------------------------------------

print("\nSaved successfully:")
print("- reports/test_evaluation.json")
print("- reports/test_predictions.csv")
print("- reports/figures/test_confusion_matrix.png")
print("- reports/figures/test_roc_curve.png")
