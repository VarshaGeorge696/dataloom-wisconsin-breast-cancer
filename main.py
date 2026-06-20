"""Run all models on Wisconsin Breast Cancer dataset."""

import csv
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from argparse import Namespace

import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix

tf.disable_v2_behavior()


NUM_EPOCHS = 3000
LOG_PATH = "./logs/"
RESULT_PATH = "./results/"
SVM_C = 5

MODELS = [
    "gru_svm",
    "mlp",
    "svm",
    "linear_regression",
    "logistic_regression",
    "nearest_neighbor",
]


def main():
    print("=== Wisconsin Breast Cancer - All Models ===\n")

    # --- GRU-SVM ---
    print("[1/6] GRU-SVM")
    tf.reset_default_graph()
    from main_gru_svm import main as gru_svm_main

    gru_svm_main()

    # --- MLP ---
    print("\n[2/6] MLP")
    tf.reset_default_graph()
    from main_mlp import main as mlp_main

    mlp_args = Namespace(
        num_epochs=NUM_EPOCHS,
        log_path=LOG_PATH + "mlp/",
        result_path=RESULT_PATH + "mlp/",
    )
    mlp_main(mlp_args)

    # --- SVM ---
    print("\n[3/6] SVM")
    tf.reset_default_graph()
    from main_svm import main as svm_main

    svm_args = Namespace(
        svm_c=SVM_C,
        num_epochs=NUM_EPOCHS,
        log_path=LOG_PATH + "svm/",
        result_path=RESULT_PATH + "svm/",
    )
    svm_main(svm_args)

    # --- Linear Regression ---
    print("\n[4/6] Linear Regression")
    tf.reset_default_graph()
    from main_linear_regression import main as lr_main

    lr_main()

    # --- Logistic Regression (Softmax) ---
    print("\n[5/6] Logistic Regression")
    tf.reset_default_graph()
    from main_logistic_regression import main as logreg_main

    logreg_main()

    # --- Nearest Neighbor ---
    print("\n[6/6] Nearest Neighbor")
    tf.reset_default_graph()
    from main_nearest_neighbor import main as nn_main

    nn_main()

    print("\n=== All models complete ===")
    print(f"Logs: {LOG_PATH}")
    print(f"Results: {RESULT_PATH}")

    # Collect metrics from result .npy files
    csv_path = collect_metrics(RESULT_PATH, MODELS)
    print(f"\nMetrics CSV: {csv_path}")

    # Generate accuracy-over-steps plot from TensorBoard logs
    from utils import plot_accuracy

    plot_path = os.path.join(RESULT_PATH, "training_accuracy.png")
    plot_accuracy(log_path=LOG_PATH, save_path=plot_path)

    print("Run: uv run tensorboard --logdir=./logs/ to view accuracy graphs")


def _get_last_step_file(files):
    """From list of .npy files, return only the file with highest step number.

    Files named like 'training-model-123.npy' -> extract 123, pick max.
    """

    def extract_step(path):
        basename = os.path.splitext(os.path.basename(path))[0]
        # Last part after final '-' is the step number
        parts = basename.rsplit("-", 1)
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return -1

    return max(files, key=extract_step)


def collect_metrics(result_path, models):
    """Parse .npy result files per model, use LAST step metrics, write CSV."""
    csv_path = os.path.join(result_path, "metrics.csv")
    rows = []

    for model_name in models:
        model_dir = os.path.join(result_path, model_name)
        if not os.path.isdir(model_dir):
            print(f"  [WARN] No results dir for {model_name}, skipping")
            continue

        files = sorted(
            [
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.endswith(".npy")
            ]
        )

        if not files:
            print(f"  [WARN] No .npy files for {model_name}, skipping")
            continue

        # Filter to training files
        train_files = [f for f in files if os.path.basename(f).startswith("training-")]
        if not train_files:
            train_files = [
                f for f in files if os.path.basename(f).startswith("testing-")
            ]
        if not train_files:
            train_files = files

        # Use LAST step only (final training accuracy, matches TensorBoard end values)
        if model_name == "nearest_neighbor":
            # NN has one file per sample, use all
            last_files = train_files
        else:
            last_file = _get_last_step_file(train_files)
            last_files = [last_file]
            step_num = os.path.splitext(os.path.basename(last_file))[0].rsplit("-", 1)[
                -1
            ]
            print(f"  {model_name}: using last step {step_num}")

        # Compute metrics from last step file(s)
        if model_name == "nearest_neighbor":
            preds, actuals = _parse_nn_results(last_files)
        else:
            preds, actuals = _parse_onehot_results(last_files)

        conf = confusion_matrix(y_true=actuals, y_pred=preds, labels=[0, 1])
        tn, fp, fn, tp = conf[0][0], conf[0][1], conf[1][0], conf[1][1]

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append(
            {
                "model": model_name,
                "accuracy": round(accuracy, 6),
                "sensitivity": round(sensitivity, 6),
                "specificity": round(specificity, 6),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "total_samples": int(total),
            }
        )
        print(
            f"  {model_name}: acc={accuracy:.4f} sens={sensitivity:.4f} spec={specificity:.4f}"
        )

    os.makedirs(result_path, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "accuracy",
                "sensitivity",
                "specificity",
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
                "total_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def _parse_onehot_results(files):
    """Parse .npy files with shape (batch, 4): pred_onehot + actual_onehot."""
    all_labels = np.array([])
    for f in files:
        batch = np.load(f)
        all_labels = np.append(all_labels, batch)

    all_labels = np.reshape(all_labels, (-1, 4))
    preds = np.argmax(all_labels[:, :2], axis=1)
    actuals = np.argmax(all_labels[:, 2:], axis=1)
    return preds, actuals


def _parse_nn_results(files):
    """Parse nearest neighbor .npy files with shape (2,): [pred, actual]."""
    preds = []
    actuals = []
    for f in files:
        arr = np.load(f)
        preds.append(int(arr[0]))
        actuals.append(int(arr[1]))
    return np.array(preds), np.array(actuals)


if __name__ == "__main__":
    main()
