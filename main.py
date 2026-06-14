"""Run all models on Wisconsin Breast Cancer dataset."""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from argparse import Namespace

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


NUM_EPOCHS = 3000
LOG_PATH = "./logs/"
RESULT_PATH = "./results/"
SVM_C = 5


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
    print("Run: uv run tensorboard --logdir=./logs/ to view accuracy graphs")


if __name__ == "__main__":
    main()
