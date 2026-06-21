"""Generate JSON-LD for paper conclusions using dtreg (PID Type Registry).

Uses Algorithm_Evaluation schema from:
  https://typeregistry.lab.pidconsortium.net/#objects/21.T11969/5e782e67e70d0b2a022a

Produces 3 conclusion statements:
  1. All six ML algorithms exceeded 90% training accuracy.
  2. Algorithms that achieved highest test accuracy.
  3. WDBC features are linearly separable (scatter plot evidence).
"""

import csv
import json
import os

import pandas as pd
from dtreg.load_datatype import load_datatype
from dtreg.to_jsonld import to_jsonld

SCHEMA_ID = "https://doi.org/21.T11969/5e782e67e70d0b2a022a"

MODEL_NAMES = {
    "gru_svm": "GRU-SVM",
    "linear_regression": "Linear Regression",
    "mlp": "Multilayer Perceptron (MLP)",
    "nearest_neighbor": "Nearest Neighbor",
    "logistic_regression": "Softmax Regression",
    "svm": "Support Vector Machine (SVM)",
}

TASK_LABEL = "Binary Classification (Breast Cancer Detection)"
DATASET_LABEL = "Wisconsin Diagnostic Breast Cancer (WDBC)"
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)"
)


def load_metrics(metrics_path: str) -> list:
    """Read metrics CSV into list of dicts."""
    with open(metrics_path, newline="") as f:
        return list(csv.DictReader(f))


def build_statement_1(dt, rows):
    """Statement 1: All six ML algorithms exceeded 90% training accuracy."""
    algorithms = []
    for row in rows:
        algo = dt.algorithm(label=MODEL_NAMES.get(row["model"], row["model"]))
        algorithms.append(algo)

    df = pd.DataFrame(
        {
            "algorithm": [MODEL_NAMES.get(r["model"], r["model"]) for r in rows],
            "training_accuracy": [float(r["accuracy"]) for r in rows],
        }
    )
    df.name = "training_accuracy_all_models"

    output = dt.data_item(
        label="Training accuracy for all six ML algorithms",
        source_table=df,
        comment=[
            "All six machine learning algorithms exceeded 90% training accuracy.",
            "Training data metrics (last-step values from TensorBoard logs).",
        ],
    )

    evaluations = []
    for row in rows:
        algo = dt.algorithm(label=MODEL_NAMES.get(row["model"], row["model"]))
        task = dt.task(label=TASK_LABEL)
        inp = dt.data_item(
            label=DATASET_LABEL,
            source_url=dt.url(label=DATASET_URL),
            comment=["70% training / 30% testing split"],
        )

        ev = dt.algorithm_evaluation(
            label=(
                "All six ML algorithms exceeded 90% training accuracy — "
                + MODEL_NAMES.get(row["model"], row["model"])
            ),
            evaluates=algo,
            evaluates_for=task,
            has_input=[inp],
            has_output=[output],
        )
        evaluations.append(ev)

    return evaluations


def build_statement_2(dt, rows):
    """Statement 2: Algorithms that achieved highest test accuracy."""
    sorted_rows = sorted(rows, key=lambda r: float(r["accuracy"]), reverse=True)
    top = sorted_rows[:2]

    df = pd.DataFrame(
        {
            "algorithm": [MODEL_NAMES.get(r["model"], r["model"]) for r in top],
            "accuracy": [float(r["accuracy"]) for r in top],
            "sensitivity": [float(r["sensitivity"]) for r in top],
            "specificity": [float(r["specificity"]) for r in top],
        }
    )
    df.name = "highest_accuracy_models"

    output = dt.data_item(
        label="Highest test accuracy algorithms",
        source_table=df,
        comment=[
            (
                "The algorithms that achieved the highest test accuracy are "
                + " and ".join(MODEL_NAMES.get(r["model"], r["model"]) for r in top)
                + "."
            ),
            "MLP achieved ~99.04% (100% sensitivity/specificity).",
            "SVM achieved ~99.22% test accuracy.",
        ],
    )

    evaluations = []
    for row in top:
        algo = dt.algorithm(label=MODEL_NAMES.get(row["model"], row["model"]))
        task = dt.task(label=TASK_LABEL)
        inp = dt.data_item(
            label=DATASET_LABEL,
            source_url=dt.url(label=DATASET_URL),
            comment=["70% training / 30% testing split"],
        )

        ev = dt.algorithm_evaluation(
            label=(
                "Highest test accuracy — " + MODEL_NAMES.get(row["model"], row["model"])
            ),
            evaluates=algo,
            evaluates_for=task,
            has_input=[inp],
            has_output=[output],
        )
        evaluations.append(ev)

    return evaluations


def build_statement_3(dt):
    """Statement 3: WDBC features are linearly separable."""
    algo = dt.algorithm(label="Feature Analysis (Scatter Plots)")
    task = dt.task(label="Linear Separability Analysis")
    inp = dt.data_item(
        label=DATASET_LABEL,
        source_url=dt.url(label=DATASET_URL),
        comment=[
            "Features computed from digitized images of FNA tests on a breast mass."
        ],
    )
    output = dt.data_item(
        label="Linear separability evidence from scatter plots",
        comment=[
            (
                "The WDBC dataset features are linearly separable, as demonstrated "
                "by scatter plots of mean features, error features, and worst features."
            ),
            "Scatter plots of mean, SE, and worst feature groups show clear class separation.",
            "This linear separability explains the high accuracy of linear classifiers.",
        ],
    )

    ev = dt.algorithm_evaluation(
        label="WDBC features are linearly separable (scatter plot analysis)",
        evaluates=algo,
        evaluates_for=task,
        has_input=[inp],
        has_output=[output],
    )
    return [ev]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, "results", "metrics.csv")
    output_path = os.path.join(base_dir, "results", "paper-conclusions.jsonld")

    print("Loading dtreg schema for Algorithm_Evaluation...")
    dt = load_datatype(SCHEMA_ID)

    print("Reading metrics CSV...")
    rows = load_metrics(metrics_path)

    print("Building statement 1: All models > 90% training accuracy...")
    s1 = build_statement_1(dt, rows)

    print("Building statement 2: Highest test accuracy algorithms...")
    s2 = build_statement_2(dt, rows)

    print("Building statement 3: Linear separability of WDBC features...")
    s3 = build_statement_3(dt)

    all_evaluations = s1 + s2 + s3

    jsonld_strings = [to_jsonld(ev) for ev in all_evaluations]
    jsonld_objects = [json.loads(s) for s in jsonld_strings]

    doc = {
        "@context": jsonld_objects[0].get("@context", {}),
        "statements": [
            {
                "conclusion": (
                    "All six machine learning algorithms exceeded "
                    "90% training accuracy."
                ),
                "algorithm_evaluations": jsonld_objects[: len(s1)],
            },
            {
                "conclusion": (
                    "The algorithms that achieved the highest test accuracy "
                    "are MLP (~99.04%) and SVM (~99.22%)."
                ),
                "algorithm_evaluations": jsonld_objects[len(s1) : len(s1) + len(s2)],
            },
            {
                "conclusion": (
                    "The WDBC dataset features are linearly separable, "
                    "as demonstrated by scatter plots of mean features, "
                    "error features, and worst features."
                ),
                "algorithm_evaluations": jsonld_objects[len(s1) + len(s2) :],
            },
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2)

    print(f"\nJSON-LD written to {output_path}")
    print(f"  3 conclusion statements, {len(all_evaluations)} total evaluations")


if __name__ == "__main__":
    main()
