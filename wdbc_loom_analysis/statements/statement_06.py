"""Statement 06

Nearest Neighbor training accuracy on the WDBC dataset.
"""

from pathlib import Path

import pandas as pd
from dtreg.load_datatype import load_datatype
from dtreg.to_jsonld import to_jsonld

DATA_ANALYSIS_SCHEMA = "https://doi.org/21.T11969/feeb33ad3e4440682a4d"
SCHEMA_ID = "https://doi.org/21.T11969/5e782e67e70d0b2a022a"

results_df = pd.DataFrame({
    "accuracy": [0.929825],
    "sensitivity": [0.943925],
    "specificity": [0.90625],
    "true_positives": [101],
    "true_negatives": [58],
    "false_positives": [6],
    "false_negatives": [6],
    "total_samples": [171],
})
results_df.name = "nearest_neighbor_metrics"

# Data Analysis and Algorithm Evaluation datatypes

dt_da = load_datatype(DATA_ANALYSIS_SCHEMA)
dt_ae = load_datatype(SCHEMA_ID)

algorithm_evaluation = dt_ae.algorithm_evaluation(
    label="Training accuracy evaluation of Nearest Neighbor",
    executes=dt_ae.software_method(
        label="models.nearest_neighbor.NearestNeighbor.predict",
        is_implemented_by="NearestNeighbor.predict(...)",
        has_support_url="https://www.tensorflow.org/api_docs/python/tf",
        part_of=dt_ae.software_library(
            label="TensorFlow",
            has_support_url="https://www.tensorflow.org/",
            part_of=dt_ae.software(
                label="Python",
                version_info="3.12.3",
            ),
        ),
    ),
    evaluates=dt_ae.algorithm(label="Nearest Neighbor"),
    evaluates_for=dt_ae.task(label="Binary Classification (Breast Cancer Detection)"),
    has_input=dt_ae.data_item(
        label="WDBC dataset",
        source_url="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)",
    ),
    has_output=dt_ae.data_item(
        source_table=results_df,
        has_part=[
            dt_ae.component(label="Accuracy"),
            dt_ae.component(label="Sensitivity"),
            dt_ae.component(label="Specificity"),
            dt_ae.component(label="True Positives"),
            dt_ae.component(label="True Negatives"),
            dt_ae.component(label="False Positives"),
            dt_ae.component(label="False Negatives"),
            dt_ae.component(label="Total Samples"),
        ],
    ),
)

instance = dt_da.data_analysis(
    label="Nearest Neighbor training accuracy statement",
    is_implemented_by="main_nearest_neighbor.py",
    has_part=algorithm_evaluation,
)

GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
output_file = GENERATED_DIR / "statement_06.json"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(to_jsonld(instance))

print(f"JSON-LD written to {output_file}")