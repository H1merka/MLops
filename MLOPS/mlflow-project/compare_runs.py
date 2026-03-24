import argparse
import os

import mlflow
from dotenv import load_dotenv


METRIC_COLUMNS = [
    "metrics.f1_macro",
    "metrics.accuracy",
    "metrics.precision_macro",
    "metrics.recall_macro",
]


RUN_COLUMNS = [
    "run_id",
    "start_time",
    "params.model_name",
    "params.test_size",
    "params.random_state",
] + METRIC_COLUMNS


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Compare MLflow runs for burnout task"
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-csv", default="runs_comparison.csv")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "student_burnout_classification"
    )

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Train models first.")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_macro DESC"],
    )

    if runs_df.empty:
        raise ValueError("No runs found in the experiment. Train models first.")

    available_cols = [c for c in RUN_COLUMNS if c in runs_df.columns]
    result_df = runs_df[available_cols].head(args.top_n).copy()
    result_df.to_csv(args.output_csv, index=False)

    best = result_df.iloc[0]
    print("Top runs by f1_macro:")
    print(result_df.to_string(index=False))
    print("\nSelected best run:")
    print(f"run_id: {best['run_id']}")
    print(f"model: {best.get('params.model_name', 'n/a')}")
    print(f"f1_macro: {best.get('metrics.f1_macro', 'n/a')}")
    print(f"accuracy: {best.get('metrics.accuracy', 'n/a')}")
    print("\nRationale: best run selected by maximum f1_macro with accuracy as secondary check.")


if __name__ == "__main__":
    main()
