import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import argparse

def plot_cv_metrics():
    parser = argparse.ArgumentParser(description="Plot CV Metrics")
    parser.add_argument("--tracking_uri", default="sqlite:////mnt/mdpm/d03/jhyl/deepstem_results/mlflow_runs.db")
    parser.add_argument("--output_dir", default="/srv/home/jhyl/Afib_recurrence/diplomka/results")
    parser.add_argument("--latest", action="store_true", help="Use the latest CV run automatically")
    args = parser.parse_args()

    tracking_uri = args.tracking_uri
    output_dir = args.output_dir

    print(f"Connecting to MLFlow at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get all experiments
    experiments = client.search_experiments()
    exp_ids = [exp.experiment_id for exp in experiments]

    # Find runs, ordered descending
    runs = client.search_runs(
        experiment_ids=exp_ids,
        order_by=["start_time DESC"],
        max_results=2000
    )

    child_runs = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    
    if not child_runs:
        print("Could not find any nested runs (with parentRunId) in MLflow.")
        return

    unique_groups = []
    seen_pids = set()
    for r in child_runs:
        pid = r.data.tags.get("mlflow.parentRunId")
        if pid not in seen_pids:
            seen_pids.add(pid)
            try:
                parent_run = client.get_run(pid)
                p_name = parent_run.data.tags.get("mlflow.runName", "Unnamed")
                start_time = parent_run.info.start_time
            except Exception:
                p_name = "Unknown Parent"
                start_time = r.info.start_time

            # Find all children of this parent
            children = [c for c in child_runs if c.data.tags.get("mlflow.parentRunId") == pid]
            
            date_str = pd.to_datetime(start_time, unit='ms').strftime("%Y-%m-%d %H:%M:%S")
            desc = f"CV Parent: '{p_name}' | {len(children)} folds | Date: {date_str} | ID: {pid}"
            unique_groups.append({
                'parent_id': pid,
                'p_name': p_name,
                'time': start_time,
                'desc': desc
            })
            
    # Sort by time descending
    unique_groups.sort(key=lambda x: x['time'], reverse=True)
            
    cv_runs = []
    if args.latest or len(unique_groups) == 1:
        selected_group = unique_groups[0]
        print(f"Automatically selected latest CV run: {selected_group['desc']}")
    else:
        print("\nAvailable Nested Cross Validation Runs:")
        for idx, g in enumerate(unique_groups[:20]): # Show top 20
            print(f" [{idx}] {g['desc']}")
        
        while True:
            try:
                choice = input(f"Select a CV run to plot [0-{min(len(unique_groups)-1, 19)}]: ")
                choice_idx = int(choice.strip())
                if 0 <= choice_idx < min(len(unique_groups), 20):
                    selected_group = unique_groups[choice_idx]
                    break
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")
                
    parent_id = selected_group['parent_id']
    cv_runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=f"tags.`mlflow.parentRunId` = '{parent_id}'",
        order_by=["start_time ASC"]
    )

    print(f"\nFound {len(cv_runs)} child runs for the selected parent:")
    for r in cv_runs:
        print(f" - {r.data.tags.get('mlflow.runName')} (run_id: {r.info.run_id})")

    metrics_to_fetch = ["train_loss", "valid_loss", "train_auroc", "valid_auroc"]
    
    records = []
    for run in cv_runs:
        run_name = run.data.tags.get("mlflow.runName", "Unknown_Fold")
        run_id = run.info.run_id
        
        for metric in metrics_to_fetch:
            try:
                history = client.get_metric_history(run_id, metric)
                for m in history:
                    records.append({
                        "run_id": run_id,
                        "fold": run_name,
                        "metric": metric,
                        "step": m.step,
                        "value": m.value
                    })
            except Exception as e:
                print(f"Could not fetch {metric} for {run_name}: {e}")
                
    if not records:
        print("No metrics found in these runs.")
        return
        
    df = pd.DataFrame(records)
    
    # Compute aggregates (mean, std, count) grouped by metric and step (epoch)
    agg_df = df.groupby(["metric", "step"])['value'].agg(['mean', 'std', 'count']).reset_index()
    
    specific_output_dir = os.path.join(output_dir, selected_group.get('p_name', 'Unnamed_CV_Run'))
    os.makedirs(specific_output_dir, exist_ok=True)
    
    # 1. Plot Loss
    plt.figure(figsize=(10, 6))
    for m, color, label in [("train_loss", "blue", "Train Loss"), ("valid_loss", "red", "Validation Loss")]:
        data = agg_df[agg_df["metric"] == m].sort_values("step")
        if data.empty: continue
        steps = data["step"]
        means = data["mean"]
        # Use fillna(0) for std if there is only 1 fold for that step
        stds = data["std"].fillna(0)
        
        plt.plot(steps, means, marker='o', color=color, label=label)
        plt.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2)
        
    plt.title("Cross Validation Loss (Mean ± Std)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(specific_output_dir, "cv_loss_plot.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    
    # 2. Plot AUC
    plt.figure(figsize=(10, 6))
    for m, color, label in [("train_auroc", "blue", "Train AUROC"), ("valid_auroc", "red", "Validation AUROC")]:
        data = agg_df[agg_df["metric"] == m].sort_values("step")
        if data.empty: continue
        steps = data["step"]
        means = data["mean"]
        stds = data["std"].fillna(0)
        
        plt.plot(steps, means, marker='s', color=color, label=label)
        plt.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2)
        
    plt.title("Cross Validation AUROC (Mean ± Std)")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    auc_plot_path = os.path.join(specific_output_dir, "cv_auroc_plot.png")
    plt.savefig(auc_plot_path, dpi=150)
    plt.close()
    
    print(f"Plots successfully generated and saved to {specific_output_dir}")
    print("\nAggregate Statistics at Final Extracted Epoch:")
    max_step = agg_df['step'].max()
    final_stats = agg_df[agg_df['step'] == max_step]
    print(final_stats.to_string(index=False))

if __name__ == '__main__':
    plot_cv_metrics()
