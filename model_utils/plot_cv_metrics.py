import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

try:
    import optuna
except ImportError:
    optuna = None

# --- GLOBAL PLOT SETTINGS FOR A4 PRINTING (Vertical Stack) ---
plt.rcParams.update({
    'mathtext.fontset': 'cm',         
    'font.family': 'serif',           
    'axes.unicode_minus': False,
    'figure.figsize': (6.27, 8.0),    
    'figure.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3
})
# -------------------------------------------------------------

def plot_cv_metrics():
    parser = argparse.ArgumentParser(description="Plot CV Metrics")
    parser.add_argument("--tracking_uri", default="sqlite:////mnt/mdpm/d03/jhyl/diplomka/_AFIB_code/mlflow.db")
    parser.add_argument("--output_dir", default="/srv/home/jhyl/Afib_recurrence/diplomka/results")
    parser.add_argument("--latest", action="store_true", help="Use the latest CV run automatically")
    
    # NEW ARGUMENT: Direct trial selection
    parser.add_argument("--trial_number", type=int, default=None, 
                        help="Specific Optuna trial number to plot (e.g., 42). Bypasses interactive menu.")
                        
    parser.add_argument("--optuna_db", default=None,
                        help="Path to Optuna SQLite DB. When given, only trials from this study are shown.")
    parser.add_argument("--study_name", default=None,
                        help="Optuna study name inside the DB.")
    args = parser.parse_args()

    tracking_uri = args.tracking_uri
    output_dir = args.output_dir

    print(f"Connecting to MLFlow at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # --- Optuna filtering ---
    optuna_trial_names = None
    if args.optuna_db is not None:
        if optuna is None:
            print("WARNING: optuna is not installed — ignoring --optuna_db filter.")
        else:
            db_path = args.optuna_db
            if not db_path.startswith("sqlite://"):
                db_path = f"sqlite:///{db_path}"
            print(f"Loading Optuna study from {db_path}")
            try:
                if args.study_name:
                    study = optuna.load_study(study_name=args.study_name, storage=db_path)
                else:
                    summaries = optuna.get_all_study_summaries(storage=db_path)
                    if not summaries:
                        print("No studies found in the Optuna DB — ignoring filter.")
                    else:
                        study = optuna.load_study(study_name=summaries[0].study_name, storage=db_path)
                        print(f"Auto-selected Optuna study: '{study.study_name}'")
                optuna_trial_names = {f"Trial_{t.number}" for t in study.trials}
                print(f"Found {len(optuna_trial_names)} trials in study '{study.study_name}'")
            except Exception as e:
                print(f"Could not load Optuna study: {e} — ignoring filter.")
    # -------------------------

    experiments = client.search_experiments()
    exp_ids = [exp.experiment_id for exp in experiments]

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

            if optuna_trial_names is not None and p_name not in optuna_trial_names:
                continue

            children = [c for c in child_runs if c.data.tags.get("mlflow.parentRunId") == pid]
            date_str = pd.to_datetime(start_time, unit='ms').strftime("%Y-%m-%d %H:%M:%S")
            desc = f"CV Parent: '{p_name}' | {len(children)} folds | Date: {date_str} | ID: {pid}"
            unique_groups.append({
                'parent_id': pid,
                'p_name': p_name,
                'time': start_time,
                'desc': desc
            })
            
    unique_groups.sort(key=lambda x: x['time'], reverse=True)
            
    # --- RUN SELECTION LOGIC ---
    cv_runs = []
    selected_group = None
    
    if args.trial_number is not None:
        target_name = f"Trial_{args.trial_number}"
        # Search for the requested trial in the collected groups
        for g in unique_groups:
            if g['p_name'] == target_name:
                selected_group = g
                break
                
        if selected_group is not None:
            print(f"Automatically selected requested run: {selected_group['desc']}")
        else:
            print(f"\nERROR: Could not find an MLflow run named '{target_name}'.")
            print("Please ensure the trial exists or run without --trial_number to see the interactive list.")
            return
            
    elif args.latest or len(unique_groups) == 1:
        selected_group = unique_groups[0]
        print(f"Automatically selected latest CV run: {selected_group['desc']}")
    else:
        print("\nAvailable Nested Cross Validation Runs:")
        for idx, g in enumerate(unique_groups[:100]):
            print(f" [{idx}] {g['desc']}")
        
        while True:
            try:
                choice = input(f"Select a CV run to plot [0-{min(len(unique_groups)-1, 99)}]: ")
                choice_idx = int(choice.strip())
                if 0 <= choice_idx < min(len(unique_groups), 100):
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
    pivot_df = df.pivot_table(index=["metric", "step"], columns="fold", values="value")
    pivot_df = pivot_df.groupby(level="metric").ffill()
    filled_df = pivot_df.reset_index().melt(id_vars=["metric", "step"], value_name="value").dropna()
    agg_df = filled_df.groupby(["metric", "step"])['value'].agg(['mean', 'std', 'count']).reset_index()
    
    specific_output_dir = os.path.join(output_dir, selected_group.get('p_name', 'Unnamed_CV_Run'))
    os.makedirs(specific_output_dir, exist_ok=True)
    
    # ==========================================
    # --- COMBINED PLOTTING LOGIC ---
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Panel A: Loss
    for m, color, label in [("train_loss", "blue", "Train Loss"), ("valid_loss", "red", "Validation Loss")]:
        data = agg_df[agg_df["metric"] == m].sort_values("step")
        if data.empty: continue
        steps = data["step"] + 1
        means = data["mean"]
        stds = data["std"].fillna(0)
        
        ax1.plot(steps, means, marker='o', color=color, label=label, markersize=4)
        ax1.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2)
        
    ax1.set_title(r"Cross Validation Loss (Mean $\pm$ Std)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    
    # Panel B: AUROC
    for m, color, label in [("train_auroc", "blue", "Train AUROC"), ("valid_auroc", "red", "Validation AUROC")]:
        data = agg_df[agg_df["metric"] == m].sort_values("step")
        if data.empty: continue
        steps = data["step"] + 1
        means = data["mean"]
        stds = data["std"].fillna(0)
        
        ax2.plot(steps, means, marker='s', color=color, label=label, markersize=4)
        ax2.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2)
        
    ax2.set_title(r"Cross Validation AUROC (Mean $\pm$ Std)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUROC")
    ax2.legend(loc="lower right")

    # Add Panel Letters (A and B)
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='bottom')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='bottom')

    # Give a 0.5 buffer on each side so the dots fit perfectly inside the box
    ax1.set_xlim([0.5, 20.5])
    ax2.set_xlim([0.5, 20.5])
    
    # Explicitly print 1, then count by 5s
    ticks = [1, 5, 10, 15, 20]
    ax1.set_xticks(ticks)
    ax2.set_xticks(ticks)

    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_path = os.path.join(specific_output_dir, "cv_learning_curves_combined.png")
    plt.savefig(combined_plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot successfully generated and saved to {combined_plot_path}")
    
    print("\nAggregate Statistics at Final Extracted Epoch:")
    max_step = agg_df['step'].max()
    final_stats = agg_df[agg_df['step'] == max_step]
    print(final_stats.to_string(index=False))

if __name__ == '__main__':
    plot_cv_metrics()