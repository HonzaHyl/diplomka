import optuna
import numpy as np
import os
import mlflow
import torch
import multiprocessing
from main_code import training_code_kfold, CONFIG
from device_selector import DeviceSelector

# --- HPO CONFIGURATION ---
DATA_DIR = "/srv/home/jhyl/Afib_recurrence/finetune_data_all/train"
MODEL_DIR = "/srv/home/jhyl/Afib_recurrence/diplomka/results/hpo_runs"
N_TOTAL_TRIALS = 150
K_FOLDS = 4
N_GPUS = 3
# -------------------------

def objective(trial, device):
    head_lr = trial.suggest_float("head_lr", 1e-6, 1e-3, log=True)
    decay_factor = trial.suggest_float("layer_decay", 0.1, 0.5)
    backbone_wd = trial.suggest_float("backbone_weight_decay", 1e-4, 1e-1, log=True)
    head_wd = trial.suggest_float("head_weight_decay", 1e-4, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.7)
    
    rb_4_lr = head_lr * decay_factor
    rb_3_lr = head_lr * (decay_factor ** 2)
    early_rb_lr = head_lr * (decay_factor ** 3)
    
    trial_config = CONFIG.copy()
    trial_config.update({
        "dropout_rate": dropout_rate,
        "layer_tuning": {
            "conv": {"trainable": False, "lr": 0, "weight_decay": 0},
            "bn":   {"trainable": False, "lr": 0, "weight_decay": 0},
            "rb_0": {"trainable": True, "lr": early_rb_lr, "weight_decay": backbone_wd},
            "rb_1": {"trainable": True, "lr": early_rb_lr, "weight_decay": backbone_wd},
            "rb_2": {"trainable": True, "lr": early_rb_lr, "weight_decay": backbone_wd},
            "rb_3": {"trainable": True, "lr": rb_3_lr,     "weight_decay": backbone_wd},
            "rb_4": {"trainable": True, "lr": rb_4_lr,     "weight_decay": backbone_wd},
            "head": {"trainable": True, "lr": head_lr,     "weight_decay": head_wd}
        }
    })
    
    trial_model_dir = os.path.join(MODEL_DIR, f"trial_{trial.number}")
    os.makedirs(trial_model_dir, exist_ok=True)
    
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_param("device", str(device))
        
        try:
            fold_aurocs = training_code_kfold(
                data_directory=DATA_DIR,
                model_directory=trial_model_dir,
                k_folds=K_FOLDS,
                config=trial_config,
                device=device,
                trial=trial
            )
            
            valid_scores = [f[0] for f in fold_aurocs]
            mean_auroc = np.mean(valid_scores)
            std_auroc  = np.std(valid_scores)
            
            mlflow.log_metric("mean_auroc", mean_auroc)
            mlflow.log_metric("std_auroc",  std_auroc)
            
            # We now return both metrics to let Optuna find the Pareto front.
            return mean_auroc, std_auroc
            
        except optuna.TrialPruned:
            print(f"Trial {trial.number} was pruned.")
            raise 
        except Exception as e:
            print(f"Trial {trial.number} on {device} failed: {e}")
            return 0.0, 1.0

def run_optimize(device, study_name, storage_name, n_trials):
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(lambda trial: objective(trial, device), n_trials=n_trials)

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("AFib_Recurrence_HPO_Parallel")
    
    study_name = "afib_hpo_parallel_2.5.1"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["maximize", "minimize"],
        load_if_exists=True,
    )
    
    selector = DeviceSelector()
    try:
        devices = selector.select(count=N_GPUS)
        print(f"Selected GPUs for parallel HPO: {devices}")
    except Exception as e:
        print(f"Failed to select {N_GPUS} GPUs: {e}")
        devices = [torch.device("cuda:0")]
        N_GPUS = 1

    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    trials_per_gpu = N_TOTAL_TRIALS // len(devices)
    processes = []
    
    for i in range(len(devices)):
        p = mp.Process(
            target=run_optimize, 
            args=(devices[i], study_name, storage_name, trials_per_gpu)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("\n" + "="*30)
    print("PARALLEL HPO COMPLETE")
    print(f"Best Trials (Pareto Front):")
    for best_trial in study.best_trials:
        print(f"  Trial #{best_trial.number}")
        print(f"    Mean Valid AUROC: {best_trial.values[0]:.4f}")
        print(f"    Std Valid AUROC:  {best_trial.values[1]:.4f}")
        print(f"    Params: {best_trial.params}")
    
    # --- VISUALIZATION EXPORT ---
    print("\nExporting visualizations...")
    try:
        import optuna.visualization as vis
        
        fig_hist = vis.plot_optimization_history(study)
        fig_hist.write_html(os.path.join(MODEL_DIR, "opt_history_auroc.html"))
        
        fig_imp = vis.plot_param_importances(study)
        fig_imp.write_html(os.path.join(MODEL_DIR, "param_importances_auroc.html"))

        fig_slice = vis.plot_slice(study)
        fig_slice.write_html(os.path.join(MODEL_DIR, "slice_auroc.html"))
        
        print(f"Interactive plots saved to: {MODEL_DIR}")
    except Exception as e:
        print(f"Could not generate plots: {e}")
        
    print("="*30)