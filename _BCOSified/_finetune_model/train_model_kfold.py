import sys
from main_code import training_code_kfold
import mlflow
import datetime
import os

if __name__ == '__main__':
    #################### MlFlow Setup ####################
    db_url = "sqlite:////srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/mlflow.db"
    experiment_name = "BCOSified_finetuning"
    artifact_path = "file:///mnt/mdpm/d03/jhyl/diplomka/results/mlruns"

    mlflow.set_tracking_uri(db_url)

    try:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_path)
    except mlflow.exceptions.MlflowException:
        pass # Experiment already exists
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Training_KFold_{timestamp}"
    run_dir = os.path.join("/mnt/mdpm/d03/jhyl/diplomka/results", run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)
    print(f"[INFO] Run directory created: {run_dir}")
    mlflow.start_run(run_name=run_name)
    
    # Change this note for each experiment so you don't get lost!
    experiment_note = "Linear head, rb_0 to rb_4 trainable, label smoothing 0.1, weight decay 1e-3, global average pooling, no polarity flip, only AVGPOOL, increase LR, onecycleLR with 0.1 warm up"
    mlflow.set_tag("experiment_note", experiment_note)

    print(f"[INFO] MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"[INFO] Experiment ID: {mlflow.get_experiment_by_name(experiment_name).experiment_id}")

    #################### MlFlow Setup ####################
    USE_ARGS = False
    if USE_ARGS:
        # Parse arguments
        if len(sys.argv) not in [3, 4]:
            raise Exception('Include the data, model folders and optionally resume checkpoint as arguments, e.g., python train_model_kfold.py data model [resume_checkpoint].')

        data_directory = sys.argv[1]
        model_directory = sys.argv[2]
        resume_checkpoint = sys.argv[3] if len(sys.argv) == 4 else None
    else:
        data_directory = "/srv/home/jhyl/Afib_recurrence/finetune_data_all/train"
        model_directory = os.path.join(run_dir, "model")
        # To resume training, set the resume_checkpoint path here:
        #resume_checkpoint = "/srv/home/jhyl/Afib_recurrence/diplomka/results/Training_20260324_130349/model/checkpoint_epoch_96.pth"
        resume_checkpoint = None

    training_code_kfold(data_directory, model_directory, k_folds=4, resume_checkpoint=resume_checkpoint) 

    print('Done.')
