# ECG Analysis and Atrial Fibrillation Recurrence Prediction Pipeline

This repository contains tools, preprocessing pipelines, and model architectures for processing electrocardiogram (ECG) data, running hyperparameter optimizations, training neural networks, evaluating their performance, and interpreting their predictions.

---

## Directory & Script Directory Overview

### 1. `model_utils/`
Utility scripts for model management, extraction, and learning curve analysis.

* **[extract_fold_checkpoints.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/model_utils/extract_fold_checkpoints.py)**: Extracts and copies model checkpoints for each fold of a cross-validation run from MLflow artifacts based on the best/selected trial in an Optuna database.
* **[plot_cv_metrics.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/model_utils/plot_cv_metrics.py)**: Fetches train/validation loss and AUROC metrics from MLflow runs and plots cross-validation learning curves with mean and standard deviation.

---

### 2. `data_utils/`
Scripts and notebooks dedicated to ECG visualization, labeling, data splitting, and preprocessing.

* **[ecg_viewer.ipynb](file:///srv/home/jhyl/Afib_recurrence/diplomka/data_utils/ecg_viewer.ipynb)**: Jupyter notebook interface to view and interactively visualize specific leads of ECG signals using `ipywidgets` and `plotly`.
* **[ecg_viewer.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/data_utils/ecg_viewer.py)**: Defines helper functions and an interactive Plotly-based viewer using `ipywidgets` to load and display specific leads of ECG signals from `.hea` and `.mat` files.
* **[flip_labels.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/data_utils/flip_labels.py)**: Inverts the class labels (0 to 1, and 1 to 0) in the `#Dx:` diagnostic lines of ECG header (`.hea`) files within a directory.
* **[preprocess_finetune_data.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/data_utils/preprocess_finetune_data.py)**: Preprocesses raw ECG files by standardizing them to 12 leads, resampling to 500Hz, bandpass filtering, applying per-lead global z-score normalization, and saving as `.npy` arrays.
* **[separate_data.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/data_utils/separate_data.py)**: Separates ECG files (`.hea`, `.mat`, `.npy`) into `SR_before` and `pathology_before` directories depending on the patient's `is_AFIB_before` flag in `features.csv`.
* **[split_dataset.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/data_utils/split_dataset.py)**: Deterministically splits ECG files (`.hea`, `.mat`, `.npy`) in a directory into train and test sets using a fixed random seed.

---

### 3. `_AFIB_code/`
The core modeling pipeline containing model architectures, training routines, hyperparameter tuning, testing, and explainability scripts.

* **[analyze_afib.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/analyze_afib.py)**: Analyzes the impact of the `is_AFIB_before` clinical feature on model prediction probabilities and prediction shifts by comparing standard and flag-disabled model outputs.
* **[create_ensemble.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/create_ensemble.py)**: Loads individual model weights from a cross-validation trial (best or specified) and packages them into a single ensemble neural network (`EnsembleNN`) checkpoint.
* **[device_selector.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/device_selector.py)**: Selects one or more PyTorch devices (CUDA/CPU) automatically by querying GPU free memory using `nvidia-smi` or `torch.cuda.mem_get_info`, or validates user-requested device specifications.
* **[evaluate_model.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/evaluate_model.py)**: Evaluates a trained single/ensemble model on a test dataset, reporting metrics (ROC AUC with 95% CI, Sensitivity, Specificity, PPV, NPV, AUPRC, Brier Score) at a fixed decision threshold and plotting ROC curves and probability distributions.
* **[explain_model.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/explain_model.py)**: Generates post-hoc explainability visualizations for a single or ensemble ECG model using Captum (Integrated Gradients or DeepLIFT with SmoothGrad) in either classic "two-pass" (FAST/SLOW band-split) mode or attribution spectrogram mode.
* **[helper_code.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/helper_code.py)**: Provides helper functions for ECG loading and processing, such as parsing header files, extracting present leads, mapping configurations, and dynamically loading/prepping models for finetuning.
* **[hpo_optuna.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/hpo_optuna.py)**: Orchestrates parallel hyperparameter optimization (HPO) using Optuna (multi-objective: maximize mean valid AUROC and minimize standard deviation) across multiple GPUs, tracking results with MLflow.
* **[main_code.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/main_code.py)**: Implements the core dataset class (`CustomDataset`), flexible multi-layer optimizer builder, Focal Loss module, and the central training and cross-validation execution loops featuring linear learning rate warm-up, dynamic trial pruning, and stratified k-fold splits.
* **[model_structure.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/model_structure.py)**: Defines the neural network architectures, including the custom Residual Block (`MyResidualBlock`), the 1D/2D CNN model class (`NN`) using adaptive global pooling and lead/rhythm conditioning, and the multi-model averaged voting `EnsembleNN` class.
* **[plot_kfold_roc.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/plot_kfold_roc.py)**: Computes and plots cross-validation ROC curves (mean and standard deviation) for both training and validation sets of a k-fold trial, and calculates/saves the pooled optimal validation decision threshold.
* **[test_model.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/test_model.py)**: Runs batch inference/testing on a test ECG folder using a single/ensemble model and a locked decision threshold, saves the metrics (AUPRC, AUROC, macro precision/recall/F1, accuracy) and patient-level probabilities to a YAML file, and logs the run to MLflow.
* **[test_script.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/test_script.py)**: A simple scratch script to verify the loading of a pickled model classifier onto the CPU/GPU device using PyTorch's `load_state_dict` in non-strict mode.
* **[train_model.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/train_model.py)**: Script to start a standard (single-fold) model training run using a given data directory, setting up directory paths, and logging progress/checkpoints to MLflow and Tensorboard.
* **[train_model_kfold.py](file:///srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/train_model_kfold.py)**: Script to start a 4-fold cross-validation model training run using a given data directory, tracking progress on a per-fold basis via MLflow.

### 4. CINC2021 Code

The original code for the CINC2021 challenge, minimally modified to pretrain the original model on the data from the challenge's training set.
