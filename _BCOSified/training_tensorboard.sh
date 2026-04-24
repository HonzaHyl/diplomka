#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=2:mem=50gb:scratch_local=100gb
#PBS -N finetune_bcosified

# --- Job Setup ---
# This command ensures that when the job ends, any background processes (like TensorBoard) are automatically killed.
trap "kill 0" EXIT

module add python/3.11.11-gcc-10.2.1-555dlyc

DATADIR="finetune_run"
# LOGDIR_TENSORBOARD="$SCRATCHDIR/$DATADIR/model/runs" # Define the path to your TensorBoard logs

# --- Data and Environment Preparation on Scratch ---
echo "Preparing scratch directory..."
# Copy everything to scratch
cp -r $PBS_O_WORKDIR/$DATADIR $SCRATCHDIR/
cp -r $PBS_O_WORKDIR/_finetune_model $SCRATCHDIR/

# Create venv on SCRATCH
echo "Creating virtual environment..."
python -m venv $SCRATCHDIR/$DATADIR/.venv_train
source $SCRATCHDIR/$DATADIR/.venv_train/bin/activate

# For cache to relocate to SCRATCH
mkdir -p $SCRATCHDIR/pip_cache
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

export TMPDIR=$SCRATCHDIR/tmp
mkdir -p $TMPDIR

# --- Install Dependencies ---
echo "Installing requirements..."
pip install -r $SCRATCHDIR/_finetune_model/requirements.txt


# --- Launch TensorBoard & Main Scripts ---
cd $SCRATCHDIR/_finetune_model/

# # Make sure the log directory exists before launching TensorBoard
# mkdir -p $LOGDIR_TENSORBOARD

# # Launch TensorBoard in the background on a specific port (e.g., 16006)
# TENSORBOARD_PORT=16006
# echo "Starting TensorBoard..."
# tensorboard --logdir "$LOGDIR_TENSORBOARD" --host 0.0.0.0 --port $TENSORBOARD_PORT &

# # Print connection info to the job's output file for easy access
# echo "----------------------------------------------------"
# echo "TensorBoard is running on node: $(hostname)"
# echo "To connect, run this on your LOCAL machine:"
# echo "ssh -L 8888:$(hostname):$TENSORBOARD_PORT $USER@zenith.cerit-sc.cz"
# echo "Then open http://localhost:8888 in your browser."
# echo "----------------------------------------------------"


# --- Run your training, testing, and evaluation sequence ---
echo "Starting main scripts..."
set -x
# python train_model.py $SCRATCHDIR/$DATADIR/train_data $SCRATCHDIR/$DATADIR/model
python test_model.py $SCRATCHDIR/$DATADIR/model $SCRATCHDIR/$DATADIR/test_data $SCRATCHDIR/$DATADIR/test_outputs
#python evaluate_model.py $SCRATCHDIR/$DATADIR/test_data $SCRATCHDIR/$DATADIR/test_outputs
set +x
echo "Main scripts finished."

# --- Copy Final Results ---
echo "Copying final results back to home directory..."
# Use rsync for more robust copying
# rsync -avh --no-g --no-p $SCRATCHDIR/$DATADIR/model $PBS_O_WORKDIR/$DATADIR/
rsync -avh --no-g --no-p $SCRATCHDIR/$DATADIR/test_outputs $PBS_O_WORKDIR/$DATADIR/


