from mlflow.tracking import MlflowClient
db_path = "sqlite:////mnt/mdpm/d03/jhyl/deepstem_results/mlflow_runs.db"
# 1. Initialize the client (this connects to your MLflow database)
client = MlflowClient(tracking_uri=db_path)

# 2. Paste the exact Run ID from your error message
run_id = "afe55716ee3e429ba847f3fdf528e27a" 

# 3. Change the status. 
# "KILLED" or "FAILED" are best so you know it didn't finish normally.
client.set_terminated(run_id, status="KILLED")

print(f"Successfully stopped run: {run_id}. All logged data has been preserved.")