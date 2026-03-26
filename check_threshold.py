import mlflow
import os
import sys

THRESHOLD = 0.85

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", 0)

print("Run ID:", run_id)
print("Accuracy:", accuracy)

if accuracy < THRESHOLD:
    print("❌ Model failed threshold. Deployment stopped.")
    sys.exit(1)
else:
    print("✅ Model passed threshold. Proceeding to deployment.")