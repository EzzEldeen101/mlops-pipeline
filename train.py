import mlflow
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# CRITICAL: If no URI is provided, use local file logging to avoid connection errors
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    
    # Load and split data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simple model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Success! Run ID: {run_id}")
    print(f"Final Accuracy: {accuracy}")

    # Create the artifact file for the pipeline
    with open("model_info.txt", "w") as f:
        f.write(f"Run_ID: {run_id}\nAccuracy: {accuracy}")
