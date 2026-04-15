import mlflow
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Set tracking URI with a fallback to avoid errors
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run() as run:
    run_id = run.info.run_id

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")

    # Create the file required by the next stage of the pipeline
    with open("model_info.txt", "w") as f:
        f.write(str(run_id))
