import mlflow
import webbrowser
import time

def setup_mlflow():
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Unlearning_CBMS")

    ui_url = "http://localhost:5000"
    print(f"\n✓ MLflow tracking URI: ./mlruns", flush=True)
    print(f"✓ Open MLflow UI: {ui_url}\n", flush=True)

    return ui_url

