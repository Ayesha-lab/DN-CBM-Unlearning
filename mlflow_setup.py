import mlflow

def setup_mlflow():
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Training_linear_probe_celebA")

    # Use the actual hostname your browser connects through
    ui_url = "http://uller.hpc.uni-saarland.de:5000"
    print(f"\n✓ MLflow tracking URI: ./mlruns", flush=True)
    print(f"✓ Open MLflow UI: {ui_url}\n", flush=True)

    return ui_url

