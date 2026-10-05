import mlflow
from mlflow import MlflowClient
import argparse
import os
from pathlib import Path
import shutil

model_name = os.environ.get("MODEL_NAME", "iris_model")

export_dir = Path(__file__).resolve().parent / "app" / "model"

def export_model(export_dir,model_name):
    
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}@prod",
        dst_path=export_dir)
    print(f"Model exported to {export_dir}")
    return 
def main():
    export_model(export_dir,model_name)

if __name__ == "__main__":
    main()

