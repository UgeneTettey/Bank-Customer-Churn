import os

project_structure = {
    "ml_project": [
        "data/raw",
        "data/processed",
        "data/external",
        "src",
        "models/experiments",
        "notebooks",
        "config",
        "tests",
        "logs",
        "reports/figures"
    ]
}

files_to_create = [
    "src/__init__.py",
    "src/data_loader.py",
    "src/data_transformation.py",
    "src/feature_engineering.py",
    "src/model_training.py",
    "src/model_evaluation.py",
    "src/model_persistence.py",
    "src/utils.py",
    "src/constants.py",
    "models/best_model.pkl",
    "notebooks/01_eda.ipynb",
    "notebooks/02_experiments.ipynb",
    "config/config.yaml",
    "config/params.yaml",
    "tests/test_data_loader.py",
    "tests/test_models.py",
    "reports/metrics.json",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    "main.py",
    "predict.py"
]

def create_project():
    base = "ml_project"

    # Create all directories
    for root, dirs in project_structure.items():
        for d in dirs:
            path = os.path.join(base, d)
            os.makedirs(path, exist_ok=True)

    # Create files
    for file_path in files_to_create:
        full_path = os.path.join(base, file_path)
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # Create empty file if not exists
        if not os.path.exists(full_path):
            with open(full_path, "w") as f:
                pass

    print("Project structure created successfully.")

if __name__ == "__main__":
    create_project()
