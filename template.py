from pathlib import Path
import subprocess
import logging
import os

logging.basicConfig(level=logging.INFO, format="[%(asctime)s: %(message)s")

subprocess.call(["uv", "init"])

project_name = "crash_detection"
list_of_files = [
    "src/{project_name}/__init__.py",
    "src/{project_name}/pipelines/__init__.py",
    "src/{project_name}/utils/__init__.py",
    "src/{project_name}/utils/common.py",
    "src/{project_name}/config/__init__.py",
    "src/{project_name}/core/__init__.py",
    "src/{project_name}/core/common.py",
    "src/{project_name}/components/__init__.py",
    "src/{project_name}/components/training/__init__.py",
    "src/{project_name}/components/training/model.py",
    "src/{project_name}/components/training/dataset.py",
    "src/{project_name}/components/training/loss.py",
    "src/{project_name}/components/training/trainer.py",
    "src/{project_name}/components/data/__init__.py",
    "src/{project_name}/components/loss/__init__.py",
    "src/{project_name}/constants/__init__.py",
    "src/{project_name}/errors/__init__.py",
    "Dockerfile",
    "docker-compose.yaml",
    ".github/workflow/.gitkeep",
    ".gitignore",
    "working/.gitkeep",
    "schemas/schema.yaml",
    "main.py",
    "app.py",
    "config/config.yaml",
]

for file in list_of_files:
    filepath = Path(file.format(project_name=project_name))
    dirname, filename = os.path.split(filepath)

    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
        logging.info(f"Created directory {dirname} for the file {filename}")

    if not filepath.exists() or os.path.getsize(filepath) == 0:
        subprocess.call(["touch", str(filepath)])
        logging.info(f"Creating an empty file {filepath}")

    else:
        logging.info(f"File {filepath} already exists")
