# Project 2-1 Group 7

## Group Members:
- [i6403134] - [Robert Ofori Asante] - @Robert-Ofori-Asante
- [i6377960] - [Ivan Borysov] - @IvanBor771
- [i6402093] - [Maksym Chumak] - @goodmaxim 
- [i6381880] - [Moritz Ebner] - @OhrenGnom
- [i6400902] - [Omar Mahmoud] - @omar1575
- [i6381407] - [Rastislav Mintál] - @Rasto712

## Overview
This repository contains our setup for BCS2720 Project 2.1 (Machine Learning in the Unity Game Engine). We train Unity ML-Agents environments and collect data about runs (time, RAM/CPU/GPU usage, performance) for downstream ML analyses.

This README provides a reproducible setup on Windows and documents exactly which versions we use, how to run an example training (`3DBall`), and where outputs are stored.

## Prerequisites
- Git and a GitHub account
- Unity Hub + Unity Editor 2022.3 LTS
  - We standardize on Unity 2022.3.4f1 for ML-Agents tooling inside `ml-agents-toolkit/Project`.
  - The project under the root `Project/` currently targets 2022.3.0f1; prefer opening `ml-agents-toolkit/Project` unless specified otherwise.
- Visual Studio 2022 (Community) with workloads:
  - Python development
  - Desktop development with C++
  - (Optional) Game development with Unity
- Python 3.10.11 (Windows installer is readily available; 3.10.12 requires source build on Windows)

## Repository structure (relevant parts)
- `ml-agents-toolkit/` — Unity ML-Agents source (Unity packages + Python trainers). The Unity project in `ml-agents-toolkit/Project` uses local packages:
  - `com.unity.ml-agents` → `ml-agents-toolkit/com.unity.ml-agents`
  - `com.unity.ml-agents.extensions` → `ml-agents-toolkit/com.unity.ml-agents.extensions`
- `Project/` — Another Unity project. For training examples and tutorials, default to opening `ml-agents-toolkit/Project`.
- `Config/` — YAML configs:
  - `training_config.yaml` — trainer config (PPO for `3DBall`)
  - `project_config.yaml` — paths and data collection defaults
- `Data/RawData/` — run logs and raw data (created on first run)
- `Models/` — exported models and artifacts

## One-time setup
1) Clone the repository
```
git clone <this-repo-url>
cd project_2_1_group_7
```

2) Create and activate a virtual environment (Python 3.10.11)
```
python -m venv venv
./venv/Scripts/activate
```

3) Install Python dependencies (including ML-Agents from local sources)
```
pip install --upgrade pip
pip install -r requirements.txt
```

4) Install Unity
- Install Unity Hub and Unity Editor 2022.3.4f1.
- Ensure Visual Studio 2022 with the workloads listed above is installed.

## Opening the Unity project
Open the Unity project at:
- `ml-agents-toolkit/Project` (preferred; aligns with local ML-Agents packages)

Unity’s package manifest already references the local packages, so no extra action is required if you keep the folder structure intact.

## Quickstart: Train `3DBall`
With the virtual environment activated:

1) Start the trainer (PPO) using our config
```
mlagents-learn Config/training_config.yaml --run-id=3dball-quickstart --time-scale=20 --no-graphics
```

2) When prompted (or if using `--no-graphics`, it will auto-start), Unity will connect. If you prefer Editor play mode, open `ml-agents-toolkit/Project` in Unity and press Play after the trainer is waiting for an environment.

Outputs:
- TensorBoard summaries: `results/3dball-quickstart/`
- Checkpoints: inside the same `results` run folder
- Raw data and metrics (if enabled): `Data/RawData/`

Launch TensorBoard (optional):
```
tensorboard --logdir results
```

## Data collection
We will log hardware (RAM/CPU/GPU) and timing metrics per run to `Data/RawData/`. Settings are defined in `Config/project_config.yaml` under `hardware_tracking` and `data_collection`.

## Versioning and consistency
- Unity Editor: standardize on 2022.3.4f1 for the ML-Agents project in `ml-agents-toolkit/Project`.
- ML-Agents (Python): installed from local sources in this repo to ensure the same version across all group members.
- Everyone should keep the same folder layout so that Unity package paths (`file:../../com.unity.ml-agents`) resolve correctly.

## Troubleshooting (Windows)
- If `pip` fails to build dependencies, ensure Visual Studio 2022 with C++ workload is installed.
- If `mlagents-learn` is not found, re-activate the venv and reinstall `requirements.txt`.
- If Unity cannot find the ML-Agents packages, verify that you opened `ml-agents-toolkit/Project` and that the relative `file:` paths are intact.

## License
See upstream ML-Agents licenses in `ml-agents-toolkit/` and this repository’s license file (if applicable).
