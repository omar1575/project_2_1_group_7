# Project 2-1 Group 7  
**BCS2720: Machine Learning in the Unity Game Engine**

## 1. Group Members
- [i6403134] Robert Ofori Asante — @Robert-Ofori-Asante  
- [i6377960] Ivan Borysov — @IvanBor771  
- [i6402093] Maksym Chumak — @goodmaxim  
- [i6381880] Moritz Ebner — @OhrenGnom  
- [i6400902] Omar Mahmoud — @omar1575  
- [i6381407] Rastislav Mintál — @Rasto712  

---

## 2. Overview
This repository contains our setup for **Project 2.1 (Machine Learning in the Unity Game Engine)**.  
We use Unity’s **ML-Agents toolkit** to train reinforcement learning agents in 3D environments, while logging data such as runtime, RAM/CPU/GPU usage, and training performance.  

The repository provides:  
- A reproducible environment setup on **Windows**  
- Unity + ML-Agents configurations  
- Scripts for data collection and experiment tracking  
- Instructions for training baseline and custom models  

---

## 3. Table of Contents
1. [Prerequisites](#4-prerequisites)  
2. [Repository Structure](#5-repository-structure)  
3. [Setup](#6-setup)  
4. [Quickstart: Train 3DBall](#7-quickstart-train-3dball)  
5. [How to Train Your Own Model](#8-how-to-train-your-own-model)  
6. [Data Collection](#9-data-collection)  
7. [Reproducibility](#10-reproducibility)  
8. [Deliverables](#11-deliverables)  
9. [Troubleshooting](#12-troubleshooting)  
10. [License](#13-license)  

---

## 4. Prerequisites
- **Git** and a GitHub account  
- **Unity Hub + Unity Editor 2022.3 LTS**  
  - Standardized version: **2022.3.4f1**  
  - Preferred Unity project: `ml-agents-toolkit/Project`  
- **Visual Studio 2022 (Community)** with workloads:  
  - Python development  
  - Desktop development with C++  
  - (Optional) Game development with Unity  
- **Python 3.10.11**  
  - (Note: 3.10.12 requires source build on Windows; avoid it)  

---

## 5. Repository Structure
```
project_2_1_group_7/
│
├── 3DBall_environment/      # 3D Ball ML-Agents environment 
│
├── Config/                  # Training and project configs
│   ├── training_config.yaml
│   ├── standart_config.yaml
│   └── project_config.yaml
│
├── Data/                    # CSV files storing training configs + their respective results
│   
├── HelperScripts/           # Utility scripts for ML-Agents data collection + dataset preparation
│   ├── AutomaticDataCollection.py
│   ├── CombineCSV.py
│   ├── data_automation.py
│   ├── DataSplitting.py
│   └── HyperParameterEnumeration.py
│
├── ml_models/               # ML-Models for predictions
│   └── MLPRegressor.py
│
├── ml-agents-toolkit/       # ML-Agents source (Unity + Python)
│   ├── Project/             # Unity project (preferred for training)
│   ├── com.unity.ml-agents
│   └── com.unity.ml-agents.extensions
│
├── ProjectSettings/         # Project-wide configuration in YAML assets
│
├── results/                 # Training runs output
│
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## 6. Setup

### Clone and prepare environment
```bash
git clone <this-repo-url>
cd project_2_1_group_7
python -m venv venv
./venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Install Unity
1. Download **Unity Hub**: https://unity.com/download  
2. Install **Unity Editor 2022.3.4f1** (via “Installs → Install Editor → Archive → LTS → 2022.3.4f1”).  
3. Ensure Visual Studio 2022 is installed with the workloads listed above.  

---

## 7. Quickstart: Train 3DBall
With the virtual environment activated:
```bash
mlagents-learn Config/training_config.yaml --run-id=3dball-quickstart --time-scale=20 --no-graphics
```

Open Unity project (`ml-agents-toolkit/Project`), press **Play**, and the agent will start training.  

Outputs:  
- **Results**: `results/3dball-quickstart/` (TensorBoard + checkpoints)  
- **Raw data**: `Data/RawData/` (if enabled)  

TensorBoard (optional):  
```bash
tensorboard --logdir results
```

---

## 8. How to Train Your Own Model

### Additional Installation (Windows)
1. Install **Python 3.10.11** (if not already installed).  
2. Open `ml-agents-toolkit/` in the terminal.  
3. Create a virtual environment:  
   ```bash
   py -3.10 -m venv venv
   venv\Scripts\activate
   ```
4. Install libraries:  
   ```bash
   python -m pip install ./ml-agents-envs
   python -m pip install ./ml-agents
   ```

### Running
1. Open **Unity Hub** → Add project → select `ml-agents-toolkit/Project`.  
2. Open the project and select the scene you want to train.  
3. In the terminal (`ml-agents-toolkit/`):  
   ```bash
   mlagents-learn [config file path] --run-id=[runId]
   ```
   Example:  
   ```bash
   mlagents-learn config/ppo/3DBall.yaml --run-id=first3DBallRun
   ```
4. Click **Play** in Unity.  
5. Monitor results with TensorBoard:  
   ```bash
   tensorboard --logdir results
   ```
   → View at http://localhost:6006  

---

## 9. Data Collection
- Source of Data: ML-Agents writes a timer JSON per run at `results/<run-id>/run_logs/timers.json`, which is the raw input for data collection.
- Data Collection: `HelperScripts/AutomaticDataCollection.py` finds the newest `results/<run-id>/run_logs/timers.json` and runs `HelperScripts/data_automation.py`.
- Data Processing: `HelperScripts/data_automation.py` processes the raw information and builds one consolidated row with hardware specs + hyperparameters + metrics (includes `cumulative_reward_mean` and `time_elapased_seconds`).
- Dataset Update: `HelperScripts/data_automation.py` then appends the single row it created to `Data/<DATA_FILENAME>.csv`, adding column headers if required.

### Dataset Merge
- Once ready, `HelperScripts/CombineCSV.py` merges all CSVs under `Data/` into `Data/Data.csv`.

---

## 10. Reproducibility
- **Unity version** fixed at 2022.3.4f1  
- **Python version** fixed at 3.10.11    
- **Configs** (`Config/`) contain exact hyperparameters  
- **Results and raw data** stored with run IDs  

To reproduce any run:  
1. Activate venv  
2. Run `mlagents-learn` with the same config + run ID  
3. Open Unity project, click Play  

---

## 11. Deliverables
- `/docs` → extended documentation, setup notes  
- `/report` → final scientific-style report (PDF + source)  
- `/slides` → midway and final presentation slides  
- `who_did_what.md` → task division (required for submission)  

---

## 12. Troubleshooting
- **`pip` build fails** → ensure VS2022 + C++ workload installed  
- **`mlagents-learn` not found** → re-activate venv, reinstall requirements  
- **Unity packages missing** → verify you opened `ml-agents-toolkit/Project`  
- **Corrupted Unity cache** → try deleting `Library/` and reopening project  

---

## 13. License
- ML-Agents upstream license → see `ml-agents-toolkit/`  
- This repository’s license → see `LICENSE`  
