**Project 2-1 Group 7**  
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
7. [Predictive Models](#10-predictive-models) 
8. [Reproducibility](#11-reproducibility)  
9. [Deliverables](#12-deliverables)  
10. [Troubleshooting](#13-troubleshooting)  
11. [License](#14-license)

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
│  ├── MLPRegressor.py
│  ├── CatBoostRegressor.py
│  └── preprocess.py
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
├── RandomForest.ipynb
├── config.py                # Datafile name setter
├── requirements.txt         # Python dependencies
└── who_did_what.md          # Overview of task distribution
```

---

## 6. Setup

### Clone and prepare environment
```bash
git clone <this-repo-url>
cd project_2_1_group_7
py -3.10 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Install Unity
1. Download **Unity Hub**: https://unity.com/download  
2. Install **Unity Editor 2022.3.4f1** (via “Installs → Install Editor → Archive → LTS → 2022.3.4f1”).  
3. Ensure Visual Studio 2022 is installed with the workloads listed above.  

---

## 7. Quickstart: Train 3DBall
- Open **Unity Hub** → Add project → select `ml-agents-toolkit/Project` 
- Select project → Assets → ML-Agents → Examples → Scenes → 3DBall.unity
- Then go to the project file `C:\dev\project_2_1_group_7\config.py` and set the DATA_FILENAME as the name of the dataset append the consolidated row to.
- With the virtual environment activated:
```bash
mlagents-learn Config/training_config.yaml --run-id=3dball-quickstart --time-scale=20 --no-graphics
```

Inside the Unity project (`ml-agents-toolkit/Project`), press **Play**, and the agent will start training.  

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

### Manual Training
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

### Automated Training
Runs multiple `mlagents-learn` trainings by sweeping hyperparameters without opening a GUI window

- Script: `HelperScripts/HyperParameterEnumeration.py`
- What it does: enumerates ranges, rewrites `Config/training_config.yaml`, runs `mlagents-learn` with `3DBall_environment/UnityEnvironment.exe` and `--no-graphics`

Run:
```bash
python HelperScripts/HyperParameterEnumeration.py
```

This will ask you to input the starting value, the ending value and the step size for each of the hyperparameters that can be altered in the DRL training. If you do not want to change the value, put the same start and end value and any step size.

---

## 9. Data Collection
- Source of Data: ML-Agents writes a timer JSON per run at `results/<run-id>/run_logs/timers.json`, which is the raw input for data collection.
- Data Collection: `HelperScripts/AutomaticDataCollection.py` finds the newest `results/<run-id>/run_logs/timers.json` and runs `HelperScripts/data_automation.py`.
- Data Processing: `HelperScripts/data_automation.py` processes the raw information and builds one consolidated row with hardware specs + hyperparameters + metrics (includes `cumulative_reward_mean` and `time_elapased_seconds`).
- Dataset Update: `HelperScripts/data_automation.py` then appends the single row it created to `Data/<DATA_FILENAME>.csv`, adding column headers if required.

### Dataset Merge
Once ready, all CSVs can be merged under `Data/` into `Data/Data.csv`.

Run:
```bash
python HelperScripts/CombineCSV.py
```

---

## 10. Predictive Models
After data collection is completed, ML models are trained on the merged dataset to predict training time and performance from hyperparameters + hardware specs.
- Common setup:
  - Dataset: `Data/Data.csv`

### CatBoost Regressor (time + performance)
- File: `ml_models/CatBoostRegressor.py`
- Targets: `time_elapased_seconds`, `cumulative_reward_mean`
- Preprocessing: drop `run_id`/`timestamp`, drop any row with any missing value across all columns, strip units from `cpu_frequency`/`total_ram` and convert to numbers 
- Run
```bash
python ml_models/CatBoostRegressor.py
```

### MLP Regressor (time + performance)
- File: `ml_models/MLPRegressor.py`
- Targets: `time_elapased_seconds`, `cumulative_reward_mean`
- Preprocessing: drop `run_id`/`timestamp`, convert numeric strings to numeric types, one‑hot encode categoricals, fill missing values with 0, standardize features/targets
- Run
```bash
python ml_models/MLPRegressor.py
```

### Random Forest Regressor (time + performance)
- File: `RandomForest.ipynb`
- Targets: `Time Elapsed/s`, `3DBall.Environment.CumulativeReward.mean`
- Preprocessing: shuffle all rows and reset the index, drop any row with any missing value across all columns
- Run
```bash
jupyter notebook RandomForest.ipynb
```

---

## 11. Reproducibility
- **Unity version** fixed at 2022.3.4f1  
- **Python version** fixed at 3.10.11    
- **Configs** (`Config/`) contain exact hyperparameters  
- **Results and raw data** stored with run IDs  

To reproduce any run manually:  
1. Activate venv by running `.\venv\Scripts\activate` in the terminal, every python script has to be run in venv

2. Make sure that you have installed pip and the requirements. If not use this in the terminal:
`python -m pip install --upgrade pip
python -m pip install -r requirements.txt`

3. Set a name of the cvs in `config.py`. The default is set to null

4. Run `mlagents-learn Config/training_config.yaml --run-id=3dball-quickstart --time-scale=20 --no-graphics` for the first manual traning. Note that for every run you need to have a uniquid `--run-id` otherwise error will pop up.

5. Open Unity project, click Play  

6. Wait for traning to finish than check if the data were saved in Data\ your_cvs_name.cvs

To run multiple traning:
1. Run `HelperScripts/AutomaticDataCollection.py`. Make sure you run the script in venv. 

To reproduce any sequence of runs automatically:
1. Run `HyperParameterEnumeration.py` with the same ranges, config path, and environment.
    
---

## 12. Deliverables
- `/docs` → extended documentation, setup notes  
- `/report` → final scientific-style report (PDF + source)  
- `/slides` → midway and final presentation slides  
- `who_did_what.md` → task division (required for submission)  

---

## 13. Troubleshooting
- **`pip` build fails** → ensure VS2022 + C++ workload installed  
- **`mlagents-learn` not found** → re-activate venv, reinstall requirements  
- **Unity packages missing** → verify you opened `ml-agents-toolkit/Project`  
- **Corrupted Unity cache** → try deleting `Library/` and reopening project
- **PowerShell blocking virtual environment activation on windows for security reasons** → enable powershell to run local scripts without signatures, while scripts from the internet must be signed by a trusted publisher
- **Package installation failure on Windows (path length limits)** → try moving the project folder to a different directory (`C:\dev`)
- **ModuleNotFoundError: No module named 'onnxscript'** → try using an older version of PyTorch which does not use onnxscript, for example:
```bash
python -m pip uninstall torch -y
python -m pip install torch==1.13.1
```
- **RuntimeError: Missing config keys: ['DATA_FILENAME']** → make sure DATA_FILENAME is set, located at `C:\dev\project_2_1_group_7\config.py` 

---

## 14. License
- ML-Agents upstream license → see `ml-agents-toolkit/`  
- This repository’s license → see `LICENSE`  
