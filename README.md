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
├── 3DBall_environment/          # 3D Ball ML-Agents environment 
│
├── Config/                      # Training and project configs
│   ├── project_config.yaml
│   ├── standart_config.yaml
│   └── training_config.yaml
│
├── Data/                        # CSV files and data analysis
│   
├── HelperScripts/               # Utility scripts
│   ├── AutomaticDataCollection.py
│   ├── CombineCSV.py
│   ├── DataSplitting.py
│   ├── HyperParameterEnumeration.py
│   └── data_automation.py
│
├── ProjectSettings/             # Project-wide configuration in YAML assets
│
├── ml-agents-toolkit/           # ML-Agents source (Unity + Python)
│   ├── Project/                 # Unity project (preferred for training)
│   ├── com.unity.ml-agents
│   └── com.unity.ml-agents.extensions
│
├── ml_models/                   # ML-Models for predictions
│   ├── CatBoostRegressor.py
│   ├── MLModel.py
│   ├── MLPRegressor.py
│   ├── ModelComparison.py
│   ├── RandomForest.py
│   └── preprocess.py
│
├── results/                     # Training runs output
│
├── README.md                    # This file
├── config.py                    # Datafile name setter
│
├── error_distribution.png       # Model comparison
├── metric_comparison.png        # Model comparison
├── model_comparison_report.txt  # Model comparison
├── predictions_vs_actual.png    # Model comparison
│
├── requirements.txt             # Python dependencies
└── who_did_what.md              # Overview of task distribution
```

---

## 6. Setup

### Clone and prepare environment
```bash
git clone <this-repo-url>
cd project_2_1_group_7
py -3.10 -m venv venv
./venv/Scripts/activate
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
- Then edit `config.py` and set `DATA_FILENAME` to the dataset name to append the consolidated row to.
- With the virtual environment activated, run at the repo root `project_2_1_group_7/`:
```bash
mlagents-learn Config/training_config.yaml --run-id=3dball-quickstart --time-scale=20 --no-graphics
```

Inside the Unity project (`ml-agents-toolkit/Project`), press **Play**, and the agent will start training.  

Outputs:  
- **Results**: `results/3dball-quickstart/`  
- **Raw data**: `results/3dball-quickstart/run_logs/`  

TensorBoard: Optional visualization and monitoring during or after training. Run the command from the repo root while the venv is active (use another terminal during training).
```bash
tensorboard --logdir results
```
→ View at http://localhost:6006

---

## 8. How to Train Your Own Model

### Additional Installation (Windows)
1. Install **Python 3.10.11** (if not already installed).  
2. Open the repo root `project_2_1_group_7/` in the terminal.  
3. Create a virtual environment:  
   ```bash
   py -3.10 -m venv venv
   ./venv/Scripts/activate
   ```

### Manual Training
1. Open **Unity Hub** → Add project → select `ml-agents-toolkit/Project`.  
2. Open the project and select the scene you want to train.  
3. In the terminal (`project_2_1_group_7/`):  
   ```bash
   mlagents-learn Config/training_config.yaml --run-id=[runId]
   ```
   Example:  
   ```bash
   mlagents-learn Config/training_config.yaml --run-id=first3DBallRun
   ```
4. Click **Play** in Unity.  
 
### Automated Training
Run multiple `mlagents-learn` trainings by sweeping hyperparameters without opening a GUI window.

- Script: `HelperScripts/HyperParameterEnumeration.py`
- What it does: enumerates ranges, rewrites `Config/training_config.yaml`, runs `mlagents-learn` with `3DBall_environment/UnityEnvironment.exe` and `--no-graphics`

Run from the repo root while venv is active:
```bash
python HelperScripts/HyperParameterEnumeration.py
```

This will ask you to input the starting value, ending value, and step size for each of the hyperparameters that can be altered in the DRL training. If you do not want to change the value, put the same start and end value and a positive step size.

**CAUTION:**
- `batch_size`, `buffer_size`, `num_epoch`, `hidden_units`, `num_layers`, `time_horizon` and their step sizes all expect strictly integer inputs. Float inputs will cause an error.
- Ensure start value <= end value and the step size > 0, otherwise the script either yields 0 training runs or hangs indefinitely.

TensorBoard: Optional visualization and monitoring during or after training (manual or automated runs). Run the command from the repo root while the venv is active (use another terminal during training).
```bash
tensorboard --logdir results
```
→ View at http://localhost:6006

---

## 9. Data Collection
- Source of Data: ML-Agents writes a timer JSON per run at `results/<run-id>/run_logs/timers.json`, which is the raw input for data collection.
- Data Collection: `HelperScripts/AutomaticDataCollection.py` finds the newest `results/<run-id>/run_logs/timers.json` and runs `HelperScripts/data_automation.py`.
- Data Processing: `HelperScripts/data_automation.py` processes the raw information and builds one consolidated row with hardware specs + hyperparameters + metrics (includes `cumulative_reward_mean` and `time_elapased_seconds`).
- Dataset Update: `HelperScripts/data_automation.py` then appends the single row it created to `Data/<DATA_FILENAME>.csv`, adding column headers if required. Note that `DATA_FILENAME` must be specified beforehand in `config.py`.

### Dataset Merge
Once ready, run from the repo root `CombineCSV.py` to create `Data/Data.csv`.

```bash
python HelperScripts/CombineCSV.py
```

Note that `Data/combined.csv` is a legacy dataset with significantly skewed data and was therefore replaced by `Data/Data.csv` to help mitigate that skew.

---

## 10. Predictive Models
After data collection is completed, ML models are trained on the merged dataset to predict training time and performance from hyperparameters + hardware specs.
- Common setup:
  - Dataset: `Data/Data.csv`

###  ModelComparison.py
- This file is responsible for loading the dataset, running preprocessing uniformly, splitting train/test sets, training all 3 model types, evaluating their predictions, and writing plots and a report.
- First, make sure the merged dataset `Data/Data.csv` exists. Then proceed to run from the repo root:
```bash
python ml_models/ModelComparison.py
```

### Model Definitions
- Targets for all models: `time_elapased_seconds`, `cumulative_reward_mean`
- The following model classes are only definitions that are imported and used by `ml_models/ModelComparison.py` (not standalone):
  - `ml_models/CatBoostRegressor.py`
  - `ml_models/MLPRegressor.py`
  - `ml_models/RandomForest.py`

---

## 11. Reproducibility
- **Unity version** fixed at 2022.3.4f1  
- **Python version** fixed at 3.10.11    
- **Configs** (`Config/`) contain exact hyperparameters  
- **Results and raw data** stored with run IDs  

To reproduce any run manually:  
1. Activate venv by running `./venv/Scripts/activate` in the terminal, every python script has to be run in venv.

2. Make sure that you have installed pip and the requirements. If not use this in the terminal:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Set a base name of the csv file without the extension in `config.py`. The default is set to `None`.

4. Run at the repo root for the first manual training:
   ```bash
   mlagents-learn Config/training_config.yaml --run-id=3dball-quickstart --time-scale=20 --no-graphics
   ```
Note that for every run you need to have a unique id `--run-id`, otherwise an error will pop up.

6. Open Unity project, click **Play**.  

7. Wait for the training to finish and then check if the data were saved in `Data/<DATA_FILENAME>.csv`
 
To reproduce any sequence of multiple runs automatically:
1. Run `HyperParameterEnumeration.py` from the repo root with the same ranges, config path, and environment. Make sure you run the script in venv.
```bash
python HelperScripts/HyperParameterEnumeration.py
```
    
---

## 12. Deliverables
- `/docs` → extended documentation, setup notes  
- `/report` → final scientific-style report (PDF + source)  
- `/slides` → midway and final presentation slides  
- `who_did_what.md` → task division (required for submission)  

---

## 13. Troubleshooting
- **`pip` build fails** → ensure VS2022 + C++ workload installed.
- **`mlagents-learn` not found** → re-activate venv, reinstall requirements.
- **Unity packages missing** → verify you opened `ml-agents-toolkit/Project`.  
- **Corrupted Unity cache** → try deleting `Library/` and reopening project.
- **PowerShell blocking virtual environment activation on Windows for security reasons** → enable PowerShell to run local scripts without signatures, while scripts from the internet must be signed by a trusted publisher.
- **Package installation failure on Windows (path length limits)** → try moving the project folder to a different directory (`C:/dev`).
- **ModuleNotFoundError: No module named 'onnxscript'** → try using an older version of PyTorch which does not use onnxscript:
```bash
python -m pip uninstall torch -y
python -m pip install torch==1.13.1
```
- **RuntimeError: Missing config keys: ['DATA_FILENAME']** → make sure `DATA_FILENAME` inside `config.py` is set.
- **ModuleNotFoundError: No module named 'catboost'** → make sure `catboost` is installed by running `python -m pip install catboost`.
- **UnityTrainerException: Previous data from this run ID was found** → make sure to use a unique run ID or delete `results/<run-id>` to reuse it.

---

## 14. License
- ML-Agents upstream license → see `ml-agents-toolkit/`  
- This repository’s license → see `LICENSE`  
