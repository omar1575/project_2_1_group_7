"""
data_automation.py

Reads a Unity ML-Agents timer/log JSON file and appends a single consolidated row
to Data/consolidated_runs.csv (creating the file and Data/ dir if needed).

Important note: the provided requirement contained a minor contradiction:
- It said "final row ... exactly 23 columns" but then listed 24 headers.
This script follows the explicit headers list (24 columns), including:
  '3DBall.Environment.CumulativeReward.mean' and 'Time Elapsed/s'.

Usage:
    python data_automation.py /path/to/mlagents_timer.json
"""

from __future__ import annotations
import json
import csv
import os
import sys
import platform
import subprocess
import yaml
import itertools  # used inside parse; safe to import local here
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Try to import psutil for richer system info; degrade gracefully if not available.
try:
    import psutil
except Exception:
    psutil = None  # type: ignore

# --- Configuration ---
OUTPUT_DIR = "Data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "max.csv")


# Exact headers (in the exact order required)
HEADERS = [
    "run_id",
    "timestamp",
    "CPU Model",
    "CPU Cores",
    "CPU Frequency",
    "Total RAM",
    "GPU Model",
    "Operating System",
    "DRL Algorithm Used",
    "batch_size",
    "buffer_size",
    "learning_rate",
    "beta",
    "epsilon",
    "lambda",
    "num_epoch",
    "normalize",
    "hidden_units",
    "num_layers",
    "gamma",
    "max_steps",
    "time_horizon",
    "3DBall.Environment.CumulativeReward.mean",
    "Time Elapsed/s",
]

# ---------------- Placeholder / system query helpers ----------------


def get_run_id_from_path(path: str) -> str:
    """Derive run_id from file path: extract from results/<run_id>/run_logs/timers.json"""
    # Try to extract run_id from path structure
    path_parts = os.path.normpath(path).split(os.sep)
    try:
        # Look for 'results' in path and get the next directory
        if "results" in path_parts:
            results_idx = path_parts.index("results")
            if results_idx + 1 < len(path_parts):
                return path_parts[results_idx + 1]
    except Exception:
        pass

    # Fallback: use basename without extension
    base = os.path.basename(path)
    run_id = os.path.splitext(base)[0]
    return run_id


def get_cpu_model(json_meta: Dict[str, Any]) -> str:
    """Attempt to get CPU model from metadata; else query system or fallback."""
    # Common metadata keys that might contain CPU model info
    for key in ("cpu_model", "CPU", "processor", "cpu"):
        val = json_meta.get(key)
        if val:
            return str(val)
    # Try platform
    try:
        # platform.processor() can be empty on some systems
        p = platform.processor()
        if p:
            return p
    except Exception:
        pass
    # Try /proc/cpuinfo on Linux
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown CPU Model"


def get_cpu_cores(json_meta: Dict[str, Any]) -> str:
    """Return number of physical cores if available, else logical count, else Unknown."""
    for key in ("cpu_cores", "cores", "physical_cores"):
        val = json_meta.get(key)
        if val:
            return str(val)
    try:
        if psutil:
            physical = psutil.cpu_count(logical=False)
            if physical:
                return str(physical)
        logical = os.cpu_count()
        if logical:
            return str(logical)
    except Exception:
        pass
    return "Unknown"


def get_cpu_freq(json_meta: Dict[str, Any]) -> str:
    """Return CPU frequency in MHz if available or detectable."""
    for key in ("cpu_freq_mhz", "cpu_frequency", "cpu_freq"):
        val = json_meta.get(key)
        if val:
            return str(val)
    try:
        if psutil:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                # report in MHz with 2 decimal places
                return f"{freq.current:.2f} MHz"
    except Exception:
        pass
    # platform won't give freq generally; fallback
    return "Unknown"


def get_total_ram_gb(json_meta: Dict[str, Any]) -> str:
    """Return total RAM in GB (rounded to 2 decimals) from metadata or system query."""
    for key in ("total_ram_gb", "total_ram", "ram_gb", "memory_gb"):
        val = json_meta.get(key)
        if val:
            try:
                return f"{float(val):.2f} GB"
            except Exception:
                return str(val)
    try:
        if psutil:
            mem = psutil.virtual_memory()
            return f"{mem.total / (1024**3):.2f} GB"
    except Exception:
        pass
    return "Unknown"


def get_gpu_model(json_meta: Dict[str, Any]) -> str:
    """Attempt to obtain GPU model. Try metadata, GPUtil, nvidia-smi, or fallback."""
    for key in ("gpu_model", "gpu", "GPU"):
        val = json_meta.get(key)
        if val:
            return str(val)
    # Try GPUtil if available
    try:
        import GPUtil  # type: ignore

        gpus = GPUtil.getGPUs()
        if gpus:
            names = ", ".join(g.name for g in gpus)
            return names
    except Exception:
        pass
    # Try nvidia-smi (Linux/Windows if NVIDIA tools installed)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        names = ",".join([line.strip() for line in out.splitlines() if line.strip()])
        if names:
            return names
    except Exception:
        pass
    return "Unknown GPU"


def get_operating_system(json_meta: Dict[str, Any]) -> str:
    for key in ("os", "operating_system", "operating_system_name"):
        val = json_meta.get(key)
        if val:
            return str(val)
    try:
        return f"{platform.system()} {platform.release()}"
    except Exception:
        return "Unknown OS"


# ---------------- Command-line args parsing from metadata ----------------


def parse_command_line_args_string(cmdline: str) -> Dict[str, Any]:
    """
    Parse a command_line_arguments string for common ML-Agents / RL hyperparameters.
    This is heuristic: looks for tokens like --batch-size 1024 or --learning-rate=0.0003
    Returns a dict of found parameters (keys normalized).
    """
    import shlex

    parsed: Dict[str, Any] = {}
    if not cmdline:
        return parsed
    try:
        tokens = shlex.split(cmdline)
    except Exception:
        # fallback naive split
        tokens = cmdline.strip().split()
    it = iter(tokens)
    for token in it:
        if token.startswith("--"):
            # strip leading dashes and replace - with _
            key = token.lstrip("-").replace("-", "_")
            # handle = syntax
            if "=" in key:
                k, v = key.split("=", 1)
                parsed[k] = try_cast_number(v)
                continue
            # else attempt to take next token as value if it doesn't start with --
            try:
                nxt = next(it)
                if nxt.startswith("--"):
                    # boolean flag; put True and step iterator back one
                    parsed[key] = True
                    # move iterator back by making a new one starting from nxt
                    it = itertools.chain([nxt], it)  # type: ignore
                else:
                    parsed[key] = try_cast_number(nxt)
            except StopIteration:
                parsed[key] = True
    return parsed


def try_cast_number(val: Any) -> Any:
    """If val looks like int or float, cast it, else return original string."""
    if isinstance(val, (int, float)):
        return val
    s = str(val)
    try:
        if "." in s:
            return float(s)
        else:
            return int(s)
    except Exception:
        # handle true/false
        low = s.lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
        return s


# ---------------- Main extraction / consolidation logic ----------------


def extract_gauge_value(gauges: Dict[str, Any], key: str) -> Optional[float]:
    """
    Extract numeric gauge value from gauges dict at given key.
    If the 'value' field is a list, take the last numeric entry; if scalar, cast to float.
    Returns None if not found or not numeric.
    """
    node = gauges.get(key)
    if not node:
        return None
    val = node.get("value") if isinstance(node, dict) else node
    if val is None:
        return None
    # If list, take last numeric
    if isinstance(val, list):
        for v in reversed(val):
            try:
                return float(v)
            except Exception:
                continue
        return None
    # If numeric-like
    try:
        return float(val)
    except Exception:
        # maybe nested with 'mean' etc
        if isinstance(val, dict):
            for candidate in ("mean", "value", "last", 0):
                try:
                    v = val.get(candidate)  # type: ignore
                    if v is not None:
                        return float(v)
                except Exception:
                    continue
        return None


def safe_get_meta(meta: Dict[str, Any], keys):
    for key in keys:
        if key in meta:
            return meta[key]
    return None


def load_hyperparameters_from_config(results_dir: str) -> Dict[str, Any]:
    """Load hyperparameters from configuration.yaml in the results directory."""
    config_path = os.path.join(results_dir, "configuration.yaml")
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract hyperparameters from the config structure
        hyperparams = {}

        # Get behaviors section (usually has the brain name like "3DBall")
        behaviors = config.get("behaviors", {})
        if behaviors:
            # Get first behavior's settings
            brain_name = list(behaviors.keys())[0]
            brain_config = behaviors[brain_name]

            # Extract values
            trainer_type = brain_config.get("trainer_type", "Unknown")
            hyperparams["DRL Algorithm Used"] = (
                trainer_type.upper() if trainer_type else "Unknown"
            )

            hp = brain_config.get("hyperparameters", {})
            hyperparams["batch_size"] = hp.get("batch_size", "")
            hyperparams["buffer_size"] = hp.get("buffer_size", "")
            hyperparams["learning_rate"] = hp.get("learning_rate", "")
            hyperparams["beta"] = hp.get("beta", "")
            hyperparams["epsilon"] = hp.get("epsilon", "")
            hyperparams["lambda"] = hp.get(
                "lambd", ""
            )  # Note: 'lambd' not 'lambda' in config
            hyperparams["num_epoch"] = hp.get("num_epoch", "")

            network = brain_config.get("network_settings", {})
            hyperparams["normalize"] = network.get("normalize", "")
            hyperparams["hidden_units"] = network.get("hidden_units", "")
            hyperparams["num_layers"] = network.get("num_layers", "")

            reward_signals = brain_config.get("reward_signals", {})
            extrinsic = reward_signals.get("extrinsic", {})
            hyperparams["gamma"] = extrinsic.get("gamma", "")

            hyperparams["max_steps"] = brain_config.get("max_steps", "")
            hyperparams["time_horizon"] = brain_config.get("time_horizon", "")

        return hyperparams
    except Exception as e:
        print(f"Warning: Could not load configuration.yaml: {e}")
        return {}


def build_row_from_json(filepath: str, data: Dict[str, Any]) -> list:
    """
    Create the CSV row values in the exact order specified by HEADERS.
    """
    # Metadata shortcuts
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    gauges = data.get("gauges", {}) if isinstance(data, dict) else {}

    run_id = get_run_id_from_path(filepath)

    # Timestamp: prefer metadata.start_time_seconds; else current UTC ISO
    start_seconds_raw = safe_get_meta(
        meta, ("start_time_seconds", "start_time", "start")
    )
    timestamp_iso = ""
    start_seconds: Optional[float] = None
    try:
        if start_seconds_raw is not None:
            start_seconds = float(start_seconds_raw)
            timestamp_iso = (
                datetime.fromtimestamp(start_seconds, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        else:
            # fallback to current UTC
            timestamp_iso = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
    except Exception:
        timestamp_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    cpu_model = get_cpu_model(meta)
    cpu_cores = get_cpu_cores(meta)
    cpu_freq = get_cpu_freq(meta)
    total_ram = get_total_ram_gb(meta)
    gpu_model = get_gpu_model(meta)
    operating_system = get_operating_system(meta)

    # Parse hyperparams: try JSON keys first, then command line
    # Known hyperparameter keys to look for (normalize both hyphen and underscore)
    hyper_keys = {
        "DRL Algorithm Used": (
            "algorithm",
            "alg",
            "drl_algorithm",
            "behavior_name",
            "behavior",
        ),
        "batch_size": ("batch_size", "batch-size", "batchsize"),
        "buffer_size": ("buffer_size", "buffer-size", "buffersize", "buffer"),
        "learning_rate": ("learning_rate", "learning-rate", "lr"),
        "beta": ("beta",),
        "epsilon": ("epsilon",),
        "lambda": ("lambda", "lam"),
        "num_epoch": ("num_epoch", "num-epoch", "num_epochs", "num-epochs", "epochs"),
        "normalize": ("normalize", "use_normalization"),
        "hidden_units": ("hidden_units", "hidden-units", "hidden"),
        "num_layers": ("num_layers", "num-layers", "layers"),
        "gamma": ("gamma",),
        "max_steps": ("max_steps", "max-steps", "maxsteps", "num_steps"),
        "time_horizon": ("time_horizon", "time-horizon", "timehorizon"),
    }

    found: Dict[str, Any] = {}

    results_dir = os.path.dirname(os.path.dirname(filepath))
    config_params = load_hyperparameters_from_config(results_dir)

    found.update(config_params)

    # 1) search metadata top-level keys
    for out_key, candidate_keys in hyper_keys.items():
        for k in candidate_keys:
            if k in meta:
                found[out_key] = try_cast_number(meta[k])
                break

    # 2) If command_line_arguments present in metadata, parse heuristically.
    cmdline = (
        meta.get("command_line_arguments")
        or meta.get("command_line")
        or meta.get("cmd")
    )
    if cmdline and isinstance(cmdline, str):
        try:

            parsed_args = parse_command_line_args_string(cmdline)
        except Exception:
            parsed_args = {}
        # Map parsed args into found keys
        # simple mapping: if parsed arg key matches any candidate, store it
        for out_key, candidate_keys in hyper_keys.items():
            if out_key in found:
                continue
            for cand in candidate_keys:
                if cand in parsed_args:
                    found[out_key] = parsed_args[cand]
                    break

    # 3) fallback defaults if not found
    defaults = {
        "DRL Algorithm Used": "Unknown",
        "batch_size": "",
        "buffer_size": "",
        "learning_rate": "",
        "beta": "",
        "epsilon": "",
        "lambda": "",
        "num_epoch": "",
        "normalize": "",
        "hidden_units": "",
        "num_layers": "",
        "gamma": "",
        "max_steps": "",
        "time_horizon": "",
    }

    # compose final values in header order
    row_values = []
    # HEADERS:
    # 1 run_id
    row_values.append(run_id)
    # 2 timestamp
    row_values.append(timestamp_iso)
    # 3 CPU Model
    row_values.append(cpu_model)
    # 4 CPU Cores
    row_values.append(cpu_cores)
    # 5 CPU Frequency
    row_values.append(cpu_freq)
    # 6 Total RAM
    row_values.append(total_ram)
    # 7 GPU Model
    row_values.append(gpu_model)
    # 8 Operating System
    row_values.append(operating_system)
    # 9 DRL Algorithm Used
    row_values.append(found.get("DRL Algorithm Used", defaults["DRL Algorithm Used"]))
    # 10 batch_size
    row_values.append(found.get("batch_size", defaults["batch_size"]))
    # 11 buffer_size
    row_values.append(found.get("buffer_size", defaults["buffer_size"]))
    # 12 learning_rate
    row_values.append(found.get("learning_rate", defaults["learning_rate"]))
    # 13 beta
    row_values.append(found.get("beta", defaults["beta"]))
    # 14 epsilon
    row_values.append(found.get("epsilon", defaults["epsilon"]))
    # 15 lambda
    row_values.append(found.get("lambda", defaults["lambda"]))
    # 16 num_epoch
    row_values.append(found.get("num_epoch", defaults["num_epoch"]))
    # 17 normalize
    row_values.append(found.get("normalize", defaults["normalize"]))
    # 18 hidden_units
    row_values.append(found.get("hidden_units", defaults["hidden_units"]))
    # 19 num_layers
    row_values.append(found.get("num_layers", defaults["num_layers"]))
    # 20 gamma
    row_values.append(found.get("gamma", defaults["gamma"]))
    # 21 max_steps
    row_values.append(found.get("max_steps", defaults["max_steps"]))
    # 22 time_horizon
    row_values.append(found.get("time_horizon", defaults["time_horizon"]))

    # 23 3DBall.Environment.CumulativeReward.mean
    gauge_key = "3DBall.Environment.CumulativeReward.mean"
    gauge_val = extract_gauge_value(gauges, gauge_key)
    if gauge_val is None:
        # try alternative lookup in gauges dictionary by iterating keys
        for k in gauges.keys():
            if "CumulativeReward" in k and "mean" in k:
                gauge_val = extract_gauge_value(gauges, k)
                break
    if gauge_val is None:
        gauge_val_out = ""
    else:
        gauge_val_out = f"{gauge_val:.6f}"
    row_values.append(gauge_val_out)

    # 24 Time Elapsed/s: end - start
    end_seconds_raw = safe_get_meta(meta, ("end_time_seconds", "end_time", "end"))
    time_elapsed_out = ""
    try:
        if start_seconds is not None and end_seconds_raw is not None:
            end_seconds = float(end_seconds_raw)
            elapsed = end_seconds - float(start_seconds)
            time_elapsed_out = f"{round(elapsed, 3):.3f}"
        elif "total" in data:
            # fallback to data["total"]
            elapsed_raw = data.get("total")
            if elapsed_raw is not None:
                time_elapsed_out = f"{float(elapsed_raw):.3f}"
    except Exception:
        time_elapsed_out = ""
    row_values.append(time_elapsed_out)

    # ensure row has same length as HEADERS
    if len(row_values) != len(HEADERS):
        # pad with empty strings or trim (shouldn't happen)
        if len(row_values) < len(HEADERS):
            row_values.extend([""] * (len(HEADERS) - len(row_values)))
        else:
            row_values = row_values[: len(HEADERS)]

    return row_values


# ---------------- CSV file handling ----------------


def ensure_output_file():
    """Ensure output directory exists and the CSV file exists with headers if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(OUTPUT_FILE):
        # create file and write headers
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)


def append_row_to_csv(row: list):
    """Append a single row to CSV file (assumes headers present or created)."""
    # Ensure file exists + headers
    file_exists = os.path.exists(OUTPUT_FILE)
    if not file_exists:
        ensure_output_file()
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ---------------- Entry point ----------------


def main(argv):
    if len(argv) != 2:
        print("Usage: python data_automation.py /path/to/mlagents_timer.json")
        sys.exit(2)
    path = argv[1]
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        sys.exit(3)
    # Load JSON file
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON from {path}: {e}")
        sys.exit(4)

    # Build row
    row = build_row_from_json(path, data)

    # Ensure CSV exists and write/append
    ensure_output_file()
    append_row_to_csv(row)
    print(f"Appended run '{get_run_id_from_path(path)}' to {OUTPUT_FILE}")


if __name__ == "__main__":
    main(sys.argv)
