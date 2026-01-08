import os
import subprocess
import glob
import sys

# ML-Agents saves to results/<run-id>/run_logs/timers.json
search_dir = os.path.join(os.getcwd(), "results")

if not os.path.exists(search_dir):
    print("No ML-Agents timer files found. Skipping data automation.")
    sys.exit(0)

# Look for timers.json in run_logs subdirectories
json_candidates = glob.glob(os.path.join(search_dir, "**", "run_logs", "timers.json"), recursive=True)

if not json_candidates:
    print("No ML-Agents timer files found. Skipping data automation.")
    sys.exit(0)

# Sort by modification time and pick the newest one
latest_json = max(json_candidates, key=os.path.getmtime)
print(f"Found latest timer file: {latest_json}")

# Run data automation script on it
data_automation_script = os.path.join(os.getcwd(), "HelperScripts/data_automation.py")
if not os.path.exists(data_automation_script):
    print(f"Error: {data_automation_script} not found.")
    sys.exit(1)

print("Running data_automation.py...")
subprocess.run([sys.executable, data_automation_script, latest_json], check=True)
print("Data automation complete.")