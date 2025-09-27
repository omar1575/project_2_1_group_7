#!/usr/bin/env python3
"""
Complete Unity ML-Agents Project Setup Verification Script
This script verifies that EVERYTHING is set up correctly for the project.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import json

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(message):
    print(f"{Colors.GREEN}✓{Colors.END} {message}")

def print_error(message):
    print(f"{Colors.RED}✗{Colors.END} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠{Colors.END} {message}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ{Colors.END} {message}")

def print_header(message):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

class SetupVerifier:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.project_root = Path.cwd()
        
    def verify_python_environment(self):
        print_header("PYTHON ENVIRONMENT VERIFICATION")
        
        # Check Python version
        version = sys.version_info
        print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major == 3 and version.minor == 10:
            print_success("Python 3.10.x detected - Compatible!")
        else:
            self.warnings.append(f"Python {version.major}.{version.minor} may have compatibility issues. Python 3.10 recommended.")
            print_warning(f"Python {version.major}.{version.minor} detected - May have compatibility issues")
            
        # Check if we're in virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print_success("Virtual environment detected")
        else:
            self.errors.append("Not running in virtual environment")
            print_error("Not running in virtual environment")
            
        # Check pip
        try:
            import pip
            print_success("pip available")
        except ImportError:
            self.errors.append("pip not available")
            print_error("pip not available")

    def verify_required_packages(self):
        print_header("REQUIRED PACKAGES VERIFICATION")
        
        required_packages = {
            'mlagents': 'Unity ML-Agents',
            'torch': 'PyTorch',
            'numpy': 'NumPy',
            'tensorboard': 'TensorBoard',
            'matplotlib': 'Matplotlib',
            'pandas': 'Pandas',
            'sklearn': 'Scikit-learn',
            'seaborn': 'Seaborn',
            'plotly': 'Plotly',
            'yaml': 'PyYAML',
            'psutil': 'psutil',
            'gymnasium': 'Gymnasium'
        }
        
        for package, description in required_packages.items():
            try:
                if package == 'sklearn':
                    importlib.import_module('sklearn')
                else:
                    importlib.import_module(package)
                print_success(f"{description} imported successfully")
            except ImportError:
                self.errors.append(f"Failed to import {description}")
                print_error(f"Failed to import {description}")

    def verify_ml_agents_specific(self):
        print_header("ML-AGENTS SPECIFIC VERIFICATION")
        
        try:
            from mlagents_envs.environment import UnityEnvironment
            print_success("ML-Agents UnityEnvironment imported")
        except ImportError as e:
            self.errors.append(f"ML-Agents UnityEnvironment import failed: {e}")
            print_error(f"ML-Agents UnityEnvironment import failed: {e}")
            
        try:
            from mlagents.trainers import trainer_util
            print_success("ML-Agents trainers imported")
        except ImportError as e:
            self.errors.append(f"ML-Agents trainers import failed: {e}")
            print_error(f"ML-Agents trainers import failed: {e}")

    def verify_directory_structure(self):
        print_header("PROJECT DIRECTORY STRUCTURE VERIFICATION")
        
        required_dirs = [
            "Assets/Scripts",
            "Assets/Scenes", 
            "Assets/Materials",
            "Assets/Prefabs",
            "Data/RawData",
            "Data/ProcessedData",
            "Models",
            "Documentation",
            "Scripts/Python",
            "Scripts/DataCollection", 
            "Scripts/Analysis",
            "Config",
            "ml-agents-toolkit"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                print_success(f"Directory exists: {dir_path}")
            else:
                self.errors.append(f"Missing directory: {dir_path}")
                print_error(f"Missing directory: {dir_path}")

    def verify_required_files(self):
        print_header("REQUIRED FILES VERIFICATION")
        
        required_files = [
            "requirements.txt",
            ".gitignore",
            ".gitattributes",
            "Config/project_config.yaml",
            "Config/training_config.yaml"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print_success(f"File exists: {file_path}")
            else:
                self.errors.append(f"Missing file: {file_path}")
                print_error(f"Missing file: {file_path}")

    def verify_git_setup(self):
        print_header("GIT REPOSITORY VERIFICATION")
        
        # Check if .git directory exists
        git_dir = self.project_root / ".git"
        if git_dir.exists():
            print_success(".git directory exists")
        else:
            self.errors.append("Not a git repository")
            print_error("Not a git repository")
            return
            
        # Check git status
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                print_success("Git repository is accessible")
                
                # Check for remote
                result = subprocess.run(['git', 'remote', '-v'], 
                                      capture_output=True, text=True, cwd=self.project_root)
                if result.stdout.strip():
                    print_success("Git remote configured")
                else:
                    self.warnings.append("No git remote configured")
                    print_warning("No git remote configured")
                    
            else:
                self.errors.append("Git repository issues")
                print_error("Git repository issues")
        except FileNotFoundError:
            self.errors.append("Git not installed or not in PATH")
            print_error("Git not installed or not in PATH")

    def verify_ml_agents_installation(self):
        print_header("ML-AGENTS INSTALLATION VERIFICATION")
        
        ml_agents_dir = self.project_root / "ml-agents-toolkit"
        if ml_agents_dir.exists():
            print_success("ML-Agents toolkit directory exists")
            
            # Check if installed in development mode
            try:
                result = subprocess.run(['pip', 'show', 'mlagents'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    output = result.stdout
                    if str(ml_agents_dir) in output:
                        print_success("ML-Agents installed in development mode")
                    else:
                        print_warning("ML-Agents installed but not in development mode")
                else:
                    self.errors.append("ML-Agents not installed via pip")
                    print_error("ML-Agents not installed via pip")
            except FileNotFoundError:
                self.errors.append("pip command not found")
                print_error("pip command not found")
        else:
            self.errors.append("ML-Agents toolkit directory not found")
            print_error("ML-Agents toolkit directory not found")

    def verify_unity_compatibility(self):
        print_header("UNITY COMPATIBILITY CHECK")
        
        # Check ProjectSettings
        project_version_file = self.project_root / "ProjectSettings" / "ProjectVersion.txt"
        if project_version_file.exists():
            print_success("Unity ProjectVersion.txt exists")
        else:
            self.warnings.append("Unity ProjectVersion.txt not found - Unity project may not be properly configured")
            print_warning("Unity ProjectVersion.txt not found")

    def run_comprehensive_import_test(self):
        print_header("COMPREHENSIVE IMPORT TEST")
        
        test_code = """
import sys
import numpy as np
import torch
import mlagents
from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers import trainer_util
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns
import plotly
import yaml
import psutil
import gymnasium

print("All critical imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
"""
        
        try:
            exec(test_code)
            print_success("All comprehensive imports successful!")
        except Exception as e:
            self.errors.append(f"Comprehensive import test failed: {e}")
            print_error(f"Comprehensive import test failed: {e}")

    def generate_setup_report(self):
        print_header("SETUP VERIFICATION REPORT")
        
        total_checks = 8  # Number of verification sections
        
        if not self.errors and not self.warnings:
            print_success("🎉 PERFECT SETUP! Everything is configured correctly!")
            print_info("Your project is ready for development.")
            return True
        elif not self.errors:
            print_warning(f"⚠️ SETUP COMPLETE WITH WARNINGS ({len(self.warnings)} warnings)")
            print_info("Your setup works but has some minor issues:")
            for warning in self.warnings:
                print(f"   • {warning}")
            return True
        else:
            print_error(f"❌ SETUP INCOMPLETE ({len(self.errors)} errors, {len(self.warnings)} warnings)")
            print_info("Critical errors that must be fixed:")
            for error in self.errors:
                print(f"   • {error}")
            if self.warnings:
                print_info("Additional warnings:")
                for warning in self.warnings:
                    print(f"   • {warning}")
            return False

    def run_full_verification(self):
        print_header("UNITY ML-AGENTS PROJECT SETUP VERIFICATION")
        print_info(f"Verifying setup in: {self.project_root}")
        
        self.verify_python_environment()
        self.verify_required_packages()
        self.verify_ml_agents_specific()
        self.verify_ml_agents_installation()
        self.verify_directory_structure()
        self.verify_required_files()
        self.verify_git_setup()
        self.verify_unity_compatibility()
        self.run_comprehensive_import_test()
        
        return self.generate_setup_report()

def main():
    verifier = SetupVerifier()
    success = verifier.run_full_verification()
    
    if success:
        print_header("NEXT STEPS")
        print_info("Your setup is ready! You can now:")
        print("   1. Open Unity Hub and add this project")
        print("   2. Install ML-Agents package via Unity Package Manager")
        print("   3. Start working on your ML-Agents training prediction project")
        print("   4. Create your data collection and analysis scripts")
        print("   5. Begin experimenting with ML-Agents environments")
    else:
        print_header("REQUIRED ACTIONS")
        print_info("Fix the errors above before proceeding with development.")
        print("Run this script again after making corrections.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())