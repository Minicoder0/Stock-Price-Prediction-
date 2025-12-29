
# Python Installation Auto-Fixer
# This script finds Python and adds it to PATH, then runs the app

import os
import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("AUTO-FIXING PYTHON PATH AND LAUNCHING APP")
print("=" * 70)
print()

# Common Python installation paths
python_paths = [
    Path(os.environ.get('LOCALAPPDATA', '')) / "Programs" / "Python" / "Python312" / "python.exe",
    Path(os.environ.get('LOCALAPPDATA', '')) / "Programs" / "Python" / "Python311" / "python.exe",
    Path(os.environ.get('LOCALAPPDATA', '')) / "Programs" / "Python" / "Python310" / "python.exe",
    Path("C:/Python312/python.exe"),
    Path("C:/Python311/python.exe"),
    Path("C:/Program Files/Python312/python.exe"),
    Path("C:/Program Files/Python311/python.exe"),
]

python_exe = None

# Find Python
print("Searching for Python installation...")
for path in python_paths:
    if path.exists():
        python_exe = str(path)
        print(f"✓ Found Python at: {python_exe}")
        break

if not python_exe:
    print("✗ Could not find Python!")
    print()
    print("Please install Python from: https://www.python.org/downloads/")
    print("Make sure to check 'Add Python to PATH' during installation!")
    input("\nPress Enter to exit...")
    sys.exit(1)

print()
print("=" * 70)
print("INSTALLING REQUIRED PACKAGES")
print("=" * 70)
print()

packages = ["streamlit", "yfinance", "pandas", "numpy", "matplotlib", "statsmodels", "scikit-learn"]

for i, package in enumerate(packages, 1):
    print(f"[{i}/{len(packages)}] Installing {package}...")
    try:
        subprocess.run([python_exe, "-m", "pip", "install", package, "--quiet"], check=True)
        print(f"  ✓ {package} installed")
    except:
        print(f"  ⚠ {package} may already be installed")

print()
print("=" * 70)
print("STARTING WEB APPLICATION")
print("=" * 70)
print()
print("The app will open in your browser at: http://localhost:8501")
print("Press Ctrl+C to stop the server")
print()
print("=" * 70)
print()

# Change to script directory
os.chdir(Path(__file__).parent)

# Run Streamlit
try:
    subprocess.run([python_exe, "-m", "streamlit", "run", "app.py"])
except KeyboardInterrupt:
    print("\n\nApp stopped by user.")
except Exception as e:
    print(f"\n\n✗ Error: {e}")
    input("\nPress Enter to exit...")
