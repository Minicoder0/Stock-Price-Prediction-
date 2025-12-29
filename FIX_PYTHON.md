# ⚠️ PYTHON SETUP ISSUE - QUICK FIX

## Problem
Python IS installed but a Windows shortcut is preventing it from running.

## Solution (30 seconds)

### Step 1: Disable Windows App Alias
1. Press **Windows + I** (opens Settings)
2. Go to **Apps**
3. Click **Advanced app settings** (at bottom)
4. Click **App execution aliases**
5. Turn **OFF** these switches:
   - `App Installer python.exe`
   - `App Installer python3.exe`
6. Close Settings

### Step 2: Run the App
Double-click **`launch.py`** in this folder

The app will:
✅ Install all packages automatically
✅ Start the web server  
✅ Open in your browser at http://localhost:8501

---

## Alternative (If Above Doesn't Work)

Open PowerShell in this folder and run:
```powershell
C:\Users\mminh\AppData\Local\Microsoft\WindowsApps\python.exe launch.py
```

---

## Need Real Python Installation?

Download from: https://www.python.org/downloads/

Make sure to check "Add Python to PATH" during installation!
