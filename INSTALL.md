# 🚀 QUICK START GUIDE

## STEP 1: Install Python (if not already installed)

### Option A: Microsoft Store (Easiest - Windows 10/11)
1. Open Microsoft Store
2. Search for "Python 3.12"
3. Click "Get" or "Install"
4. Wait for installation to complete

### Option B: Python.org
1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.12.x"
3. Run the installer
4. **IMPORTANT**: Check "Add Python to PATH"
5. Click "Install Now"

## STEP 2: Run the App

### Double-click `run.bat`

That's it! The batch file will:
- Check if Python is installed
- Help you install it if needed
- Install all required packages
- Start the web app
- Open it in your browser

## STEP 3: Use the App

Once the browser opens at http://localhost:8501:
- Enter a stock symbol (e.g., AAPL)
- Click "Analyze Stock"
- See AI-powered predictions!

All instructions are in the app's sidebar.

---

## Manual Installation (Advanced Users)

If the batch file doesn't work:

```bash
# Install packages
python -m pip install -r requirements.txt

# Run app
python -m streamlit run app.py
```

---

## Still Having Issues?

1. **Restart your computer** after installing Python
2. **Reopen Command Prompt** or PowerShell
3. Try running `run.bat` again

---

**Need Python?** 
👉 Microsoft Store: Search "Python 3.12"
👉 Or visit: https://www.python.org/downloads/
