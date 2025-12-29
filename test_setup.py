"""
Quick test script to verify all dependencies and core functionality
"""

print("Testing dependencies...")

try:
    import streamlit
    print("✓ streamlit")
except ImportError as e:
    print(f"✗ streamlit: {e}")

try:
    import yfinance
    print("✓ yfinance")
except ImportError as e:
    print(f"✗ yfinance: {e}")

try:
    import pandas
    print("✓ pandas")
except ImportError as e:
    print(f"✗ pandas: {e}")

try:
    import numpy
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import matplotlib
    print("✓ matplotlib")
except ImportError as e:
    print(f"✗ matplotlib: {e}")

try:
    import statsmodels
    print("✓ statsmodels")
except ImportError as e:
    print(f"✗ statsmodels: {e}")

try:
    import sklearn
    print("✓ sklearn")
except ImportError as e:
    print(f"✗ sklearn: {e}")

print("\nTesting llm_explainer module...")
try:
    from llm_explainer import StockPredictionExplainer
    print("✓ llm_explainer import successful")
    
    # Quick test
    import pandas as pd
    import numpy as np
    
    explainer = StockPredictionExplainer()
    data = pd.Series(np.random.randn(100) + 100)
    
    result = explainer.analyze_prediction(
        current_price=100,
        predicted_price=105,
        historical_data=data
    )
    
    print("✓ Analysis function works")
    print(f"  - Action: {result['trading_decision']['action']}")
    print(f"  - Risk: {result['risk_assessment']['level']}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests complete!")
