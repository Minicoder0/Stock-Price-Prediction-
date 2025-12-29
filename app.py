"""
Stock Price Prediction with AI - Streamlit Web UI
A clean, professional interface for stock analysis with LLM-powered explanations
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from llm_explainer import StockPredictionExplainer

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction with AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .buy-signal {
        color: #28a745;
        font-weight: bold;
    }
    .sell-signal {
        color: #dc3545;
        font-weight: bold;
    }
    .hold-signal {
        color: #ffc107;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Stock Price Prediction with AI</h1>', unsafe_allow_html=True)

# Sidebar - Quick Guide
with st.sidebar:
    st.header("📚 Quick Guide")
    
    with st.expander("🚀 How to Use", expanded=True):
        st.markdown("""
        **Single Stock Analysis:**
        1. Enter a stock symbol (e.g., AAPL, TSLA)
        2. Select date range (default: 2 years)
        3. Click "Analyze Stock"
        4. Review AI-powered insights
        
        **Multi-Stock Comparison:**
        1. Switch to "Compare Stocks" tab
        2. Enter multiple symbols (comma-separated)
        3. Get side-by-side analysis
        """)
    
    with st.expander("📊 Understanding Metrics"):
        st.markdown("""
        **MAPE** (Prediction Accuracy):
        - < 5%: Excellent
        - 5-10%: Good
        - > 10%: Moderate
        
        **Volatility**:
        - < 20%: Low (Stable)
        - 20-40%: Moderate
        - > 40%: High (Risky)
        
        **Trading Signals**:
        - 🟢 **BUY**: Price expected to rise
        - 🔴 **SELL**: Price expected to fall
        - 🟡 **HOLD**: No strong movement
        
        **Risk Levels**:
        - 🛡️ **LOW**: Safe for conservative
        - ⚠️ **MODERATE**: Normal risk
        - 🚨 **HIGH**: High volatility
        """)
    
    with st.expander("💡 Popular Stocks"):
        st.markdown("""
        **Technology:**
        AAPL, MSFT, GOOGL, NVDA, META
        
        **Finance:**
        JPM, BAC, GS, C, WFC
        
        **Healthcare:**
        JNJ, PFE, UNH, ABBV
        
        **E-Commerce:**
        AMZN, WMT, TGT
        
        **EV/Auto:**
        TSLA, F, GM
        """)
    
    st.warning("⚠️ **Educational Tool Only**\n\nNot financial advice. Always do your own research.")

# Main tabs
tab1, tab2 = st.tabs(["📊 Single Stock Analysis", "🔄 Compare Stocks"])

# Tab 1: Single Stock Analysis
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol:", value="AAPL", help="e.g., AAPL, TSLA, MSFT").upper()
    
    with col2:
        days = st.selectbox("Historical Data:", [365, 730, 1095, 1825], index=1, 
                            format_func=lambda x: f"{x//365} Year{'s' if x>365 else ''}")
    
    if st.button("🔍 Analyze Stock", key="analyze_single"):
        if symbol:
            with st.spinner(f"Fetching data for {symbol}..."):
                try:
                    # Fetch data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    stock = yf.Ticker(symbol)
                    df = stock.history(start=start_date, end=end_date)
                    
                    if df.empty:
                        st.error(f"❌ No data found for {symbol}. Please check the symbol.")
                    else:
                        # Get company info
                        try:
                            info = stock.info
                            company_name = info.get('longName', symbol)
                            sector = info.get('sector', 'Unknown')
                        except:
                            company_name = symbol
                            sector = "Unknown"
                        
                        st.success(f"✅ Loaded {len(df)} days of data for {company_name}")
                        
                        # Prepare data
                        prices = df['Close'].copy()
                        train_size = int(len(prices) * 0.8)
                        train = prices[:train_size]
                        test = prices[train_size:]
                        
                        # Train ARIMA
                        with st.spinner("Training ARIMA model..."):
                            model = ARIMA(train, order=(1, 1, 1))
                            model_fit = model.fit()
                            forecast = model_fit.forecast(steps=len(test))
                            forecast.index = test.index
                        
                        # Calculate metrics
                        rmse = np.sqrt(mean_squared_error(test, forecast))
                        mae = mean_absolute_error(test, forecast)
                        mape = np.mean(np.abs((test - forecast) / test)) * 100
                        
                        # LLM Analysis
                        with st.spinner("Generating AI insights..."):
                            explainer = StockPredictionExplainer(model_name="ARIMA")
                            current_price = train.iloc[-1]
                            predicted_price = forecast.iloc[0]
                            
                            llm_analysis = explainer.analyze_prediction(
                                current_price=current_price,
                                predicted_price=predicted_price,
                                historical_data=train
                            )
                        
                        # Display Results
                        st.markdown("---")
                        st.subheader(f"📊 Analysis Results for {symbol}")
                        
                        # Key Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"${llm_analysis['current_price']:.2f}")
                        
                        with col2:
                            st.metric("Predicted Price", 
                                     f"${llm_analysis['predicted_price']:.2f}",
                                     f"{llm_analysis['percent_change']:+.2f}%")
                        
                        with col3:
                            accuracy_color = "🟢" if mape < 5 else "🟡" if mape < 10 else "🔴"
                            st.metric("Prediction Accuracy", f"{accuracy_color} {mape:.2f}% MAPE")
                        
                        with col4:
                            vol_color = "🟢" if llm_analysis['volatility'] < 20 else "🟡" if llm_analysis['volatility'] < 40 else "🔴"
                            st.metric("Volatility", f"{vol_color} {llm_analysis['volatility']:.1f}%")
                        
                        # Trading Decision
                        st.markdown("---")
                        st.subheader("💡 AI Trading Recommendation")
                        
                        decision = llm_analysis['trading_decision']
                        action_icon = "🟢" if decision['action'] == "BUY" else "🔴" if decision['action'] == "SELL" else "🟡"
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            action_class = "buy-signal" if decision['action'] == "BUY" else "sell-signal" if decision['action'] == "SELL" else "hold-signal"
                            st.markdown(f'<div class="metric-card"><h3>{action_icon} Action: <span class="{action_class}">{decision["action"]}</span> ({decision["strength"]})</h3></div>', 
                                       unsafe_allow_html=True)
                            st.write(f"**Rationale:** {decision['rationale']}")
                            st.write(f"**Position Size:** {decision['suggested_position_size']}")
                            st.write(f"**Time Horizon:** {decision['time_horizon']}")
                        
                        with col2:
                            risk = llm_analysis['risk_assessment']
                            risk_icon = "🛡️" if risk['level'] == "LOW" else "⚠️" if risk['level'] == "MODERATE" else "🚨"
                            st.markdown(f'<div class="metric-card"><h3>{risk_icon} Risk Level: {risk["level"]}</h3></div>', 
                                       unsafe_allow_html=True)
                            st.write(f"**Stop Loss:** {risk['recommended_stop_loss']}")
                            st.write(f"**Confidence:** {llm_analysis['confidence']}")
                            st.write(f"**Description:** {risk['description']}")
                        
                        # Explanation
                        with st.expander("📖 Detailed Explanation", expanded=False):
                            st.write(llm_analysis['explanation'])
                        
                        # Visualization
                        st.markdown("---")
                        st.subheader("📈 Price Prediction Chart")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(train.index, train, label="Training Data", alpha=0.7, color='blue')
                        ax.plot(test.index, test, label="Actual Price", color='green', linewidth=2)
                        ax.plot(forecast.index, forecast, label="ARIMA Forecast", color='red', linestyle='--', linewidth=2)
                        ax.set_title(f"{symbol} Stock Price: Actual vs Predicted", fontsize=16, fontweight='bold')
                        ax.set_xlabel("Date", fontsize=12)
                        ax.set_ylabel("Price ($)", fontsize=12)
                        ax.legend(loc='best')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Performance Metrics
                        with st.expander("📊 Model Performance Metrics"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("RMSE", f"${rmse:.2f}")
                            with col2:
                                st.metric("MAE", f"${mae:.2f}")
                            with col3:
                                st.metric("MAPE", f"{mape:.2f}%")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Please check the stock symbol and try again.")
        else:
            st.warning("⚠️ Please enter a stock symbol")

# Tab 2: Compare Stocks
with tab2:
    st.subheader("🔄 Multi-Stock Comparison")
    
    symbols_input = st.text_input("Enter stock symbols (comma-separated):", 
                                   value="AAPL,MSFT,GOOGL",
                                   help="e.g., AAPL,MSFT,GOOGL,TSLA").upper()
    
    if st.button("🔍 Compare Stocks", key="compare_multi"):
        symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
        
        if len(symbols) < 2:
            st.warning("⚠️ Please enter at least 2 stock symbols")
        else:
            comparison_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(symbols):
                status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
                progress_bar.progress((i + 1) / len(symbols))
                
                try:
                    # Fetch and analyze
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=730)
                    
                    stock = yf.Ticker(symbol)
                    df = stock.history(start=start_date, end=end_date)
                    
                    if not df.empty:
                        prices = df['Close'].copy()
                        train_size = int(len(prices) * 0.8)
                        train = prices[:train_size]
                        test = prices[train_size:]
                        
                        # ARIMA
                        model = ARIMA(train, order=(1, 1, 1))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=len(test))
                        
                        mape = np.mean(np.abs((test - forecast) / test)) * 100
                        
                        # LLM
                        explainer = StockPredictionExplainer(model_name="ARIMA")
                        llm_analysis = explainer.analyze_prediction(
                            current_price=train.iloc[-1],
                            predicted_price=forecast.iloc[0],
                            historical_data=train
                        )
                        
                        comparison_data.append({
                            'Symbol': symbol,
                            'Current Price': f"${llm_analysis['current_price']:.2f}",
                            'Predicted Change': f"{llm_analysis['percent_change']:+.2f}%",
                            'MAPE': f"{mape:.2f}%",
                            'Volatility': f"{llm_analysis['volatility']:.1f}%",
                            'Action': llm_analysis['trading_decision']['action'],
                            'Strength': llm_analysis['trading_decision']['strength'],
                            'Risk': llm_analysis['risk_assessment']['level'],
                            'Confidence': llm_analysis['confidence']
                        })
                
                except Exception as e:
                    st.warning(f"⚠️ Could not analyze {symbol}: {str(e)}")
            
            status_text.text("✅ Analysis complete!")
            progress_bar.empty()
            
            if comparison_data:
                st.markdown("---")
                st.subheader("📊 Comparison Results")
                
                df_comparison = pd.DataFrame(comparison_data)
                
                # Style the dataframe
                def highlight_action(val):
                    if val == 'BUY':
                        return 'background-color: #d4edda; color: #155724'
                    elif val == 'SELL':
                        return 'background-color: #f8d7da; color: #721c24'
                    else:
                        return 'background-color: #fff3cd; color: #856404'
                
                styled_df = df_comparison.style.applymap(highlight_action, subset=['Action'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Summary
                st.markdown("---")
                st.subheader("🏆 Recommendations")
                
                buy_stocks = df_comparison[df_comparison['Action'] == 'BUY']
                if not buy_stocks.empty:
                    st.success(f"**🟢 BUY Signals:** {', '.join(buy_stocks['Symbol'].tolist())}")
                
                low_risk = df_comparison[df_comparison['Risk'] == 'LOW']
                if not low_risk.empty:
                    st.info(f"**🛡️ Lowest Risk:** {', '.join(low_risk['Symbol'].tolist())}")
                
                # Download button
                csv = df_comparison.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison (CSV)",
                    data=csv,
                    file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Stock Price Prediction with AI | Educational Tool Only</p>
    <p><small>⚠️ Not financial advice. Always do your own research before investing.</small></p>
</div>
""", unsafe_allow_html=True)
