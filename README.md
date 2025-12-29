# 📈 Stock Price Prediction with AI

A professional web application for stock price prediction using ARIMA models with AI-powered explanations and trading recommendations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 📊 **Real-time Stock Data** - Fetches live data from Yahoo Finance
- 🤖 **AI-Powered Analysis** - ARIMA predictions with LLM explanations
- 💡 **Trading Recommendations** - BUY/SELL/HOLD signals with confidence levels
- ⚠️ **Risk Assessment** - Automatic risk analysis and stop-loss calculations
- 📈 **Multi-Stock Comparison** - Compare multiple stocks side-by-side
- 🎨 **Clean Web Interface** - Professional Streamlit UI with built-in guide
- 💾 **Export Results** - Download analysis as CSV

## 🚀 Quick Start

### ⚡ Easiest Way (Windows)

**Just double-click `run.bat`** - it will install everything and start the app!

### 📋 Manual Method

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📖 How to Use

### Single Stock Analysis
1. Enter a stock symbol (e.g., **AAPL**, **TSLA**, **MSFT**)
2. Select historical data period (default: 2 years)
3. Click **"Analyze Stock"**
4. Review AI-powered predictions and recommendations

### Multi-Stock Comparison
1. Switch to **"Compare Stocks"** tab
2. Enter multiple symbols comma-separated (e.g., AAPL,MSFT,GOOGL)
3. Click **"Compare Stocks"**
4. View side-by-side comparison table
5. Download results as CSV

## 📊 Understanding the Metrics

### Prediction Accuracy (MAPE)
- **< 5%**: Excellent accuracy 🟢
- **5-10%**: Good accuracy 🟡
- **> 10%**: Moderate accuracy (use with caution) 🔴

### Volatility
- **< 20%**: Low volatility (stable stock) 🛡️
- **20-40%**: Moderate volatility ⚠️
- **> 40%**: High volatility (risky) 🚨

### Trading Signals
- **🟢 BUY**: Stock price expected to increase
- **🔴 SELL**: Stock price expected to decrease
- **🟡 HOLD**: No significant movement predicted

### Risk Levels
- **🛡️ LOW**: Safe for conservative investors
- **⚠️ MODERATE**: Normal market risk
- **🚨 HIGH**: High volatility, risk-tolerant only

## 💡 Popular Stocks to Try

**Technology:** AAPL, MSFT, GOOGL, NVDA, META  
**Finance:** JPM, BAC, GS, C, WFC  
**Healthcare:** JNJ, PFE, UNH, ABBV, MRK  
**E-Commerce:** AMZN, WMT, TGT, COST  
**EV/Auto:** TSLA, F, GM, RIVN  
**Entertainment:** DIS, NFLX, SPOT  

## 🛠️ Technical Details

### Technologies Used
- **Streamlit** - Web interface
- **yfinance** - Stock data API
- **ARIMA** - Time series forecasting model
- **scikit-learn** - Performance metrics
- **Custom LLM Explainer** - AI-powered analysis engine

### Model Details
- **Algorithm**: ARIMA (AutoRegressive Integrated Moving Average)
- **Parameters**: (1, 1, 1)
- **Train/Test Split**: 80/20
- **Metrics**: RMSE, MAE, MAPE

## ⚠️ Important Disclaimer

**This is an educational tool for learning purposes only.**

- ❌ NOT financial advice
- ❌ NOT guaranteed to be accurate
- ❌ Past performance ≠ future results
- ❌ Always do your own research
- ❌ Consult financial professionals before investing

**Use at your own risk. Only invest what you can afford to lose.**

## 📁 Project Structure

```
Stock-Price-Prediction--main/
├── app.py                 # Streamlit web application
├── llm_explainer.py       # AI analysis engine
├── requirements.txt       # Python dependencies
├── stock_data.csv         # Sample data (NVIDIA)
└── README.md              # This file
```

## 🎓 Educational Use Cases

- Learn time series forecasting
- Understand stock market analysis
- Practice risk management
- Study trading strategies
- Build data science portfolio

## 🔧 Troubleshooting

### "No data found for symbol"
- Verify stock symbol on Yahoo Finance website
- Check if stock is actively traded
- Try different symbol

### "Module not found" error
- Run: `pip install -r requirements.txt`
- Ensure Python 3.8+ installed

### App won't start
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

## 🌟 Future Enhancements

Potential additions:
- [ ] LSTM neural network model
- [ ] Technical indicators (RSI, MACD)
- [ ] Sentiment analysis from news
- [ ] Portfolio optimization
- [ ] Real-time alerts
- [ ] Integration with real LLM APIs (GPT-4, Claude)

## 📝 License

MIT License - feel free to use for educational purposes

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share improvements

---

**Made with ❤️ for stock market education**

*Last Updated: December 29, 2025*
