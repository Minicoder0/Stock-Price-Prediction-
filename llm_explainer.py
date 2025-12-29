"""
LLM-Based Stock Price Prediction Explainer
This module provides AI-powered explanations for stock predictions and trading decisions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

class StockPredictionExplainer:
    """
    A class that uses LLM-like logic to explain stock predictions and provide trading decisions.
    """
    
    def __init__(self, model_name: str = "ARIMA"):
        """
        Initialize the explainer.
        
        Args:
            model_name: Name of the prediction model (e.g., 'ARIMA', 'Random Walk')
        """
        self.model_name = model_name
        self.prediction_history = []
        
    def analyze_prediction(self, 
                          current_price: float,
                          predicted_price: float,
                          historical_data: pd.Series,
                          volatility: Optional[float] = None) -> Dict:
        """
        Provide comprehensive analysis of a stock prediction.
        
        Args:
            current_price: Current stock price
            predicted_price: Predicted future price
            historical_data: Historical price data
            volatility: Optional volatility metric
            
        Returns:
            Dictionary containing analysis results
        """
        # Calculate key metrics
        price_change = predicted_price - current_price
        percent_change = (price_change / current_price) * 100
        
        # Calculate historical volatility if not provided
        if volatility is None:
            returns = historical_data.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Generate explanation
        explanation = self._generate_explanation(
            current_price, predicted_price, percent_change, volatility
        )
        
        # Generate trading decision
        decision = self._generate_trading_decision(
            percent_change, volatility, historical_data
        )
        
        # Calculate confidence level
        confidence = self._calculate_confidence(volatility, historical_data)
        
        # Generate risk assessment
        risk_assessment = self._assess_risk(volatility, percent_change)
        
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': self.model_name,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change': round(price_change, 2),
            'percent_change': round(percent_change, 2),
            'volatility': round(volatility * 100, 2),
            'confidence': confidence,
            'explanation': explanation,
            'trading_decision': decision,
            'risk_assessment': risk_assessment
        }
        
        self.prediction_history.append(result)
        return result
    
    def _generate_explanation(self, current_price: float, predicted_price: float, 
                            percent_change: float, volatility: float) -> str:
        """Generate detailed explanation for the prediction."""
        
        direction = "increase" if percent_change > 0 else "decrease"
        magnitude = abs(percent_change)
        
        # Determine magnitude description
        if magnitude < 1:
            change_desc = "slight"
        elif magnitude < 3:
            change_desc = "moderate"
        elif magnitude < 5:
            change_desc = "significant"
        else:
            change_desc = "substantial"
        
        # Determine volatility description
        if volatility < 0.2:
            vol_desc = "relatively stable"
        elif volatility < 0.4:
            vol_desc = "moderately volatile"
        else:
            vol_desc = "highly volatile"
        
        explanation = f"""
PREDICTION EXPLANATION:
-----------------------
Based on the {self.model_name} model analysis, the stock is expected to {direction} 
from ${current_price:.2f} to ${predicted_price:.2f}, representing a {change_desc} 
change of {abs(percent_change):.2f}%.

MARKET CONTEXT:
The stock has shown {vol_desc} behavior with an annualized volatility of {volatility*100:.2f}%. 
This {'high' if volatility > 0.3 else 'moderate' if volatility > 0.2 else 'low'} volatility 
suggests that price movements {'may be unpredictable' if volatility > 0.3 else 'are relatively predictable'}.

MODEL INSIGHTS:
The {self.model_name} model has captured the underlying trends and patterns in the 
historical data. {'However, given the high volatility, predictions should be interpreted with caution.' 
if volatility > 0.3 else 'The model shows reasonable confidence in this prediction.'}
        """.strip()
        
        return explanation
    
    def _generate_trading_decision(self, percent_change: float, volatility: float, 
                                   historical_data: pd.Series) -> Dict:
        """Generate trading decision with rationale."""
        
        # Calculate trend
        recent_trend = historical_data.tail(20).pct_change().mean()
        
        # Decision logic
        if percent_change > 2 and volatility < 0.4:
            action = "BUY"
            strength = "STRONG"
            rationale = "Significant upward prediction with manageable volatility"
        elif percent_change > 0.5 and volatility < 0.3:
            action = "BUY"
            strength = "MODERATE"
            rationale = "Positive prediction with low volatility"
        elif percent_change < -2 and volatility < 0.4:
            action = "SELL"
            strength = "STRONG"
            rationale = "Significant downward prediction suggests exit opportunity"
        elif percent_change < -0.5:
            action = "SELL"
            strength = "MODERATE"
            rationale = "Negative prediction suggests reducing position"
        else:
            action = "HOLD"
            strength = "NEUTRAL"
            rationale = "Price change not significant enough to warrant action"
        
        # Add volatility consideration
        if volatility > 0.4 and action != "HOLD":
            strength = "WEAK" if strength == "STRONG" else "WEAK"
            rationale += " (High volatility reduces confidence)"
        
        return {
            'action': action,
            'strength': strength,
            'rationale': rationale,
            'suggested_position_size': self._suggest_position_size(strength, volatility),
            'time_horizon': self._suggest_time_horizon(volatility)
        }
    
    def _calculate_confidence(self, volatility: float, historical_data: pd.Series) -> str:
        """Calculate confidence level in prediction."""
        
        # Factors affecting confidence
        volatility_score = 1 - min(volatility, 1.0)  # Lower volatility = higher confidence
        trend_consistency = self._calculate_trend_consistency(historical_data)
        
        # Combined confidence score
        confidence_score = (volatility_score * 0.6 + trend_consistency * 0.4)
        
        if confidence_score > 0.7:
            return "HIGH"
        elif confidence_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_trend_consistency(self, historical_data: pd.Series) -> float:
        """Calculate how consistent the trend has been."""
        
        if len(historical_data) < 20:
            return 0.5
        
        # Calculate directional consistency
        returns = historical_data.pct_change().dropna()
        positive_days = (returns > 0).sum()
        consistency = abs(positive_days / len(returns) - 0.5) * 2
        
        return consistency
    
    def _assess_risk(self, volatility: float, percent_change: float) -> Dict:
        """Provide detailed risk assessment."""
        
        # Risk level determination
        if volatility < 0.2 and abs(percent_change) < 2:
            risk_level = "LOW"
            risk_desc = "Stable stock with predictable movement"
        elif volatility < 0.4 and abs(percent_change) < 5:
            risk_level = "MODERATE"
            risk_desc = "Normal market risk with manageable volatility"
        else:
            risk_level = "HIGH"
            risk_desc = "Elevated risk due to high volatility or large predicted movement"
        
        # Risk factors
        risk_factors = []
        if volatility > 0.4:
            risk_factors.append("High volatility increases prediction uncertainty")
        if abs(percent_change) > 5:
            risk_factors.append("Large predicted price movement")
        if volatility > 0.5 and abs(percent_change) > 3:
            risk_factors.append("Combination of high volatility and significant price change")
        
        return {
            'level': risk_level,
            'description': risk_desc,
            'factors': risk_factors if risk_factors else ["No significant risk factors identified"],
            'recommended_stop_loss': self._calculate_stop_loss(percent_change, volatility)
        }
    
    def _suggest_position_size(self, strength: str, volatility: float) -> str:
        """Suggest position sizing based on signal strength and volatility."""
        
        if strength == "STRONG" and volatility < 0.3:
            return "25-30% of portfolio"
        elif strength == "STRONG" or (strength == "MODERATE" and volatility < 0.3):
            return "15-20% of portfolio"
        elif strength == "MODERATE":
            return "10-15% of portfolio"
        else:
            return "5-10% of portfolio (or avoid)"
    
    def _suggest_time_horizon(self, volatility: float) -> str:
        """Suggest appropriate time horizon."""
        
        if volatility < 0.2:
            return "Medium to Long-term (3-12 months)"
        elif volatility < 0.4:
            return "Short to Medium-term (1-6 months)"
        else:
            return "Short-term only (days to weeks)"
    
    def _calculate_stop_loss(self, percent_change: float, volatility: float) -> str:
        """Calculate recommended stop-loss level."""
        
        # Base stop-loss on volatility
        stop_loss_percent = max(3, volatility * 100 * 1.5)
        
        if percent_change > 0:
            # For long positions
            return f"-{stop_loss_percent:.1f}% (Place sell order at this level)"
        else:
            # For short positions or sell signals
            return f"+{stop_loss_percent:.1f}% (Place buy-back order at this level)"
    
    def generate_comprehensive_report(self, analysis_result: Dict) -> str:
        """Generate a comprehensive, formatted report."""
        
        report = f"""
================================================================================
NVIDIA STOCK PREDICTION ANALYSIS REPORT
================================================================================
Generated: {analysis_result['timestamp']}
Model: {analysis_result['model']}

PRICE PREDICTION
--------------------------------------------------------------------------------
Current Price:    ${analysis_result['current_price']:>10.2f}
Predicted Price:  ${analysis_result['predicted_price']:>10.2f}
Expected Change:  ${analysis_result['price_change']:>10.2f} ({analysis_result['percent_change']:+.2f}%)
Volatility:       {analysis_result['volatility']:>10.2f}%
Confidence:       {analysis_result['confidence']:>10s}

{analysis_result['explanation']}

TRADING RECOMMENDATION
--------------------------------------------------------------------------------
Action:           {analysis_result['trading_decision']['action']} ({analysis_result['trading_decision']['strength']})
Rationale:        {analysis_result['trading_decision']['rationale']}
Position Size:    {analysis_result['trading_decision']['suggested_position_size']}
Time Horizon:     {analysis_result['trading_decision']['time_horizon']}

RISK ASSESSMENT
--------------------------------------------------------------------------------
Risk Level:       {analysis_result['risk_assessment']['level']}
Description:      {analysis_result['risk_assessment']['description']}
Stop Loss:        {analysis_result['risk_assessment']['recommended_stop_loss']}

Risk Factors:"""
        
        for factor in analysis_result['risk_assessment']['factors']:
            report += f"\n  • {factor}"
        
        report += "\n\n" + "=" * 80
        report += "\nDISCLAIMER: This analysis is for educational purposes only."
        report += "\nNot financial advice. Always conduct your own research."
        report += "\n" + "=" * 80
        
        return report
    
    def batch_analyze(self, test_data: pd.Series, forecasts: pd.Series, 
                     historical_data: pd.Series) -> pd.DataFrame:
        """
        Analyze multiple predictions at once.
        
        Args:
            test_data: Actual test prices
            forecasts: Predicted prices
            historical_data: Historical price data for context
            
        Returns:
            DataFrame with analysis for each prediction
        """
        results = []
        
        for i, (date, actual_price) in enumerate(test_data.items()):
            predicted_price = forecasts.iloc[i]
            
            # Get relevant historical data up to this point
            hist_cutoff = historical_data.index.get_loc(date) if date in historical_data.index else len(historical_data)
            relevant_history = historical_data.iloc[:hist_cutoff]
            
            analysis = self.analyze_prediction(
                current_price=relevant_history.iloc[-1] if len(relevant_history) > 0 else actual_price,
                predicted_price=predicted_price,
                historical_data=relevant_history
            )
            
            analysis['date'] = date
            analysis['actual_price'] = actual_price
            analysis['prediction_error'] = predicted_price - actual_price
            
            results.append(analysis)
        
        return pd.DataFrame(results)


def main():
    """Example usage of the StockPredictionExplainer."""
    
    print("LLM-Based Stock Prediction Explainer")
    print("=" * 80)
    print("\nThis module provides AI-powered explanations for stock predictions.")
    print("Import this module in your Jupyter notebooks to use it.")
    print("\nExample usage:")
    print("""
    from llm_explainer import StockPredictionExplainer
    
    explainer = StockPredictionExplainer(model_name="ARIMA")
    
    analysis = explainer.analyze_prediction(
        current_price=current_price,
        predicted_price=forecast_price,
        historical_data=historical_prices
    )
    
    report = explainer.generate_comprehensive_report(analysis)
    print(report)
    """)


if __name__ == "__main__":
    main()
