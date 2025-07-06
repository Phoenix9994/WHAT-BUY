#!/usr/bin/env python
# coding: utf-8
"""
STOCK PRICE PREDICTION USING MACHINE LEARNING
==============================================

This program uses artificial intelligence to predict whether a stock price will go UP or DOWN tomorrow.

HOW IT WORKS:
1. Downloads historical stock data (prices, volumes, dividends, etc.)
2. Creates "features" - mathematical indicators that might predict future price movements
3. Uses a "Random Forest" algorithm to learn patterns from historical data
4. Tests the model on past data to see how accurate it would have been
5. Makes a prediction for tomorrow based on what it learned

MACHINE LEARNING CONCEPTS EXPLAINED:
- Features: Input variables (like price changes, volume, moving averages)
- Target: What we want to predict (will price go up or down tomorrow?)
- Training: Teaching the algorithm using historical data
- Prediction: Using the trained model to guess future outcomes
- Precision: How often the model is correct when it predicts "UP"
"""

import yfinance as yf          # Library to download stock data from Yahoo Finance
import pandas as pd            # Library for handling data tables (like Excel)
import numpy as np             # Library for mathematical calculations
from sklearn.ensemble import RandomForestClassifier  # The AI algorithm we'll use
from sklearn.metrics import precision_score, classification_report  # Tools to measure accuracy
import warnings
warnings.filterwarnings('ignore')  # Hide technical warnings to keep output clean

class StockPredictor:
    """
    A class that predicts stock price movements using machine learning.
    
    WHAT IS A RANDOM FOREST?
    Think of it like asking 300 different experts (trees) for their opinion,
    then taking the majority vote. Each expert looks at different aspects
    of the data and makes a prediction. The final prediction is based on
    what most experts agree on.
    """
    
    def __init__(self, symbol, n_estimators=300, min_samples_split=50, random_state=1):
        """
        Initialize the predictor with settings.
        
        Parameters explained:
        - symbol: Stock ticker (like "AAPL" for Apple)
        - n_estimators=300: Number of "expert trees" in our forest
        - min_samples_split=50: Minimum data points needed to make a decision
        - random_state=1: Ensures we get the same results each time (for testing)
        """
        self.symbol = symbol
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,      # 300 decision trees
            min_samples_split=min_samples_split,  # Need 50+ samples to split
            random_state=random_state,      # For reproducible results
            max_depth=10,                   # Limit tree depth to prevent overfitting
            min_samples_leaf=5              # Each leaf needs at least 5 samples
        )
        self.data = None           # Will store our stock data
        self.predictors = []       # Will store our feature names
        
    def fetch_data(self, period="max"):
        """
        Download stock data and prepare basic features.
        
        WHAT IS STOCK DATA?
        - Open: Price when market opened
        - High: Highest price during the day
        - Low: Lowest price during the day
        - Close: Price when market closed
        - Volume: Number of shares traded
        - Dividends: Cash payments to shareholders
        - Stock Splits: When 1 share becomes 2 shares (price halves)
        """
        print(f"Downloading data for {self.symbol}...")
        
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {self.symbol}")
        
        # DIVIDEND AND SPLIT FEATURES
        # These events can affect stock prices, so we track them
        
        # Fill missing values with 0 (no dividend/split that day)
        data["Dividends"] = data["Dividends"].fillna(0)
        data["Stock Splits"] = data["Stock Splits"].fillna(0)
        
        # Create binary indicators (1 if event happened, 0 if not)
        data["Has_Dividend"] = (data["Dividends"] > 0).astype(int)
        data["Has_Split"] = (data["Stock Splits"] > 0).astype(int)
        
        # Calculate dividend yield (annual dividend as % of price)
        data["Dividend_Yield"] = (data["Dividends"] / data["Close"]) * 4  # Assume quarterly
        
        # Track cumulative dividends over past year (252 trading days)
        data["Cumulative_Dividends"] = data["Dividends"].rolling(window=252, min_periods=1).sum()
        
        # Track how many days since last dividend/split (might indicate patterns)
        data["Days_Since_Split"] = self._days_since_event(data["Has_Split"])
        data["Days_Since_Dividend"] = self._days_since_event(data["Has_Dividend"])
        
        # CREATE THE TARGET VARIABLE (what we want to predict)
        # "Tomorrow" = tomorrow's closing price
        # "Target" = 1 if tomorrow's price > today's price, 0 otherwise
        data["Tomorrow"] = data["Close"].shift(-1)  # Shift prices up by 1 day
        data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
        
        # Remove early data to ensure we have enough history for calculations
        # We need at least 252 days (1 trading year) for some features
        start_idx = max(252, len(data) - len(data.loc["1990-01-01":]) if "1990-01-01" in data.index else 252)
        data = data.iloc[start_idx:].copy()
        
        self.data = data
        print(f"Downloaded {len(data)} days of data")
        return data
    
    def _days_since_event(self, event_series):
        """
        Calculate how many days have passed since the last dividend or split.
        
        WHY THIS MATTERS:
        Companies often pay dividends on regular schedules (quarterly, annually).
        The time since last dividend might help predict when the next one is coming.
        """
        days_since = []
        last_event_idx = -999  # Start with a large negative number
        
        for i, has_event in enumerate(event_series):
            if has_event:
                last_event_idx = i  # Remember when the event happened
            
            # Calculate days since last event (or 999 if no event yet)
            days_since.append(i - last_event_idx if last_event_idx != -999 else 999)
        
        return pd.Series(days_since, index=event_series.index)
    
    def create_features(self):
        """
        Create technical indicators and features for the machine learning model.
        
        WHAT ARE FEATURES?
        Features are the "inputs" to our AI model. Just like a human trader
        might look at charts, trends, and patterns, our AI looks at mathematical
        features that capture these same concepts.
        
        TYPES OF FEATURES WE CREATE:
        1. Price-based: How much did price change?
        2. Volume-based: How much trading activity?
        3. Technical indicators: RSI, MACD, Bollinger Bands
        4. Trend indicators: Moving averages, trend strength
        5. Pattern recognition: Consecutive up/down days
        """
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")
        
        print("Creating features (technical indicators)...")
        data = self.data.copy()
        
        # =================================================================
        # PRICE-BASED FEATURES
        # =================================================================
        # These measure how much the price changed over different time periods
        
        data["Price_Change"] = data["Close"].pct_change()        # % change from yesterday
        data["Price_Change_2d"] = data["Close"].pct_change(periods=2)  # % change from 2 days ago
        data["Price_Change_5d"] = data["Close"].pct_change(periods=5)  # % change from 5 days ago
        
        # =================================================================
        # VOLATILITY FEATURES
        # =================================================================
        # Volatility measures how much the price "jumps around"
        # High volatility = price changes a lot, Low volatility = price is stable
        
        data["Volatility_5d"] = data["Price_Change"].rolling(5).std()   # 5-day volatility
        data["Volatility_20d"] = data["Price_Change"].rolling(20).std() # 20-day volatility
        
        # =================================================================
        # VOLUME FEATURES
        # =================================================================
        # Volume is how many shares were traded
        # High volume often indicates strong interest/news
        
        data["Volume_Change"] = data["Volume"].pct_change()  # % change in volume
        
        # Volume ratios: Is today's volume higher than average?
        data["Volume_Ratio_5d"] = data["Volume"] / data["Volume"].rolling(5).mean()
        data["Volume_Ratio_20d"] = data["Volume"] / data["Volume"].rolling(20).mean()
        
        # =================================================================
        # PRICE POSITION FEATURES
        # =================================================================
        # These help understand where the closing price is relative to the day's range
        
        data["High_Low_Ratio"] = data["High"] / data["Low"]      # How wide was the day's range?
        data["Close_to_High"] = data["Close"] / data["High"]     # Did it close near the high?
        data["Close_to_Low"] = data["Close"] / data["Low"]       # Did it close near the low?
        
        # =================================================================
        # MOVING AVERAGES AND TREND INDICATORS
        # =================================================================
        # Moving averages smooth out price data to show trends
        # If price > moving average, it might be in an uptrend
        
        horizons = [5, 10, 20, 50, 100, 200]  # Different time periods
        
        for horizon in horizons:
            # Calculate moving average (average price over N days)
            ma_col = f"MA_{horizon}"
            data[ma_col] = data["Close"].rolling(horizon).mean()
            
            # Price to moving average ratio
            # > 1.0 means price is above average (bullish)
            # < 1.0 means price is below average (bearish)
            ratio_col = f"Close_MA_Ratio_{horizon}"
            data[ratio_col] = data["Close"] / data[ma_col]
            
            # Trend strength: How many up days in the last N days?
            trend_col = f"Trend_{horizon}"
            data[trend_col] = data.shift(1).rolling(horizon).sum()["Target"]
            
            # Volume moving average
            vol_ma_col = f"Volume_MA_{horizon}"
            data[vol_ma_col] = data["Volume"].rolling(horizon).mean()
        
        # =================================================================
        # TECHNICAL INDICATORS
        # =================================================================
        # These are popular indicators that traders use
        
        # RSI (Relative Strength Index): Measures if stock is overbought/oversold
        # Values: 0-100. >70 = overbought, <30 = oversold
        data["RSI"] = self._calculate_rsi(data["Close"])
        
        # MACD: Shows relationship between two moving averages
        # Used to identify trend changes
        data["MACD"], data["MACD_Signal"] = self._calculate_macd(data["Close"])
        
        # Bollinger Bands: Price channel based on standard deviation
        # If price hits upper band, might be overbought
        data["BB_Upper"], data["BB_Lower"], data["BB_Position"] = self._calculate_bollinger_bands(data["Close"])
        
        # =================================================================
        # GAP ANALYSIS
        # =================================================================
        # A gap occurs when today's opening price is different from yesterday's close
        # Gaps often indicate news or events
        
        data["Gap"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)
        data["Gap_Size"] = abs(data["Gap"])  # Size of gap (regardless of direction)
        
        # =================================================================
        # CONSECUTIVE PATTERNS
        # =================================================================
        # How many days in a row has the stock gone up or down?
        # Long streaks might indicate trend continuation or reversal
        
        data["Consecutive_Up"] = self._consecutive_days(data["Target"])
        data["Consecutive_Down"] = self._consecutive_days(1 - data["Target"])
        
        # Clean up data (remove rows with missing values)
        self.data = data.dropna()
        
        # Define which columns are our features (inputs to the AI model)
        # We exclude the target and intermediate calculation columns
        exclude_cols = ["Tomorrow", "Target", "Adj Close"]
        self.predictors = [col for col in self.data.columns if col not in exclude_cols]
        
        print(f"Created {len(self.predictors)} features")
        return self.data
    
    def _calculate_rsi(self, prices, window=14):
        """
        Calculate RSI (Relative Strength Index).
        
        RSI EXPLAINED:
        - Measures momentum: is the stock rising or falling?
        - Scale: 0 to 100
        - Above 70: Might be overbought (price too high)
        - Below 30: Might be oversold (price too low)
        - Used to identify potential reversal points
        """
        delta = prices.diff()  # Daily price changes
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Average gains
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Average losses
        rs = gain / loss  # Relative strength
        rsi = 100 - (100 / (1 + rs))  # Convert to 0-100 scale
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD EXPLAINED:
        - Compares two moving averages to identify trend changes
        - MACD line: Fast moving average - Slow moving average
        - Signal line: Smoothed version of MACD line
        - When MACD crosses above signal line: Bullish signal
        - When MACD crosses below signal line: Bearish signal
        """
        exp1 = prices.ewm(span=fast).mean()    # Fast exponential moving average
        exp2 = prices.ewm(span=slow).mean()    # Slow exponential moving average
        macd = exp1 - exp2                     # MACD line
        signal_line = macd.ewm(span=signal).mean()  # Signal line
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """
        Calculate Bollinger Bands.
        
        BOLLINGER BANDS EXPLAINED:
        - Three lines: Upper, Middle (moving average), Lower
        - Upper/Lower bands are 2 standard deviations from middle
        - When price hits upper band: Might be overbought
        - When price hits lower band: Might be oversold
        - BB_Position shows where current price is within the bands
        """
        rolling_mean = prices.rolling(window).mean()  # Middle line
        rolling_std = prices.rolling(window).std()    # Standard deviation
        
        upper_band = rolling_mean + (rolling_std * num_std)  # Upper band
        lower_band = rolling_mean - (rolling_std * num_std)  # Lower band
        
        # Position: 0 = at lower band, 1 = at upper band, 0.5 = in middle
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        
        return upper_band, lower_band, bb_position
    
    def _consecutive_days(self, series):
        """
        Count consecutive days of the same outcome.
        
        CONSECUTIVE PATTERNS EXPLAINED:
        - Tracks "streaks" of up or down days
        - Long streaks might indicate strong trends
        - Very long streaks might indicate trend exhaustion
        - Helps identify potential reversal points
        """
        consecutive = []
        current_streak = 0
        
        for i, val in enumerate(series):
            if i == 0:
                current_streak = 1
            elif val == series.iloc[i-1]:  # Same as previous day
                current_streak += 1
            else:                          # Different from previous day
                current_streak = 1
            consecutive.append(current_streak)
        
        return pd.Series(consecutive, index=series.index)
    
    def predict(self, train, test, predictors, model):
        """
        Train the model and make predictions.
        
        MACHINE LEARNING PROCESS:
        1. Train: Show the AI historical data and correct answers
        2. Predict: Ask the AI to guess on new data
        3. Threshold: Convert probability to binary decision (UP/DOWN)
        
        PROBABILITY vs BINARY PREDICTION:
        - Model outputs probability (0.0 to 1.0)
        - We use threshold (0.55) to convert to binary decision
        - Above 0.55 = Predict UP, Below 0.55 = Predict DOWN
        """
        # Train the model on historical data
        model.fit(train[predictors], train["Target"])
        
        # Get probabilities for test data
        preds = model.predict_proba(test[predictors])[:, 1]  # Probability of UP
        
        # Convert probabilities to binary predictions using threshold
        threshold = 0.55  # 55% confidence required to predict UP
        preds_binary = (preds >= threshold).astype(int)
        
        # Combine actual results with predictions for evaluation
        preds_series = pd.Series(preds_binary, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], preds_series], axis=1)
        return combined
    
    def backtest(self, start=500, step=100):
        """
        Test the model on historical data to see how it would have performed.
        
        BACKTESTING EXPLAINED:
        - Simulates what would have happened if we used our model in the past
        - Uses data from day 1 to day 500 to train
        - Tests on days 501-600
        - Then uses days 1-600 to train, tests on days 601-700
        - And so on...
        - This shows us how accurate our model would have been
        
        WHY BACKTEST?
        - Gives us confidence in our model
        - Shows historical accuracy
        - Helps identify if model is overfitting (too specific to training data)
        """
        if self.data is None or len(self.predictors) == 0:
            raise ValueError("Data and features not prepared. Call fetch_data() and create_features() first.")
        
        print("Running backtest (testing model on historical data)...")
        all_predictions = []
        data = self.data
        
        # Walk forward through time, always training on past data
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()      # All data up to day i
            test = data.iloc[i:(i + step)].copy()  # Next 'step' days
            
            if len(test) == 0:
                break
                
            predictions = self.predict(train, test, self.predictors, self.model)
            all_predictions.append(predictions)
        
        return pd.concat(all_predictions)
    
    def evaluate_model(self, predictions):
        """
        Evaluate how well our model performed.
        
        EVALUATION METRICS:
        - Precision: When model predicts UP, how often is it correct?
        - If precision = 0.60, model is right 60% of the time when it says UP
        - Higher precision = more reliable UP predictions
        - We focus on precision because we want to be confident when we predict UP
        """
        # Calculate precision (accuracy when predicting UP)
        precision = precision_score(predictions["Target"], predictions["Predictions"])
        
        print(f"\n--- Model Performance for {self.symbol} ---")
        print(f"Precision Score: {precision:.4f}")
        print(f"This means: When model predicts UP, it's correct {precision:.1%} of the time")
        
        print(f"\nPrediction Distribution:")
        print(predictions["Predictions"].value_counts())
        print(f"Actual Distribution:")
        print(predictions["Target"].value_counts())
        print(f"Total Predictions: {len(predictions)}")
        
        return precision
    
    def get_tomorrow_prediction(self):
        """
        Get prediction for tomorrow with confidence level.
        
        TOMORROW'S PREDICTION:
        - Uses ALL available historical data to train
        - Makes prediction for the next trading day
        - Returns direction (UP/DOWN) and confidence level
        - Higher confidence = model is more certain
        """
        if self.data is None or len(self.predictors) == 0:
            raise ValueError("Data and features not prepared. Call fetch_data() and create_features() first.")
        
        # Use all available data except the last day (which has no target)
        train_data = self.data.iloc[:-1].copy()
        latest_data = self.data.iloc[-1:].copy()
        
        # Train model on all available data
        self.model.fit(train_data[self.predictors], train_data["Target"])
        
        # Get prediction probability for tomorrow
        confidence = self.model.predict_proba(latest_data[self.predictors])[:, 1][0]
        
        # Use higher threshold for tomorrow's prediction (be more conservative)
        threshold = 0.65  # Need 65% confidence to predict UP
        direction = "UP" if confidence >= threshold else "DOWN"
        
        return direction, confidence
    
    def get_feature_importance(self):
        """
        Show which features are most important for predictions.
        
        FEATURE IMPORTANCE:
        - Shows which indicators the model relies on most
        - Higher importance = more influence on predictions
        - Helps us understand what drives the model's decisions
        - Can reveal market patterns and relationships
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model needs to be trained first")
            return None
        
        # Create DataFrame with features and their importance scores
        importance_df = pd.DataFrame({
            'feature': self.predictors,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

# =================================================================
# MAIN ANALYSIS FUNCTION
# =================================================================

def analyze_stock(symbol, period="10y"):
    """
    Analyze a single stock and generate predictions.
    
    COMPLETE ANALYSIS PROCESS:
    1. Download stock data (with fallback periods if not enough data)
    2. Create technical features
    3. Train and test model on historical data
    4. Evaluate performance
    5. Make prediction for tomorrow
    6. Show most important features
    
    FALLBACK SYSTEM:
    If stock doesn't have enough data for the requested period,
    we automatically try shorter periods: 10y → 5y → 3y → 2y → 1y
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING {symbol}")
    print(f"{'='*60}")
    
    # List of periods to try in order (from longest to shortest)
    periods_to_try = ["10y", "5y", "3y", "2y", "1y"]
    
    # If a specific period was requested, start with that
    if period not in periods_to_try:
        periods_to_try.insert(0, period)
    else:
        # Start from the requested period onward
        start_index = periods_to_try.index(period)
        periods_to_try = periods_to_try[start_index:]
    
    predictor = None
    
    # Try each period until we get enough data
    for attempt_period in periods_to_try:
        try:
            print(f"Trying to fetch {attempt_period} of data for {symbol}...")
            
            # Initialize predictor
            predictor = StockPredictor(symbol)
            
            # Step 1: Download and prepare data
            data = predictor.fetch_data(period=attempt_period)
            
            # Check if we have enough data for meaningful analysis
            # We need at least 500 days for backtesting (our default start point)
            if len(data) < 500:
                print(f"Not enough data with {attempt_period} period ({len(data)} days). Trying shorter period...")
                continue
            
            print(f"Successfully loaded {len(data)} days of data using {attempt_period} period")
            break
            
        except Exception as e:
            print(f"Failed to get data for {attempt_period} period: {str(e)}")
            if attempt_period == periods_to_try[-1]:  # Last attempt
                print(f"All periods failed for {symbol}")
                return None, None, None, None, None
            continue
    
    # If we couldn't get data with any period
    if predictor is None or predictor.data is None:
        print(f"Could not fetch sufficient data for {symbol} with any time period")
        return None, None, None, None, None
    
    try:
        # Step 2: Create features (technical indicators)
        predictor.create_features()
        
        # Check again if we have enough data after feature creation and cleanup
        if len(predictor.data) < 500:
            print(f"Not enough data after feature creation ({len(predictor.data)} days). Need at least 500 days.")
            return None, None, None, None, None
        
        # Step 3: Run backtest (test on historical data)
        predictions = predictor.backtest()
        
        # Step 4: Evaluate performance
        precision = predictor.evaluate_model(predictions)
        
        # Step 5: Get tomorrow's prediction
        direction, confidence = predictor.get_tomorrow_prediction()
        
        # Step 6: Show important features
        importance = predictor.get_feature_importance()
        if importance is not None:
            print(f"\nTop 10 Most Important Features for {symbol}:")
            print(importance.head(10))
        
        # Step 7: Display tomorrow's prediction
        print(f"\n--- TOMORROW'S PREDICTION ---")
        print(f"Direction: {direction}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Historical Precision: {precision:.1%}")
        
        return predictor, predictions, precision, direction, confidence
        
    except Exception as e:
        print(f"Error during analysis of {symbol}: {str(e)}")
        return None, None, None, None, None

# Example: Analyze multiple individual stocks
if __name__ == "__main__":
    # Test with popular individual stocks
    stocks = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN"]
    
    results = {}
    
    for stock in stocks:
        predictor, predictions, precision, direction, confidence = analyze_stock(stock)
        if predictor is not None:
            results[stock] = {
                'predictor': predictor,
                'predictions': predictions,
                'precision': precision,
                'direction': direction,
                'confidence': confidence
            }
    
    # Summary of results
    print("\n" + "="*60)
    print("TOMORROW'S PREDICTIONS SUMMARY")
    print("="*60)
    for stock, result in results.items():
        direction = result['direction']
        confidence = result['confidence']
        precision = result['precision']
        
        print(f"{stock:5} | Tomorrow: {direction:4} | Confidence: {confidence:5.1%} | Model Precision: {precision:.1%}")
    
    print("\n" + "="*60)

"""
============================================================
TOMORROW'S PREDICTIONS SUMMARY
============================================================
STOCK | Tomorrow: UP or DOWN   | Confidence: 72.3% (The Likelyhood that it will go up) | Model Precision: 100.0% (Based on historical data, how accurate was it)
============================================================
"""