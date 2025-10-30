import os
import sys
import numpy as np
import pandas as pd
from binance.client import Client
import concurrent.futures
from binance.exceptions import BinanceAPIException
import time
import talib
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Import modules
import numpy as np
import pandas as pd
from binance.client import Client
import concurrent.futures
from binance.exceptions import BinanceAPIException
import time
import talib
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Import TensorFlow after setting environment variables
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Further suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

class Trader:
    def __init__(self, file):
        self.connect(file)
        self.model_path = 'lstm_model.keras'
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Load or create the LSTM model
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print("Loaded existing LSTM model.")
        else:
            self.model = self.build_model()
            print("Created new LSTM model.")
    
    def connect(self, file):
        with open(file) as f:
            lines = [line.rstrip('\n') for line in f]
        key, secret = lines[0], lines[1]
        self.client = Client(key, secret)
    
    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(10, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def get_klines(self, symbol, interval, limit=100):
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            # Extract OHLC data
            open_prices = [float(entry[1]) for entry in klines]
            high_prices = [float(entry[2]) for entry in klines]
            low_prices = [float(entry[3]) for entry in klines]
            close_prices = [float(entry[4]) for entry in klines]
            volumes = [float(entry[5]) for entry in klines]
            
            return open_prices, high_prices, low_prices, close_prices, volumes
        except BinanceAPIException as e:
            print(f"API Error for {symbol} {interval}: {e}")
            return None, None, None, None, None
    
    def preprocess_data(self, data):
        # Replace NaN values with 0
        data = np.nan_to_num(data)
        # Replace 0 values with a very small number to avoid division by zero
        data[data == 0] = 1e-10
        return data
    
    def calculate_ht_sine(self, close_prices):
        # Preprocess data
        close_prices = self.preprocess_data(np.array(close_prices))
        
        # Calculate HT_SINE using TA-Lib
        sine, leadsine = talib.HT_SINE(close_prices)
        
        # Handle NaN values in the result
        sine = np.nan_to_num(sine)
        leadsine = np.nan_to_num(leadsine)
        
        return sine, leadsine
    
    def enhanced_fft_forecast(self, close_prices, forecast_period=5):
        """Enhanced FFT forecast for short-term predictions"""
        # Preprocess data
        close_prices = self.preprocess_data(np.array(close_prices))
        
        # Apply FFT
        fft_values = fft(close_prices)
        
        # Create a new array with zeros for the forecast period
        forecast = np.zeros(forecast_period, dtype=complex)
        
        # Combine the original FFT values with the forecast zeros
        fft_combined = np.concatenate([fft_values, forecast])
        
        # Apply inverse FFT to get the forecasted values
        ifft_values = ifft(fft_combined)
        
        # Extract the real part of the inverse FFT (the forecasted values)
        forecasted_values = np.real(ifft_values[-forecast_period:])
        
        # Apply smoothing to the forecasted values for better accuracy
        smoothed_forecast = np.convolve(forecasted_values, np.ones(3)/3, mode='valid')
        
        # If smoothing reduces the array size, pad with the last value
        if len(smoothed_forecast) < forecast_period:
            padding = np.full(forecast_period - len(smoothed_forecast), smoothed_forecast[-1])
            smoothed_forecast = np.concatenate([smoothed_forecast, padding])
        
        return smoothed_forecast
    
    def prepare_lstm_data(self, data, time_step=10):
        # Preprocess data
        data = self.preprocess_data(np.array(data))
        
        # Normalize data
        data = data.reshape(-1, 1)
        data = self.scaler.fit_transform(data)
        
        # Create dataset
        X, y = [], []
        for i in range(len(data)-time_step-1):
            X.append(data[i:(i+time_step), 0])
            y.append(data[i+time_step, 0])
        
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, historical_data, time_step=10, epochs=10):
        # Prepare data for LSTM
        X, y = self.prepare_lstm_data(historical_data, time_step)
        
        # Reshape input to be [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Train the model
        self.model.fit(X, y, epochs=epochs, batch_size=1, verbose=0)
        
        # Save the model
        self.model.save(self.model_path)
        print("LSTM model trained and saved.")
    
    def lstm_predict(self, data, time_step=10):
        # Preprocess data
        data = self.preprocess_data(np.array(data))
        
        # Normalize data
        data = data.reshape(-1, 1)
        data = self.scaler.transform(data)
        
        # Get the last sequence for prediction
        last_sequence = data[-time_step:].reshape(1, time_step, 1)
        
        # Make prediction
        predicted_price = self.model.predict(last_sequence, verbose=0)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        
        return predicted_price[0][0]
    
    def volume_analysis(self, volumes, close_prices):
        """Analyze volume for momentum confirmation"""
        volumes = np.array(volumes)
        close_prices = np.array(close_prices)
        
        # Calculate volume moving average
        volume_ma = talib.SMA(volumes, timeperiod=10)
        
        # Calculate volume rate of change
        volume_roc = talib.ROC(volumes, timeperiod=5)
        
        # Calculate price rate of change
        price_roc = talib.ROC(close_prices, timeperiod=5)
        
        # Determine volume momentum
        current_volume = volumes[-1]
        avg_volume = volume_ma[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume confirmation signals
        volume_signals = []
        
        # High volume with price increase = bullish
        if volume_ratio > 1.2 and price_roc[-1] > 0:
            volume_signals.append("Bullish Volume Confirmation")
        
        # High volume with price decrease = bearish
        if volume_ratio > 1.2 and price_roc[-1] < 0:
            volume_signals.append("Bearish Volume Confirmation")
        
        # Low volume = weak momentum
        if volume_ratio < 0.8:
            volume_signals.append("Weak Volume - Low Momentum")
        
        return {
            'volume_ratio': volume_ratio,
            'volume_roc': volume_roc[-1],
            'price_roc': price_roc[-1],
            'signals': volume_signals
        }
    
    def momentum_indicators(self, close_prices, high_prices, low_prices):
        """Calculate momentum indicators"""
        close_prices = np.array(close_prices)
        high_prices = np.array(high_prices)
        low_prices = np.array(low_prices)
        
        # RSI
        rsi = talib.RSI(close_prices, timeperiod=14)
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, 
                                   fastk_period=5, slowk_period=3, slowd_period=3)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # CCI
        cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Determine momentum signals
        momentum_signals = []
        
        # RSI signals
        if rsi[-1] > 70:
            momentum_signals.append("RSI Overbought")
        elif rsi[-1] < 30:
            momentum_signals.append("RSI Oversold")
        
        # Stochastic signals
        if slowk[-1] > 80 and slowd[-1] > 80:
            momentum_signals.append("Stochastic Overbought")
        elif slowk[-1] < 20 and slowd[-1] < 20:
            momentum_signals.append("Stochastic Oversold")
        
        # MACD signals
        if macd[-1] > macdsignal[-1] and macdhist[-1] > 0:
            momentum_signals.append("MACD Bullish")
        elif macd[-1] < macdsignal[-1] and macdhist[-1] < 0:
            momentum_signals.append("MACD Bearish")
        
        # CCI signals
        if cci[-1] > 100:
            momentum_signals.append("CCI Overbought")
        elif cci[-1] < -100:
            momentum_signals.append("CCI Oversold")
        
        return {
            'rsi': rsi[-1],
            'slowk': slowk[-1],
            'slowd': slowd[-1],
            'macd': macd[-1],
            'macdsignal': macdsignal[-1],
            'macdhist': macdhist[-1],
            'cci': cci[-1],
            'signals': momentum_signals
        }
    
    def calculate_profit_targets(self, current_close, direction):
        """Calculate 2% profit targets with min and max thresholds"""
        profit_percentage = 0.02  # 2% profit target
        
        if direction == "Upward":
            target_price = current_close * (1 + profit_percentage)
            min_threshold = current_close * (1 + profit_percentage * 0.8)  # 80% of target
            max_threshold = current_close * (1 + profit_percentage * 1.2)  # 120% of target
        else:  # Downward
            target_price = current_close * (1 - profit_percentage)
            min_threshold = current_close * (1 - profit_percentage * 1.2)  # 120% of target (lower)
            max_threshold = current_close * (1 - profit_percentage * 0.8)  # 80% of target (higher)
        
        return {
            'target_price': target_price,
            'min_threshold': min_threshold,
            'max_threshold': max_threshold,
            'profit_percentage': profit_percentage
        }
    
    def run_analysis(self):
        # Only use 1m, 3m, and 5m timeframes for fast targets
        intervals = ['1m', '3m', '5m']
        symbol = 'BTCUSDC'
        
        # Get historical data for training the LSTM model
        print("Fetching historical data for LSTM model training...")
        _, _, _, historical_close, _ = self.get_klines(symbol, '5m', limit=500)
        
        if historical_close is not None:
            # Train the LSTM model
            self.train_lstm_model(historical_close, time_step=10, epochs=10)
        else:
            print("Failed to fetch historical data. LSTM model will not be trained.")
        
        # Store results for each timeframe
        timeframe_results = {}
        
        # Sequential processing to maintain order
        for interval in intervals:
            print(f"Processing {interval} timeframe...")
            open_prices, high_prices, low_prices, close_prices, volumes = self.get_klines(symbol, interval, 100)
            
            if open_prices is not None and high_prices is not None and low_prices is not None and close_prices is not None and volumes is not None:
                # Convert to numpy arrays for efficient operations
                high_array = np.array(high_prices)
                low_array = np.array(low_prices)
                close_array = np.array(close_prices)
                volume_array = np.array(volumes)
                
                # Find indices of lowest low and highest high using argmin and argmax
                argmin_low_index = np.argmin(low_array)
                argmax_high_index = np.argmax(high_array)
                
                # Get the actual lowest low and highest high values
                lowest_low = low_array[argmin_low_index]
                highest_high = high_array[argmax_high_index]
                
                # Get current close price (most recent)
                current_close = close_array[-1]
                
                # Determine which is more recent
                if argmin_low_index > argmax_high_index:
                    pattern = "DIP"
                    pattern_index = argmin_low_index
                    pattern_price = lowest_low
                    cycle_direction = "Up cycle"  # DIP is most recent, suggesting an upcoming up cycle
                else:
                    pattern = "TOP"
                    pattern_index = argmax_high_index
                    pattern_price = highest_high
                    cycle_direction = "Down cycle"  # TOP is most recent, suggesting an upcoming down cycle
                
                # Calculate total range between min and max
                total_range = highest_high - lowest_low
                
                # Calculate middle threshold
                middle_threshold = lowest_low + (total_range / 2)
                
                # Determine if current close is above or below middle threshold
                if current_close > middle_threshold:
                    middle_position = "above"
                else:
                    middle_position = "below"
                
                # Calculate differences
                diff_to_min = current_close - lowest_low
                diff_to_max = highest_high - current_close
                
                # Calculate symmetrical percentages based on total range
                perc_to_min = (diff_to_min / total_range) * 100
                perc_to_max = (diff_to_max / total_range) * 100
                
                # Calculate HT_SINE
                sine, leadsine = self.calculate_ht_sine(close_prices)
                
                # Get the most recent sine and leadsine values
                current_sine = sine[-1]
                current_leadsine = leadsine[-1]
                
                # Enhanced FFT forecast for short-term predictions
                forecasted_prices = self.enhanced_fft_forecast(close_prices, forecast_period=5)
                
                # Predict using LSTM
                lstm_prediction = self.lstm_predict(close_prices)
                
                # Volume analysis
                volume_results = self.volume_analysis(volumes, close_prices)
                
                # Momentum indicators
                momentum_results = self.momentum_indicators(high_prices, low_prices, close_prices)
                
                # Determine reversal based on all indicators
                reversal_signals = []
                
                # HT_SINE signal
                if current_sine < current_leadsine:
                    reversal_signals.append("Upward")
                elif current_sine > current_leadsine:
                    reversal_signals.append("Downward")
                
                # FFT signal
                if forecasted_prices[-1] > current_close:
                    reversal_signals.append("Upward")
                elif forecasted_prices[-1] < current_close:
                    reversal_signals.append("Downward")
                
                # LSTM signal
                if lstm_prediction > current_close:
                    reversal_signals.append("Upward")
                elif lstm_prediction < current_close:
                    reversal_signals.append("Downward")
                
                # Volume signals
                if "Bullish Volume Confirmation" in volume_results['signals']:
                    reversal_signals.append("Upward")
                elif "Bearish Volume Confirmation" in volume_results['signals']:
                    reversal_signals.append("Downward")
                
                # Momentum signals
                if "RSI Oversold" in momentum_results['signals'] or "Stochastic Oversold" in momentum_results['signals']:
                    reversal_signals.append("Upward")
                elif "RSI Overbought" in momentum_results['signals'] or "Stochastic Overbought" in momentum_results['signals']:
                    reversal_signals.append("Downward")
                
                if "MACD Bullish" in momentum_results['signals']:
                    reversal_signals.append("Upward")
                elif "MACD Bearish" in momentum_results['signals']:
                    reversal_signals.append("Downward")
                
                # Determine the dominant reversal signal
                up_signals = reversal_signals.count("Upward")
                down_signals = reversal_signals.count("Downward")
                
                if up_signals > down_signals:
                    reversal_signal = "Upward reversal expected"
                    direction = "Upward"
                    # Combine FFT and LSTM predictions for target price
                    target_price = (forecasted_prices[-1] + lstm_prediction) / 2
                elif down_signals > up_signals:
                    reversal_signal = "Downward reversal expected"
                    direction = "Downward"
                    # Combine FFT and LSTM predictions for target price
                    target_price = (forecasted_prices[-1] + lstm_prediction) / 2
                else:
                    reversal_signal = "No clear reversal signal"
                    direction = "Neutral"
                    target_price = current_close
                
                # Calculate 2% profit targets with min and max thresholds
                profit_targets = self.calculate_profit_targets(current_close, direction)
                
                # Format all floats to 25 decimal places
                format_float = lambda x: "{0:.25f}".format(x)
                
                # Store results for MTF analysis
                timeframe_results[interval] = {
                    'current_close': current_close,
                    'lowest_low': lowest_low,
                    'highest_high': highest_high,
                    'total_range': total_range,
                    'pattern': pattern,
                    'cycle_direction': cycle_direction,
                    'middle_threshold': middle_threshold,
                    'middle_position': middle_position,
                    'perc_to_min': perc_to_min,
                    'perc_to_max': perc_to_max,
                    'current_sine': current_sine,
                    'current_leadsine': current_leadsine,
                    'reversal_signal': reversal_signal,
                    'target_price': target_price,
                    'lstm_prediction': lstm_prediction,
                    'forecasted_prices': forecasted_prices,
                    'volume_results': volume_results,
                    'momentum_results': momentum_results,
                    'profit_targets': profit_targets,
                    'up_signals': up_signals,
                    'down_signals': down_signals,
                    'direction': direction
                }
                
                print(f"\nTimeframe: {interval}")
                print(f"  Current Close: {format_float(current_close)}")
                print(f"  Lowest Low: {format_float(lowest_low)} at index {argmin_low_index}")
                print(f"  Highest High: {format_float(highest_high)} at index {argmax_high_index}")
                print(f"  Total Range: {format_float(total_range)}")
                print(f"  Middle Threshold: {format_float(middle_threshold)}")
                print(f"  Current Close is {middle_position} the middle threshold")
                print(f"  Pattern Confirmed: {pattern} at index {pattern_index} (price: {format_float(pattern_price)})")
                print(f"  Cycle Direction: {cycle_direction}")
                print(f"  Difference to Min: {format_float(diff_to_min)}")
                print(f"  Difference to Max: {format_float(diff_to_max)}")
                print(f"  Percentage to Min: {format_float(perc_to_min)}%")
                print(f"  Percentage to Max: {format_float(perc_to_max)}%")
                print(f"  HT_SINE: {format_float(current_sine)}")
                print(f"  HT_LEADSINE: {format_float(current_leadsine)}")
                print(f"  LSTM Prediction: {format_float(lstm_prediction)}")
                print(f"  FFT Forecast: {format_float(forecasted_prices[-1])}")
                print(f"  Volume Ratio: {format_float(volume_results['volume_ratio'])}")
                print(f"  Volume ROC: {format_float(volume_results['volume_roc'])}")
                print(f"  Price ROC: {format_float(volume_results['price_roc'])}")
                print(f"  Volume Signals: {', '.join(volume_results['signals']) if volume_results['signals'] else 'None'}")
                print(f"  RSI: {format_float(momentum_results['rsi'])}")
                print(f"  Stochastic: {format_float(momentum_results['slowk'])}/{format_float(momentum_results['slowd'])}")
                print(f"  MACD: {format_float(momentum_results['macd'])}/{format_float(momentum_results['macdsignal'])}")
                print(f"  CCI: {format_float(momentum_results['cci'])}")
                print(f"  Momentum Signals: {', '.join(momentum_results['signals']) if momentum_results['signals'] else 'None'}")
                print(f"  Reversal Signal: {reversal_signal} (Up: {up_signals}, Down: {down_signals})")
                print(f"  Direction: {direction}")
                print(f"  2% Profit Target: {format_float(profit_targets['target_price'])}")
                print(f"  Min Threshold: {format_float(profit_targets['min_threshold'])}")
                print(f"  Max Threshold: {format_float(profit_targets['max_threshold'])}")
                print(f"  Combined Target Price: {format_float(target_price)}\n")
            else:
                print(f"Failed to retrieve data for {interval} timeframe\n")
            
            # Add delay to avoid rate limits
            time.sleep(0.5)
        
        # Perform MTF analysis
        self.perform_mtf_analysis(timeframe_results)
    
    def perform_mtf_analysis(self, timeframe_results):
        print("\n" + "="*50)
        print("MULTI-TIMEFRAME (MTF) ANALYSIS - FAST TARGETS")
        print("="*50)
        
        # Count up and down cycles across all timeframes
        up_cycles = sum(1 for result in timeframe_results.values() if result['cycle_direction'] == "Up cycle")
        down_cycles = sum(1 for result in timeframe_results.values() if result['cycle_direction'] == "Down cycle")
        
        # Determine overall MTF cycle direction
        if up_cycles > down_cycles:
            overall_mtf_cycle = "POWER MTF UP CYCLE"
        elif down_cycles > up_cycles:
            overall_mtf_cycle = "POWER MTF DOWN CYCLE"
        else:
            overall_mtf_cycle = "BALANCED MTF CYCLE"
        
        print(f"\nOverall MTF Cycle Direction: {overall_mtf_cycle}")
        print(f"Up Cycles: {up_cycles}/{len(timeframe_results)}")
        print(f"Down Cycles: {down_cycles}/{len(timeframe_results)}")
        
        # Calculate compounded MTF sine wave
        print("\nMTF Compounded Sine Wave Analysis:")
        
        # Get the most recent close prices across all timeframes
        recent_closes = [result['current_close'] for result in timeframe_results.values()]
        
        # Normalize the close prices to [0, 1] range
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_closes = scaler.fit_transform(np.array(recent_closes).reshape(-1, 1))
        
        # Calculate a compounded sine wave using the normalized values
        compounded_sine = np.sin(np.pi * normalized_closes)
        
        # Calculate the MTF power score
        mtf_power_score = np.mean(compounded_sine)
        
        # Determine if the MTF power is positive or negative
        if mtf_power_score > 0:
            mtf_power_direction = "Positive (Bullish)"
        else:
            mtf_power_direction = "Negative (Bearish)"
        
        # Format the MTF power score to 25 decimal places
        format_float = lambda x: "{0:.25f}".format(x)
        
        print(f"MTF Power Score: {format_float(mtf_power_score)}")
        print(f"MTF Power Direction: {mtf_power_direction}")
        
        # Calculate MTF target price based on the compounded sine wave
        # Get the target prices from each timeframe
        target_prices = [result['target_price'] for result in timeframe_results.values()]
        
        # Calculate a weighted average of target prices, giving more weight to longer timeframes
        weights = np.linspace(0.5, 1.5, len(target_prices))  # Higher weights for longer timeframes
        weighted_target_prices = np.array(target_prices) * weights
        mtf_target_price = np.sum(weighted_target_prices) / np.sum(weights)
        
        print(f"MTF Target Price: {format_float(mtf_target_price)}")
        
        # Calculate MTF confidence level based on the agreement between timeframes
        reversal_signals = [result['reversal_signal'] for result in timeframe_results.values()]
        up_reversals = sum(1 for signal in reversal_signals if "Upward" in signal)
        down_reversals = sum(1 for signal in reversal_signals if "Downward" in signal)
        
        if up_reversals > down_reversals:
            mtf_reversal_direction = "Upward"
            mtf_confidence = up_reversals / len(reversal_signals) * 100
        elif down_reversals > up_reversals:
            mtf_reversal_direction = "Downward"
            mtf_confidence = down_reversals / len(reversal_signals) * 100
        else:
            mtf_reversal_direction = "Neutral"
            mtf_confidence = 50.0
        
        print(f"MTF Reversal Direction: {mtf_reversal_direction}")
        print(f"MTF Confidence Level: {format_float(mtf_confidence)}%")
        
        # Generate trading recommendation based on MTF analysis
        print("\nMTF Trading Recommendation:")
        
        if overall_mtf_cycle == "POWER MTF UP CYCLE" and mtf_reversal_direction == "Upward" and mtf_confidence > 70:
            recommendation = "STRONG BUY"
        elif overall_mtf_cycle == "POWER MTF UP CYCLE" and mtf_reversal_direction == "Upward":
            recommendation = "BUY"
        elif overall_mtf_cycle == "POWER MTF DOWN CYCLE" and mtf_reversal_direction == "Downward" and mtf_confidence > 70:
            recommendation = "STRONG SELL"
        elif overall_mtf_cycle == "POWER MTF DOWN CYCLE" and mtf_reversal_direction == "Downward":
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        print(f"Recommendation: {recommendation}")
        
        # Calculate risk/reward ratio
        current_price = timeframe_results['5m']['current_close']  # Use 5m price as reference
        if mtf_reversal_direction == "Upward":
            risk = current_price - timeframe_results['5m']['lowest_low']
            reward = mtf_target_price - current_price
        elif mtf_reversal_direction == "Downward":
            risk = timeframe_results['5m']['highest_high'] - current_price
            reward = current_price - mtf_target_price
        else:
            risk = 0
            reward = 0
        
        if risk > 0:
            risk_reward_ratio = reward / risk
        else:
            risk_reward_ratio = 0
        
        print(f"Risk/Reward Ratio: {format_float(risk_reward_ratio)}")
        
        # MTF Volume Analysis
        print("\nMTF Volume Analysis:")
        
        # Get volume results from all timeframes
        all_volume_ratios = [result['volume_results']['volume_ratio'] for result in timeframe_results.values()]
        all_volume_rocs = [result['volume_results']['volume_roc'] for result in timeframe_results.values()]
        all_price_rocs = [result['volume_results']['price_roc'] for result in timeframe_results.values()]
        
        # Calculate averages
        avg_volume_ratio = np.mean(all_volume_ratios)
        avg_volume_roc = np.mean(all_volume_rocs)
        avg_price_roc = np.mean(all_price_rocs)
        
        print(f"Average Volume Ratio: {format_float(avg_volume_ratio)}")
        print(f"Average Volume ROC: {format_float(avg_volume_roc)}")
        print(f"Average Price ROC: {format_float(avg_price_roc)}")
        
        # Determine volume momentum
        if avg_volume_ratio > 1.2 and avg_price_roc > 0:
            volume_momentum = "Strong Bullish Volume Momentum"
        elif avg_volume_ratio > 1.2 and avg_price_roc < 0:
            volume_momentum = "Strong Bearish Volume Momentum"
        elif avg_volume_ratio < 0.8:
            volume_momentum = "Weak Volume Momentum"
        else:
            volume_momentum = "Neutral Volume Momentum"
        
        print(f"Volume Momentum: {volume_momentum}")
        
        # MTF Momentum Analysis
        print("\nMTF Momentum Analysis:")
        
        # Get momentum results from all timeframes
        all_rsi = [result['momentum_results']['rsi'] for result in timeframe_results.values()]
        all_slowk = [result['momentum_results']['slowk'] for result in timeframe_results.values()]
        all_slowd = [result['momentum_results']['slowd'] for result in timeframe_results.values()]
        all_macd = [result['momentum_results']['macd'] for result in timeframe_results.values()]
        all_macdsignal = [result['momentum_results']['macdsignal'] for result in timeframe_results.values()]
        all_cci = [result['momentum_results']['cci'] for result in timeframe_results.values()]
        
        # Calculate averages
        avg_rsi = np.mean(all_rsi)
        avg_slowk = np.mean(all_slowk)
        avg_slowd = np.mean(all_slowd)
        avg_macd = np.mean(all_macd)
        avg_macdsignal = np.mean(all_macdsignal)
        avg_cci = np.mean(all_cci)
        
        print(f"Average RSI: {format_float(avg_rsi)}")
        print(f"Average Stochastic: {format_float(avg_slowk)}/{format_float(avg_slowd)}")
        print(f"Average MACD: {format_float(avg_macd)}/{format_float(avg_macdsignal)}")
        print(f"Average CCI: {format_float(avg_cci)}")
        
        # Determine momentum direction
        momentum_signals = []
        
        if avg_rsi > 70:
            momentum_signals.append("Overbought")
        elif avg_rsi < 30:
            momentum_signals.append("Oversold")
        
        if avg_slowk > 80 and avg_slowd > 80:
            momentum_signals.append("Overbought")
        elif avg_slowk < 20 and avg_slowd < 20:
            momentum_signals.append("Oversold")
        
        if avg_macd > avg_macdsignal:
            momentum_signals.append("Bullish MACD")
        else:
            momentum_signals.append("Bearish MACD")
        
        if avg_cci > 100:
            momentum_signals.append("Overbought")
        elif avg_cci < -100:
            momentum_signals.append("Oversold")
        
        print(f"Momentum Signals: {', '.join(momentum_signals) if momentum_signals else 'Neutral'}")
        
        # MTF FFT Forecast Analysis
        print("\nMTF FFT Forecast Analysis:")
        
        # Get FFT forecasts from all timeframes
        all_fft_forecasts = [result['forecasted_prices'][-1] for result in timeframe_results.values()]
        
        # Calculate average FFT forecast
        avg_fft_forecast = np.mean(all_fft_forecasts)
        
        # Calculate FFT forecast direction
        fft_directions = []
        for result in timeframe_results.values():
            if result['forecasted_prices'][-1] > result['current_close']:
                fft_directions.append("Upward")
            else:
                fft_directions.append("Downward")
        
        up_fft = fft_directions.count("Upward")
        down_fft = fft_directions.count("Downward")
        
        print(f"Average FFT Forecast: {format_float(avg_fft_forecast)}")
        print(f"FFT Direction: Upward: {up_fft}, Downward: {down_fft}")
        
        # MTF Profit Target Analysis
        print("\nMTF Profit Target Analysis:")
        
        # Get profit targets from all timeframes
        all_profit_targets = [result['profit_targets']['target_price'] for result in timeframe_results.values()]
        all_min_thresholds = [result['profit_targets']['min_threshold'] for result in timeframe_results.values()]
        all_max_thresholds = [result['profit_targets']['max_threshold'] for result in timeframe_results.values()]
        
        # Calculate averages
        avg_profit_target = np.mean(all_profit_targets)
        avg_min_threshold = np.mean(all_min_thresholds)
        avg_max_threshold = np.mean(all_max_thresholds)
        
        print(f"Average 2% Profit Target: {format_float(avg_profit_target)}")
        print(f"Average Min Threshold: {format_float(avg_min_threshold)}")
        print(f"Average Max Threshold: {format_float(avg_max_threshold)}")
        
        # Calculate final MTF target price with all adjustments
        print("\nFinal MTF Target Price Calculation:")
        
        # Combine all target prices
        all_targets = [mtf_target_price, avg_fft_forecast, avg_profit_target]
        final_mtf_target = np.mean(all_targets)
        
        # Calculate confidence in the target
        target_confidence = 0
        if mtf_reversal_direction == "Upward" and final_mtf_target > current_price:
            target_confidence = mtf_confidence
        elif mtf_reversal_direction == "Downward" and final_mtf_target < current_price:
            target_confidence = mtf_confidence
        else:
            target_confidence = 50.0
        
        print(f"Final MTF Target Price: {format_float(final_mtf_target)}")
        print(f"Target Confidence: {format_float(target_confidence)}%")
        
        # Generate final trading signal with volume and momentum confirmation
        print("\nFinal Trading Signal:")
        
        # Check for volume and momentum confirmation
        volume_confirmation = False
        momentum_confirmation = False
        
        if "Strong Bullish Volume Momentum" == volume_momentum and mtf_reversal_direction == "Upward":
            volume_confirmation = True
        elif "Strong Bearish Volume Momentum" == volume_momentum and mtf_reversal_direction == "Downward":
            volume_confirmation = True
        
        if "Bullish MACD" in momentum_signals and mtf_reversal_direction == "Upward":
            momentum_confirmation = True
        elif "Bearish MACD" in momentum_signals and mtf_reversal_direction == "Downward":
            momentum_confirmation = True
        
        # Generate final signal
        if volume_confirmation and momentum_confirmation and target_confidence > 70:
            final_signal = "STRONG SIGNAL - Volume and Momentum Confirmed"
        elif volume_confirmation or momentum_confirmation and target_confidence > 60:
            final_signal = "MODERATE SIGNAL - Partial Confirmation"
        else:
            final_signal = "WEAK SIGNAL - Low Confirmation"
        
        print(f"Volume Confirmation: {'Yes' if volume_confirmation else 'No'}")
        print(f"Momentum Confirmation: {'Yes' if momentum_confirmation else 'No'}")
        print(f"Final Signal: {final_signal}")
        
        # Calculate expected profit percentage
        if mtf_reversal_direction == "Upward":
            expected_profit = (final_mtf_target - current_price) / current_price * 100
        else:
            expected_profit = (current_price - final_mtf_target) / current_price * 100
        
        print(f"Expected Profit: {format_float(expected_profit)}%")
        
        # Time to target estimation (in minutes)
        if mtf_reversal_direction == "Upward":
            time_to_target = ((final_mtf_target - current_price) / (avg_price_roc / 100)) if avg_price_roc > 0 else 0
        else:
            time_to_target = ((current_price - final_mtf_target) / (-avg_price_roc / 100)) if avg_price_roc < 0 else 0
        
        print(f"Estimated Time to Target: {format_float(time_to_target)} minutes")
        
        print("\n" + "="*50)
        print("END OF MTF ANALYSIS - FAST TARGETS")
        print("="*50 + "\n")

if __name__ == "__main__":
    trader = Trader('api.txt')
    trader.run_analysis()