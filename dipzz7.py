import numpy as np
import pandas as pd
from binance.client import Client
import time
import talib as ta
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LinearRegression
from scipy.fft import fft, ifft

class TradingBot:
    def __init__(self):
        # Load Binance API credentials from a file
        with open("credentials.txt", "r") as f:
            api_key = f.readline().strip()
            api_secret = f.readline().strip()
        self.client = Client(api_key, api_secret)
        self.timeframes = ["1d", "4h", "1h", "15m", "5m"]
        pd.set_option('display.float_format', '{:.8f}'.format)  # Set float display format
        pd.set_option('display.max_rows', None)  # To display all rows
        pd.set_option('display.max_columns', None)  # To display all columns

    def get_usdc_pairs(self):
        """Get all pairs traded against USDC on the spot market."""
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [
            symbol['symbol']
            for symbol in exchange_info['symbols']
            if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING'
        ]
        return trading_pairs

    def get_valid_candles(self, symbol, timeframe, limit=1000):
        """Retrieve valid candlesticks for a specified symbol and timeframe."""
        klines = self.client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        valid_candles = []
        for k in klines:
            open_ = float(k[1])
            high = float(k[2])
            low = float(k[3])
            close = float(k[4])
            volume = float(k[5])

            if all(x != 0 for x in [open_, high, low, close, volume]) and \
               not any(np.isnan([open_, high, low, close, volume])):
                valid_candles.append({
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                })
        return valid_candles

    def analyze_market(self, symbol):
        """Analyze market trends for a specific symbol across different timeframes."""
        dips_found = {}

        for interval in self.timeframes:
            candles = self.get_valid_candles(symbol, interval)
            if not candles:
                continue

            close_prices = np.array([candle["close"] for candle in candles], dtype=float)
            volumes = np.array([candle["volume"] for candle in candles], dtype=float)

            if len(close_prices) < 14:  # Ensure we have enough data for momentum calculation
                continue

            # Calculate bullish and bearish volumes
            bullish_volume = np.sum(volumes[1:][close_prices[1:] > close_prices[:-1]])  # Future price higher than previous
            bearish_volume = np.sum(volumes[1:][close_prices[1:] < close_prices[:-1]])  # Future price lower than previous
            
            # Linear regression for price forecasting
            X = np.arange(len(close_prices)).reshape(-1, 1)  # Reshape for sklearn
            model = LinearRegression()
            model.fit(X, close_prices)

            # Forecast the current price using linear regression
            future_index = np.array([[len(close_prices),]])  # Next time step
            forecast_price = model.predict(future_index)[0]
            forecast_channel = model.predict(X)

            # Calculate the standard deviation of the residuals
            residuals = close_prices - forecast_channel
            std_dev = np.std(residuals)
            k = 1  # Channel width factor

            # Upper and lower channel calculations
            upper_channel = forecast_channel + (k * std_dev)
            lower_channel = forecast_channel - (k * std_dev)

            # Best fit line calculation
            best_fit_line = np.poly1d(np.polyfit(np.arange(len(close_prices)), close_prices, 1))(np.arange(len(close_prices)))

            dips_found[interval] = {
                "Current Close": close_prices[-1],
                "Best Fit Line": best_fit_line[-1],
                "Forecast Price": forecast_price,
                "Upper Channel": upper_channel[-1],
                "Lower Channel": lower_channel[-1],
                "Momentum": None,
                "Trend Status": None,
                "RSI": None,  # Calculation is kept, but print is removed
                "Momentum Reversal": None,  # Calculation is kept, but print is removed
                "FFT Forecast Price": None,  # Placeholder for FFT forecast price
                "Volume Status": None
            }

            # Apply FFT for forecasting and sinusoidal wave modeling
            if len(close_prices) > 1:  # Ensure we have enough data points for FFT
                n = len(close_prices)
                yf = fft(close_prices)  # Perform FFT on closing prices
                frequencies = np.fft.fftfreq(n)

                # Keep only the positive frequencies and their corresponding FFT values
                half_n = n // 2
                significant_yf = yf[:half_n]
                significant_frequencies = frequencies[:half_n]

                # Number of future steps to forecast
                future_steps = 5
                future_time = np.arange(n + future_steps)

                # Rebuild the forecasted prices using the most significant frequencies.
                future_prices = np.zeros(future_steps)

                for i in range(len(significant_yf)):
                    amplitude = np.abs(significant_yf[i]) / n  # Amplitude of the corresponding frequency
                    phase = np.angle(significant_yf[i])        # Phase of the corresponding frequency
                    frequency = significant_frequencies[i]  # Frequency

                    # At each future step, add the contribution of this frequency
                    for k in range(future_steps):
                        future_prices[k] += amplitude * np.cos(2 * np.pi * frequency * (n + k) / n + phase)

                # Assign the last forecasted value as the FFT forecast price (5-step projection)
                dips_found[interval]['FFT Forecast Price'] = future_prices[-1]

            # Momentum calculation
            momentum = ta.MOM(close_prices, timeperiod=14)
            current_momentum = momentum[-1] if len(momentum) > 0 else 0  # Default to 0 if N/A
            dips_found[interval]["Momentum"] = current_momentum
            
            # Calculate momentum reversal metric
            negative_momentum = [m for m in momentum if m < 0]
            dips_found[interval]["Momentum Reversal"] = abs(min(negative_momentum)) if negative_momentum else None
            
            # RSI calculation
            rsi = ta.RSI(close_prices, timeperiod=14)
            dips_found[interval]["RSI"] = rsi[-1] if len(rsi) > 0 else None
            
            # Analyze volume status
            if bullish_volume > bearish_volume:
                dips_found[interval]["Volume Status"] = "Bullish"
            elif bearish_volume > bullish_volume:
                dips_found[interval]["Volume Status"] = "Bearish"
            else:
                dips_found[interval]["Volume Status"] = "Neutral"

            # Trend conditions
            current_close = close_prices[-1]
            best_fit_val = best_fit_line[-1]
            if current_close < lower_channel[-1]:  # Price below lower channel
                dips_found[interval]["Trend Status"] = "Potential Buy"
            elif current_close > upper_channel[-1]:  # Price above upper channel
                dips_found[interval]["Trend Status"] = "Potential Sell"
            elif current_close < best_fit_val * 0.99:  # Detect dip scenario
                dips_found[interval]["Trend Status"] = "Dip"
            elif current_close > best_fit_val * 1.01:  # Detect uptrend scenario
                dips_found[interval]["Trend Status"] = "Uptrend"
            else:
                dips_found[interval]["Trend Status"] = "Neutral"

        return dips_found

    def display_results(self, overall_analysis):
        """Display analysis results for each timeframe in separate tables without truncation."""
        all_results = {}
        for interval in self.timeframes:
            data = {
                symbol: analysis[interval] for symbol, analysis in overall_analysis.items() if interval in analysis
            }
            if data:
                all_results[interval] = pd.DataFrame.from_dict(data, orient='index')

        # Print all results in one go for full output
        for interval, df in all_results.items():
            print(f"\nMarket Analysis for Timeframe: {interval.upper()}")
            print(df.to_string(index=True))  # Display full DataFrame

    def get_top_uptrend_assets(self, overall_analysis):
        """Identify and return assets with daily uptrends and dips on smaller timeframes."""
        uptrend_assets = {}
        
        for symbol, analysis in overall_analysis.items():
            # Check for daily uptrend
            if analysis.get('1d', {}).get('Trend Status') == "Uptrend":
                # Check for dips in smaller timeframes
                small_tf_dips = sum(
                    1 for interval in ["4h", "1h", "15m"]
                    if analysis.get(interval, {}).get('Trend Status') == "Dip"
                )
                
                if small_tf_dips > 0:  # If there are dips on smaller timeframes
                    uptrend_assets[symbol] = {
                        'small_tf_dips': small_tf_dips,
                        'fft_forecast': analysis['5m']['FFT Forecast Price'] if '5m' in analysis else None
                    }

        return uptrend_assets

    def get_top_dip_assets(self, overall_analysis):
        """Identify and return assets with daily dips and lowest momentum."""
        dip_assets = {}

        for symbol, analysis in overall_analysis.items():
            # Check for daily dip
            if analysis.get('1d', {}).get('Trend Status') == "Dip":
                dip_count = sum(
                    1 for intervals in analysis.values()
                    if 'Trend Status' in intervals and intervals['Trend Status'] == "Dip" 
                )

                # Identify the lowest momentum value observed
                lowest_momentum = None
                for intervals in analysis.values():
                    if intervals.get("Momentum") is not None:
                        if lowest_momentum is None or intervals["Momentum"] < lowest_momentum:
                            lowest_momentum = intervals["Momentum"]

                last_close_price = analysis['1d'].get('Current Close') if '1d' in analysis else None

                if dip_count > 0 and lowest_momentum is not None and last_close_price is not None:
                    dip_assets[symbol] = {
                        'dip_count': dip_count,
                        'lowest_momentum': lowest_momentum,
                        'last_close_price': last_close_price,
                        'fft_forecast': analysis['5m']['FFT Forecast Price'] if '5m' in analysis else None
                    }

        return dip_assets

    def build_overall_table(self, uptrend_assets, dip_assets, overall_analysis):
        """Build the overall table summarizing top dips and uptrends."""
        overall_data = []

        # Gather dip assets data
        for symbol, dip_info in dip_assets.items():
            last_close = overall_analysis[symbol]['1d'].get('Current Close') if '1d' in overall_analysis[symbol] else None
            lowest_momentum = dip_info['lowest_momentum']
            number_of_dips = dip_info['dip_count']
            fft_forecast = dip_info['fft_forecast']

            if lowest_momentum is not None and last_close is not None:
                overall_data.append({
                    'Symbol': symbol,
                    'Category': 'Dip with Lowest Momentum',
                    'Number of Dips': number_of_dips,
                    'Avg FFT Forecast Price': fft_forecast,
                    'Last Close Price': last_close,
                    'Lowest Momentum': lowest_momentum
                })

        # Gather uptrend assets data
        for symbol, uptrend_info in uptrend_assets.items():
            last_close_uptrend = overall_analysis[symbol]['1d'].get('Current Close', None)
            uptrend_avg_forecast = uptrend_info['fft_forecast']
            number_of_dips = uptrend_info['small_tf_dips']

            if uptrend_avg_forecast is not None and last_close_uptrend is not None:
                overall_data.append({
                    'Symbol': symbol,
                    'Category': 'Uptrend with Dips',
                    'Number of Dips': number_of_dips,
                    'Avg FFT Forecast Price': uptrend_avg_forecast,
                    'Last Close Price': last_close_uptrend,
                    'Lowest Momentum': None  # As this is an uptrend asset, no momentum to report
                })

        # Create the overall DataFrame
        overall_df = pd.DataFrame(overall_data)

        # Ensure numeric values are handled properly and avoid KeyErrors
        overall_df.fillna({'Avg FFT Forecast Price': 0, 'Lowest Momentum': 0}, inplace=True)
        overall_df['Avg FFT Forecast Price'] = pd.to_numeric(overall_df['Avg FFT Forecast Price'], errors='coerce')
        overall_df['Lowest Momentum'] = pd.to_numeric(overall_df['Lowest Momentum'], errors='coerce')

        # Sort by 'Category' (for proper grouping) and 'Number of Dips' (descending)
        overall_df.sort_values(by=['Category', 'Number of Dips'], ascending=[True, False], inplace=True)

        print("\nOverall Table: Top Dips and Uptrends")
        print(overall_df.to_string(index=False))

    def scan_assets(self):
        """Scan all USDC trading pairs and analyze each concurrently."""
        symbols = self.get_usdc_pairs()
        overall_analysis = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.analyze_market, symbol): symbol for symbol in symbols}
            for future in futures:
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        overall_analysis[symbol] = result
                except Exception as e:
                    print(f"Error analyzing {symbol}: {str(e)}")

        self.display_results(overall_analysis)

        # Get and combine the top uptrend and dip assets
        uptrend_assets = self.get_top_uptrend_assets(overall_analysis)
        dip_assets = self.get_top_dip_assets(overall_analysis)

        # Build the overall summary table
        self.build_overall_table(uptrend_assets, dip_assets, overall_analysis)  # Show the combined table

# Main program loop
if __name__ == "__main__":
    bot = TradingBot()
    start_time = time.time()  # Start the timer
    bot.scan_assets()
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Scan completed in {elapsed_time:.2f} seconds.")