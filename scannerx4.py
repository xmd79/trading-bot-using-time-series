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
            if len(close_prices) < 14:  # Ensure we have enough data for momentum calculation
                continue
            
            # Linear regression for price forecasting
            X = np.arange(len(close_prices)).reshape(-1, 1)  # Reshape for sklearn
            model = LinearRegression()
            model.fit(X, close_prices)

            # Forecast the next price using linear regression
            future_index = np.array([[len(close_prices)]])  # Next time step
            forecast_price = model.predict(future_index)[0]
            forecast_channel = model.predict(X)  # For linear regression channel

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
                "RSI": None,
                "Momentum Reversal": None,
                "FFT Forecast Price": None  # Placeholder for FFT forecast price
            }

            # Apply FFT for forecasting
            if len(close_prices) > 1:  # Ensure we have enough data points for FFT
                fft_results = fft(close_prices)  # Perform FFT on closing prices
                
                # Simulate future prices
                future_index = np.arange(len(close_prices), len(close_prices) + 1)  # Only next point
                fft_extended = np.zeros(len(future_index), dtype=complex)  # Extended FFT results
                
                # Perform IFFT only on available FFT results:
                future_prices = np.real(ifft(fft_results)[:len(future_index)])  # Get real part of the inverse FFT
                
                # Assign the next forecast price
                dips_found[interval]['FFT Forecast Price'] = future_prices[-1]  # Predict value for the next timestep

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
            
            # Analyze trend status
            current_close = close_prices[-1]
            best_fit_val = best_fit_line[-1]

            # Check for additional trend conditions based on the linear regression channel
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
        """Identify and return assets with the most uptrends across timeframes and their FFT forecast prices."""
        uptrend_assets = {}
        dip_assets = {}
        
        for symbol, analysis in overall_analysis.items():
            uptrend_count = sum(
                1 for intervals in analysis.values() 
                if 'Best Fit Line' in intervals and intervals['Current Close'] > intervals['Best Fit Line']
            )
            dip_count = sum(
                1 for intervals in analysis.values()
                if 'Trend Status' in intervals and intervals['Trend Status'] == "Dip" 
                and intervals['Current Close'] < intervals['Lower Channel']
            )

            if uptrend_count > 0:
                forecast_prices = []
                for interval in analysis:
                    forecast_prices.append(analysis[interval].get("FFT Forecast Price"))
                avg_forecast_price = np.mean(list(filter(None, forecast_prices)))
                uptrend_assets[symbol] = {
                    'uptrend_count': uptrend_count,
                    'avg_forecast_price': avg_forecast_price
                }

            if dip_count > 0:
                dip_assets[symbol] = {
                    'dip_count': dip_count,
                    'last_close_price': analysis[self.timeframes[0]]['Current Close'] if self.timeframes[0] in analysis else None,
                    'avg_forecast_price': np.mean(list(filter(lambda x: x is not None, [analysis[interval].get("FFT Forecast Price") for interval in analysis])))
                }

        top_uptrends = sorted(uptrend_assets.items(), key=lambda item: item[1]['uptrend_count'], reverse=True)
        print("\nTop Uptrend Assets with Forecast Prices:")
        for asset, data in top_uptrends:
            print(f"{asset}: Uptrend in {data['uptrend_count']} timeframes, Avg. Forecast Price: {data['avg_forecast_price']:.8f}")

        print("\nAssets Capturing Dips across Multiple Timeframes:")
        for asset, data in dip_assets.items():
            print(f"{asset}: Dips in {data['dip_count']} timeframes, Last Close: {data['last_close_price']:.8f}, Avg. Forecast Price: {data['avg_forecast_price']:.8f}")

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
        self.get_top_uptrend_assets(overall_analysis)

# Main program loop
if __name__ == "__main__":
    bot = TradingBot()
    start_time = time.time()  # Start the timer
    bot.scan_assets()
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Scan completed in {elapsed_time:.2f} seconds.")