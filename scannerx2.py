import numpy as np
import pandas as pd
from binance.client import Client
import time
import talib as ta
from concurrent.futures import ThreadPoolExecutor

class TradingBot:
    def __init__(self):
        # Load Binance API credentials from a file
        with open("credentials.txt", "r") as f:
            api_key = f.readline().strip()
            api_secret = f.readline().strip()
        self.client = Client(api_key, api_secret)
        self.timeframes = ["1d", "4h", "1h", "15m", "5m"]
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
            
            # Best fit line calculation
            y = np.arange(len(close_prices))
            best_fit_line = np.poly1d(np.polyfit(y, close_prices, 1))(y)

            dips_found[interval] = {
                "Current Close": close_prices[-1],
                "Best Fit Line": best_fit_line[-1],
                "Momentum": None,  # Temporary None value to be filled later
                "Trend Status": None,  # Temporary None value to be filled later
                "RSI": None,  # Added RSI placeholder
                "Momentum Reversal": None  # Added closest momentum to negative reversal
            }

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

            # Check for trends and dip detection
            if current_close < best_fit_val * 0.99:  # Detect dip scenario
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
        """Identify and return assets with the most uptrends across timeframes."""
        uptrend_assets = {}
        for symbol, analysis in overall_analysis.items():
            uptrend_count = sum(
                1 for intervals in analysis.values() 
                if 'Best Fit Line' in intervals and intervals['Current Close'] > intervals['Best Fit Line']
            )
            if uptrend_count > 0:
                uptrend_assets[symbol] = uptrend_count
        
        top_uptrends = sorted(uptrend_assets.items(), key=lambda item: item[1], reverse=True)
        print("\nTop Uptrend Assets:")
        for asset, count in top_uptrends:
            print(f"{asset}: Uptrend in {count} timeframes")

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