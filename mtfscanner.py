import numpy as np
import pandas as pd
from binance.client import Client
import time
import talib as ta
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LinearRegression
from scipy.fft import fft

class TradingBot:
    def __init__(self):
        # Load Binance API credentials from a file
        with open("credentials.txt", "r") as f:
            api_key = f.readline().strip()
            api_secret = f.readline().strip()
        self.client = Client(api_key, api_secret)
        self.timeframes = ["1d", "4h", "1h", "15m", "5m", "1m"]
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

            if all(x > 0 for x in [open_, high, low, close, volume]):
                valid_candles.append({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        return valid_candles

    def analyze_market(self, symbol):
        """Analyze market trends for a specific symbol across different timeframes."""
        dips_found = {}

        for interval in self.timeframes:
            candles = self.get_valid_candles(symbol, interval)
            if len(candles) < 14:  # Ensure enough data for analysis
                continue

            close_prices = np.array([candle["close"] for candle in candles], dtype=float)
            volumes = np.array([candle["volume"] for candle in candles], dtype=float)

            # Calculate bullish and bearish volumes
            bullish_volume = np.sum(volumes[1:][close_prices[1:] > close_prices[:-1]])
            bearish_volume = np.sum(volumes[1:][close_prices[1:] < close_prices[:-1]])  # Fixed parenthesis error
            avg_volume = np.mean(volumes)

            # Linear regression for price forecasting
            X = np.arange(len(close_prices)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, close_prices)

            # Forecast the current price using linear regression
            future_index = np.array([[len(close_prices),]]) 
            forecast_price = model.predict(future_index)[0]

            dips_found[interval] = {
                "Current Close": close_prices[-1],
                "Average Volume": avg_volume,
                "Bullish Volume": bullish_volume,
                "Bearish Volume": bearish_volume,
                "Forecast Price": forecast_price,
                "Trend Status": self.determine_trend_status(close_prices[-1], forecast_price)
            }

            # Hurst Exponent Calculation
            hurst_value = self.hurst_exponent(close_prices)  
            dips_found[interval]["Hurst Exponent"] = hurst_value if hurst_value is not None else np.nan

            # FFT Analysis
            if len(close_prices) > 1:
                significant_category, avg_neg_freq, avg_neutral_freq, avg_pos_freq = self.analyze_fft(close_prices)
                dips_found[interval]["FFT Analysis"] = significant_category
                dips_found[interval]["Avg Negative Frequency"] = avg_neg_freq
                dips_found[interval]["Avg Neutral Frequency"] = avg_neutral_freq
                dips_found[interval]["Avg Positive Frequency"] = avg_pos_freq

            dips_found[interval]['Market Sentiment'] = 'Bullish' if bullish_volume > bearish_volume else 'Bearish'

        return dips_found

    def determine_trend_status(self, current_close, forecast_price):
        """Determine trend status based on current close price and forecast price."""
        if current_close < forecast_price * 0.98:
            return "Dip"
        elif current_close > forecast_price * 1.02:
            return "Uptrend"
        return "Neutral"

    def hurst_exponent(self, ts):
        """Calculate the Hurst Exponent to analyze long-term memory."""
        lags = range(2, 100)
        try:
            ts = np.asarray(ts)  # Ensure the input is a numpy array
            tau = [np.std(np.subtract(ts[l:], ts[:-l])) for l in lags if len(ts) > l]
            if not tau or any(np.array(tau) <= 0):  # Correctly check against 'tau'
                return None
            
            log_lags = np.log(lags)
            log_tau = np.log(tau)

            hurst_exponent = np.polyfit(log_lags, log_tau, 1)[0]
            return hurst_exponent
        except Exception as e:
            print(f"Error calculating Hurst exponent: {e}")
            return None

    def analyze_fft(self, close_prices):
        """Perform FFT Analysis."""
        freq_data = fft(close_prices)
        magnitudes = np.abs(freq_data[:25])
        
        negative_freqs = magnitudes[:12]
        neutral_freq = magnitudes[12]
        positive_freqs = magnitudes[13:25]

        avg_neg_freq = np.mean(negative_freqs) if len(negative_freqs) > 0 else np.nan
        avg_neutral_freq = neutral_freq if neutral_freq else np.nan
        avg_pos_freq = np.mean(positive_freqs) if len(positive_freqs) > 0 else np.nan

        if avg_pos_freq > avg_neg_freq and avg_pos_freq > avg_neutral_freq:
            significant_category = "Positive"
        elif avg_neg_freq > avg_pos_freq and avg_neg_freq > avg_neutral_freq:
            significant_category = "Negative"
        else:
            significant_category = "Neutral"

        return significant_category, avg_neg_freq, avg_neutral_freq, avg_pos_freq

    def display_results(self, overall_analysis):
        """Display analysis results for each timeframe in separate tables."""
        all_results = {}
        for interval in self.timeframes:
            data = {
                symbol: analysis[interval] for symbol, analysis in overall_analysis.items() if interval in analysis
            }
            if data:
                all_results[interval] = pd.DataFrame.from_dict(data, orient='index')
        
        for interval, df in all_results.items():
            print(f"\nMarket Analysis for Timeframe: {interval.upper()}")
            print(df.to_string(index=True))

    def get_top_dip_assets(self, overall_analysis):
        """Identify and return assets with daily dips and provide data from all timeframes."""
        dip_assets = {}
        for symbol, analysis in overall_analysis.items():
            if analysis.get('1d', {}).get('Trend Status') == "Dip":
                dip_assets[symbol] = {interval: analysis.get(interval, {}) for interval in self.timeframes}
        return dip_assets

    def build_overall_table(self, dip_assets, overall_analysis):
        """Build the overall table summarizing top dips."""
        overall_data = []
        for symbol, dip_info in dip_assets.items():
            last_close = dip_info['1d'].get('Current Close')
            if last_close is not None:
                overall_data.append({
                    'Symbol': symbol,
                    'Last Close Price': last_close,
                    '1D Trend Status': dip_info['1d'].get('Trend Status')
                })
        
        overall_df = pd.DataFrame(overall_data)
        overall_df.fillna('N/A', inplace=True)
        print("\nOverall Table: Top Dips")
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
        dip_assets = self.get_top_dip_assets(overall_analysis)
        self.build_overall_table(dip_assets, overall_analysis)

# Main program loop
if __name__ == "__main__":
    bot = TradingBot()
    start_time = time.time()
    bot.scan_assets()
    elapsed_time = time.time() - start_time
    print(f"Scan completed in {elapsed_time:.2f} seconds.")