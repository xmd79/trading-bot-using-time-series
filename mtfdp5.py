import numpy as np
import concurrent.futures
from binance.client import Client
from datetime import datetime, timedelta

class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    def get_usdc_pairs(self):
        exchange_info = self.client.get_exchange_info()
        trading_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING'
        ]
        return trading_pairs

    def calculate_volume(self, symbol, interval):
        klines = self.client.get_klines(symbol=symbol, interval=interval)
        volumes = [float(entry[5]) for entry in klines]
        close_prices = [float(entry[4]) for entry in klines]

        if not close_prices or not volumes:
            return symbol, {'bullish': 0, 'bearish': 0}

        bullish_volume = sum(volumes[i] for i in range(len(close_prices) - 1) if close_prices[i] < close_prices[i + 1])
        bearish_volume = sum(volumes[i] for i in range(len(close_prices) - 1) if close_prices[i] > close_prices[i + 1])

        return symbol, {'bullish': bullish_volume, 'bearish': bearish_volume, 'prices': close_prices}

    def envelope_and_forecast(self, prices):
        N = len(prices)
        if N < 10:
            return None, None, None, None

        support = np.min(prices)
        resistance = np.max(prices)

        return support, resistance

    def next_reversal_price(self, amplitude, sym_center):
        return sym_center + (amplitude * 1.1)  # extend the target slightly to ensure bullish outlook

def main():
    filename = 'credentials.txt'
    trader = Trader(filename)

    trading_pairs = trader.get_usdc_pairs()
    volume_data = {}
    timeframes_volume = ['1m']
    timeframes_sine = ['5m']

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_symbol = {
            executor.submit(trader.calculate_volume, symbol, tf): (symbol, tf)
            for symbol in trading_pairs for tf in timeframes_volume
        }

        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol, tf = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    symbol, vol = result
                    if symbol not in volume_data:
                        volume_data[symbol] = {}
                    volume_data[symbol][tf] = vol
            except Exception as e:
                print(f'Error fetching data for {symbol} in {tf}: {e}')

    highest_ratio = 0
    best_symbol = None

    for symbol, volumes in volume_data.items():
        total_bullish = sum(volumes[tf]['bullish'] for tf in volumes if 'bullish' in volumes[tf])
        total_bearish = sum(volumes[tf]['bearish'] for tf in volumes if 'bearish' in volumes[tf])

        if total_bearish > 0:
            ratio = total_bullish / total_bearish
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_symbol = (symbol, total_bullish, total_bearish)

    if best_symbol:
        symbol, total_bullish, total_bearish = best_symbol
        print(f'The asset with the highest bullish/bearish volume ratio is: {symbol}')

        total_volume = total_bullish + total_bearish
        bullish_percentage = (total_bullish / total_volume) * 100 if total_volume > 0 else 0
        bearish_percentage = (total_bearish / total_volume) * 100 if total_volume > 0 else 0

        print(f'Bullish Volume: {total_bullish:.2f} ({bullish_percentage:.2f}%)')
        print(f'Bearish Volume: {total_bearish:.2f} ({bearish_percentage:.2f}%)')

        # Get the latest price data for the selected symbol using the 5-minute timeframe
        klines = trader.client.get_klines(symbol=symbol, interval='5m')
        latest_prices = [float(entry[4]) for entry in klines]
        support, resistance = trader.envelope_and_forecast(latest_prices)

        if latest_prices:
            current_price = latest_prices[-1]
            max_threshold = np.max(latest_prices)
            min_threshold = np.min(latest_prices)

            # Revised next reversal price logic
            forecast_price = trader.next_reversal_price(max_threshold - current_price, current_price)

            # Calculate upper and lower channel based on volume resistance and projections
            volume_resistance = max_threshold * 1.05  # 5% extension potential resistance based on volume
            volume_support = max_threshold * 0.95      # 5% extension potential support based on volume
            
            # Calculate distances to min/max
            distance_to_min = (current_price - min_threshold) / (max_threshold - min_threshold) * 100
            distance_to_max = (max_threshold - current_price) / (max_threshold - min_threshold) * 100

            # Calculate distance from current price to forecast price
            distance_to_forecast = ((forecast_price - current_price) / current_price) * 100 if current_price > 0 else float('inf')

            # Estimate time to target more realistically
            recent_price_changes = [latest_prices[i] - latest_prices[i-1] for i in range(1, len(latest_prices))]
            average_price_change_per_5m = np.mean(recent_price_changes)  # Average price change per 5 minutes

            # Calculate how many 5-minute intervals it will take to reach the forecast price
            if average_price_change_per_5m != 0:
                time_to_target_intervals = (forecast_price - current_price) / average_price_change_per_5m 
                if time_to_target_intervals < 0:
                    time_to_target_intervals = 0
            else:
                time_to_target_intervals = float('inf')

            time_to_target_mins = time_to_target_intervals * 5  # Convert to minutes
            
            # Convert time to target into days, hours, minutes, and seconds
            days = int(time_to_target_mins // 1440)
            hours = int((time_to_target_mins % 1440) // 60)
            minutes = int(time_to_target_mins % 60)
            seconds = int((time_to_target_mins - int(time_to_target_mins)) * 60)

            # Current local time and projected target time
            current_local_time = datetime.now()
            target_time = current_local_time + timedelta(minutes=time_to_target_mins)

            # Final output display
            print("\n" + "=" * 80)
            print(f"{'Symbol':<20}{'Timeframe':<10}{'Min Threshold':<25}{'Max Threshold':<25}{'Projected Price':<25}{'Upper Channel':<25}{'Lower Channel':<25}{'Market Mood':<20}")
            print("-" * 80)
            print(f"{symbol:<20}{'5m':<10}{min_threshold:<25.4f}{max_threshold:<25.4f}{forecast_price:<25.4f}{volume_resistance:<25.4f}{volume_support:<25.4f}{'Bullish' if total_bullish > total_bearish else 'Bearish':<20}")
            print(f"{'Distance to Min (%):':<45}{distance_to_min:<20.2f}")
            print(f"{'Distance to Max (%):':<45}{distance_to_max:<20.2f}")
            print(f"{'Distance to Forecast Price (%):':<45}{distance_to_forecast:<20.2f}")
            # Removed prints for Time to Target and Normalized Distance to Target
            print(f"{'Bullish Volume Percentage (%):':<45}{bullish_percentage:<20.2f}%")
            print(f"{'Bearish Volume Percentage (%):':<45}{bearish_percentage:<20.2f}%")
            print("=" * 80)

        else:
            print('No suitable asset found.')
    else:
        print('No assets found.')

if __name__ == "__main__":
    main()
