from binance.client import Client
import numpy as np
import os
import sys
import concurrent.futures
import matplotlib.pyplot as plt


class Trader:
    def __init__(self, file):
        self.connect(file)

    def connect(self, file):
        """ Connect to Binance using API key and secret. """
        lines = [line.rstrip('\n') for line in open(file)]
        key = lines[0]
        secret = lines[1]
        self.client = Client(key, secret)

    def get_usdc_pairs(self):
        """ Retrieve all trading pairs that use USDC as the quote asset. """
        exchange_info = self.client.get_exchange_info()
        return [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING'
        ]

    def calculate_volume(self, symbol, interval):
        """ Calculate bullish and bearish volume based on price movements. """
        klines = self.client.get_klines(symbol=symbol, interval=interval)
        volumes = [float(entry[5]) for entry in klines]
        close_prices = [float(entry[4]) for entry in klines]

        if not close_prices or not volumes:
            return symbol, {'bullish': 0, 'bearish': 0, 'prices': []}

        bullish_volume = sum(volumes[i] for i in range(len(close_prices) - 1) if close_prices[i] < close_prices[i + 1])
        bearish_volume = sum(volumes[i] for i in range(len(close_prices) - 1) if close_prices[i] > close_prices[i + 1])

        return symbol, {'bullish': bullish_volume, 'bearish': bearish_volume, 'prices': close_prices}

    def envelope_and_forecast(self, prices, cycles=(5, 10, 20)):
        """ Create sine waves for price envelope and identify support/resistance. """
        N = len(prices)
        if N < 10:  # Ensure there's enough data
            return None, None, None, None

        x = np.arange(N)
        offset = np.mean(prices)

        sine_waves = []
        for cycle in cycles:
            frequency = 2 * np.pi / cycle
            amplitude = (np.max(prices) - np.min(prices)) / 2
            sine_wave = amplitude * np.sin(frequency * x) + offset
            sine_waves.append(sine_wave)

        support = np.min(prices)
        resistance = np.max(prices)
        return sine_waves, support, resistance, offset

    def next_reversal_price(self, amplitude, sym_center, peak=True):
        """ Calculate next potential price reversal based on amplitude. """
        return sym_center + (amplitude if peak else -amplitude)

    def calculate_dips(self, symbol):
        """ Calculate significant dips including daily and 5m historical close prices. """
        # Fetch 5-minute candles
        klines_5m = self.client.get_klines(symbol=symbol, interval='5m')
        close_prices_5m = np.array([float(entry[4]) for entry in klines_5m])
        
        # Fetch daily candles
        klines_daily = self.client.get_klines(symbol=symbol, interval='1d')
        close_prices_daily = np.array([float(entry[4]) for entry in klines_daily])

        # Check if enough data is available
        if len(close_prices_5m) < 2 and len(close_prices_daily) < 2:
            return symbol, None, None, None, None  # Not enough data for analysis

        # Find dips in the 5m timeframe
        dips_5m = np.where(np.diff(close_prices_5m) < 0)[0] + 1  # Find index of dips
        significant_dip_5m = close_prices_5m[dips_5m].min() if dips_5m.size > 0 else None

        # Find dips in the daily timeframe
        dips_daily = np.where(np.diff(close_prices_daily) < 0)[0] + 1  # Find index of daily dips
        significant_dip_daily = close_prices_daily[dips_daily].min() if dips_daily.size > 0 else None

        # Find the major reversal points - at these lowest lows
        reversal_points_5m = close_prices_5m[np.where(close_prices_5m == significant_dip_5m)[0]] if significant_dip_5m is not None else None
        reversal_points_daily = close_prices_daily[np.where(close_prices_daily == significant_dip_daily)[0]] if significant_dip_daily is not None else None

        return symbol, significant_dip_5m, significant_dip_daily, reversal_points_5m, reversal_points_daily

    def market_mood_metrics(self, current_price, forecast_price):
        """ Calculate market mood metrics based on current and forecast prices. """
        distance = abs(current_price - forecast_price)
        potential_equilibrium = (current_price + forecast_price) / 2
        angle = np.degrees(np.arctan2(forecast_price - current_price, 1))
        intensity = distance / potential_equilibrium if potential_equilibrium else 0
        energy = (current_price + forecast_price) / 2
        angular_momentum = current_price * np.sin(np.radians(angle))

        return intensity, energy, angular_momentum, angle


if __name__ == "__main__":
    filename = 'credentials.txt'
    trader = Trader(filename)

    trading_pairs = trader.get_usdc_pairs()
    volume_data = {}
    timeframes_volume = ['1m']
    timeframes_dips = ['1d']

    # Calculate volume data concurrently
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

    # Calculate average dips
    dip_data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_symbol = {
            executor.submit(trader.calculate_dips, symbol): symbol
            for symbol in trading_pairs
        }

        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    symbol, sig_dip_5m, sig_dip_daily, rev_points_5m, rev_points_daily = result
                    dip_data[symbol] = {
                        '5m_dip': sig_dip_5m,
                        'daily_dip': sig_dip_daily,
                        '5m_reversals': rev_points_5m,
                        'daily_reversals': rev_points_daily
                    }
            except Exception as e:
                print(f'Error fetching dip data for {symbol}: {e}')

    # Sort symbols by average dip
    sorted_dips = sorted(dip_data.items(), key=lambda x: (x[1]['5m_dip'] is not None, x[1]['5m_dip']))
    sorted_symbols = [symbol for symbol, dip_info in sorted_dips if dip_info['5m_dip'] is not None]

    # Calculate bullish-to-bearish ratios for the best asset
    best_symbol = None
    highest_ratio = 0

    for symbol in volume_data.keys():
        total_bullish = sum(volume['bullish'] for volume in volume_data[symbol].values() if 'bullish' in volume)
        total_bearish = sum(volume['bearish'] for volume in volume_data[symbol].values() if 'bearish' in volume)

        if total_bearish > 0:  # Avoid division by zero
            ratio = total_bullish / total_bearish
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_symbol = (symbol, total_bullish, total_bearish)

    # Enhanced: Check if best_symbol was found
    if best_symbol:
        symbol, total_bullish, total_bearish = best_symbol
        print(f'The asset with the highest bullish/bearish volume ratio is: {symbol}')

        # Calculate and print dominance ratios
        total_volume = total_bullish + total_bearish
        bullish_percentage = (total_bullish / total_volume) * 100 if total_volume > 0 else 0
        bearish_percentage = (total_bearish / total_volume) * 100 if total_volume > 0 else 0

        print(f'Bullish Volume: {total_bullish:.25f} ({bullish_percentage:.25f}%)')
        print(f'Bearish Volume: {total_bearish:.25f} ({bearish_percentage:.25f}%)')

        # Get the latest price data for the selected symbol using 5-minute timeframe
        klines = trader.client.get_klines(symbol=symbol, interval='5m')
        latest_prices = [float(entry[4]) for entry in klines]
        sine_waves, support, resistance, offset = trader.envelope_and_forecast(latest_prices)

        if sine_waves is not None and len(sine_waves) > 0:
            latest_volume_prices = volume_data[symbol]['1m']['prices']
            current_price = latest_prices[-1]
            amplitude = np.max(latest_prices) - np.min(latest_prices)

            # Calculate forecast price
            forecast_price = trader.next_reversal_price(amplitude, current_price, peak=current_price < (current_price + amplitude))

            # Print market mood metrics
            intensity, energy, angular_momentum, angle = trader.market_mood_metrics(current_price, forecast_price)
            print(f'Current Close Price: {current_price:.25f}')
            print(f'Forecasted Price for next reversal: {forecast_price:.25f}')
            print(f'Market Mood Intensity: {intensity:.25f}')
            print(f'Market Energy: {energy:.25f}')
            print(f'Angular Momentum: {angular_momentum:.25f}')
            print(f'Market Mood Angle: {angle:.25f} degrees')

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(latest_prices, label='Price', color='blue')

            key_points = {}
            for i, sine_wave in enumerate(sine_waves):
                x = np.arange(len(sine_wave))
                plt.plot(sine_wave, label=f'Sine Wave {i + 1}', linestyle='--')

                key_point_index = np.abs(sine_wave - current_price).argmin()
                key_points[f'Wave {i + 1}'] = (x[key_point_index], sine_wave[key_point_index])
                plt.annotate(f'Key Point {i + 1}: {sine_wave[key_point_index]:.25f}',
                             xy=(x[key_point_index], sine_wave[key_point_index]),
                             xytext=(x[key_point_index], sine_wave[key_point_index] + 0.5),
                             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=8)

            plt.axhline(y=support, color='red', label='Support', linestyle='--')
            plt.axhline(y=resistance, color='purple', label='Resistance', linestyle='--')
            plt.axhline(y=current_price, color='orange', label='Current Close', linestyle='-.')
            plt.axhline(y=forecast_price, color='green', label='Forecast Price', linestyle='-.')

            plt.title(f"Price Forecast with Sine Wave Projection for {symbol}")
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print('No suitable asset found for projection.')
    else:
        print("No suitable symbol found based on volume analysis. Please check the trends or intervals.")

    sys.exit(0)
