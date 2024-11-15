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

    def calculate_45_degree_price(self, current_price):
        """ Calculate the price value at a 45-degree angle relative to current price. """
        return current_price / np.sqrt(2)  # 45 degrees implies the price increment in equal x,y.

if __name__ == "__main__":
    filename = 'credentials.txt'
    trader = Trader(filename)

    trading_pairs = trader.get_usdc_pairs()
    dip_data = {}
    volume_data = {}
    timeframes_volume = ['1m', '3m', '5m']
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
    for symbol in trading_pairs:
        result = trader.calculate_dips(symbol)
        if result:
            symbol, sig_dip_5m, sig_dip_daily, rev_points_5m, rev_points_daily = result
            dip_data[symbol] = {
                '5m_dip': sig_dip_5m,
                'daily_dip': sig_dip_daily,
                '5m_reversals': rev_points_5m,
                'daily_reversals': rev_points_daily
            }

    # Select the most bullish dip while ensuring current price is below the 45-degree angle price
    best_symbol = None
    best_bullish_volume = 0
    best_dip_info = None
    best_price_at_45_degree = None

    for symbol in dip_data.keys():
        if dip_data[symbol]['5m_dip'] is not None:
            total_bullish = volume_data.get(symbol, {}).get('1m', {}).get('bullish', 0)
            current_price = trader.client.get_symbol_ticker(symbol=symbol)['price']
            price_at_45_degree = trader.calculate_45_degree_price(float(current_price))

            # Check if current price is below the 45-degree angle price
            if float(current_price) < price_at_45_degree:
                if total_bullish > best_bullish_volume:
                    best_bullish_volume = total_bullish
                    best_symbol = symbol
                    best_dip_info = dip_data[symbol]
                    best_price_at_45_degree = price_at_45_degree

    # Output result
    if best_symbol:
        print(f'The asset with the best bullish dip is: {best_symbol}')
        print(f'Best Dip (5m): {best_dip_info["5m_dip"]}')
        print(f'Current Price: {current_price}')
        print(f'Price at 45 Degree: {best_price_at_45_degree:.25f}')
    else:
        print('No suitable asset found below the 45 degree angle.')

    sys.exit(0)
