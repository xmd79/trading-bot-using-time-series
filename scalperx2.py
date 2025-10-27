import numpy as np
import pandas as pd
import math
from decimal import Decimal, getcontext
from binance import Client
from binance.enums import *
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import os

# Set Decimal precision
getcontext().prec = 25

class BinanceSineAnalyzer:
    def __init__(self):
        """Initialize the Binance Sine Analyzer"""
        # Read API credentials from file
        api_key, api_secret = self.read_api_credentials()
        
        self.client = Client(api_key, api_secret)
        # Timeframes in descending order (from highest to lowest)
        self.timeframes = [
            ('24h', Client.KLINE_INTERVAL_1DAY),
            ('12h', Client.KLINE_INTERVAL_12HOUR),
            ('8h', Client.KLINE_INTERVAL_8HOUR),
            ('6h', Client.KLINE_INTERVAL_6HOUR),
            ('4h', Client.KLINE_INTERVAL_4HOUR),
            ('2h', Client.KLINE_INTERVAL_2HOUR),
            ('1h', Client.KLINE_INTERVAL_1HOUR),
            ('30min', Client.KLINE_INTERVAL_30MINUTE),
            ('15min', Client.KLINE_INTERVAL_15MINUTE),
            ('5min', Client.KLINE_INTERVAL_5MINUTE),
            ('3min', Client.KLINE_INTERVAL_3MINUTE),
            ('1min', Client.KLINE_INTERVAL_1MINUTE)
        ]
        
        # Trading state
        self.in_trade = False
        self.current_trade_symbol = None
        self.initial_usdc_balance = Decimal('0')
        self.current_asset_balance = Decimal('0')
        
    def read_api_credentials(self):
        """Read API key and secret from api.txt file"""
        try:
            with open('api.txt', 'r') as file:
                lines = file.readlines()
                api_key = lines[0].strip()
                api_secret = lines[1].strip()
                return api_key, api_secret
        except Exception as e:
            print(f"Error reading API credentials: {e}")
            print("Please create an api.txt file with your Binance API key on the first line and API secret on the second line.")
            return None, None
    
    def format_decimal(self, value, decimal_places=25):
        """Format a value to a fixed number of decimal places"""
        if isinstance(value, str):
            try:
                value = Decimal(value)
            except:
                return value
        
        if isinstance(value, (int, float)):
            value = Decimal(str(value))
        
        if isinstance(value, Decimal):
            return format(value, f'.{decimal_places}f')
        
        return value
    
    def get_all_usdc_pairs(self):
        """Get all trading pairs with USDC"""
        try:
            exchange_info = self.client.get_exchange_info()
            usdc_pairs = [s['symbol'] for s in exchange_info['symbols'] 
                         if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING']
            return usdc_pairs
        except Exception as e:
            print(f"Error fetching USDC pairs: {e}")
            return []
    
    def get_klines(self, symbol, interval, limit=100):
        """Get kline data for a symbol and interval"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return klines
        except Exception as e:
            print(f"Error fetching klines for {symbol} on {interval}: {e}")
            return []
    
    def calculate_sine_wave_with_fft(self, prices, forecast_periods=5):
        """Calculate sine wave values using FFT and forecast using inverse FFT"""
        if len(prices) < 10:
            return None, None, None, None, None, None, None
            
        # Convert to numpy array
        prices_array = np.array(prices, dtype=float)
        
        # Detrend the price series
        x = np.arange(len(prices_array))
        y = prices_array
        
        # Fit a polynomial to detrend
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        trend = p(x)
        
        # Detrend
        detrended = y - trend
        
        # Calculate FFT
        fft_values = np.fft.fft(detrended)
        frequencies = np.fft.fftfreq(len(detrended))
        
        # Find dominant frequency
        positive_freq_idx = np.where(frequencies > 0)[0]
        if len(positive_freq_idx) == 0:
            return None, None, None, None, None, None, None
            
        dominant_freq_idx = positive_freq_idx[np.argmax(np.abs(fft_values[positive_freq_idx]))]
        dominant_freq = frequencies[dominant_freq_idx]
        
        # Create a filtered FFT - keep only dominant frequency and its harmonics
        filtered_fft = np.zeros_like(fft_values)
        filtered_fft[dominant_freq_idx] = fft_values[dominant_freq_idx]
        filtered_fft[-dominant_freq_idx] = fft_values[-dominant_freq_idx]  # Keep symmetric component
        
        # Reconstruct the signal using inverse FFT
        reconstructed = np.real(np.fft.ifft(filtered_fft))
        
        # Create sine wave
        sine_wave = np.sin(2 * np.pi * dominant_freq * x)
        
        # Get min and max of sine wave
        sine_min = np.min(sine_wave)
        sine_max = np.max(sine_wave)
        
        # Get corresponding prices
        min_idx = np.argmin(sine_wave)
        max_idx = np.argmax(sine_wave)
        
        price_at_sine_min = prices_array[min_idx]
        price_at_sine_max = prices_array[max_idx]
        
        # Forecast using inverse FFT
        # Extend the x array for forecasting
        x_extended = np.arange(len(prices_array), len(prices_array) + forecast_periods)
        
        # Create extended trend
        trend_extended = p(x_extended)
        
        # Create extended sine wave
        sine_extended = np.sin(2 * np.pi * dominant_freq * x_extended)
        
        # Create extended detrended signal using the same FFT pattern
        # We'll extend the reconstructed signal by repeating the pattern
        pattern_length = len(reconstructed)
        extended_reconstructed = np.zeros(forecast_periods)
        for i in range(forecast_periods):
            extended_reconstructed[i] = reconstructed[i % pattern_length]
        
        # Combine extended trend and extended detrended signal
        forecast_prices = trend_extended + extended_reconstructed
        
        # Calculate forecast min, max, and middle
        forecast_min = np.min(forecast_prices)
        forecast_max = np.max(forecast_prices)
        forecast_middle = (forecast_min + forecast_max) / 2
        
        return sine_min, sine_max, price_at_sine_min, price_at_sine_max, forecast_min, forecast_max, forecast_middle
    
    def analyze_pair_on_tf(self, symbol, tf_name, tf_interval):
        """Analyze a single trading pair on a specific timeframe"""
        try:
            klines = self.get_klines(symbol, tf_interval)
            if not klines:
                return None
                
            # Extract OHLC data
            closes = [float(kline[4]) for kline in klines]
            highs = [float(kline[2]) for kline in klines]
            lows = [float(kline[3]) for kline in klines]
            volumes = [float(kline[5]) for kline in klines]
            
            # Calculate sine wave values with FFT forecasting
            sine_values = self.calculate_sine_wave_with_fft(closes)
            if sine_values[0] is None:
                return None
            
            sine_min, sine_max, price_at_sine_min, price_at_sine_max, forecast_min, forecast_max, forecast_middle = sine_values
            
            # Determine if we're closer to a dip or top
            current_close = closes[-1]
            dist_to_low = abs(current_close - price_at_sine_min)
            dist_to_high = abs(current_close - price_at_sine_max)
            
            # Calculate percentage from low and high
            pct_from_low = ((current_close - price_at_sine_min) / price_at_sine_min) * 100
            pct_from_high = ((price_at_sine_max - current_close) / price_at_sine_max) * 100
            
            # Determine cycle phase
            if dist_to_low < dist_to_high:
                cycle_phase = "DIP"
                is_dip = True
            else:
                cycle_phase = "TOP"
                is_dip = False
            
            # Get current price and 24h statistics
            ticker = self.client.get_ticker(symbol=symbol)
            current_price = float(ticker['lastPrice'])
            quote_volume_24h = float(ticker['quoteVolume'])
            
            # Return timeframe data
            return {
                'symbol': symbol,
                'current_price': current_price,
                'quote_volume_24h': quote_volume_24h,
                'tf_name': tf_name,
                'sine_min': sine_min,
                'sine_max': sine_max,
                'price_at_sine_min': price_at_sine_min,
                'price_at_sine_max': price_at_sine_max,
                'current_close': current_close,
                'dist_to_low': dist_to_low,
                'dist_to_high': dist_to_high,
                'pct_from_low': pct_from_low,
                'pct_from_high': pct_from_high,
                'cycle_phase': cycle_phase,
                'high': max(highs),
                'low': min(lows),
                'volume': sum(volumes),
                'forecast_min': forecast_min,
                'forecast_max': forecast_max,
                'forecast_middle': forecast_middle,
                'is_dip': is_dip
            }
        except Exception as e:
            print(f"Error analyzing {symbol} on {tf_name}: {e}")
            return None
    
    def scan_assets_on_tf(self, symbols, tf_name, tf_interval):
        """Scan a list of assets on a specific timeframe using multithreading"""
        print(f"\nScanning {len(symbols)} assets on {tf_name} timeframe...")
        print("=" * 60)
        
        dip_assets = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(self.analyze_pair_on_tf, symbol, tf_name, tf_interval): symbol for symbol in symbols}
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result and result['is_dip']:
                        dip_assets.append(result)
                        completed_count += 1
                        # Print immediate result for this asset
                        print(f"✓ {result['symbol']} - DIP found on {tf_name} ({completed_count}/{len(symbols)})")
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
        
        print(f"\nFound {len(dip_assets)} assets in DIP on {tf_name} timeframe")
        return dip_assets
    
    def gradual_mtf_scan(self):
        """Perform gradual MTF scan from highest to lowest timeframe"""
        # Get all USDC pairs
        all_pairs = self.get_all_usdc_pairs()
        if not all_pairs:
            print("No USDC pairs found")
            return None
        
        print(f"Found {len(all_pairs)} USDC pairs. Starting gradual MTF scan...")
        
        # Initialize with all pairs
        current_assets = all_pairs
        
        # Dictionary to store results for each asset across all timeframes
        asset_results = {}
        
        # Scan each timeframe from highest to lowest
        for i, (tf_name, tf_interval) in enumerate(self.timeframes):
            print(f"\n{'='*20} SCANNING {tf_name} TIMEFRAME {'='*20}")
            
            # If this is the first timeframe, scan all assets
            if i == 0:
                dip_results = self.scan_assets_on_tf(current_assets, tf_name, tf_interval)
            else:
                # For subsequent timeframes, only scan assets that were dips on all previous timeframes
                dip_results = self.scan_assets_on_tf(current_assets, tf_name, tf_interval)
            
            # Update current_assets to only include those that are dips on this timeframe
            current_assets = [result['symbol'] for result in dip_results]
            
            # Store results for each asset
            for result in dip_results:
                symbol = result['symbol']
                if symbol not in asset_results:
                    asset_results[symbol] = {
                        'symbol': symbol,
                        'current_price': result['current_price'],
                        'quote_volume_24h': result['quote_volume_24h'],
                        'timeframes': {}
                    }
                
                asset_results[symbol]['timeframes'][tf_name] = result
            
            # If no assets left, break early
            if not current_assets:
                print(f"\nNo assets in DIP on {tf_name}. Stopping scan.")
                break
            
            print(f"\nAssets remaining for next timeframe: {len(current_assets)}")
        
        # Convert the dictionary to a list
        final_results = list(asset_results.values())
        
        # Sort by daily (24h) price_at_sine_min to find the lowest low
        if final_results and '24h' in final_results[0]['timeframes']:
            final_results.sort(key=lambda x: x['timeframes']['24h']['price_at_sine_min'])
        
        # Return the best asset if we have results
        if final_results:
            return final_results[0]
        return None
    
    def get_spot_balance(self, asset='USDC'):
        """Get spot balance for a specific asset"""
        try:
            balance_info = self.client.get_asset_balance(asset=asset)
            if balance_info:
                return Decimal(balance_info['free'])
            return Decimal('0')
        except Exception as e:
            print(f"Error getting balance for {asset}: {e}")
            return Decimal('0')
    
    def place_market_buy_order(self, symbol, usdc_amount):
        """Place a market buy order for a symbol using USDC"""
        try:
            # Get current price for calculation
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # Calculate quantity to buy (accounting for precision)
            quantity = usdc_amount / Decimal(str(current_price))
            
            # Get symbol info to determine precision
            symbol_info = self.client.get_symbol_info(symbol)
            filters = symbol_info['filters']
            lot_size_filter = next(filter(lambda f: f['filterType'] == 'LOT_SIZE', filters), None)
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                # Round quantity to the appropriate step size
                quantity = self.round_step_size(quantity, step_size)
            
            # Place the order
            print(f"Placing market buy order for {symbol}: {self.format_decimal(quantity)} at ~{self.format_decimal(current_price)}")
            order = self.client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(quantity)
            )
            
            print(f"Buy order executed: {order}")
            return True
        except Exception as e:
            print(f"Error placing buy order for {symbol}: {e}")
            return False
    
    def place_market_sell_order(self, symbol):
        """Place a market sell order for all holdings of a symbol"""
        try:
            # Get current balance
            asset_balance = self.get_spot_balance(symbol.replace('USDC', ''))
            if asset_balance <= Decimal('0'):
                print(f"No balance to sell for {symbol}")
                return False
            
            # Get symbol info to determine precision
            symbol_info = self.client.get_symbol_info(symbol)
            filters = symbol_info['filters']
            lot_size_filter = next(filter(lambda f: f['filterType'] == 'LOT_SIZE', filters), None)
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                # Round quantity to the appropriate step size
                quantity = self.round_step_size(asset_balance, step_size)
            else:
                quantity = asset_balance
            
            # Place the order
            print(f"Placing market sell order for {symbol}: {self.format_decimal(quantity)}")
            order = self.client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(quantity)
            )
            
            print(f"Sell order executed: {order}")
            return True
        except Exception as e:
            print(f"Error placing sell order for {symbol}: {e}")
            return False
    
    def round_step_size(self, quantity, step_size):
        """Round quantity to the nearest step size"""
        precision = int(round(-math.log(step_size, 10), 0))
        return float(Decimal(str(quantity)).quantize(Decimal(f'1e-{precision}')))
    
    def check_profit_target(self):
        """Check if we've reached the 2% profit target"""
        if not self.in_trade or not self.current_trade_symbol:
            return False
        
        try:
            # Get current asset balance
            asset = self.current_trade_symbol.replace('USDC', '')
            asset_balance = self.get_spot_balance(asset)
            
            if asset_balance <= Decimal('0'):
                print(f"No balance found for {asset}")
                return False
            
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=self.current_trade_symbol)
            current_price = Decimal(ticker['price'])
            
            # Calculate current value in USDC
            current_value = asset_balance * current_price
            
            # Calculate profit percentage
            profit_pct = ((current_value - self.initial_usdc_balance) / self.initial_usdc_balance) * Decimal('100')
            
            print(f"Current profit: {self.format_decimal(profit_pct)}% (Target: 2%)")
            
            # Check if we've reached the profit target
            if profit_pct >= Decimal('2.0'):
                print(f"Profit target reached! Exiting trade.")
                return True
            
            return False
        except Exception as e:
            print(f"Error checking profit target: {e}")
            return False
    
    def run_trading_bot(self):
        """Main trading bot loop"""
        print("Starting trading bot...")
        print("Bot will scan for MTF dip assets every 5 seconds")
        print("Will enter trades with entire USDC balance and exit at 2% profit")
        
        while True:
            try:
                # Get current USDC balance
                usdc_balance = self.get_spot_balance('USDC')
                print(f"\nCurrent USDC balance: {self.format_decimal(usdc_balance)}")
                
                # Check if we're in a trade
                if self.in_trade:
                    print(f"Currently in trade: {self.current_trade_symbol}")
                    
                    # Check if we've reached the profit target
                    if self.check_profit_target():
                        # Exit the trade
                        if self.place_market_sell_order(self.current_trade_symbol):
                            # Reset trade state
                            self.in_trade = False
                            self.current_trade_symbol = None
                            self.initial_usdc_balance = Decimal('0')
                            self.current_asset_balance = Decimal('0')
                            
                            # Get updated USDC balance
                            usdc_balance = self.get_spot_balance('USDC')
                            print(f"Trade exited. New USDC balance: {self.format_decimal(usdc_balance)}")
                else:
                    print("Not in trade. Scanning for best MTF asset...")
                    
                    # Scan for best MTF asset
                    best_asset = self.gradual_mtf_scan()
                    
                    if best_asset:
                        print(f"\nBest MTF asset found: {best_asset['symbol']}")
                        
                        # Check if we have enough balance to trade
                        if usdc_balance >= Decimal('10'):  # Minimum balance to trade
                            # Enter the trade
                            print(f"Entering trade with {self.format_decimal(usdc_balance)} USDC")
                            
                            # Place buy order
                            if self.place_market_buy_order(best_asset['symbol'], usdc_balance):
                                # Update trade state
                                self.in_trade = True
                                self.current_trade_symbol = best_asset['symbol']
                                self.initial_usdc_balance = usdc_balance
                                
                                # Get asset balance
                                asset = best_asset['symbol'].replace('USDC', '')
                                self.current_asset_balance = self.get_spot_balance(asset)
                                
                                print(f"Trade entered. Asset balance: {self.format_decimal(self.current_asset_balance)}")
                        else:
                            print(f"Insufficient balance to trade: {self.format_decimal(usdc_balance)} USDC")
                    else:
                        print("No suitable MTF asset found")
                
                # Wait for 5 seconds before next iteration
                print("\nWaiting 5 seconds for next iteration...")
                time.sleep(5)
                
                # Clean up memory
                gc.collect()
                
            except KeyboardInterrupt:
                print("\nBot stopped by user")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(5)  # Wait before retrying

def main():
    """Main function to run the analyzer"""
    print("Binance USDC Pairs Gradual MTF Sine Wave Trading Bot")
    print("===================================================")
    
    # Initialize analyzer
    analyzer = BinanceSineAnalyzer()
    
    # Run the trading bot
    analyzer.run_trading_bot()

if __name__ == "__main__":
    main()