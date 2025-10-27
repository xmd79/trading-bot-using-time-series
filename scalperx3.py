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
import requests
import json
from scipy.signal import find_peaks

# Set Decimal precision
getcontext().prec = 25

class BinanceSineAnalyzer:
    def __init__(self, webhook_url=None):
        """Initialize the Binance Sine Analyzer"""
        # Read API credentials from file
        api_key, api_secret = self.read_api_credentials()
        
        self.client = Client(api_key, api_secret, testnet=False)
        # Webhook URL for sending data
        self.webhook_url = webhook_url
        
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
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # Minimum time between requests in seconds
        self.request_weight = 0
        self.max_request_weight = 1200  # Binance's default limit per minute
        self.weight_reset_time = time.time() + 60  # Reset weight every minute
        
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
    
    def rate_limit(self):
        """Implement rate limiting to avoid API bans"""
        current_time = time.time()
        
        # Reset request weight if a minute has passed
        if current_time > self.weight_reset_time:
            self.request_weight = 0
            self.weight_reset_time = current_time + 60
        
        # Check if we're approaching the weight limit
        if self.request_weight >= self.max_request_weight * 0.9:  # 90% of limit
            sleep_time = self.weight_reset_time - current_time
            if sleep_time > 0:
                print(f"Approaching rate limit. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.request_weight = 0
                self.weight_reset_time = time.time() + 60
        
        # Enforce minimum time between requests
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def make_request(self, func, *args, weight=1, **kwargs):
        """Make an API request with rate limiting and error handling"""
        self.rate_limit()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                self.request_weight += weight
                return result
            except Exception as e:
                if hasattr(e, 'code') and e.code == -1003:  # Rate limit error
                    retry_after = 60  # Default retry time
                    if hasattr(e, 'response') and e.response:
                        retry_after = int(e.response.headers.get('Retry-After', 60))
                    
                    print(f"Rate limit exceeded. Waiting {retry_after} seconds before retry...")
                    time.sleep(retry_after)
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Request failed. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Request failed after {max_retries} attempts: {e}")
                    raise
    
    def get_all_usdc_pairs(self):
        """Get all trading pairs with USDC"""
        try:
            exchange_info = self.make_request(self.client.get_exchange_info, weight=10)
            usdc_pairs = [s['symbol'] for s in exchange_info['symbols'] 
                         if s['quoteAsset'] == 'USDC' and s['status'] == 'TRADING']
            return usdc_pairs
        except Exception as e:
            print(f"Error fetching USDC pairs: {e}")
            return []
    
    def get_klines(self, symbol, interval, limit=100):
        """Get kline data for a symbol and interval"""
        try:
            klines = self.make_request(
                self.client.get_klines,
                symbol=symbol,
                interval=interval,
                limit=limit,
                weight=1
            )
            return klines
        except Exception as e:
            print(f"Error fetching klines for {symbol} on {interval}: {e}")
            return []
    
    def calculate_sine_wave_with_fft(self, prices, forecast_periods=5):
        """Calculate sine wave values using FFT and forecast using inverse FFT with improved accuracy"""
        if len(prices) < 20:  # Increased minimum length for better FFT results
            return None, None, None, None, None, None, None, None, None, None
            
        # Convert to numpy array
        prices_array = np.array(prices, dtype=float)
        
        # Find the most significant minima and maxima in the last 1200 values (or all if less than 1200)
        window_size = min(1200, len(prices_array))
        recent_prices = prices_array[-window_size:]
        
        # Find minima and maxima using scipy's find_peaks
        minima_indices, _ = find_peaks(-recent_prices, distance=5)
        maxima_indices, _ = find_peaks(recent_prices, distance=5)
        
        # Get the most recent minima and maxima
        recent_min_idx = minima_indices[-1] if len(minima_indices) > 0 else np.argmin(recent_prices)
        recent_max_idx = maxima_indices[-1] if len(maxima_indices) > 0 else np.argmax(recent_prices)
        
        # Get prices at minima and maxima
        price_at_recent_min = recent_prices[recent_min_idx]
        price_at_recent_max = recent_prices[recent_max_idx]
        
        # Get current close (last price in the array)
        current_close = prices_array[-1]
        
        # Calculate position in range as percentage
        total_range = price_at_recent_max - price_at_recent_min
        if total_range > 0:
            position_from_min = current_close - price_at_recent_min
            pct_from_min = (position_from_min / total_range) * 100
            pct_from_max = 100 - pct_from_min
        else:
            pct_from_min = 50  # Default to middle if range is zero
            pct_from_max = 50
        
        # Detrend the price series for FFT analysis
        x = np.arange(len(prices_array))
        y = prices_array
        
        # Fit a polynomial to detrend (using higher degree for better fit)
        z = np.polyfit(x, y, 5)
        p = np.poly1d(z)
        trend = p(x)
        
        # Detrend
        detrended = y - trend
        
        # Calculate FFT
        fft_values = np.fft.fft(detrended)
        frequencies = np.fft.fftfreq(len(detrended))
        
        # Find significant frequencies (top 5 frequencies by magnitude)
        positive_freq_idx = np.where(frequencies > 0)[0]
        if len(positive_freq_idx) == 0:
            return None, None, None, None, None, None, None, None, None, None
            
        # Get magnitudes and sort to find top frequencies
        magnitudes = np.abs(fft_values[positive_freq_idx])
        sorted_indices = np.argsort(magnitudes)[::-1]  # Descending order
        
        # Take top 5 significant frequencies
        top_n = min(5, len(sorted_indices))
        significant_indices = positive_freq_idx[sorted_indices[:top_n]]
        
        # Create filtered FFT with significant frequencies
        filtered_fft = np.zeros_like(fft_values)
        for idx in significant_indices:
            filtered_fft[idx] = fft_values[idx]
            filtered_fft[-idx] = fft_values[-idx]  # Keep symmetric component
        
        # Reconstruct the signal using inverse FFT
        reconstructed = np.real(np.fft.ifft(filtered_fft))
        
        # Normalize reconstructed signal to match price scale
        reconstructed = reconstructed * np.std(y) / np.std(reconstructed)
        
        # Get the current sine value (last value in the reconstructed signal)
        current_sine_value = reconstructed[-1]
        
        # Forecast using inverse FFT
        # Extend the x array for forecasting
        x_extended = np.arange(len(prices_array), len(prices_array) + forecast_periods)
        
        # Create extended trend
        trend_extended = p(x_extended)
        
        # Create extended reconstructed signal by continuing the pattern
        extended_reconstructed = np.zeros(forecast_periods)
        
        for i in range(forecast_periods):
            # Continue the pattern with phase shift
            for idx in significant_indices:
                freq = frequencies[idx]
                magnitude = np.abs(fft_values[idx])
                phase = np.angle(fft_values[idx])
                extended_reconstructed[i] += magnitude * np.cos(2 * np.pi * freq * (len(prices_array) + i) + phase)
        
        # Normalize extended reconstructed signal
        extended_reconstructed = extended_reconstructed * np.std(y) / np.std(reconstructed)
        
        # Combine extended trend and extended detrended signal
        forecast_prices = trend_extended + extended_reconstructed
        
        # Calculate forecast min, max, and middle
        forecast_min = np.min(forecast_prices)
        forecast_max = np.max(forecast_prices)
        forecast_middle = (forecast_min + forecast_max) / 2
        
        # Calculate distance percentages
        dist_to_min_pct = ((current_close - forecast_min) / current_close) * 100
        dist_to_max_pct = ((forecast_max - current_close) / current_close) * 100
        
        # Only return if forecast shows at least 2% movement
        if abs(dist_to_min_pct) < 2 and abs(dist_to_max_pct) < 2:
            return None, None, None, None, None, None, None, None, None, None
        
        return (price_at_recent_min, price_at_recent_max, current_sine_value, 
                forecast_min, forecast_max, forecast_middle, dist_to_min_pct, dist_to_max_pct,
                pct_from_min, pct_from_max)
    
    def calculate_vwap(self, klines):
        """Calculate Volume Weighted Average Price (VWAP)"""
        if not klines:
            return None
            
        total_volume = 0
        total_volume_price = 0
        
        for kline in klines:
            high = float(kline[2])
            low = float(kline[3])
            close = float(kline[4])
            volume = float(kline[5])
            
            typical_price = (high + low + close) / 3
            total_volume_price += typical_price * volume
            total_volume += volume
        
        if total_volume > 0:
            return total_volume_price / total_volume
        return None
    
    def build_market_profile(self, klines, num_bins=20):
        """Build a simple market profile (volume profile)"""
        if not klines:
            return None
            
        # Extract high, low, close, and volume
        highs = [float(kline[2]) for kline in klines]
        lows = [float(kline[3]) for kline in klines]
        volumes = [float(kline[5]) for kline in klines]
        
        # Determine price range
        min_price = min(lows)
        max_price = max(highs)
        price_range = max_price - min_price
        
        if price_range <= 0:
            return None
            
        # Create price bins
        bin_size = price_range / num_bins
        bins = [min_price + i * bin_size for i in range(num_bins + 1)]
        
        # Initialize volume for each bin
        bin_volumes = [0] * num_bins
        
        # Assign volume to bins
        for i in range(len(klines)):
            high = highs[i]
            low = lows[i]
            volume = volumes[i]
            
            # Determine which bins this candle covers
            start_bin = max(0, min(int((low - min_price) / bin_size), num_bins - 1))
            end_bin = max(0, min(int((high - min_price) / bin_size), num_bins - 1))
            
            # Distribute volume evenly across bins
            if start_bin == end_bin:
                bin_volumes[start_bin] += volume
            else:
                volume_per_bin = volume / (end_bin - start_bin + 1)
                for bin_idx in range(start_bin, end_bin + 1):
                    bin_volumes[bin_idx] += volume_per_bin
        
        # Create profile data
        profile = []
        for i in range(num_bins):
            profile.append({
                'price_low': bins[i],
                'price_high': bins[i + 1],
                'price_mid': (bins[i] + bins[i + 1]) / 2,
                'volume': bin_volumes[i]
            })
        
        # Find Point of Control (POC) - price level with highest volume
        poc_idx = bin_volumes.index(max(bin_volumes))
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Calculate Value Area (price range containing 70% of volume)
        total_volume = sum(bin_volumes)
        target_volume = total_volume * 0.7
        sorted_bins = sorted(enumerate(bin_volumes), key=lambda x: x[1], reverse=True)
        
        value_area_bins = []
        accumulated_volume = 0
        for idx, vol in sorted_bins:
            value_area_bins.append(idx)
            accumulated_volume += vol
            if accumulated_volume >= target_volume:
                break
        
        if value_area_bins:
            value_area_low = bins[min(value_area_bins)]
            value_area_high = bins[max(value_area_bins) + 1]
        else:
            value_area_low = min_price
            value_area_high = max_price
        
        return {
            'profile': profile,
            'poc_price': poc_price,
            'value_area_low': value_area_low,
            'value_area_high': value_area_high,
            'total_volume': total_volume
        }
    
    def send_to_webhook(self, data):
        """Send data to webhook URL"""
        if not self.webhook_url:
            return
            
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                self.webhook_url,
                data=json.dumps(data),
                headers=headers
            )
            if response.status_code == 200:
                print("Data successfully sent to webhook")
            else:
                print(f"Failed to send data to webhook: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error sending data to webhook: {e}")
    
    def analyze_pair_on_tf(self, symbol, tf_name, tf_interval):
        """Analyze a single trading pair on a specific timeframe"""
        try:
            klines = self.get_klines(symbol, tf_interval)
            if not klines:
                return None
                
            # Extract OHLC data
            opens = [float(kline[1]) for kline in klines]
            highs = [float(kline[2]) for kline in klines]
            lows = [float(kline[3]) for kline in klines]
            closes = [float(kline[4]) for kline in klines]
            volumes = [float(kline[5]) for kline in klines]
            
            # Calculate sine wave values with improved FFT forecasting
            sine_values = self.calculate_sine_wave_with_fft(closes)
            if sine_values[0] is None:
                return None
            
            (price_at_recent_min, price_at_recent_max, current_sine_value, 
             forecast_min, forecast_max, forecast_middle, dist_to_min_pct, dist_to_max_pct,
             pct_from_min, pct_from_max) = sine_values
            
            # Calculate VWAP
            vwap = self.calculate_vwap(klines)
            
            # Build market profile
            market_profile = self.build_market_profile(klines)
            
            # Get current close
            current_close = closes[-1]
            
            # Determine if we're in a dip based on sine value and position in range
            # Lower sine value indicates a deeper dip
            # Being closer to the minima (lower pct_from_min) also indicates a dip
            sine_threshold = -0.5  # Adjust based on testing
            position_threshold = 30  # Within 30% of the minima
            
            is_dip = (current_sine_value < sine_threshold) and (pct_from_min < position_threshold)
            
            # Determine cycle phase
            if is_dip:
                cycle_phase = "DIP"
            else:
                cycle_phase = "TOP"
            
            # Get current price and 24h statistics
            ticker = self.make_request(self.client.get_ticker, symbol=symbol, weight=1)
            current_price = float(ticker['lastPrice'])
            quote_volume_24h = float(ticker['quoteVolume'])
            price_change_24h = float(ticker['priceChange'])
            price_change_pct_24h = float(ticker['priceChangePercent'])
            
            # Calculate additional technical indicators
            # Simple Moving Averages
            sma_5 = sum(closes[-5:]) / min(5, len(closes))
            sma_10 = sum(closes[-10:]) / min(10, len(closes))
            sma_20 = sum(closes[-20:]) / min(20, len(closes))
            
            # Exponential Moving Averages
            ema_12 = self.calculate_ema(closes, 12)
            ema_26 = self.calculate_ema(closes, 26)
            
            # MACD
            macd_line = ema_12 - ema_26
            signal_line = self.calculate_ema([macd_line], 9)
            macd_histogram = macd_line - signal_line
            
            # RSI
            rsi = self.calculate_rsi(closes, 14)
            
            # Bollinger Bands
            bb_middle = sma_20
            bb_std = np.std(closes[-20:]) if len(closes) >= 20 else 0
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # Return timeframe data
            return {
                'symbol': symbol,
                'current_price': current_price,
                'quote_volume_24h': quote_volume_24h,
                'price_change_24h': price_change_24h,
                'price_change_pct_24h': price_change_pct_24h,
                'tf_name': tf_name,
                'price_at_recent_min': price_at_recent_min,
                'price_at_recent_max': price_at_recent_max,
                'current_sine_value': current_sine_value,
                'current_close': current_close,
                'total_range': price_at_recent_max - price_at_recent_min,
                'pct_from_min': pct_from_min,
                'pct_from_max': pct_from_max,
                'cycle_phase': cycle_phase,
                'open': opens[-1],
                'high': max(highs),
                'low': min(lows),
                'close': current_close,
                'volume': sum(volumes),
                'forecast_min': forecast_min,
                'forecast_max': forecast_max,
                'forecast_middle': forecast_middle,
                'dist_to_min_pct': dist_to_min_pct,
                'dist_to_max_pct': dist_to_max_pct,
                'vwap': vwap,
                'market_profile': market_profile,
                'is_dip': is_dip,
                'sine_threshold': sine_threshold,
                'position_threshold': position_threshold,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'macd_line': macd_line,
                'signal_line': signal_line,
                'macd_histogram': macd_histogram,
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower
            }
        except Exception as e:
            print(f"Error analyzing {symbol} on {tf_name}: {e}")
            return None
    
    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
            
        multiplier = 2 / (period + 1)
        ema = [sum(prices[:period]) / period]
        
        for price in prices[period:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
            
        return ema[-1]
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50  # Default RSI value
            
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - (100./(1.+rs))
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100. - (100./(1.+rs))
            
        return rsi[-1]
    
    def scan_assets_on_tf(self, symbols, tf_name, tf_interval):
        """Scan a list of assets on a specific timeframe using multithreading"""
        print(f"\nScanning {len(symbols)} assets on {tf_name} timeframe...")
        print("=" * 60)
        
        dip_assets = []
        completed_count = 0
        
        # Reduce number of workers to avoid rate limits
        with ThreadPoolExecutor(max_workers=5) as executor:
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
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
        
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
        
        # Lists to store assets that are dips on each timeframe
        daily_dips = []
        twelve_hour_dips = []
        eight_hour_dips = []
        six_hour_dips = []
        four_hour_dips = []
        two_hour_dips = []
        one_hour_dips = []
        thirty_min_dips = []
        fifteen_min_dips = []
        five_min_dips = []
        three_min_dips = []
        one_min_dips = []
        
        # Scan each timeframe from highest to lowest
        for i, (tf_name, tf_interval) in enumerate(self.timeframes):
            print(f"\n{'='*20} SCANNING {tf_name} TIMEFRAME {'='*20}")
            
            # If this is the first timeframe, scan all assets
            if i == 0:
                dip_results = self.scan_assets_on_tf(current_assets, tf_name, tf_interval)
                daily_dips = [result['symbol'] for result in dip_results]
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
                        'price_change_24h': result['price_change_24h'],
                        'price_change_pct_24h': result['price_change_pct_24h'],
                        'timeframes': {}
                    }
                
                asset_results[symbol]['timeframes'][tf_name] = result
            
            # Store dips for each timeframe
            if tf_name == '12h':
                twelve_hour_dips = current_assets
            elif tf_name == '8h':
                eight_hour_dips = current_assets
            elif tf_name == '6h':
                six_hour_dips = current_assets
            elif tf_name == '4h':
                four_hour_dips = current_assets
            elif tf_name == '2h':
                two_hour_dips = current_assets
            elif tf_name == '1h':
                one_hour_dips = current_assets
            elif tf_name == '30min':
                thirty_min_dips = current_assets
            elif tf_name == '15min':
                fifteen_min_dips = current_assets
            elif tf_name == '5min':
                five_min_dips = current_assets
            elif tf_name == '3min':
                three_min_dips = current_assets
            elif tf_name == '1min':
                one_min_dips = current_assets
            
            # If no assets left, break early
            if not current_assets:
                print(f"\nNo assets in DIP on {tf_name}. Stopping scan.")
                break
            
            print(f"\nAssets remaining for next timeframe: {len(current_assets)}")
        
        # Print summary of dips found at each timeframe
        print("\n" + "="*60)
        print("SUMMARY OF DIPS FOUND AT EACH TIMEFRAME")
        print("="*60)
        print(f"Daily (24h) dips: {len(daily_dips)} assets")
        print(f"12h dips: {len(twelve_hour_dips)} assets")
        print(f"8h dips: {len(eight_hour_dips)} assets")
        print(f"6h dips: {len(six_hour_dips)} assets")
        print(f"4h dips: {len(four_hour_dips)} assets")
        print(f"2h dips: {len(two_hour_dips)} assets")
        print(f"1h dips: {len(one_hour_dips)} assets")
        print(f"30min dips: {len(thirty_min_dips)} assets")
        print(f"15min dips: {len(fifteen_min_dips)} assets")
        print(f"5min dips: {len(five_min_dips)} assets")
        print(f"3min dips: {len(three_min_dips)} assets")
        print(f"1min dips: {len(one_min_dips)} assets")
        
        # Convert the dictionary to a list
        final_results = list(asset_results.values())
        
        # Sort by sine value (lower is better for a dip) and then by position from minima
        if final_results and '24h' in final_results[0]['timeframes']:
            final_results.sort(key=lambda x: (
                x['timeframes']['24h']['current_sine_value'],  # Lower sine value is better
                x['timeframes']['24h']['pct_from_min']  # Lower percentage from min is better
            ))
        
        # Return the best asset if we have results
        if final_results:
            return final_results[0]
        return None
    
    def print_technical_data(self, asset_data):
        """Print comprehensive technical data for the best MTF asset"""
        if not asset_data:
            print("No asset data to display")
            return
            
        print("\n" + "="*60)
        print("COMPREHENSIVE TECHNICAL DATA FOR BEST MTF ASSET")
        print("="*60)
        print(f"Symbol: {asset_data['symbol']}")
        print(f"Current Price: {self.format_decimal(asset_data['current_price'])}")
        print(f"24h Volume: {self.format_decimal(asset_data['quote_volume_24h'])}")
        print(f"24h Price Change: {self.format_decimal(asset_data['price_change_24h'])} ({self.format_decimal(asset_data['price_change_pct_24h'])}%)")
        
        # Print data for each timeframe
        for tf_name, tf_data in asset_data['timeframes'].items():
            print(f"\n{'='*20} {tf_name} TIMEFRAME {'='*20}")
            print(f"OHLC Data:")
            print(f"  Open: {self.format_decimal(tf_data['open'])}")
            print(f"  High: {self.format_decimal(tf_data['high'])}")
            print(f"  Low: {self.format_decimal(tf_data['low'])}")
            print(f"  Close: {self.format_decimal(tf_data['close'])}")
            print(f"  Volume: {self.format_decimal(tf_data['volume'])}")
            
            print(f"\nMinima/Maxima Analysis:")
            print(f"  Recent Minima Price: {self.format_decimal(tf_data['price_at_recent_min'])}")
            print(f"  Recent Maxima Price: {self.format_decimal(tf_data['price_at_recent_max'])}")
            print(f"  Total Range: {self.format_decimal(tf_data['total_range'])}")
            print(f"  Position from Minima: {self.format_decimal(tf_data['pct_from_min'])}%")
            print(f"  Position from Maxima: {self.format_decimal(tf_data['pct_from_max'])}%")
            print(f"  Current Sine Value: {self.format_decimal(tf_data['current_sine_value'])}")
            print(f"  Sine Threshold: {self.format_decimal(tf_data['sine_threshold'])}")
            print(f"  Position Threshold: {self.format_decimal(tf_data['position_threshold'])}%")
            print(f"  Cycle Phase: {tf_data['cycle_phase']}")
            
            print(f"\nForecast Analysis:")
            print(f"  Forecast Min: {self.format_decimal(tf_data['forecast_min'])}")
            print(f"  Forecast Max: {self.format_decimal(tf_data['forecast_max'])}")
            print(f"  Forecast Middle: {self.format_decimal(tf_data['forecast_middle'])}")
            print(f"  Distance to Min (%): {self.format_decimal(tf_data['dist_to_min_pct'])}%")
            print(f"  Distance to Max (%): {self.format_decimal(tf_data['dist_to_max_pct'])}%")
            
            print(f"\nTechnical Indicators:")
            print(f"  VWAP: {self.format_decimal(tf_data['vwap']) if tf_data['vwap'] else 'N/A'}")
            print(f"  SMA 5: {self.format_decimal(tf_data['sma_5'])}")
            print(f"  SMA 10: {self.format_decimal(tf_data['sma_10'])}")
            print(f"  SMA 20: {self.format_decimal(tf_data['sma_20'])}")
            print(f"  EMA 12: {self.format_decimal(tf_data['ema_12'])}")
            print(f"  EMA 26: {self.format_decimal(tf_data['ema_26'])}")
            print(f"  MACD Line: {self.format_decimal(tf_data['macd_line'])}")
            print(f"  Signal Line: {self.format_decimal(tf_data['signal_line'])}")
            print(f"  MACD Histogram: {self.format_decimal(tf_data['macd_histogram'])}")
            print(f"  RSI: {self.format_decimal(tf_data['rsi'])}")
            print(f"  Bollinger Upper: {self.format_decimal(tf_data['bb_upper'])}")
            print(f"  Bollinger Middle: {self.format_decimal(tf_data['bb_middle'])}")
            print(f"  Bollinger Lower: {self.format_decimal(tf_data['bb_lower'])}")
            
            # Print market profile data if available
            if tf_data['market_profile']:
                mp = tf_data['market_profile']
                print(f"\nMarket Profile:")
                print(f"  Point of Control (POC): {self.format_decimal(mp['poc_price'])}")
                print(f"  Value Area Low: {self.format_decimal(mp['value_area_low'])}")
                print(f"  Value Area High: {self.format_decimal(mp['value_area_high'])}")
                print(f"  Total Volume: {self.format_decimal(mp['total_volume'])}")
                
                # Print top 5 price levels by volume
                print("\n  Top 5 Price Levels by Volume:")
                sorted_profile = sorted(mp['profile'], key=lambda x: x['volume'], reverse=True)[:5]
                for i, level in enumerate(sorted_profile, 1):
                    print(f"  {i}. Price: {self.format_decimal(level['price_mid'])}, Volume: {self.format_decimal(level['volume'])}")
    
    def get_spot_balance(self, asset='USDC'):
        """Get spot balance for a specific asset"""
        try:
            balance_info = self.make_request(self.client.get_asset_balance, asset=asset, weight=2)
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
            ticker = self.make_request(self.client.get_symbol_ticker, symbol=symbol, weight=1)
            current_price = float(ticker['price'])
            
            # Calculate quantity to buy (accounting for precision)
            quantity = usdc_amount / Decimal(str(current_price))
            
            # Get symbol info to determine precision
            symbol_info = self.make_request(self.client.get_symbol_info, symbol=symbol, weight=2)
            filters = symbol_info['filters']
            lot_size_filter = next(filter(lambda f: f['filterType'] == 'LOT_SIZE', filters), None)
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                # Round quantity to the appropriate step size
                quantity = self.round_step_size(quantity, step_size)
            
            # Place the order
            print(f"Placing market buy order for {symbol}: {self.format_decimal(quantity)} at ~{self.format_decimal(current_price)}")
            order = self.make_request(
                self.client.create_order,
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(quantity),
                weight=5
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
            symbol_info = self.make_request(self.client.get_symbol_info, symbol=symbol, weight=2)
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
            order = self.make_request(
                self.client.create_order,
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(quantity),
                weight=5
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
            ticker = self.make_request(self.client.get_symbol_ticker, symbol=self.current_trade_symbol, weight=1)
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
        print("Bot will scan for MTF dip assets every 30 seconds")
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
                        
                        # Print comprehensive technical data
                        self.print_technical_data(best_asset)
                        
                        # Send data to webhook if URL is provided
                        if self.webhook_url:
                            self.send_to_webhook(best_asset)
                        
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
                
                # Wait for 30 seconds before next iteration (increased from 5)
                print("\nWaiting 30 seconds for next iteration...")
                time.sleep(30)
                
                # Clean up memory
                gc.collect()
                
            except KeyboardInterrupt:
                print("\nBot stopped by user")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(30)  # Wait before retrying

def main():
    """Main function to run the analyzer"""
    print("Binance USDC Pairs Gradual MTF Sine Wave Trading Bot")
    print("===================================================")
    
    # Initialize analyzer with optional webhook URL
    # Replace with your actual webhook URL if needed
    webhook_url = None  # "https://your-webhook-url.com"
    analyzer = BinanceSineAnalyzer(webhook_url=webhook_url)
    
    # Run the trading bot
    analyzer.run_trading_bot()

if __name__ == "__main__":
    main()