import pandas as pd
from binance import Client, BinanceAPIException
import numpy as np
import time
from datetime import datetime
import os
import json
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import warnings

# Try to import TA-Lib
try:
    import talib
    from talib import abstract
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not installed. Sine wave comparison will be skipped.")

@dataclass
class TimeframeAnalysis:
    """Data class to store timeframe analysis results"""
    timeframe: str
    current_close: float
    min_low: float
    max_high: float
    min_threshold: float
    max_threshold: float
    middle_threshold: float
    low_idx: int
    high_idx: int
    proximity_to_low: float
    near_low: bool
    above_low: bool
    low_recent: bool
    valid: bool

@dataclass
class AssetAnalysis:
    """Data class to store asset analysis results"""
    symbol: str
    timeframe_results: Dict[str, TimeframeAnalysis]
    valid_tfs_count: int
    passed_all_tfs: bool
    sine_value: Optional[float] = None

class BinanceTradingBot:
    def __init__(self, api_file: str = 'api.txt'):
        """Initialize the trading bot with API credentials from file."""
        self.api_key, self.api_secret = self._load_api_credentials(api_file)
        self.client = Client(self.api_key, self.api_secret)
        self.timeframes = ['2h', '1h', '30m', '15m', '5m', '3m', '1m']
        self.lookback = 1200
        self.proximity_threshold = 0.02  # 2% proximity to low
        self.max_workers = 10  # Max concurrent threads
        self.results = {}
        self.top_assets = []
        self.best_mtf_dip = None
        self.lock = threading.Lock()
        self.usdc_balance = 0.0
        
    def _load_api_credentials(self, api_file: str) -> Tuple[str, str]:
        """Load API credentials from file."""
        if not os.path.exists(api_file):
            raise FileNotFoundError(f"API file {api_file} not found")
        
        with open(api_file, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                raise ValueError("API file must contain at least 2 lines (API key and secret)")
            api_key = lines[0].strip()
            api_secret = lines[1].strip()
            
        return api_key, api_secret
    
    def get_usdc_balance(self) -> float:
        """Get USDC spot balance."""
        try:
            account_info = self.client.get_account()
            for balance in account_info['balances']:
                if balance['asset'] == 'USDC':
                    return float(balance['free'])
            return 0.0
        except BinanceAPIException as e:
            print(f"Error fetching USDC balance: {e}")
            return 0.0
    
    def get_usdc_pairs(self) -> List[str]:
        """Get all USDC trading pairs on Binance spot."""
        try:
            exchange_info = self.client.get_exchange_info()
            usdc_pairs = [
                s['symbol'] for s in exchange_info['symbols'] 
                if s['quoteAsset'] == 'USDC' 
                and s['status'] == 'TRADING' 
                and s['isSpotTradingAllowed']
            ]
            return usdc_pairs
        except BinanceAPIException as e:
            print(f"Error fetching exchange info: {e}")
            return []
    
    def fetch_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a symbol and timeframe."""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=self.lookback
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_vol', 'trades',
                'taker_buy_vol', 'taker_buy_quote_vol', 'ignore'
            ])
            
            # Convert to numeric and set timestamp
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except BinanceAPIException as e:
            print(f"Error fetching data for {symbol} {timeframe}: {e}")
            return None
    
    def clean_data_for_talib(self, data: np.ndarray) -> np.ndarray:
        """Clean data for TA-Lib by completely removing NaN and zero values."""
        # Convert to pandas Series for easier handling
        series = pd.Series(data)
        
        # Remove zeros and NaN values
        series = series.replace(0, np.nan)  # Replace zeros with NaN
        series = series.dropna()  # Remove all NaN values
        
        # If we have no data left, return a small array with default values
        if len(series) == 0:
            return np.array([1.0, 1.0001, 0.9999])  # Small variation
        
        # If we have less than 3 values, pad with slight variations
        if len(series) < 3:
            base_value = series.iloc[0]
            return np.array([base_value, base_value * 1.0001, base_value * 0.9999])
        
        # Return as numpy array
        return series.values
    
    def calculate_ht_sine(self, symbol: str) -> Optional[float]:
        """Calculate HT_SINE value for an asset using 1min timeframe."""
        if not TALIB_AVAILABLE:
            return None
            
        try:
            # Use 1min timeframe as requested
            df = self.fetch_ohlcv(symbol, '1m')
            if df is None or df.empty:
                print(f"⚠️ No 1m data for {symbol}, trying 5m")
                df = self.fetch_ohlcv(symbol, '5m')
                if df is None or df.empty:
                    print(f"⚠️ No 5m data for {symbol}, trying 15m")
                    df = self.fetch_ohlcv(symbol, '15m')
                    if df is None or df.empty:
                        print(f"⚠️ No data available for {symbol}")
                        return None
            
            # Extract close prices
            close_prices = df['close'].values
            
            # Clean the data completely
            close_prices_clean = self.clean_data_for_talib(close_prices)
            
            # Ensure we have enough data for HT_SINE (minimum 32 values)
            if len(close_prices_clean) < 32:
                print(f"⚠️ Not enough clean data for {symbol} after cleaning")
                return None
            
            # Calculate HT_SINE
            sine, leadsine = talib.HT_SINE(close_prices_clean)
            
            # Remove NaN values from the result
            sine_clean = sine[~np.isnan(sine)]
            
            if len(sine_clean) == 0:
                print(f"⚠️ No valid sine values for {symbol}")
                return None
            
            # Return the last valid sine value
            last_sine = sine_clean[-1]
            return last_sine
            
        except Exception as e:
            print(f"Error calculating HT_SINE for {symbol}: {e}")
            return None
    
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Optional[TimeframeAnalysis]:
        """Analyze a single timeframe for extrema and conditions."""
        if df is None or df.empty:
            return None
        
        # Find extrema indices and values
        low_idx = df['low'].argmin()
        high_idx = df['high'].argmax()
        
        # Current price
        current_close = df['close'].iloc[-1]
        min_low = df['low'].min()
        max_high = df['high'].max()
        
        # Calculate thresholds
        min_threshold = min_low
        max_threshold = max_high
        middle_threshold = (min_low + max_high) / 2
        
        # Calculate proximity to low (percentage)
        proximity_to_low = (current_close - min_low) / min_low if min_low > 0 else float('inf')
        
        # Check conditions
        near_low = proximity_to_low <= self.proximity_threshold
        above_low = current_close > min_low
        low_recent = low_idx > high_idx
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            current_close=current_close,
            min_low=min_low,
            max_high=max_high,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            middle_threshold=middle_threshold,
            low_idx=low_idx,
            high_idx=high_idx,
            proximity_to_low=proximity_to_low,
            near_low=near_low,
            above_low=above_low,
            low_recent=low_recent,
            valid=near_low and above_low and low_recent
        )
    
    def analyze_asset_timeframe(self, symbol: str, timeframe: str) -> Optional[TimeframeAnalysis]:
        """Analyze a single asset for a specific timeframe."""
        df = self.fetch_ohlcv(symbol, timeframe)
        return self.analyze_timeframe(df, timeframe)
    
    def hierarchical_scan(self) -> None:
        """Perform hierarchical MTF scan starting from 2h down to 1min."""
        print("\n🔍 Starting Hierarchical MTF Dip Scan")
        print("=" * 60)
        
        # Get USDC balance and pairs
        self.usdc_balance = self.get_usdc_balance()
        print(f"💰 USDC Spot Balance: {self.usdc_balance}")
        
        all_pairs = self.get_usdc_pairs()
        print(f"📊 Total USDC Trading Pairs: {len(all_pairs)}")
        
        # Start with all pairs for 2h scan
        current_assets = all_pairs
        scan_results = {}
        
        # Process each timeframe sequentially
        for i, tf in enumerate(self.timeframes):
            print(f"\n⏱️ Scanning {tf} timeframe...")
            print("-" * 60)
            
            # If no assets passed previous scan, break
            if not current_assets:
                print(f"No assets passed {self.timeframes[i-1]} scan. Stopping.")
                break
            
            # Process assets in parallel for current timeframe
            tf_results = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.analyze_asset_timeframe, symbol, tf): symbol 
                    for symbol in current_assets
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        if result and result.valid:
                            tf_results[symbol] = result
                            print(f"✅ {symbol}: Valid dip found on {tf}")
                        else:
                            print(f"❌ {symbol}: No valid dip on {tf}")
                    except Exception as e:
                        print(f"⚠️ Error processing {symbol} on {tf}: {e}")
            
            # Update scan results
            for symbol, result in tf_results.items():
                if symbol not in scan_results:
                    scan_results[symbol] = {}
                scan_results[symbol][tf] = result
            
            # Update current_assets for next timeframe
            current_assets = list(tf_results.keys())
            print(f"\n📈 Assets passing {tf} scan: {len(current_assets)}")
        
        # Process final results
        self.process_scan_results(scan_results)
    
    def process_scan_results(self, scan_results: Dict) -> None:
        """Process and rank the scan results."""
        print("\n📊 Processing Scan Results")
        print("=" * 60)
        
        # Convert scan results to AssetAnalysis objects
        for symbol, tf_data in scan_results.items():
            valid_tfs_count = len(tf_data)
            passed_all_tfs = valid_tfs_count == len(self.timeframes)
            
            # Calculate HT_SINE for tie-breaking
            sine_value = self.calculate_ht_sine(symbol)
            
            asset_analysis = AssetAnalysis(
                symbol=symbol,
                timeframe_results=tf_data,
                valid_tfs_count=valid_tfs_count,
                passed_all_tfs=passed_all_tfs,
                sine_value=sine_value
            )
            
            self.results[symbol] = asset_analysis
        
        # Rank assets by number of valid timeframes
        self.top_assets = sorted(
            self.results.values(), 
            key=lambda x: (x.valid_tfs_count, x.passed_all_tfs), 
            reverse=True
        )
        
        # Find best MTF dip using sine wave comparison if needed
        self.find_best_mtf_dip()
        
        print(f"✅ Analysis complete. Found {len(self.top_assets)} assets with MTF dips.")
    
    def find_best_mtf_dip(self) -> None:
        """Find the best MTF dip using sine wave comparison for ties."""
        if not self.top_assets:
            self.best_mtf_dip = None
            return
        
        # Group assets by valid_tfs_count
        groups = {}
        for asset in self.top_assets:
            count = asset.valid_tfs_count
            if count not in groups:
                groups[count] = []
            groups[count].append(asset)
        
        # Find the group with the highest count
        max_count = max(groups.keys())
        best_group = groups[max_count]
        
        if len(best_group) == 1:
            self.best_mtf_dip = best_group[0]
        else:
            # Multiple assets with same number of valid timeframes
            if TALIB_AVAILABLE:
                # Filter assets with valid sine values
                valid_sine_assets = [a for a in best_group if a.sine_value is not None]
                
                if valid_sine_assets:
                    # Find asset with the lowest sine value (most oversold)
                    self.best_mtf_dip = min(valid_sine_assets, key=lambda x: x.sine_value)
                    print(f"\n🌊 Used sine wave comparison to select best asset")
                    print(f"   {self.best_mtf_dip.symbol}: Sine value = {self.best_mtf_dip.sine_value:.4f}")
                else:
                    # Fallback to first asset if no valid sine values
                    self.best_mtf_dip = best_group[0]
                    print("\n⚠️ No valid sine values. Selected first asset in group.")
            else:
                # Without TA-Lib, pick the first asset
                self.best_mtf_dip = best_group[0]
                print("\n⚠️ TA-Lib not available. Selected first asset in group.")
    
    def display_scan_summary(self) -> None:
        """Display a summary of the scan results."""
        print("\n📋 MTF DIP SCAN SUMMARY")
        print("=" * 80)
        print(f"USDC Balance: {self.usdc_balance}")
        print(f"Timeframes Scanned: {', '.join(self.timeframes)}")
        print(f"Total Assets with Dips: {len(self.top_assets)}")
        
        # Count assets by number of valid timeframes
        tf_counts = {}
        for asset in self.top_assets:
            count = asset.valid_tfs_count
            if count not in tf_counts:
                tf_counts[count] = 0
            tf_counts[count] += 1
        
        print("\nAssets by Valid Timeframes:")
        for count in sorted(tf_counts.keys(), reverse=True):
            print(f"  {count}/{len(self.timeframes)} TFs: {tf_counts[count]} assets")
        
        print("\nTop Assets by Valid Timeframes:")
        print(f"{'Symbol':<12} {'Valid TFs':<12} {'Passed All':<12} {'Sine Value':<12}")
        print("-" * 50)
        
        for asset in self.top_assets[:10]:
            sine_str = f"{asset.sine_value:.4f}" if asset.sine_value is not None else "N/A"
            print(f"{asset.symbol:<12} {asset.valid_tfs_count}/{len(self.timeframes):<12} "
                  f"{str(asset.passed_all_tfs):<12} {sine_str:<12}")
        
        print("=" * 80)
    
    def display_best_mtf_dip(self) -> None:
        """Display the best MTF dip found."""
        if not self.best_mtf_dip:
            print("\nNo MTF dip found meeting criteria.")
            return
        
        print("\n🎯 BEST MTF DIP FOUND 🎯")
        print("=" * 100)
        print(f"Symbol: {self.best_mtf_dip.symbol}")
        print(f"Valid Timeframes: {self.best_mtf_dip.valid_tfs_count}/{len(self.timeframes)}")
        print(f"Passed All Timeframes: {self.best_mtf_dip.passed_all_tfs}")
        if self.best_mtf_dip.sine_value is not None:
            print(f"HT_SINE Value: {self.best_mtf_dip.sine_value:.4f}")
        print("\nTimeframe Analysis:")
        print(f"{'TF':<8} {'Current':<12} {'Min Low':<12} {'Max High':<12} {'Min Thr':<12} {'Max Thr':<12} {'Mid Thr':<12} {'Proximity':<12}")
        print("-" * 100)
        
        for tf, analysis in self.best_mtf_dip.timeframe_results.items():
            print(f"{tf:<8} {analysis.current_close:<12.8f} {analysis.min_low:<12.8f} "
                  f"{analysis.max_high:<12.8f} {analysis.min_threshold:<12.8f} "
                  f"{analysis.max_threshold:<12.8f} {analysis.middle_threshold:<12.8f} "
                  f"{analysis.proximity_to_low*100:<12.4f}%")
        
        print("\nThreshold Analysis:")
        print(f"{'TF':<8} {'Min Thr':<12} {'Max Thr':<12} {'Mid Thr':<12} {'Current':<12} {'Position':<15}")
        print("-" * 80)
        
        for tf, analysis in self.best_mtf_dip.timeframe_results.items():
            # Determine position relative to thresholds
            if analysis.current_close < analysis.min_threshold:
                position = "Below Min"
            elif analysis.current_close > analysis.max_threshold:
                position = "Above Max"
            elif analysis.current_close < analysis.middle_threshold:
                position = "Lower Half"
            else:
                position = "Upper Half"
            
            print(f"{tf:<8} {analysis.min_threshold:<12.8f} {analysis.max_threshold:<12.8f} "
                  f"{analysis.middle_threshold:<12.8f} {analysis.current_close:<12.8f} {position:<15}")
        
        print("=" * 100)

def main():
    """Main function to run the trading bot."""
    print("🚀 Binance Trading Bot - Hierarchical MTF Dip Scanner with Sine Wave Analysis")
    print("="*90)
    
    try:
        # Initialize the bot
        bot = BinanceTradingBot()
        
        # Perform hierarchical scan
        start_time = time.time()
        bot.hierarchical_scan()
        elapsed = time.time() - start_time
        
        print(f"\n⏱️ Scan completed in {elapsed:.2f} seconds")
        
        # Display results
        bot.display_scan_summary()
        bot.display_best_mtf_dip()
        
    except Exception as e:
        print(f"❌ Error running trading bot: {e}")


if __name__ == "__main__":
    main()