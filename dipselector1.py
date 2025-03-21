from binance.client import Client
import numpy as np
import pandas as pd
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        trading_pairs = [symbol['symbol'] for symbol in exchange_info['symbols'] 
                         if symbol['quoteAsset'] == 'USDC' and symbol['status'] == 'TRADING']
        return trading_pairs

def find_major_reversals(candles, current_close, min_threshold, max_threshold):
    lows = [float(candle[3]) for candle in candles if float(candle[3]) >= min_threshold]
    highs = [float(candle[2]) for candle in candles if float(candle[2]) <= max_threshold]
    
    last_bottom = np.nanmin(lows) if lows else None
    last_top = np.nanmax(highs) if highs else None
    
    closest_reversal = None
    closest_type = None
    
    if last_bottom and (closest_reversal is None or abs(last_bottom - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_bottom
        closest_type = 'DIP'
    
    if last_top and (closest_reversal is None or abs(last_top - current_close) < abs(closest_reversal - current_close)):
        closest_reversal = last_top
        closest_type = 'TOP'
    
    if closest_type == 'TOP' and closest_reversal <= current_close:
        closest_type = None
        closest_reversal = None
    elif closest_type == 'DIP' and closest_reversal >= current_close:
        closest_type = None
        closest_reversal = None
    
    return last_bottom, last_top, closest_reversal, closest_type

def check_dip(trader, symbol, interval):
    try:
        klines = trader.client.get_klines(symbol=symbol, interval=interval, limit=1000)
        if not klines:
            logging.debug(f"No data returned for {symbol} on {interval} (empty klines)")
            return False
        
        close = [float(entry[4]) for entry in klines]
        if len(close) < 2:
            logging.debug(f"Insufficient data for {symbol} on {interval}: {len(close)} candles")
            return False
        
        if any(np.isnan(c) or np.isinf(c) for c in close):
            logging.warning(f"Invalid data (NaN/Inf) for {symbol} on {interval}")
            return False
        
        if all(c == close[0] for c in close):
            logging.warning(f"Degenerate data (all identical) for {symbol} on {interval}")
            return False
        
        x = close
        y = range(len(x))
        best_fit_line1 = np.poly1d(np.polyfit(y, x, 1))(y)
        best_fit_line3 = best_fit_line1 * 0.99
        
        return x[-1] < best_fit_line3[-1]
    
    except Exception as e:
        logging.error(f"Error in check_dip for {symbol} on {interval}: {str(e)}")
        return False

def find_largest_timeframe_with_data(trader, symbol):
    timeframes = [
        ('1d', 'Daily'),
        ('4h', '4h'),
        ('1h', '1h'),
        ('15m', '15m'),
        ('5m', '5m'),
        ('1m', '1m')
    ]
    for tf, name in timeframes:
        klines = trader.client.get_klines(symbol=symbol, interval=tf, limit=1000)
        close = [float(entry[4]) for entry in klines] if klines else []
        if len(close) >= 2:
            logging.info(f"Found data for {symbol} on {name} ({len(close)} candles)")
            return tf, name, timeframes[timeframes.index((tf, name)):]
        logging.warning(f"No or insufficient data for {symbol} on {name}")
    return None, None, []  # No timeframe with data

def analyze_symbol(trader, symbol):
    results = {'symbol': symbol}
    
    # Find the largest timeframe with data and remaining timeframes
    largest_tf, largest_tf_name, active_timeframes = find_largest_timeframe_with_data(trader, symbol)
    if not largest_tf:
        logging.warning(f"No usable data found for {symbol} on any timeframe")
        results.update({f'{name}_dip': False for _, name in active_timeframes})
        results['mtf_dip'] = False
        results['largest_tf'] = 'None'
        return results
    
    results['largest_tf'] = largest_tf_name
    
    # Check dips from the largest timeframe with data down to 1m
    for tf, name in active_timeframes:
        results[f'{name}_dip'] = check_dip(trader, symbol, tf)
    
    # Fill in higher timeframes (if any) as False
    all_timeframes = {'Daily', '4h', '1h', '15m', '5m', '1m'}
    active_names = {name for _, name in active_timeframes}
    for name in all_timeframes - active_names:
        results[f'{name}_dip'] = False
    
    # MTF dip confirmation on 1m
    if results['1m_dip']:
        try:
            klines = trader.client.get_klines(symbol=symbol, interval='1m', limit=1000)
            close = [float(entry[4]) for entry in klines]
            current_close = close[-1]
            min_threshold = current_close * 0.8
            max_threshold = current_close * 1.2
            
            last_bottom, last_top, closest_reversal, closest_type = find_major_reversals(
                klines, current_close, min_threshold, max_threshold
            )
            
            if closest_type == 'DIP':
                price_range = last_top - last_bottom if last_top and last_bottom else 0
                dist_to_min = abs(current_close - last_bottom) if last_bottom else float('inf')
                dist_to_max = abs(current_close - last_top) if last_top else float('inf')
                perc_to_min = ((current_close - last_bottom) / price_range * 100) if price_range > 0 else float('inf')
                perc_to_max = ((last_top - current_close) / price_range * 100) if price_range > 0 else float('inf')
                
                results.update({
                    'last_bottom': last_bottom,
                    'last_top': last_top,
                    'dist_to_min': dist_to_min,
                    'dist_to_max': dist_to_max,
                    'perc_to_min': perc_to_min,
                    'perc_to_max': perc_to_max,
                    'mtf_dip': True
                })
            else:
                results['mtf_dip'] = False
        except Exception as e:
            logging.error(f"Error in 1m analysis for {symbol}: {str(e)}")
            results['mtf_dip'] = False
    else:
        results['mtf_dip'] = False
    
    return results

def main():
    filename = 'credentials.txt'
    trader = Trader(filename)
    trading_pairs = trader.get_usdc_pairs()
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(analyze_symbol, trader, symbol): symbol 
                          for symbol in trading_pairs}
        for future in as_completed(future_to_symbol):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Thread error for {future_to_symbol[future]}: {str(e)}")
    
    df = pd.DataFrame(results)
    df['dip_count'] = df[['Daily_dip', '4h_dip', '1h_dip', '15m_dip', '5m_dip', '1m_dip']].sum(axis=1)
    
    mtf_dips = df[df['mtf_dip']].sort_values(by=['dip_count', 'perc_to_min'], 
                                            ascending=[False, True])
    
    print("\n=== All Analyzed Pairs ===")
    print(df[['symbol', 'largest_tf', 'Daily_dip', '4h_dip', '1h_dip', '15m_dip', '5m_dip', '1m_dip', 'dip_count', 'mtf_dip']])
    
    if not mtf_dips.empty:
        print("\n=== MTF Dip Analysis ===")
        print(mtf_dips[['symbol', 'largest_tf', 'dip_count', 'perc_to_min', 'perc_to_max', 
                       'dist_to_min', 'dist_to_max', 'last_bottom', 'last_top']])
        
        best_dip = mtf_dips.iloc[0]
        print(f"\nBest MTF Dip: {best_dip['symbol']}")
        print(f"Largest timeframe with data: {best_dip['largest_tf']}")
        print(f"Dip on {int(best_dip['dip_count'])} timeframes")
        print(f"Distance to min: {best_dip['dist_to_min']:.25f} ({best_dip['perc_to_min']:.2f}%)")
        print(f"Distance to max: {best_dip['dist_to_max']:.25f} ({best_dip['perc_to_max']:.2f}%)")
        print(f"Last bottom: {best_dip['last_bottom']:.25f}")
        print(f"Last top: {best_dip['last_top']:.25f}")
    else:
        print("\nNo MTF dips found")

if __name__ == "__main__":
    main()
    sys.exit(0)