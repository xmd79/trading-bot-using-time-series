import numpy as np
import talib
import requests
from datetime import datetime
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from scipy.fft import fft, ifft

# Load credentials from file
with open("credentials.txt", "r") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    api_secret = lines[1].strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

# Define a function to get candles
def get_candles(symbol, timeframe, limit=1000):
    try:
        klines = client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        candles = [{
            "time": k[0] / 1000,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        } for k in klines]
        return candles
    except BinanceAPIException as e:
        print(f"Error fetching candles for {symbol} at {timeframe}: {e}")
        return []

# Define functions for indicators
def calculate_vwap(candles):
    close_prices = np.array([c["close"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    return np.sum(close_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else np.nan

def calculate_ema(candles, timeperiod):
    close_prices = np.array([c["close"] for c in candles])
    return talib.EMA(close_prices, timeperiod=timeperiod)

def calculate_rsi(candles, timeperiod=14):
    close_prices = np.array([c["close"] for c in candles])
    return talib.RSI(close_prices, timeperiod=timeperiod)

def calculate_macd(candles, fastperiod=12, slowperiod=26, signalperiod=9):
    close_prices = np.array([c["close"] for c in candles])
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    return macd, macdsignal, macdhist

def calculate_momentum(candles, timeperiod=10):
    close_prices = np.array([c["close"] for c in candles])
    return talib.MOM(close_prices, timeperiod=timeperiod)

def calculate_regression_channels(candles):
    if len(candles) < 50:
        return np.nan, np.nan
    close_prices = np.array([c["close"] for c in candles])
    x = np.arange(len(close_prices))
    coeffs = np.polyfit(x, close_prices, 1)
    regression_line = np.polyval(coeffs, x)
    deviation = close_prices - regression_line
    regression_upper = regression_line + np.std(deviation)
    regression_lower = regression_line - np.std(deviation)
    return regression_lower[-1], regression_upper[-1]

def calculate_fibonacci_retracement(high, low):
    diff = high - low
    return {
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "61.8%": high - 0.618 * diff
    }

def calculate_fft(candles):
    close_prices = np.array([c["close"] for c in candles])
    fft_result = fft(close_prices)
    return fft_result

def calculate_ifft(fft_result):
    return ifft(fft_result).real

def calculate_all_talib_indicators(candles):
    results = {}
    close_prices = np.array([c["close"] for c in candles])
    high_prices = np.array([c["high"] for c in candles])
    low_prices = np.array([c["low"] for c in candles])
    open_prices = np.array([c["open"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])

    # Overlap Studies
    results['HT_TRENDLINE'] = talib.HT_TRENDLINE(close_prices)
    results['SAR'] = talib.SAR(high_prices, low_prices)
    results['SMA'] = talib.SMA(close_prices)
    results['T3'] = talib.T3(close_prices)
    results['TRIMA'] = talib.TRIMA(close_prices)
    results['WMA'] = talib.WMA(close_prices)

    # Momentum Indicators
    results['ADX'] = talib.ADX(high_prices, low_prices, close_prices)
    results['ADXR'] = talib.ADXR(high_prices, low_prices, close_prices)
    results['APO'] = talib.APO(close_prices)
    results['CCI'] = talib.CCI(high_prices, low_prices, close_prices)
    results['CMO'] = talib.CMO(close_prices)
    results['DX'] = talib.DX(high_prices, low_prices, close_prices)
    results['MACD'], results['MACD_SIGNAL'], results['MACD_HIST'] = talib.MACD(close_prices)
    results['MFI'] = talib.MFI(high_prices, low_prices, close_prices, volumes)
    results['MINUS_DI'] = talib.MINUS_DI(high_prices, low_prices, close_prices)
    results['MINUS_DM'] = talib.MINUS_DM(high_prices, low_prices)
    results['MOM'] = talib.MOM(close_prices)
    results['PLUS_DI'] = talib.PLUS_DI(high_prices, low_prices, close_prices)
    results['PLUS_DM'] = talib.PLUS_DM(high_prices, low_prices)
    results['PPO'] = talib.PPO(close_prices)
    results['ROC'] = talib.ROC(close_prices)
    results['ROCP'] = talib.ROCP(close_prices)
    results['ROCR'] = talib.ROCR(close_prices)
    results['ROCR100'] = talib.ROCR100(close_prices)
    results['RSI'] = talib.RSI(close_prices)
    results['STOCH_K'], results['STOCH_D'] = talib.STOCH(high_prices, low_prices, close_prices)
    results['STOCHF_K'], results['STOCHF_D'] = talib.STOCHF(high_prices, low_prices, close_prices)
    results['STOCHRSI_K'], results['STOCHRSI_D'] = talib.STOCHRSI(close_prices)
    results['TRIX'] = talib.TRIX(close_prices)
    results['ULTOSC'] = talib.ULTOSC(high_prices, low_prices, close_prices)
    results['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices)

    # Volume Indicators
    results['AD'] = talib.AD(high_prices, low_prices, close_prices, volumes)
    results['ADOSC'] = talib.ADOSC(high_prices, low_prices, close_prices, volumes)
    results['OBV'] = talib.OBV(close_prices, volumes)

    # Volatility Indicators
    results['ATR'] = talib.ATR(high_prices, low_prices, close_prices)
    results['NATR'] = talib.NATR(high_prices, low_prices, close_prices)
    results['TRANGE'] = talib.TRANGE(high_prices, low_prices, close_prices)

    # Price Transform
    results['AVGPRICE'] = talib.AVGPRICE(open_prices, high_prices, low_prices, close_prices)
    results['MEDPRICE'] = talib.MEDPRICE(high_prices, low_prices)
    results['TYPPRICE'] = talib.TYPPRICE(high_prices, low_prices, close_prices)
    results['WCLPRICE'] = talib.WCLPRICE(high_prices, low_prices, close_prices)

    # Cycle Indicators
    results['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_prices)
    results['HT_DCPHASE'] = talib.HT_DCPHASE(close_prices)
    results['HT_PHASOR_INPHASE'], results['HT_PHASOR_QUADRATURE'] = talib.HT_PHASOR(close_prices)
    results['HT_SINE'], results['HT_LEADSINE'] = talib.HT_SINE(close_prices)
    results['HT_TRENDMODE'] = talib.HT_TRENDMODE(close_prices)

    # Math Transform
    results['ACOS'] = talib.ACOS(close_prices)
    results['ASIN'] = talib.ASIN(close_prices)
    results['ATAN'] = talib.ATAN(close_prices)
    results['CEIL'] = talib.CEIL(close_prices)
    results['COS'] = talib.COS(close_prices)
    results['COSH'] = talib.COSH(close_prices)
    results['EXP'] = talib.EXP(close_prices)
    results['FLOOR'] = talib.FLOOR(close_prices)
    results['LN'] = talib.LN(close_prices)
    results['LOG10'] = talib.LOG10(close_prices)
    results['SIN'] = talib.SIN(close_prices)
    results['SINH'] = talib.SINH(close_prices)
    results['SQRT'] = talib.SQRT(close_prices)
    results['TAN'] = talib.TAN(close_prices)
    results['TANH'] = talib.TANH(close_prices)

    # Math Operators
    results['ADD'] = talib.ADD(close_prices, close_prices)
    results['DIV'] = talib.DIV(close_prices, close_prices)
    results['MAX'] = talib.MAX(close_prices)
    results['MAXINDEX'] = talib.MAXINDEX(close_prices)
    results['MIN'] = talib.MIN(close_prices)
    results['MININDEX'] = talib.MININDEX(close_prices)
    results['MINMAX'] = talib.MINMAX(close_prices)
    results['MINMAXINDEX'] = talib.MINMAXINDEX(close_prices)
    results['MULT'] = talib.MULT(close_prices, close_prices)
    results['SUB'] = talib.SUB(close_prices, close_prices)
    results['SUM'] = talib.SUM(close_prices)

    # Statistical Functions
    results['BETA'] = talib.BETA(high_prices, low_prices)
    results['CORREL'] = talib.CORREL(high_prices, low_prices)
    results['LINEARREG'] = talib.LINEARREG(close_prices)
    results['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close_prices)
    results['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close_prices)
    results['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close_prices)
    results['STDDEV'] = talib.STDDEV(close_prices)
    results['TSF'] = talib.TSF(close_prices)
    results['VAR'] = talib.VAR(close_prices)

    return results

# Main logic
symbol = "BTCUSDC"
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

for timeframe in timeframes:
    candles = get_candles(symbol, timeframe)
    if not candles:
        continue

    current_price = candles[-1]["close"]
    poc = np.max([c["high"] for c in candles])  # Simplified POC approximation
    vwap = calculate_vwap(candles)
    ema = calculate_ema(candles, timeperiod=14)[-1] if len(candles) >= 14 else np.nan
    rsi = calculate_rsi(candles, timeperiod=14)[-1] if len(candles) >= 14 else np.nan
    macd, macdsignal, macdhist = calculate_macd(candles)
    macd_value = macd[-1] if len(macd) > 0 else np.nan
    macdsignal_value = macdsignal[-1] if len(macdsignal) > 0 else np.nan
    macdhist_value = macdhist[-1] if len(macdhist) > 0 else np.nan
    momentum = calculate_momentum(candles, timeperiod=10)[-1] if len(candles) >= 10 else np.nan
    regression_lower, regression_upper = calculate_regression_channels(candles)
    fib_levels = calculate_fibonacci_retracement(np.max([c["high"] for c in candles]), np.min([c["low"] for c in candles]))
    fft_result = calculate_fft(candles)
    ifft_result = calculate_ifft(fft_result)
    talib_indicators = calculate_all_talib_indicators(candles)

    # Print details
    print(f"Timeframe: {timeframe}")
    print(f"Current Price: {current_price}")
    print(f"POC: {poc}")
    print(f"VWAP: {vwap}")
    print(f"EMA: {ema}")
    print(f"RSI: {rsi}")
    print(f"MACD: {macd_value}, MACD Signal: {macdsignal_value}, MACD Histogram: {macdhist_value}")
    print(f"Momentum: {momentum}")
    print(f"Regression Lower: {regression_lower}")
    print(f"Regression Upper: {regression_upper}")
    print(f"Fibonacci Retracement Levels: {fib_levels}")
    print(f"FFT Result (first 5 components): {fft_result[:5]}")
    print(f"IFFT Result (first 5 components): {ifft_result[:5]}")
    print("\nTA-Lib Indicators:")
    for key, value in talib_indicators.items():
        print(f"{key}: {value[-1] if isinstance(value, np.ndarray) else value}")
    print("\n" + "="*30 + "\n")
