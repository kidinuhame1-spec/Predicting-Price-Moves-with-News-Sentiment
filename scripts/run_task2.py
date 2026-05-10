#!/usr/bin/env python3
"""Run Task 2: load price data, compute indicators (SMA/EMA/RSI/MACD), save cleaned data and figures.

Usage:
    python scripts/run_task2.py
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

BASE = Path(__file__).resolve().parents[1]
DATA_CSV = BASE / 'data' / 'raw' / 'prices.csv'
OUT_DIR = BASE / 'outputs_task2'
OUT_DIR.mkdir(exist_ok=True)

try:
    import yfinance as yf
    YFINANCE = True
except Exception:
    YFINANCE = False

try:
    import talib
    TALIB = True
except Exception:
    TALIB = False

def load_prices():
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV, parse_dates=['Date'])
    else:
        if not YFINANCE:
            raise RuntimeError('No data/raw/prices.csv and yfinance not available')
        print('Downloading SPY sample data via yfinance...')
        df = yf.download('SPY', period='2y', auto_adjust=False)
        df = df.reset_index().rename(columns={'Adj Close':'Adj_Close'})
    return df

def prepare(df):
    # ensure Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = df.reset_index().rename(columns={'index':'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # coerce numeric
    for c in ['Open','High','Low','Close','Adj_Close','Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.ffill().dropna().reset_index(drop=True)
    return df

def compute_indicators(df):
    price_col = 'Close' if 'Close' in df.columns else 'Adj_Close'
    windows = [10,20,50,200]
    for w in windows:
        sma = f'SMA_{w}'
        ema = f'EMA_{w}'
        if TALIB:
            df[sma] = talib.SMA(df[price_col].values, timeperiod=w)
            df[ema] = talib.EMA(df[price_col].values, timeperiod=w)
        else:
            df[sma] = df[price_col].rolling(window=w, min_periods=1).mean()
            df[ema] = df[price_col].ewm(span=w, adjust=False).mean()
    # RSI
    rsi_period = 14
    if TALIB:
        df['RSI'] = talib.RSI(df[price_col].values, timeperiod=rsi_period)
    else:
        delta = df[price_col].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(com=rsi_period-1, adjust=False).mean()
        ma_down = down.ewm(com=rsi_period-1, adjust=False).mean()
        rs = ma_up / ma_down
        df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    if TALIB:
        macd, macd_signal, macd_hist = talib.MACD(df[price_col].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
    else:
        ema12 = df[price_col].ewm(span=12, adjust=False).mean()
        ema26 = df[price_col].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def plot_and_save(df):
    price_col = 'Close' if 'Close' in df.columns else 'Adj_Close'
    fig, axes = plt.subplots(3,1, figsize=(12,10), sharex=True)
    axes[0].plot(df['Date'], df[price_col], label='Close', color='black')
    for w in [10,50,200]:
        col = f'EMA_{w}' if f'EMA_{w}' in df.columns else f'SMA_{w}'
        if col in df.columns:
            axes[0].plot(df['Date'], df[col], label=col)
    axes[0].legend(loc='upper left')
    axes[0].set_title('Price with MAs')

    axes[1].plot(df['Date'], df['RSI'], color='purple')
    axes[1].axhline(70, color='red', linestyle='--')
    axes[1].axhline(30, color='green', linestyle='--')
    axes[1].set_title('RSI (14)')

    axes[2].plot(df['Date'], df['MACD'], label='MACD', color='blue')
    axes[2].plot(df['Date'], df['MACD_Signal'], label='Signal', color='orange')
    axes[2].bar(df['Date'], df['MACD_Hist'], label='Hist', color='grey')
    axes[2].legend()
    axes[2].set_title('MACD')

    plt.tight_layout()
    out_png = OUT_DIR / 'task2_indicators.png'
    fig.savefig(out_png)
    plt.close(fig)
    print('Saved', out_png)

    # also save cleaned csv
    out_csv = OUT_DIR / 'task2_cleaned_prices.csv'
    df.to_csv(out_csv, index=False)
    print('Saved', out_csv)

def main():
    try:
        df = load_prices()
    except Exception as e:
        print('Error loading prices:', e)
        sys.exit(1)
    df = prepare(df)
    df = compute_indicators(df)
    plot_and_save(df)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
