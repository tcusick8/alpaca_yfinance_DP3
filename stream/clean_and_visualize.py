#!/usr/bin/env python3
"""
Clean and visualize Alpaca bar data from DuckDB.
- Deduplicates bars
- Aggregates to hourly bars
- Computes moving averages
- Plots line and candlestick charts
"""

import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf  

# --------------------------
# Configuration
# --------------------------
DUCKDB_PATH = "data/market_data.duckdb"
AGGREGATED_TABLE = "bars_hourly"
SMA_WINDOW = 5  # number of periods for simple moving average

# --------------------------
# Connect to DuckDB
# --------------------------
conn = duckdb.connect(DUCKDB_PATH)

# --------------------------
# Clean & aggregate
# --------------------------
# Deduplicate and aggregate to hourly bars
conn.execute(f"""
CREATE OR REPLACE TABLE {AGGREGATED_TABLE} AS
SELECT
    symbol,
    DATE_TRUNC('hour', timestamp) AS hour,
    MIN(low) AS low,
    MAX(high) AS high,
    FIRST(open) AS open,
    LAST(close) AS close,
    SUM(volume) AS volume
FROM (
    SELECT DISTINCT symbol, timestamp, open, high, low, close, volume
    FROM bars
) sub
GROUP BY symbol, DATE_TRUNC('hour', timestamp)
ORDER BY symbol, hour;
""")

# --------------------------
# Load aggregated data into Pandas
# --------------------------
df = conn.execute(f"SELECT * FROM {AGGREGATED_TABLE}").df()
df['hour'] = pd.to_datetime(df['hour'])

# --------------------------
# Plotting
# --------------------------
symbols = df['symbol'].unique()
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df.set_index('hour', inplace=True)

    # Compute simple moving average
    symbol_df['SMA'] = symbol_df['close'].rolling(SMA_WINDOW).mean()

    # Line chart: Close + SMA
    plt.figure(figsize=(12,6))
    plt.plot(symbol_df.index, symbol_df['close'], label='Close', color='blue')
    plt.plot(symbol_df.index, symbol_df['SMA'], label=f'{SMA_WINDOW}-period SMA', color='orange')
    plt.title(f'{symbol} Close Price & SMA')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Candlestick chart using mplfinance
    mpf_df = symbol_df[['open', 'high', 'low', 'close', 'volume']].copy()
    mpf.plot(
        mpf_df,
        type='candle',
        style='charles',
        title=f'{symbol} Hourly Candlestick',
        volume=True,
        figsize=(12,6)
    )


