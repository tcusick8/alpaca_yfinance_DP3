"""
fetch_hist_data.py
==================
Historical Market Data Fetcher using yfinance

Purpose: Fetch 20 years of historical market data (2005-2025) for recession analysis
Author: Market Data Engineering Team
Date: 2025

Features:
- Robust error handling and retry logic
- Progress tracking with tqdm
- Data validation and cleaning
- Saves data in multiple formats (CSV, Parquet)
- Logs all operations
- Can handle large datasets efficiently
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
import uuid
import yfinance as yf
from tqdm.auto import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for data fetching"""
    
    # Date range
    START_DATE = "2005-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    # Stock universe
    STOCK_GROUPS = {
        'Financial': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 'USB'],
        'Housing': ['DHI', 'LEN', 'PHM', 'KBH', 'TOL', 'MTH', 'TMHC', 'MHO', 'BZH', 'LGIH'],
        'Consumer_Discretionary': ['AMZN', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'TJX', 'ROST', 'DG'],
        'Consumer_Staples': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CLX'],
        'Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CSCO', 'ORCL'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
        'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'LLY'],
        'Indices': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VEA', 'VWO', 'EEM', 'GLD', 'TLT']
    }
    
    # Output directories
    DATA_DIR = Path("data")
    LOG_DIR = DATA_DIR / "logs"
    # DuckDB file path (default persisted DB file in project root)
    DUCKDB_PATH = os.getenv('DUCKDB_PATH', 'historical_market_data.duckdb')
    
    # Data intervals
    INTERVAL = "1d"  # Daily data
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging"""
    Config.create_directories()
    
    log_file = Config.LOG_DIR / f"fetch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

class DataFetcher:
    """Class to handle data fetching operations"""
    
    def __init__(self, logger):
        self.logger = logger
        self.failed_symbols = []
        self.successful_symbols = []
        # DuckDB connection (in-memory by default)
        try:
            self.conn = duckdb.connect(Config.DUCKDB_PATH)
        except Exception:
            self.conn = None
        
    def fetch_single_symbol(self, symbol, start_date, end_date, interval='1d', retries=0):
        """
        Fetch data for a single symbol with retry logic
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        interval : str
            Data interval (1d, 1wk, 1mo)
        retries : int
            Current retry attempt
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame with OHLCV data, or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return None
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            
            # Select and reorder columns
            columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            df = df[columns]
            
            # Validate data
            if self._validate_data(df, symbol):
                self.successful_symbols.append(symbol)
                self.logger.info(f"✓ {symbol}: {len(df)} bars fetched")
                return df
            else:
                self.logger.warning(f"Data validation failed for {symbol}")
                return None
                
        except Exception as e:
            if retries < Config.MAX_RETRIES:
                self.logger.warning(f"Error fetching {symbol} (attempt {retries + 1}/{Config.MAX_RETRIES}): {str(e)}")
                import time
                time.sleep(Config.RETRY_DELAY)
                return self.fetch_single_symbol(symbol, start_date, end_date, interval, retries + 1)
            else:
                self.logger.error(f"✗ Failed to fetch {symbol} after {Config.MAX_RETRIES} attempts: {str(e)}")
                self.failed_symbols.append(symbol)
                return None
    
    def _validate_data(self, df, symbol):
        """
        Validate fetched data
        
        Checks:
        - Non-empty DataFrame
        - Required columns present
        - No all-NaN columns
        - Reasonable price values
        """
        if df is None or df.empty:
            return False
        
        required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Missing required columns for {symbol}")
            return False
        
        # Check for all-NaN price columns
        price_cols = ['open', 'high', 'low', 'close']
        if df[price_cols].isna().all().any():
            self.logger.warning(f"All-NaN price columns found for {symbol}")
            return False
        
        # Check for negative prices
        if (df[price_cols] < 0).any().any():
            self.logger.warning(f"Negative prices found for {symbol}")
            return False
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            self.logger.warning(f"High < Low found for {symbol}")
            return False
        
        return True
    
    def fetch_sector(self, sector_name, symbols, start_date, end_date):
        """
        Fetch data for all symbols in a sector
        
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame for all symbols in sector
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Fetching {sector_name} sector ({len(symbols)} symbols)")
        self.logger.info(f"{'='*60}")
        
        sector_data = []
        
        for symbol in tqdm.tqdm(symbols, desc=f"Downloading {sector_name}", unit="stock"):
            df = self.fetch_single_symbol(symbol, start_date, end_date)
            if df is not None:
                sector_data.append(df)
        
        if sector_data:
            combined_df = pd.concat(sector_data, ignore_index=True)
            combined_df = combined_df.sort_values(['symbol', 'date']).reset_index(drop=True)
            
            self.logger.info(f"✓ {sector_name}: Successfully fetched {len(sector_data)}/{len(symbols)} symbols")
            self.logger.info(f"  Total bars: {len(combined_df):,}")
            self.logger.info(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            
            return combined_df
        else:
            self.logger.warning(f"✗ {sector_name}: No data fetched")
            return pd.DataFrame()
    
    def save_sector_data(self, df, sector_name):
        """Persist sector data into DuckDB (in-memory by default).

        This avoids writing CSV/Parquet files to disk. Data from all sectors
        is appended into a single table named `bars`.
        """
        if df.empty:
            self.logger.warning(f"No data to save for {sector_name}")
            return

        if self.conn is None:
            self.logger.error("No DuckDB connection available; skipping save")
            return

        # Register a temporary table name and insert into persistent 'bars' table
        tmp_table = f"tmp_{sector_name.lower()}_{uuid.uuid4().hex}"
        try:
            self.conn.register(tmp_table, df)

            # If 'bars' doesn't exist yet, create it from the first sector
            exists = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'bars'"
            ).fetchone()[0]

            if not exists:
                self.conn.execute(f"CREATE TABLE bars AS SELECT * FROM {tmp_table}")
            else:
                # Append rows
                self.conn.execute(f"INSERT INTO bars SELECT * FROM {tmp_table}")

            # Also store simple metadata in a fetch_metadata table
            metadata = {
                'sector': sector_name,
                'symbols': sorted(df['symbol'].unique().tolist()),
                'num_symbols': int(df['symbol'].nunique()),
                'num_bars': int(len(df)),
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'fetch_timestamp': datetime.now().isoformat()
            }

            # Upsert metadata row
            try:
                self.conn.execute(
                    "CREATE TABLE IF NOT EXISTS fetch_metadata (sector VARCHAR, symbols VARCHAR, num_symbols INTEGER, num_bars INTEGER, start DATE, end DATE, fetch_timestamp VARCHAR)"
                )
                # store symbols as JSON string
                self.conn.execute(
                    "INSERT INTO fetch_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        metadata['sector'],
                        json.dumps(metadata['symbols']),
                        metadata['num_symbols'],
                        metadata['num_bars'],
                        metadata['start'],
                        metadata['end'],
                        metadata['fetch_timestamp'],
                    ],
                )
            except Exception:
                # metadata is best-effort—don't fail the whole pipeline
                self.logger.warning("Failed to write fetch metadata to DuckDB")

            self.logger.info(f"  Saved sector '{sector_name}' into DuckDB table 'bars'")

        finally:
            # Attempt to unregister the temporary table (ignore errors)
            try:
                self.conn.unregister(tmp_table)
            except Exception:
                pass


# ============================================================================
# DATA STATISTICS
# ============================================================================

def generate_fetch_summary(fetcher, all_data):
    """Generate and save summary statistics of fetched data"""
    logger = fetcher.logger
    
    logger.info(f"\n{'='*80}")
    logger.info("FETCH SUMMARY")
    logger.info(f"{'='*80}")
    
    # Overall statistics
    total_symbols = len(fetcher.successful_symbols) + len(fetcher.failed_symbols)
    total_bars = sum(len(df) for df in all_data.values() if not df.empty)
    
    logger.info(f"\nTotal symbols attempted: {total_symbols}")
    logger.info(f"Successful: {len(fetcher.successful_symbols)}")
    logger.info(f"Failed: {len(fetcher.failed_symbols)}")
    logger.info(f"Success rate: {len(fetcher.successful_symbols)/total_symbols*100:.1f}%")
    logger.info(f"Total bars fetched: {total_bars:,}")
    
    if fetcher.failed_symbols:
        logger.warning(f"\nFailed symbols: {', '.join(fetcher.failed_symbols)}")
    
    # Sector breakdown
    logger.info(f"\n{'='*80}")
    logger.info("SECTOR BREAKDOWN")
    logger.info(f"{'='*80}")
    
    summary_data = []
    for sector, df in all_data.items():
        if not df.empty:
            summary_data.append({
                'Sector': sector,
                'Symbols': df['symbol'].nunique(),
                'Bars': len(df),
                'Start_Date': df['date'].min().strftime('%Y-%m-%d'),
                'End_Date': df['date'].max().strftime('%Y-%m-%d'),
                'Days': (df['date'].max() - df['date'].min()).days
            })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    
    # Save summary
    summary_path = Config.DATA_DIR / "fetch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to: {summary_path}")
    
    return summary_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Setup
    logger = setup_logging()
    Config.create_directories()
    
    logger.info("="*80)
    logger.info("HISTORICAL MARKET DATA FETCHER")
    logger.info("="*80)
    logger.info(f"Start Date: {Config.START_DATE}")
    logger.info(f"End Date: {Config.END_DATE}")
    logger.info(f"Interval: {Config.INTERVAL}")
    logger.info(f"Total Sectors: {len(Config.STOCK_GROUPS)}")
    logger.info(f"Total Symbols: {sum(len(symbols) for symbols in Config.STOCK_GROUPS.values())}")
    logger.info(f"Output Directory: {Config.DATA_DIR}")
    
    # Initialize fetcher
    fetcher = DataFetcher(logger)
    
    # Fetch data for each sector
    all_data = {}
    
    for sector_name, symbols in Config.STOCK_GROUPS.items():
        df = fetcher.fetch_sector(
            sector_name=sector_name,
            symbols=symbols,
            start_date=Config.START_DATE,
            end_date=Config.END_DATE
        )
        
        all_data[sector_name] = df
        
        # Save immediately after fetching each sector
        fetcher.save_sector_data(df, sector_name)
    
    # Generate summary
    summary_df = generate_fetch_summary(fetcher, all_data)
    
    # Create combined dataset
    logger.info(f"\n{'='*80}")
    logger.info("CREATING COMBINED DATASET")
    logger.info(f"{'='*80}")
    
    non_empty_data = [df for df in all_data.values() if not df.empty]
    if non_empty_data:
        combined_df = pd.concat(non_empty_data, ignore_index=True)
        combined_df = combined_df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # Data is persisted to DuckDB by `save_sector_data`; summarize here
        logger.info(f"Combined dataset prepared in-memory: {len(combined_df):,} bars")
        logger.info(f"Persistent storage: DuckDB file at {Config.DUCKDB_PATH} (table 'bars')")
    
    logger.info(f"\n{'='*80}")
    logger.info("DATA FETCH COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"\nNext step: Run clean_and_visualize.py to analyze the data")
    logger.info(f"Data location: {Config.DATA_DIR}")
    
    return all_data, summary_df


if __name__ == "__main__":
    # Check dependencies
    try:
        import yfinance
        import tqdm
        import pandas
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install with: pip install yfinance tqdm pandas pyarrow")
        sys.exit(1)
    
    # Run main function
    all_data, summary = main()