"""
clean_and_visualize.py
======================
Data Cleaning and Visualization Module

Purpose: Load, clean, and visualize historical market data for recession analysis
Author: Market Data Engineering Team
Date: 2025

Features:
- Loads data from multiple formats (CSV, Parquet)
- Advanced data cleaning and validation
- Statistical analysis
- Comprehensive visualizations
- Recession pattern detection
- Current vs 2008 comparison
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import duckdb
import json
import uuid

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for analysis and visualization"""
    
    # Directories
    DATA_DIR = Path("data")
    RAW_DIR = DATA_DIR / "raw"
    OUTPUT_DIR = DATA_DIR / "visualizations"
    
    # Recession periods for analysis
    RECESSION_PERIODS = {
        'Dot-com Crash': ('2000-03-01', '2002-10-01'),
        'Financial Crisis': ('2007-12-01', '2009-06-30'),
        'COVID-19': ('2020-02-01', '2020-04-30'),
    }
    
    # Visualization settings
    plt.style.use('seaborn-v0_8-darkgrid')
    FIGURE_SIZE = (18, 10)
    DPI = 150
    
    @classmethod
    def create_directories(cls):
        """Create output directories"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """Handle loading of historical data"""
    
    def __init__(self):
        self.data = {}
        self.combined_data = None
        # DuckDB path (file created by fetch script)
        self.db_path = os.getenv('DUCKDB_PATH', 'historical_market_data.duckdb')
        self.conn = None
        
    def load_sector(self, sector_name, use_parquet=True):
        """
        Load data for a specific sector
        
        Parameters:
        -----------
        sector_name : str
            Name of the sector
        use_parquet : bool
            Use parquet format (faster) if True, else CSV
            
        Returns:
        --------
        pd.DataFrame
        """
        ext = 'parquet' if use_parquet else 'csv'
        file_path = Config.RAW_DIR / f"{sector_name}_raw.{ext}"
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            return pd.DataFrame()
        
        try:
            if use_parquet:
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, parse_dates=['date'])
            
            print(f"✓ Loaded {sector_name}: {len(df):,} bars, {df['symbol'].nunique()} symbols")
            return df
            
        except Exception as e:
            print(f"✗ Error loading {sector_name}: {str(e)}")
            return pd.DataFrame()
    
    def load_all_sectors(self):
        """Load all available sector data from DuckDB `bars` table.

        Expects `historical_market_data.duckdb` (or `DUCKDB_PATH` env) to contain
        a `bars` table and optional `fetch_metadata` table created by the fetch script.
        """
        print("="*80)
        print("LOADING DATA FROM DUCKDB")
        print("="*80)

        # Connect to DuckDB
        try:
            self.conn = duckdb.connect(self.db_path)
        except Exception as e:
            print(f"✗ Failed to open DuckDB at {self.db_path}: {e}")
            return False

        # Load combined bars table
        try:
            combined_df = self.conn.execute("SELECT * FROM bars").df()
        except Exception:
            print(f"⚠️  No 'bars' table found in {self.db_path}. Run fetch_hist_data.py first!")
            return False

        # Ensure date is datetime
        if 'date' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['date'])

        self.combined_data = combined_df

        # Attempt to read fetch_metadata to split by sector
        sectors = {}
        try:
            rows = self.conn.execute("SELECT sector, symbols FROM fetch_metadata").fetchall()
            for sector, symbols_json in rows:
                try:
                    symbols = json.loads(symbols_json)
                except Exception:
                    symbols = []
                sectors[sector] = symbols
        except Exception:
            # fallback: place all symbols under 'All'
            sectors = {'All': sorted(combined_df['symbol'].unique().tolist())}

        # Build per-sector dataframes
        for sector_name, symbols in sectors.items():
            if symbols:
                df = combined_df[combined_df['symbol'].isin(symbols)].copy()
            else:
                df = pd.DataFrame()
            self.data[sector_name] = df

        print(f"✓ Loaded combined dataset from {self.db_path}: {len(self.combined_data):,} bars")
        print(f"  Sectors found: {len(self.data)}")
        return True


# ============================================================================
# DATA CLEANING
# ============================================================================

class DataCleaner:
    """Handle data cleaning and validation"""
    
    @staticmethod
    def clean_data(df):
        """
        Clean and validate data
        
        Cleaning steps:
        1. Remove duplicates
        2. Handle missing values
        3. Fix data type issues
        4. Remove outliers
        5. Sort by date
        """
        print(f"\nCleaning data: {len(df)} initial rows")
        
        # Make a copy
        df_clean = df.copy()
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_clean['date']):
            df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Remove duplicates
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['symbol', 'date'])
        if len(df_clean) < initial_len:
            print(f"  Removed {initial_len - len(df_clean)} duplicate rows")
        
        # Handle missing values in price columns
        price_cols = ['open', 'high', 'low', 'close']
        missing_before = df_clean[price_cols].isna().sum().sum()
        
        if missing_before > 0:
            print(f"  Found {missing_before} missing price values")
            # Forward fill missing prices
            df_clean[price_cols] = df_clean.groupby('symbol')[price_cols].fillna(method='ffill')
            # Backward fill any remaining
            df_clean[price_cols] = df_clean.groupby('symbol')[price_cols].fillna(method='bfill')
            missing_after = df_clean[price_cols].isna().sum().sum()
            print(f"  Filled missing values, {missing_after} remaining")
        
        # Remove rows with negative prices
        negative_mask = (df_clean[price_cols] < 0).any(axis=1)
        if negative_mask.sum() > 0:
            print(f"  Removed {negative_mask.sum()} rows with negative prices")
            df_clean = df_clean[~negative_mask]
        
        # Validate high >= low
        invalid_high_low = df_clean['high'] < df_clean['low']
        if invalid_high_low.sum() > 0:
            print(f"  Fixed {invalid_high_low.sum()} rows where high < low")
            # Swap high and low
            df_clean.loc[invalid_high_low, ['high', 'low']] = df_clean.loc[invalid_high_low, ['low', 'high']].values
        
        # Sort by symbol and date
        df_clean = df_clean.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        print(f"✓ Cleaned data: {len(df_clean)} rows remaining")
        return df_clean
    
    @staticmethod
    def add_technical_indicators(df):
        """Add technical indicators for analysis"""
        print("\nCalculating technical indicators...")
        
        df_with_indicators = df.copy()
        
        # Returns
        df_with_indicators['daily_return'] = df_with_indicators.groupby('symbol')['close'].pct_change()
        
        # Moving averages
        for window in [20, 50, 200]:
            df_with_indicators[f'ma_{window}'] = df_with_indicators.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Volatility (20-day rolling)
        df_with_indicators['volatility_20d'] = df_with_indicators.groupby('symbol')['daily_return'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std() * np.sqrt(252)
        )
        
        # Volume moving average
        df_with_indicators['volume_ma_20'] = df_with_indicators.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )
        
        print("✓ Technical indicators calculated")
        return df_with_indicators


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

class MarketAnalyzer:
    """Analyze market data and patterns"""
    
    @staticmethod
    def calculate_performance_metrics(df):
        """Calculate key performance metrics for each symbol"""
        
        metrics_list = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) < 2:
                continue
            
            # Basic metrics
            first_price = symbol_data['close'].iloc[0]
            last_price = symbol_data['close'].iloc[-1]
            total_return = ((last_price - first_price) / first_price) * 100
            
            # Volatility
            returns = symbol_data['daily_return'].dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Max drawdown
            cummax = symbol_data['close'].cummax()
            drawdown = ((symbol_data['close'] - cummax) / cummax) * 100
            max_drawdown = drawdown.min()
            
            # Sharpe ratio (assuming 0% risk-free rate for simplicity)
            if volatility > 0:
                sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe = 0
            
            metrics_list.append({
                'Symbol': symbol,
                'Start_Price': first_price,
                'End_Price': last_price,
                'Total_Return_%': total_return,
                'Annualized_Volatility_%': volatility,
                'Max_Drawdown_%': max_drawdown,
                'Sharpe_Ratio': sharpe,
                'Trading_Days': len(symbol_data)
            })
        
        return pd.DataFrame(metrics_list)
    
    @staticmethod
    def analyze_recession_periods(df, recession_periods):
        """Analyze performance during recession periods"""
        
        results = {}
        
        for period_name, (start, end) in recession_periods.items():
            # Handle timezone-aware dates
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            # Make timezone-aware if df dates are timezone-aware
            if df['date'].dt.tz is not None:
                start_dt = start_dt.tz_localize(df['date'].dt.tz)
                end_dt = end_dt.tz_localize(df['date'].dt.tz)
            
            period_data = df[(df['date'] >= start_dt) & 
                            (df['date'] <= end_dt)]
            
            if period_data.empty:
                continue
            
            period_returns = []
            
            for symbol in period_data['symbol'].unique():
                symbol_data = period_data[period_data['symbol'] == symbol].sort_values('date')
                if len(symbol_data) >= 2:
                    ret = ((symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0]) - 1) * 100
                    period_returns.append(ret)
            
            if period_returns:
                results[period_name] = {
                    'mean_return': np.mean(period_returns),
                    'median_return': np.median(period_returns),
                    'worst_return': np.min(period_returns),
                    'best_return': np.max(period_returns),
                    'num_stocks': len(period_returns)
                }
        
        return pd.DataFrame(results).T
    
    @staticmethod
    def compare_current_to_2008(df):
        """Compare current market conditions to pre-2008 crisis"""
        
        # Define periods
        pre_2008 = ('2006-01-01', '2007-06-30')
        current = ((datetime.now() - timedelta(days=540)).strftime("%Y-%m-%d"), 
                   datetime.now().strftime("%Y-%m-%d"))
        
        def get_period_metrics(start, end):
            # Handle timezone-aware dates
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            if df['date'].dt.tz is not None:
                start_dt = start_dt.tz_localize(df['date'].dt.tz)
                end_dt = end_dt.tz_localize(df['date'].dt.tz)
            
            period_df = df[(df['date'] >= start_dt) & 
                          (df['date'] <= end_dt)]
            
            if period_df.empty:
                return None
            
            # Calculate average volatility
            vol = period_df.groupby('symbol')['volatility_20d'].mean().mean()
            
            # Calculate average returns
            returns = []
            for symbol in period_df['symbol'].unique():
                sym_data = period_df[period_df['symbol'] == symbol]
                if len(sym_data) >= 2:
                    ret = ((sym_data['close'].iloc[-1] / sym_data['close'].iloc[0]) - 1) * 100
                    returns.append(ret)
            
            return {
                'avg_volatility': vol,
                'avg_return': np.mean(returns) if returns else 0,
                'num_stocks': len(returns)
            }
        
        pre_2008_metrics = get_period_metrics(*pre_2008)
        current_metrics = get_period_metrics(*current)
        
        comparison = pd.DataFrame({
            'Pre-2008 Crisis': pre_2008_metrics,
            'Current Period': current_metrics
        })
        
        return comparison


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

class Visualizer:
    """Create comprehensive visualizations"""
    
    def __init__(self):
        Config.create_directories()
    
    def plot_sector_performance(self, data_dict):
        """Plot normalized performance for each sector"""
        
        print("\nGenerating sector performance plots...")
        
        for sector, df in data_dict.items():
            if df.empty:
                continue
            
            fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE)
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_values('date')
                normalized = (symbol_data['close'] / symbol_data['close'].iloc[0]) * 100
                ax.plot(symbol_data['date'], normalized, label=symbol, linewidth=2, alpha=0.7)
            
            # Add recession shading
            for period_name, (start, end) in Config.RECESSION_PERIODS.items():
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                ax.axvspan(start_dt, end_dt, alpha=0.15, color='red')
            
            ax.axhline(y=100, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_title(f'{sector} - Normalized Performance (2005-2025)', 
                        fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Year', fontsize=14)
            ax.set_ylabel('Normalized Price (Base=100)', fontsize=14)
            ax.legend(loc='best', fontsize=10, ncol=2)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(Config.OUTPUT_DIR / f'{sector}_performance.png', dpi=Config.DPI)
            plt.close()
        
        print(f"✓ Saved sector performance plots to {Config.OUTPUT_DIR}")
    
    def plot_volatility_analysis(self, df):
        """Plot volatility over time"""
        
        print("\nGenerating volatility analysis...")
        
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE)
        
        # Calculate average volatility by date
        daily_vol = df.groupby('date')['volatility_20d'].mean()
        
        ax.plot(daily_vol.index, daily_vol.values, linewidth=2, color='steelblue')
        ax.fill_between(daily_vol.index, daily_vol.values, alpha=0.3, color='steelblue')
        
        # Add recession shading
        for period_name, (start, end) in Config.RECESSION_PERIODS.items():
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                      alpha=0.2, color='red', label=period_name)
        
        ax.set_title('Market Volatility Over Time (20-Day Rolling)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Annualized Volatility', fontsize=14)
        
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'market_volatility.png', dpi=Config.DPI)
        plt.close()
        
        print(f"✓ Saved volatility analysis")
    
    def plot_drawdown_analysis(self, df):
        """Plot maximum drawdown over time"""
        
        print("\nGenerating drawdown analysis...")
        
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))
        
        # Plot 1: Individual stocks
        ax1 = axes[0]
        
        for symbol in df['symbol'].unique()[:20]:  # Plot top 20 for readability
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            cummax = symbol_data['close'].cummax()
            drawdown = ((symbol_data['close'] - cummax) / cummax) * 100
            ax1.plot(symbol_data['date'], drawdown, alpha=0.5, linewidth=1)
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_title('Individual Stock Drawdowns', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Drawdown (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average drawdown
        ax2 = axes[1]
        
        # Calculate average drawdown across all stocks
        all_drawdowns = []
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            cummax = symbol_data['close'].cummax()
            drawdown = ((symbol_data['close'] - cummax) / cummax) * 100
            symbol_data['drawdown'] = drawdown
            all_drawdowns.append(symbol_data[['date', 'drawdown']])
        
        combined_dd = pd.concat(all_drawdowns)
        avg_drawdown = combined_dd.groupby('date')['drawdown'].mean()
        
        ax2.plot(avg_drawdown.index, avg_drawdown.values, linewidth=2, color='darkred')
        ax2.fill_between(avg_drawdown.index, avg_drawdown.values, 0, alpha=0.3, color='darkred')
        
        # Add recession shading
        for period_name, (start, end) in Config.RECESSION_PERIODS.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            ax2.axvspan(start_dt, end_dt, alpha=0.2, color='red')
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_title('Average Market Drawdown', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Average Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'drawdown_analysis.png', dpi=Config.DPI)
        plt.close()
        
        print(f"✓ Saved drawdown analysis")
    
    def plot_recession_comparison(self, recession_analysis):
        """Compare returns during different recession periods"""
        
        print("\nGenerating recession comparison...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        periods = recession_analysis.index
        mean_returns = recession_analysis['mean_return']
        
        colors = ['red' if x < 0 else 'green' for x in mean_returns]
        bars = ax.bar(periods, mean_returns, color=colors, alpha=0.7, edgecolor='black')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_title('Average Returns During Recession Periods', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Average Return (%)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'recession_comparison.png', dpi=Config.DPI)
        plt.close()
        
        print(f"✓ Saved recession comparison")
    
    def plot_current_vs_2008(self, comparison_df):
        """Plot comparison between current market and pre-2008"""
        
        print("\nGenerating current vs 2008 comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Volatility comparison
        ax1 = axes[0]
        periods = comparison_df.columns
        volatility = comparison_df.loc['avg_volatility']
        
        bars = ax1.bar(periods, volatility, color=['orange', 'blue'], alpha=0.7, edgecolor='black')
        ax1.set_title('Average Market Volatility Comparison', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Annualized Volatility', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Returns comparison
        ax2 = axes[1]
        returns = comparison_df.loc['avg_return']
        colors = ['orange', 'blue']
        
        bars = ax2.bar(periods, returns, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_title('Average Returns Comparison', 
                     fontsize=16, fontweight='bold')
        ax2.set_ylabel('Average Return (%)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'current_vs_2008.png', dpi=Config.DPI)
        plt.close()
        
        print(f"✓ Saved current vs 2008 comparison")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*80)
    print("HISTORICAL DATA CLEANING AND VISUALIZATION")
    print("="*80)
    
    # Load data
    loader = DataLoader()
    if not loader.load_all_sectors():
        print("\n⚠️  No data found. Please run fetch_hist_data.py first!")
        return
    
    # Clean data
    print("\n" + "="*80)
    print("CLEANING DATA")
    print("="*80)
    
    cleaner = DataCleaner()
    
    # Clean sector data
    cleaned_sector_data = {}
    for sector, df in loader.data.items():
        cleaned_df = cleaner.clean_data(df)
        cleaned_df = cleaner.add_technical_indicators(cleaned_df)
        cleaned_sector_data[sector] = cleaned_df
        # Persist cleaned data back to DuckDB if available, otherwise write parquet
        if loader.conn is not None:
            tmp_name = f"tmp_clean_{sector.lower()}_{uuid.uuid4().hex}"
            try:
                loader.conn.register(tmp_name, cleaned_df)
                exists = loader.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'bars_cleaned'").fetchone()[0]
                if not exists:
                    loader.conn.execute(f"CREATE TABLE bars_cleaned AS SELECT * FROM {tmp_name}")
                else:
                    loader.conn.execute(f"INSERT INTO bars_cleaned SELECT * FROM {tmp_name}")
                print(f"✓ Persisted cleaned sector '{sector}' to DuckDB table 'bars_cleaned'")
            finally:
                try:
                    loader.conn.unregister(tmp_name)
                except Exception:
                    pass
        else:
            # fallback to write cleaned parquet into data/logs (no processed dir)
            logs_dir = Config.DATA_DIR / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            output_path = logs_dir / f"{sector}_cleaned.parquet"
            cleaned_df.to_parquet(output_path, index=False)
            print(f"✓ Saved cleaned data to logs: {output_path}")
    
    # Clean combined data
    if loader.combined_data is not None:
        print("\nCleaning combined dataset...")
        combined_cleaned = cleaner.clean_data(loader.combined_data)
        combined_cleaned = cleaner.add_technical_indicators(combined_cleaned)
        # Persist combined cleaned dataset
        if loader.conn is not None:
            tmp_name = f"tmp_combined_clean_{uuid.uuid4().hex}"
            try:
                loader.conn.register(tmp_name, combined_cleaned)
                exists = loader.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'bars_cleaned'").fetchone()[0]
                if not exists:
                    loader.conn.execute(f"CREATE TABLE bars_cleaned AS SELECT * FROM {tmp_name}")
                else:
                    loader.conn.execute(f"INSERT INTO bars_cleaned SELECT * FROM {tmp_name}")
                print("✓ Persisted combined cleaned dataset to DuckDB table 'bars_cleaned'")
            finally:
                try:
                    loader.conn.unregister(tmp_name)
                except Exception:
                    pass
        else:
            logs_dir = Config.DATA_DIR / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            output_path = logs_dir / "all_sectors_cleaned.parquet"
            combined_cleaned.to_parquet(output_path, index=False)
            print(f"✓ Saved combined cleaned data to logs: {output_path}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYZING DATA")
    print("="*80)
    
    analyzer = MarketAnalyzer()
    
    # Performance metrics
    print("\nCalculating performance metrics...")
    metrics = analyzer.calculate_performance_metrics(combined_cleaned)
    # Persist metrics into DuckDB if available, otherwise write to data/logs
    if loader.conn is not None:
        try:
            loader.conn.execute(
                "CREATE TABLE IF NOT EXISTS performance_metrics (Symbol VARCHAR, Start_Price DOUBLE, End_Price DOUBLE, Total_Return_% DOUBLE, Annualized_Volatility_% DOUBLE, Max_Drawdown_% DOUBLE, Sharpe_Ratio DOUBLE, Trading_Days INTEGER)"
            )
            # Insert rows
            for _, row in metrics.iterrows():
                loader.conn.execute(
                    "INSERT INTO performance_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        row['Symbol'],
                        float(row['Start_Price']),
                        float(row['End_Price']),
                        float(row['Total_Return_%']),
                        float(row['Annualized_Volatility_%']),
                        float(row['Max_Drawdown_%']),
                        float(row['Sharpe_Ratio']),
                        int(row['Trading_Days']),
                    ],
                )
            print("✓ Persisted performance metrics to DuckDB table 'performance_metrics'")
        except Exception as e:
            print(f"✗ Failed to persist metrics to DuckDB: {e}")
            # fallback to logs
            logs_dir = Config.DATA_DIR / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = logs_dir / "performance_metrics.csv"
            metrics.to_csv(metrics_path, index=False)
            print(f"✓ Saved metrics to logs: {metrics_path}")
    else:
        logs_dir = Config.DATA_DIR / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = logs_dir / "performance_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        print(f"✓ Saved metrics to logs: {metrics_path}")
    print("\nTop 10 performers:")
    print(metrics.nlargest(10, 'Total_Return_%')[['Symbol', 'Total_Return_%', 'Annualized_Volatility_%']])
    
    # Recession analysis
    print("\nAnalyzing recession periods...")
    recession_analysis = analyzer.analyze_recession_periods(combined_cleaned, Config.RECESSION_PERIODS)
    print("\n", recession_analysis)
    
    # Current vs 2008
    print("\nComparing current market to pre-2008...")
    comparison = analyzer.compare_current_to_2008(combined_cleaned)
    print("\n", comparison)
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    visualizer = Visualizer()
    visualizer.plot_sector_performance(cleaned_sector_data)
    visualizer.plot_volatility_analysis(combined_cleaned)
    visualizer.plot_drawdown_analysis(combined_cleaned)
    visualizer.plot_recession_comparison(recession_analysis)
    visualizer.plot_current_vs_2008(comparison)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {Config.OUTPUT_DIR}")
    print("Cleaned data saved to DuckDB table 'bars_cleaned' (or data/logs if no DuckDB)")
    
    print("\n📊 RECESSION PREDICTION INSIGHTS:")
    print("="*80)
    if comparison is not None and not comparison.empty:
        vol_2008 = comparison.loc['avg_volatility', 'Pre-2008 Crisis']
        vol_current = comparison.loc['avg_volatility', 'Current Period']
        vol_ratio = (vol_current / vol_2008) * 100
        print(f"Current volatility is {vol_ratio:.1f}% of pre-2008 crisis levels")
        
        if vol_ratio > 80:
            print("⚠️  WARNING: Volatility approaching pre-2008 crisis levels!")
        elif vol_ratio > 60:
            print("⚠️  CAUTION: Elevated volatility detected")
        else:
            print("✓ Volatility remains below pre-2008 crisis levels")
    
    print("\nNext steps:")
    print("1. Review visualizations in data/visualizations/")
    print("2. Analyze performance_metrics.csv for stock-level insights")
    print("3. Run producer/consumer scripts for real-time stream simulation")


if __name__ == "__main__":
    main()