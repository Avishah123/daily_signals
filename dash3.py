import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import random
import concurrent.futures

# Page config
st.set_page_config(page_title="Active Trades Dashboard", page_icon="📈", layout="wide")

# Database connection using Streamlit secrets
@st.cache_resource
def connect_to_db():
    try:
        # Get database credentials from secrets.toml
        pg_connection_string = f"postgresql://{st.secrets['db']['user']}:{st.secrets['db']['password']}@{st.secrets['db']['host']}:{st.secrets['db']['port']}/{st.secrets['db']['database']}"
        
        # Create SQLAlchemy engine
        engine = create_engine(pg_connection_string)
        
        # Test connection
        with engine.connect() as conn:
            pass
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.error("""
Please ensure you have correctly set up your database credentials in secrets.toml:

For local development:
1. Create a .streamlit folder in your project directory
2. Create a secrets.toml file inside the .streamlit folder
3. Add your database credentials in this format:

[db]
host = "your_host"
port = "your_port"
database = "your_database"
user = "your_username"
password = "your_password"

For hosted environments (like Streamlit Cloud):
Add these same secrets in your hosting platform's secrets management.
        """)
        return None

# Fetch data from database with caching
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_table(_engine, table_name):
    if _engine is None:
        return pd.DataFrame()
        
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, _engine)
        
        # Convert date columns to datetime
        date_columns = ['SignalDate', 'EntryDate', 'ExitDate', 'BacktestDate', 'date', 'LastUpdated']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error fetching data from table {table_name}: {e}")
        return pd.DataFrame()

# Cache price data for 5 minutes to avoid repeated API calls
@st.cache_data(ttl=300)
def get_symbol_price_data(symbol_with_suffix):
    """Fetch price data for a single symbol with timeout handling"""
    try:
        ticker = yf.Ticker(symbol_with_suffix)
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            return {
                'current_price': hist['Close'].iloc[-1],
                'open_price': hist['Open'].iloc[-1]
            }
        return None
    except Exception:
        return None

# Function to fetch stock data with retry but with a limited timeout
def fetch_stock_data_with_timeout(symbol, timeout=10):
    """Fetch stock data with a maximum timeout"""
    clean_symbol = symbol.split('.')[0] if '.' in symbol else symbol
    
    # Try these suffixes in order
    suffixes = [".NS", ".BO", ""]  # First try NSE, then BSE, then raw
    
    # Set a start time to track overall timeout
    start_time = time.time()
    
    for suffix in suffixes:
        # Check if we've exceeded our timeout
        if time.time() - start_time > timeout:
            return {'current_price': None, 'open_price': None}
        
        symbol_with_suffix = f"{clean_symbol}{suffix}"
        result = get_symbol_price_data(symbol_with_suffix)
        
        if result:
            return result
    
    # If we get here, no data was found for any variant
    return {'current_price': None, 'open_price': None}

# Process symbols in parallel to speed up data fetching
def process_symbols_in_parallel(symbols, max_workers=10):
    """Fetch price data for multiple symbols in parallel"""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a future-to-symbol mapping
        future_to_symbol = {executor.submit(fetch_stock_data_with_timeout, symbol): symbol for symbol in symbols}
        
        # Process completed futures as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                results[symbol] = data
            except Exception as e:
                st.warning(f"Error processing {symbol}: {e}")
                results[symbol] = {'current_price': None, 'open_price': None}
            
            # Update progress (approximately, since these complete out of order)
            progress = (i + 1) / len(symbols)
            st.session_state.progress = min(progress, 1.0)
    
    return results

# Function to get current prices for symbols using yfinance (not cached)
def get_current_prices(symbols):
    """Get current prices for a list of symbols in parallel with progress tracking"""
    if not symbols:
        return {}
    
    try:
        # Initialize progress tracking
        st.session_state.progress = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Start progress updating in a separate thread
        def update_progress_bar():
            while st.session_state.progress < 1.0:
                progress_bar.progress(st.session_state.progress)
                status_text.text(f"Fetching price data... {int(st.session_state.progress * 100)}%")
                time.sleep(0.1)
        
        # Start fetching data
        ticker_data = process_symbols_in_parallel(symbols)
        
        # Complete the progress bar
        progress_bar.progress(100)
        status_text.empty()
        
        return ticker_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return {}

# Function to update the trade data with market prices
def update_trade_data_with_prices(df):
    """Update trade dataframe with current prices where needed"""
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    updated_df = df.copy()
    
    # Get list of unique symbols that need price data
    symbols_needing_entry_price = updated_df[updated_df['EntryPrice'].isna()]['Symbol'].unique().tolist()
    symbols_needing_exit_price = updated_df['Symbol'].unique().tolist()
    
    # Combine lists and remove duplicates
    all_symbols = list(set(symbols_needing_entry_price + symbols_needing_exit_price))
    
    # Show how many symbols we're processing
    st.info(f"Fetching current prices for {len(all_symbols)} symbols...")
    
    # Get current prices for all needed symbols
    price_data = get_current_prices(all_symbols)
    
    # Count how many prices were successfully fetched
    successful_fetches = sum(1 for symbol in price_data if price_data[symbol]['current_price'] is not None)
    st.success(f"Successfully fetched prices for {successful_fetches} out of {len(all_symbols)} symbols")
    
    # Update EntryPrice where missing with Open price
    for idx, row in updated_df.iterrows():
        symbol = row['Symbol']
        if symbol in price_data:
            # Fill missing entry prices with today's open
            if pd.isna(row['EntryPrice']) and price_data[symbol]['open_price'] is not None:
                updated_df.at[idx, 'EntryPrice'] = price_data[symbol]['open_price']
            
            # Update exit price with current price for all trades
            if price_data[symbol]['current_price'] is not None:
                updated_df.at[idx, 'ExitPrice'] = price_data[symbol]['current_price']
                
                # Recalculate PnL if we have both prices
                if not pd.isna(updated_df.at[idx, 'EntryPrice']):
                    entry_price = updated_df.at[idx, 'EntryPrice']
                    exit_price = updated_df.at[idx, 'ExitPrice']
                    direction = updated_df.at[idx, 'Direction']
                    
                    # Handle Counter-Long and Counter-Short correctly
                    if direction == 'Counter-Long':  # Short position
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                    elif direction == 'Counter-Short':  # Long position
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        # For other directions if any
                        if 'LONG' in direction:
                            pnl_pct = (exit_price - entry_price) / entry_price * 100
                        else:
                            pnl_pct = (entry_price - exit_price) / entry_price * 100
                    
                    updated_df.at[idx, 'PnL_Pct'] = pnl_pct
                    
                    # Recalculate absolute PnL if possible
                    if 'PnL_Abs' in updated_df.columns:
                        # Simple calculation for demo - adjust as needed based on actual position sizing
                        updated_df.at[idx, 'PnL_Abs'] = pnl_pct * entry_price / 100
    
    # Add a column to indicate when the data was last updated
    updated_df['LastRefreshed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return updated_df

def main():
    st.title("Active Trades Dashboard")
    
    # Initialize session state for progress tracking
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    
    # Add a toggle for live price updates
    update_prices = st.sidebar.checkbox("Update prices from Yahoo Finance", value=True)
    
    # Connect to database
    engine = connect_to_db()
    
    if engine is None:
        st.error("Failed to connect to database. Please check your secrets.toml configuration.")
        st.stop()
    else:
        st.success("Database connection successful!")
    
    # Fetch active trades data
    active_trades = get_table(engine, "active_trades")
    
    st.header("Active Trades")
    if active_trades.empty:
        st.info("No active trades found")
    else:
        # Update with current market prices if enabled
        if update_prices:
            with st.spinner("Fetching current market prices..."):
                active_trades = update_trade_data_with_prices(active_trades)
            st.success("Price updates completed!")
        else:
            st.warning("Live price updates are disabled. Toggle the checkbox in the sidebar to enable.")
        
        # Create filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbols = st.multiselect("Select Symbols", options=sorted(active_trades["Symbol"].unique()))
        
        with col2:
            directions = st.multiselect("Select Directions", options=sorted(active_trades["Direction"].unique()))
        
        with col3:
            status = st.multiselect("Select Status", options=sorted(active_trades["Status"].unique()))
        
        # Apply filters
        filtered_trades = active_trades.copy()
        if symbols:
            filtered_trades = filtered_trades[filtered_trades["Symbol"].isin(symbols)]
        
        if directions:
            filtered_trades = filtered_trades[filtered_trades["Direction"].isin(directions)]
        
        if status:
            filtered_trades = filtered_trades[filtered_trades["Status"].isin(status)]
        
        # Display filtered data
        st.dataframe(filtered_trades, use_container_width=True)
        
        # Display summary metrics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Active Trades", len(filtered_trades))
        
        with col2:
            avg_pnl = filtered_trades["PnL_Pct"].mean() if "PnL_Pct" in filtered_trades.columns else 0
            st.metric("Average PnL %", f"{avg_pnl:.2f}%")
        
        with col3:
            # Count long positions (Counter-Short is Long)
            long_count = len(filtered_trades[filtered_trades["Direction"] == "Counter-Short"]) if "Direction" in filtered_trades.columns else 0
            st.metric("Long Positions", long_count)
        
        with col4:
            # Count short positions (Counter-Long is Short)
            short_count = len(filtered_trades[filtered_trades["Direction"] == "Counter-Long"]) if "Direction" in filtered_trades.columns else 0
            st.metric("Short Positions", short_count)
        
        # Create a chart of PnL by symbol
        st.subheader("PnL by Symbol")
        if "Symbol" in filtered_trades.columns and "PnL_Pct" in filtered_trades.columns and not filtered_trades.empty:
            pnl_by_symbol = filtered_trades.groupby("Symbol")["PnL_Pct"].mean().reset_index()
            fig = px.bar(pnl_by_symbol, x="Symbol", y="PnL_Pct", 
                         title="Average PnL % by Symbol",
                         color="PnL_Pct",
                         color_continuous_scale=["red", "green"],
                         labels={"PnL_Pct": "PnL %", "Symbol": "Ticker"})
            st.plotly_chart(fig, use_container_width=True)
        
        # Add last refresh time at the bottom
        st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add refresh button
        if st.button("Refresh Data"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
