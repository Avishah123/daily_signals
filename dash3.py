import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

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

# Function to get current prices for symbols using yfinance (not cached)
def get_current_prices(symbols):
    """Get current prices for a list of symbols using yfinance"""
    if not symbols:
        return {}
    
    try:
        # For Indian markets, we'll try with .NS suffix first
        ticker_data = {}
        
        for symbol in symbols:
            try:
                # Remove any existing suffix if present
                clean_symbol = symbol.split('.')[0] if '.' in symbol else symbol
                
                # First try with .NS suffix for NSE symbols (most common for Indian stocks)
                ticker = yf.Ticker(f"{clean_symbol}.NS")
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    ticker_data[symbol] = {
                        'current_price': hist['Close'].iloc[-1],
                        'open_price': hist['Open'].iloc[-1]
                    }
                    # st.info(f"Successfully fetched data for {clean_symbol}.NS")
                else:
                    # If no data with .NS, try with .BO suffix for BSE symbols
                    ticker = yf.Ticker(f"{clean_symbol}.BO")
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        ticker_data[symbol] = {
                            'current_price': hist['Close'].iloc[-1],
                            'open_price': hist['Open'].iloc[-1]
                        }
                        # st.info(f"Successfully fetched data for {clean_symbol}.BO")
                    else:
                        # As a last resort, try the symbol as-is
                        ticker = yf.Ticker(clean_symbol)
                        hist = ticker.history(period="1d")
                        
                        if not hist.empty:
                            ticker_data[symbol] = {
                                'current_price': hist['Close'].iloc[-1],
                                'open_price': hist['Open'].iloc[-1]
                            }
                            # st.info(f"Successfully fetched data for {clean_symbol}")
                        else:
                            st.warning(f"Could not fetch data for {symbol} - please check the symbol")
                            ticker_data[symbol] = {
                                'current_price': None,
                                'open_price': None
                            }
            except Exception as e:
                st.warning(f"Error fetching data for {symbol}: {e}")
                ticker_data[symbol] = {
                    'current_price': None,
                    'open_price': None
                }
        
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
    
    # Get current prices for all needed symbols
    price_data = get_current_prices(all_symbols)
    
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
        # Update with current market prices
        with st.spinner("Fetching current market prices..."):
            active_trades = update_trade_data_with_prices(active_trades)
        
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
