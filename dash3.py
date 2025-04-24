import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np



# Page config
st.set_page_config(
    page_title="Trading System Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to connect to the database
# @st.cache_resource
def get_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"]
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Function to fetch active signals
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_active_signals():
    """Fetch active signals from the database"""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                id, symbol, signal_date, entry_date, exit_date, entry_price, 
                direction, net_value_pct_change, lookback, holding,
                current_price, current_pnl_pct, current_pnl_abs,
                days_in_trade, days_remaining, last_updated
            FROM active_signals
            ORDER BY days_remaining ASC
        """
        df = pd.read_sql(query, conn)
        
        # Convert date columns to datetime
        date_columns = ['signal_date', 'entry_date', 'exit_date', 'last_updated']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error fetching active signals: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Function to fetch backtest results
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def get_backtest_results(days=30, limit=1000):
    """Fetch backtest results from the database"""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        # Date filter
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = f"""
            SELECT 
                id, symbol, signal_date, entry_date, exit_date, 
                entry_price, exit_price, direction, pnl_pct, pnl_abs,
                net_value_pct_change, lookback, holding, price_type_used,
                status, backtest_date, days_in_trade
            FROM backtest_results
            WHERE backtest_date >= '{cutoff_date.strftime('%Y-%m-%d')}'
            ORDER BY backtest_date DESC, entry_date DESC
            LIMIT {limit}
        """
        df = pd.read_sql(query, conn)
        
        # Convert date columns to datetime
        date_columns = ['signal_date', 'entry_date', 'exit_date', 'backtest_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error fetching backtest results: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Function to fetch backtest summaries
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def get_backtest_summaries(limit=20):
    """Fetch backtest summaries from the database"""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = f"""
            SELECT 
                id, backtest_date, days_period, win_rate_threshold,
                total_trades, winning_trades, win_rate, avg_pnl, total_pnl,
                price_type_used, best_trade_symbol, best_trade_pnl,
                worst_trade_symbol, worst_trade_pnl, created_at
            FROM backtest_summary
            ORDER BY backtest_date DESC
            LIMIT {limit}
        """
        df = pd.read_sql(query, conn)
        
        # Convert date columns to datetime
        date_columns = ['backtest_date', 'created_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"Error fetching backtest summaries: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Function to show active trades tab
def show_active_trades_tab():
    st.header("Active Trades")
    
    # Fetch active signals
    with st.spinner("Fetching active trades..."):
        active_signals = get_active_signals()
    
    if active_signals.empty:
        st.info("No active trades found in the database.")
        return
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Active Trades", len(active_signals))
    
    with col2:
        winning_trades = (active_signals['current_pnl_pct'] > 0).sum()
        win_rate = winning_trades / len(active_signals) * 100
        st.metric("Winning Positions", f"{winning_trades} ({win_rate:.1f}%)")
    
    with col3:
        avg_pnl = active_signals['current_pnl_pct'].mean()
        st.metric("Average PnL", f"{avg_pnl:.2f}%", 
                 delta=f"{'+' if avg_pnl > 0 else ''}{avg_pnl:.2f}%")
    
    with col4:
        total_pnl = active_signals['current_pnl_abs'].sum()
        st.metric("Total PnL (₹)", f"{total_pnl:.2f}", 
                 delta=f"{'+' if total_pnl > 0 else ''}{total_pnl:.2f}")
    
    # Due-for-exit trades
    due_trades = active_signals[active_signals['days_remaining'] <= 2]
    if not due_trades.empty:
        st.subheader(f"Trades Due for Exit Soon (≤ 2 days): {len(due_trades)}")
        
        # Format the due trades table
        due_trades_display = due_trades.copy()
        due_trades_display = due_trades_display[[
            'symbol', 'direction', 'entry_date', 'exit_date', 
            'entry_price', 'current_price', 'current_pnl_pct', 'days_remaining'
        ]]
        
        # Rename columns for display
        due_trades_display.columns = [
            'Symbol', 'Direction', 'Entry Date', 'Exit Date', 
            'Entry Price', 'Current Price', 'PnL %', 'Days Left'
        ]
        
        # Format numeric columns
        due_trades_display['PnL %'] = due_trades_display['PnL %'].map('{:,.2f}%'.format)
        due_trades_display['Entry Price'] = due_trades_display['Entry Price'].map('₹{:,.2f}'.format)
        due_trades_display['Current Price'] = due_trades_display['Current Price'].map('₹{:,.2f}'.format)
        
        # Style the DataFrame
        st.dataframe(
            due_trades_display,
            use_container_width=True,
            hide_index=True
        )
    
    # Create tabs for different views
    trade_tabs = st.tabs(["All Trades", "Performance by Symbol", "Charts"])
    
    with trade_tabs[0]:
        st.subheader("All Active Trades")
        
        # Add filters in the sidebar
        st.sidebar.header("Active Trades Filters")
        
        # Symbol filter
        all_symbols = ['All'] + sorted(active_signals['symbol'].unique().tolist())
        selected_symbol = st.sidebar.selectbox("Filter by Symbol", all_symbols)
        
        # Direction filter
        all_directions = ['All'] + sorted(active_signals['direction'].unique().tolist())
        selected_direction = st.sidebar.selectbox("Filter by Direction", all_directions)
        
        # PnL filter
        pnl_options = ['All', 'Profit', 'Loss']
        selected_pnl = st.sidebar.selectbox("Filter by PnL", pnl_options)
        
        # Apply filters
        filtered_signals = active_signals.copy()
        
        if selected_symbol != 'All':
            filtered_signals = filtered_signals[filtered_signals['symbol'] == selected_symbol]
            
        if selected_direction != 'All':
            filtered_signals = filtered_signals[filtered_signals['direction'] == selected_direction]
            
        if selected_pnl == 'Profit':
            filtered_signals = filtered_signals[filtered_signals['current_pnl_pct'] > 0]
        elif selected_pnl == 'Loss':
            filtered_signals = filtered_signals[filtered_signals['current_pnl_pct'] <= 0]
        
        # Sort options
        sort_options = {
            'Days Remaining (Asc)': ('days_remaining', True),
            'Days Remaining (Desc)': ('days_remaining', False),
            'PnL % (Best First)': ('current_pnl_pct', False),
            'PnL % (Worst First)': ('current_pnl_pct', True),
            'Entry Date (Newest First)': ('entry_date', False),
            'Entry Date (Oldest First)': ('entry_date', True)
        }
        
        sort_selection = st.sidebar.selectbox("Sort by", list(sort_options.keys()))
        sort_column, sort_ascending = sort_options[sort_selection]
        
        filtered_signals = filtered_signals.sort_values(by=sort_column, ascending=sort_ascending)
        
        # Format the table for display
        display_df = filtered_signals.copy()
        display_columns = [
            'symbol', 'direction', 'entry_date', 'exit_date', 
            'entry_price', 'current_price', 'current_pnl_pct', 
            'days_in_trade', 'days_remaining'
        ]
        
        display_df = display_df[display_columns].copy()
        
        # Rename columns for display
        display_df.columns = [
            'Symbol', 'Direction', 'Entry Date', 'Exit Date', 
            'Entry Price', 'Current Price', 'PnL %', 
            'Days in Trade', 'Days Remaining'
        ]
        
        # Format numeric columns
        display_df['PnL %'] = display_df['PnL %'].map('{:,.2f}%'.format)
        display_df['Entry Price'] = display_df['Entry Price'].map('₹{:,.2f}'.format)
        display_df['Current Price'] = display_df['Current Price'].map('₹{:,.2f}'.format)
        
        # Apply conditional formatting
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Displaying {len(filtered_signals)} of {len(active_signals)} active trades")
        
    with trade_tabs[1]:
        st.subheader("Performance by Symbol")
        
        # Group by symbol and calculate metrics
        symbol_performance = active_signals.groupby('symbol').agg({
            'id': 'count',
            'current_pnl_pct': ['mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0],
            'current_pnl_abs': 'sum'
        })
        
        # Flatten multi-level columns
        symbol_performance.columns = [
            'Trade Count', 'Avg PnL %', 'Win Rate %', 'Total PnL (₹)'
        ]
        
        # Reset index
        symbol_performance = symbol_performance.reset_index()
        
        # Sort by total PnL
        symbol_performance = symbol_performance.sort_values('Total PnL (₹)', ascending=False)
        
        # Display as table
        st.dataframe(
            symbol_performance,
            use_container_width=True,
            hide_index=True
        )
        
        # Create chart
        symbol_chart = px.bar(
            symbol_performance, 
            x='symbol', 
            y='Total PnL (₹)',
            color='Avg PnL %',
            color_continuous_scale='RdYlGn',
            title='PnL by Symbol',
            hover_data=['Trade Count', 'Win Rate %', 'Avg PnL %']
        )
        
        symbol_chart.update_layout(xaxis_title="Symbol", yaxis_title="Total PnL (₹)")
        st.plotly_chart(symbol_chart, use_container_width=True)
        
    with trade_tabs[2]:
        st.subheader("Visual Analysis")
        
        # PnL distribution
        fig1 = px.histogram(
            active_signals, 
            x='current_pnl_pct',
            nbins=20,
            title='PnL Distribution',
            color_discrete_sequence=['lightblue']
        )
        
        fig1.add_vline(x=0, line_dash="dash", line_color="red")
        fig1.update_layout(xaxis_title="PnL %", yaxis_title="Number of Trades")
        st.plotly_chart(fig1, use_container_width=True)
        
        # PnL by days in trade - FIXED VERSION
        plot_data = active_signals.copy()
        # Calculate absolute value for sizing and handle NaN values
        plot_data['abs_pnl'] = plot_data['current_pnl_abs'].abs()
        
        # Fix: Replace NaN values with a default value of 1
        plot_data['abs_pnl'] = plot_data['abs_pnl'].fillna(1)
        
        # Add a column to indicate positive or negative PnL for coloring
        plot_data['pnl_status'] = plot_data['current_pnl_pct'].apply(
            lambda x: 'Profit' if x > 0 else 'Loss'
        )
        
        fig2 = px.scatter(
            plot_data, 
            x='days_in_trade', 
            y='current_pnl_pct',
            color='pnl_status',  # Color by profit/loss status
            size='abs_pnl',  # Use absolute value for sizing
            size_max=30,  # Limit maximum bubble size
            color_discrete_map={'Profit': 'green', 'Loss': 'red'},
            hover_data=['symbol', 'direction', 'entry_date', 'current_price', 'current_pnl_pct'],
            title='PnL vs Days in Trade'
        )
        
        fig2.add_hline(y=0, line_dash="dash", line_color="black")
        fig2.update_layout(
            xaxis_title="Days in Trade", 
            yaxis_title="PnL %",
            legend_title="Result"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # PnL by direction
        direction_performance = active_signals.groupby('direction').agg({
            'id': 'count',
            'current_pnl_pct': ['mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0],
            'current_pnl_abs': 'sum'
        })
        
        # Flatten multi-level columns
        direction_performance.columns = [
            'Trade Count', 'Avg PnL %', 'Win Rate %', 'Total PnL (₹)'
        ]
        
        # Reset index
        direction_performance = direction_performance.reset_index()
        
        fig3 = px.bar(
            direction_performance,
            x='direction',
            y='Total PnL (₹)',
            color='Avg PnL %',
            title='Performance by Direction',
            hover_data=['Trade Count', 'Win Rate %']
        )
        
        fig3.update_layout(xaxis_title="Direction", yaxis_title="Total PnL (₹)")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Add a new chart: PnL by holding duration (grouped)
        if len(active_signals) > 0:
            # Create holding period groups
            active_signals['holding_group'] = pd.cut(
                active_signals['days_in_trade'],
                bins=[0, 2, 5, 10, 20, 100],
                labels=['0-2 days', '3-5 days', '6-10 days', '11-20 days', '21+ days']
            )
            
            holding_perf = active_signals.groupby('holding_group').agg({
                'id': 'count',
                'current_pnl_pct': ['mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0],
                'current_pnl_abs': 'sum'
            })
            
            # Flatten multi-level columns
            holding_perf.columns = ['Trade Count', 'Avg PnL %', 'Win Rate %', 'Total PnL (₹)']
            holding_perf = holding_perf.reset_index()
            
            fig4 = px.bar(
                holding_perf,
                x='holding_group',
                y=['Avg PnL %', 'Win Rate %'],
                barmode='group',
                title='Performance by Holding Duration',
                labels={'value': 'Percentage', 'variable': 'Metric'},
            )
            
            fig4.update_layout(xaxis_title="Holding Duration", yaxis_title="Percentage")
            st.plotly_chart(fig4, use_container_width=True)

# Function to show backtest results tab
def show_backtest_tab():
    st.header("Backtest Results")
    
    # Display backtest summaries
    with st.spinner("Loading backtest summaries..."):
        backtest_summaries = get_backtest_summaries()
    
    if backtest_summaries.empty:
        st.info("No backtest summaries found in the database.")
    else:
        st.subheader("Recent Backtest Summaries")
        
        # Format for display
        summary_display = backtest_summaries.copy()
        display_columns = [
            'backtest_date', 'days_period', 'win_rate_threshold',
            'total_trades', 'winning_trades', 'win_rate', 'avg_pnl', 'total_pnl'
        ]
        
        summary_display = summary_display[display_columns].copy()
        
        # Rename columns
        summary_display.columns = [
            'Backtest Date', 'Days Period', 'Win Rate Threshold',
            'Total Trades', 'Winning Trades', 'Win Rate %', 'Avg PnL %', 'Total PnL (₹)'
        ]
        
        # Format numbers
        summary_display['Win Rate %'] = summary_display['Win Rate %'].map('{:,.2f}%'.format)
        summary_display['Avg PnL %'] = summary_display['Avg PnL %'].map('{:,.2f}%'.format)
        summary_display['Total PnL (₹)'] = summary_display['Total PnL (₹)'].map('₹{:,.2f}'.format)
        
        st.dataframe(summary_display, use_container_width=True, hide_index=True)
        
        # Create chart of backtest results over time
        chart_data = backtest_summaries.copy()
        chart_data['backtest_date'] = pd.to_datetime(chart_data['backtest_date']).dt.strftime('%Y-%m-%d')
        
        fig = px.line(
            chart_data, 
            x='backtest_date', 
            y=['win_rate', 'avg_pnl'],
            title='Backtest Performance Over Time',
            labels={'value': 'Percentage', 'variable': 'Metric', 'backtest_date': 'Date'},
            color_discrete_map={'win_rate': 'blue', 'avg_pnl': 'green'}
        )
        
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    # Fetch backtest results
    st.sidebar.header("Backtest Results Filters")
    
    # Date range filter
    days_options = [7, 15, 30, 60, 90, 180, 365]
    selected_days = st.sidebar.selectbox("Show results from last X days", days_options, index=2)  # Default 30
    
    # Limit number of records
    limit_options = [100, 500, 1000, 5000]
    selected_limit = st.sidebar.selectbox("Maximum records to fetch", limit_options, index=2)  # Default 1000
    
    with st.spinner(f"Fetching backtest results from the last {selected_days} days..."):
        backtest_results = get_backtest_results(days=selected_days, limit=selected_limit)
    
    if backtest_results.empty:
        st.info(f"No backtest results found from the last {selected_days} days.")
        return
    
    # Create tabs for different views
    backtest_tabs = st.tabs(["Performance Metrics", "Trade List", "Analysis by Symbol"])
    
    with backtest_tabs[0]:
        st.subheader("Backtest Performance Metrics")
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(backtest_results))
        
        with col2:
            winning_trades = (backtest_results['pnl_pct'] > 0).sum()
            win_rate = winning_trades / len(backtest_results) * 100
            st.metric("Win Rate", f"{win_rate:.2f}%")
        
        with col3:
            avg_pnl = backtest_results['pnl_pct'].mean()
            st.metric("Average PnL", f"{avg_pnl:.2f}%", 
                     delta=f"{'+' if avg_pnl > 0 else ''}{avg_pnl:.2f}%")
        
        with col4:
            total_pnl = backtest_results['pnl_abs'].sum()
            st.metric("Total PnL (₹)", f"{total_pnl:.2f}", 
                     delta=f"{'+' if total_pnl > 0 else ''}{total_pnl:.2f}")
        
        # PnL distribution
        fig1 = px.histogram(
            backtest_results, 
            x='pnl_pct',
            nbins=25,
            title='PnL Distribution',
            color_discrete_sequence=['lightgreen']
        )
        
        fig1.add_vline(x=0, line_dash="dash", line_color="red")
        fig1.update_layout(xaxis_title="PnL %", yaxis_title="Number of Trades")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Performance by direction
        direction_perf = backtest_results.groupby('direction').agg({
            'id': 'count',
            'pnl_pct': ['mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0],
            'pnl_abs': 'sum'
        })
        
        # Flatten multi-level columns
        direction_perf.columns = ['Trade Count', 'Avg PnL %', 'Win Rate %', 'Total PnL (₹)']
        direction_perf = direction_perf.reset_index()
        
        fig2 = px.bar(
            direction_perf,
            x='direction',
            y=['Trade Count', 'Win Rate %', 'Avg PnL %'],
            barmode='group',
            title='Performance by Direction',
            labels={'value': 'Value', 'variable': 'Metric'},
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Performance over time
        backtest_results['entry_month'] = backtest_results['entry_date'].dt.strftime('%Y-%m')
        time_perf = backtest_results.groupby('entry_month').agg({
            'id': 'count',
            'pnl_pct': ['mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0],
            'pnl_abs': 'sum'
        })
        
        # Flatten multi-level columns
        time_perf.columns = ['Trade Count', 'Avg PnL %', 'Win Rate %', 'Total PnL (₹)']
        time_perf = time_perf.reset_index()
        
        fig3 = px.line(
            time_perf,
            x='entry_month',
            y=['Win Rate %', 'Avg PnL %'],
            title='Performance by Month',
            labels={'value': 'Percentage', 'variable': 'Metric'},
        )
        
        fig3.update_layout(hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True)
        
    with backtest_tabs[1]:
        st.subheader("Backtest Trade List")
        
        # Add filters
        # Symbol filter
        all_symbols = ['All'] + sorted(backtest_results['symbol'].unique().tolist())
        selected_symbol = st.sidebar.selectbox("Filter by Symbol (Backtest)", all_symbols)
        
        # Direction filter
        all_directions = ['All'] + sorted(backtest_results['direction'].unique().tolist())
        selected_direction = st.sidebar.selectbox("Filter by Direction (Backtest)", all_directions)
        
        # Result filter
        result_options = ['All', 'Winners', 'Losers']
        selected_result = st.sidebar.selectbox("Filter by Result", result_options)
        
        # Apply filters
        filtered_results = backtest_results.copy()
        
        if selected_symbol != 'All':
            filtered_results = filtered_results[filtered_results['symbol'] == selected_symbol]
            
        if selected_direction != 'All':
            filtered_results = filtered_results[filtered_results['direction'] == selected_direction]
            
        if selected_result == 'Winners':
            filtered_results = filtered_results[filtered_results['pnl_pct'] > 0]
        elif selected_result == 'Losers':
            filtered_results = filtered_results[filtered_results['pnl_pct'] <= 0]
        
        # Sort options
        sort_options = {
            'PnL % (Best First)': ('pnl_pct', False),
            'PnL % (Worst First)': ('pnl_pct', True),
            'Entry Date (Newest First)': ('entry_date', False),
            'Entry Date (Oldest First)': ('entry_date', True),
            'Exit Date (Newest First)': ('exit_date', False),
            'Exit Date (Oldest First)': ('exit_date', True)
        }
        
        sort_selection = st.sidebar.selectbox("Sort Backtest Results by", list(sort_options.keys()))
        sort_column, sort_ascending = sort_options[sort_selection]
        
        filtered_results = filtered_results.sort_values(by=sort_column, ascending=sort_ascending)
        
        # Format for display
        display_df = filtered_results.copy()
        display_columns = [
            'symbol', 'direction', 'entry_date', 'exit_date', 
            'entry_price', 'exit_price', 'pnl_pct', 'pnl_abs',
            'days_in_trade', 'status'
        ]
        
        display_df = display_df[display_columns].copy()
        
        # Add result column
        display_df['result'] = display_df['pnl_pct'].apply(
            lambda x: 'Win' if x > 0 else 'Loss'
        )
        
        # Rename columns
        display_df.columns = [
            'Symbol', 'Direction', 'Entry Date', 'Exit Date',
            'Entry Price', 'Exit Price', 'PnL %', 'PnL (₹)',
            'Days in Trade', 'Status', 'Result'
        ]
        
        # Format numeric columns
        display_df['PnL %'] = display_df['PnL %'].map('{:,.2f}%'.format)
        display_df['PnL (₹)'] = display_df['PnL (₹)'].map('₹{:,.2f}'.format)
        display_df['Entry Price'] = display_df['Entry Price'].map('₹{:,.2f}'.format)
        display_df['Exit Price'] = display_df['Exit Price'].map('₹{:,.2f}'.format)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Displaying {len(filtered_results)} of {len(backtest_results)} backtest trades")
    
    with backtest_tabs[2]:
        st.subheader("Analysis by Symbol")
        
        # Group by symbol
        symbol_perf = backtest_results.groupby('symbol').agg({
            'id': 'count',
            'pnl_pct': ['mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0],
            'pnl_abs': 'sum'
        })
        
        # Flatten multi-level columns
        symbol_perf.columns = ['Trade Count', 'Avg PnL %', 'Win Rate %', 'Total PnL (₹)']
        symbol_perf = symbol_perf.reset_index()
        
        # Sort by total PnL
        symbol_perf = symbol_perf.sort_values('Total PnL (₹)', ascending=False)
        
        # Display table
        st.dataframe(symbol_perf, use_container_width=True, hide_index=True)
        
        # Create charts
        # Top 10 symbols by PnL
        top_symbols = symbol_perf.sort_values('Total PnL (₹)', ascending=False).head(10)
        
        fig1 = px.bar(
            top_symbols,
            x='symbol',
            y='Total PnL (₹)',
            color='Win Rate %',
            color_continuous_scale='RdYlGn',
            title='Top 10 Symbols by PnL',
            hover_data=['Trade Count', 'Avg PnL %']
        )
        
        fig1.update_layout(xaxis_title="Symbol", yaxis_title="Total PnL (₹)")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Bottom 10 symbols by PnL
        bottom_symbols = symbol_perf.sort_values('Total PnL (₹)', ascending=True).head(10)
        
        fig2 = px.bar(
            bottom_symbols,
            x='symbol',
            y='Total PnL (₹)',
            color='Win Rate %',
            color_continuous_scale='RdYlGn',
            title='Bottom 10 Symbols by PnL',
            hover_data=['Trade Count', 'Avg PnL %']
        )
        
        fig2.update_layout(xaxis_title="Symbol", yaxis_title="Total PnL (₹)")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Win rate vs avg PnL by symbol (bubble chart)
        fig3 = px.scatter(
            symbol_perf,
            x='Win Rate %',
            y='Avg PnL %',
            size='Trade Count',
            color='Total PnL (₹)',
            hover_name='symbol',
            title='Win Rate vs Avg PnL by Symbol'
        )
        
        fig3.update_layout(xaxis_title="Win Rate %", yaxis_title="Avg PnL %")
        st.plotly_chart(fig3, use_container_width=True)

# Main application
def main():
    st.title("Trading System Dashboard")
    
    # Create a last updated indicator
    st.sidebar.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create tabs for different views
    tabs = st.tabs(["Active Trades", "Backtest Results"])
    
    # Display tabs
    with tabs[0]:
        show_active_trades_tab()
    
    with tabs[1]:
        show_backtest_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Dashboard v1.0")
    st.sidebar.markdown("Refresh data using the button below")
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if __name__ == "__main__":
    main()