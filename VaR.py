import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import datetime

# Configuration
st.set_page_config(layout="wide", page_title="Portfolio VaR and Stress Testing")

# Initial Constant Setup (Will be overridden by user input)
INITIAL_PORTFOLIO_VALUE = 0
CASH_WEIGHT = 0.0
TICKERS = []
WEIGHTS = np.array([])


# Financial Functions

@st.cache_data
def fetch_data(tickers, start_date, end_date):
    """Fetches adjusted closing prices for the given tickers."""
    try:
        # Using auto_adjust=True ensures 'Close' is the adjusted price and simplifies the column structure.
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        
        # If only one ticker was fetched, yfinance returns a Series. Convert it back to a DataFrame.
        if isinstance(data, pd.Series):
            ticker_name = tickers[0] if isinstance(tickers, list) else tickers
            data = data.to_frame(name=ticker_name)

        # Check if the downloaded data is truly empty or just has a single row
        if data.empty or len(data) <= 1:
            st.error("Data fetch failed: No historical data returned for the specified dates/tickers.")
            return pd.DataFrame()
            
        return data.dropna() # Drop any rows with missing data after fetch
        
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance. Please check ticker symbols or connection. Error: {e}")
        return pd.DataFrame()

def calculate_portfolio_returns(data, weights):
    """Calculates daily log returns and portfolio returns (cash component is implicitly 0 return)."""
    log_returns = np.log(data / data.shift(1)).dropna()
    portfolio_returns = log_returns @ weights
    return log_returns, portfolio_returns

def calculate_historical_var(portfolio_returns, confidence_level, time_horizon=1, initial_value=100000):
    """Calculates Historical VaR (Non-Parametric)."""
    alpha = 1 - confidence_level
    t_day_returns = portfolio_returns * np.sqrt(time_horizon)

    var_return = -np.quantile(t_day_returns, alpha)
    var_value = var_return * initial_value
    return var_return, var_value

def calculate_parametric_var(portfolio_returns, confidence_level, time_horizon=1, initial_value=100000):
    """Calculates Parametric (Variance-Covariance) VaR."""
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    
    alpha = 1 - confidence_level
    z_score = norm.ppf(alpha)

    # VaR formula: VaR = - (mu*T + Z_alpha * sigma * sqrt(T))
    var_return = -(mu * time_horizon + z_score * sigma * np.sqrt(time_horizon))
    var_value = var_return * initial_value
    return var_return, var_value

def run_stress_test(log_returns, weights, market_shock_percent, rate_shock_percent, rate_sensitive_tickers, confidence_level, time_horizon=1, initial_value=100000):
    """Applies a dual-factor stress scenario (Market + Rate Shock) and calculates the stressed VaR."""
    
    market_shock_factor = market_shock_percent / 100.0
    rate_shock_factor = rate_shock_percent / 100.0

    stressed_log_returns = log_returns.copy()
    
    # 1. Apply general market shock to all risky assets
    stressed_log_returns = stressed_log_returns + market_shock_factor
    
    # 2. Apply additional rate-sensitive shock only to specific assets
    # This simulates a bond price change due to an interest rate move, independent of the general market shock.
    rate_sensitive_cols = [t for t in rate_sensitive_tickers if t in log_returns.columns]
    
    if rate_sensitive_cols:
        stressed_log_returns[rate_sensitive_cols] += rate_shock_factor
    
    stressed_portfolio_returns = stressed_log_returns @ weights

    stressed_hist_var_return, stressed_hist_var_value = calculate_historical_var(
        stressed_portfolio_returns, confidence_level, time_horizon, initial_value
    )
    
    stressed_mean = stressed_portfolio_returns.mean() * time_horizon
    stressed_loss_value = -stressed_mean * initial_value
    
    return stressed_hist_var_return, stressed_hist_var_value, stressed_loss_value


# Streamlit Dashboard Layout

st.title("Portfolio VaR & Stress Testing Dashboard")
st.markdown("---")

# Sidebar Inputs
st.sidebar.header("Configuration")

# --- New Capital Input ---
initial_capital = st.sidebar.number_input(
    "1. Investment Capital ($)",
    min_value=1000,
    value=100000,
    step=1000,
    format="%d",
    help="Total capital available for investment and cash reserve."
)
INITIAL_PORTFOLIO_VALUE = initial_capital


# --- Portfolio Input Section ---
st.sidebar.markdown("### 2. Portfolio Definition (Risky Assets)")
ticker_input = st.sidebar.text_area(
    "Asset Tickers (Comma-separated)",
    "AAPL, MSFT, GOOGL, AMZN, BND, VNQ"
)
weight_input = st.sidebar.text_area(
    "Weights (Comma-separated, sum <= 1.0)",
    "0.20, 0.15, 0.15, 0.10, 0.20, 0.10",
    help="Weights for securities. Any remaining weight (1.0 - sum) is allocated to CASH."
)

# Parse and Validate Portfolio Inputs
try:
    TICKERS = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    user_weights_list = [float(w.strip()) for w in weight_input.split(',') if w.strip()]
    WEIGHTS = np.array(user_weights_list)
    
    if not TICKERS:
        st.error("Please enter valid Ticker Symbols.")
        st.stop()
    if not user_weights_list:
        st.error("Please enter valid portfolio Weights.")
        st.stop()
    if len(TICKERS) != len(WEIGHTS):
        st.error("Number of Tickers must match the number of Weights.")
        st.stop()
    
    security_weight_sum = np.sum(WEIGHTS)

    if security_weight_sum > 1.0:
        st.error(f"Security weights sum must not exceed 1.0. Current sum: {security_weight_sum:.4f}")
        st.stop()
    
    # Calculate Cash Weight
    CASH_WEIGHT = 1.0 - security_weight_sum


except ValueError:
    st.error("Invalid input format for Tickers or Weights. Ensure they are comma-separated and weights are numerical.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during input parsing: {e}")
    st.stop()

# --- Date and VaR Parameters ---
start_date = st.sidebar.date_input("3. Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("4. End Date", datetime.date.today())

st.sidebar.markdown("### 5. VaR Parameters")
confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95, 1) / 100.0
var_horizon = st.sidebar.slider("VaR Horizon (Days)", 1, 30, 1, 1)

# Data Fetching and Preparation
if start_date >= end_date:
    st.error("The start date must be before the end date.")
    st.stop()

# Load data
st.info(f"Fetching market data for {TICKERS}... please wait.")
data = fetch_data(TICKERS, start_date, end_date)
st.empty() # Clear the info message after fetching

if data.empty:
    st.warning("Application halted because necessary financial data could not be loaded. Please check the date range and console for errors.")
    st.stop()

# Prepare returns
log_returns, portfolio_returns = calculate_portfolio_returns(data, WEIGHTS)

# Check if returns calculation was successful (ensures data wasn't too short)
if portfolio_returns.empty:
    st.error("Returns calculation failed. Ensure the time range contains at least two trading days.")
    st.stop()

st.header("Portfolio Overview")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Investment Capital", f"${INITIAL_PORTFOLIO_VALUE:,.2f}")
    st.metric("Security Investment", f"${security_weight_sum * INITIAL_PORTFOLIO_VALUE:,.2f}")
    st.metric("Cash Reserve", f"${CASH_WEIGHT * INITIAL_PORTFOLIO_VALUE:,.2f}")

with col2:
    # Display the user-defined portfolio components including CASH
    display_tickers = TICKERS + ['CASH']
    display_weights = np.append(WEIGHTS, CASH_WEIGHT)
    
    weight_df = pd.Series(display_weights, index=display_tickers).to_frame(name="Weight")
    weight_df.index.name = "Asset"
    st.dataframe(weight_df, use_container_width=True)


# 1. VaR Calculation Section
st.header("1. Value at Risk (VaR) Calculation")

# Pass INITIAL_PORTFOLIO_VALUE to the VaR functions
hist_var_return, hist_var_value = calculate_historical_var(portfolio_returns, confidence_level, var_horizon, INITIAL_PORTFOLIO_VALUE)
param_var_return, param_var_value = calculate_parametric_var(portfolio_returns, confidence_level, var_horizon, INITIAL_PORTFOLIO_VALUE)

var_cols = st.columns(4)

with var_cols[0]:
    st.metric(f"Confidence Level", f"{confidence_level*100:.0f}%")
with var_cols[1]:
    st.metric(f"Horizon", f"{var_horizon} Days")

with var_cols[2]:
    st.markdown("### Historical VaR")
    st.metric("Loss (%)", f"{hist_var_return * 100:.2f}%")
    st.metric("Loss ($)", f"${hist_var_value:,.2f}", delta_color="inverse")
    st.caption(f"Max expected loss in {var_horizon} days (Historical Method).")

with var_cols[3]:
    st.markdown("### Parametric VaR")
    st.metric("Loss (%)", f"{param_var_return * 100:.2f}%")
    st.metric("Loss ($)", f"${param_var_value:,.2f}", delta_color="inverse")
    st.caption(f"Max expected loss in {var_horizon} days (Gaussian Method).")

st.markdown("---")

# 2. Stress Testing Section
st.header("2. Stress Testing (Dual-Factor Scenario)")
st.caption("Define simultaneous shocks to the general market and to interest rate-sensitive assets.")

stress_test_cols = st.columns([1, 2])
with stress_test_cols[0]:
    market_shock_percent = st.number_input(
        "1. General Market Shock (%)",
        value=-10.0,
        step=1.0,
        format="%.1f",
        help="Applied to all risky assets."
    )
    rate_shock_percent = st.number_input(
        "2. Rate-Sensitive Shock (%)",
        value=-5.0,
        step=1.0,
        format="%.1f",
        help="An ADDITIONAL shock applied to selected rate-sensitive assets (e.g., bonds fall 5% in price due to rate hike)."
    )
    rate_sensitive_assets = st.multiselect(
        "3. Select Rate-Sensitive Assets",
        options=TICKERS,
        default=[t for t in TICKERS if t in ['BND', 'VNQ']] # Example defaults
    )

    run_stress = st.button("Run Stress Test", type="primary")

with stress_test_cols[1]:
    if run_stress:
        stressed_var_return, stressed_var_value, stressed_loss_value = run_stress_test(
            log_returns, WEIGHTS, market_shock_percent, rate_shock_percent, rate_sensitive_assets, confidence_level, var_horizon, INITIAL_PORTFOLIO_VALUE
        )

        st.markdown(f"**Scenario:** Market drops by **{market_shock_percent:.1f}%**, and selected rate assets drop an **additional {rate_shock_percent:.1f}%**.")
        
        stress_metrics = st.columns(2)
        with stress_metrics[0]:
            st.markdown("#### Stressed VaR (Historical)")
            st.metric("Loss (%)", f"{stressed_var_return * 100:.2f}%")
            st.metric("Loss ($)", f"${stressed_var_value:,.2f}", delta_color="inverse")
            st.caption("VaR recalculated using the historical distribution shifted by the dual shock.")
        
        with stress_metrics[1]:
            st.markdown("#### Expected Loss (Scenario)")
            st.metric("Expected Loss (%)", f"{-stressed_loss_value / INITIAL_PORTFOLIO_VALUE * 100:.2f}%")
            st.metric("Loss ($)", f"${stressed_loss_value:,.2f}", delta_color="inverse")
            st.caption("The expected mean loss for the portfolio's risky assets under this specific shock scenario.")

    else:
        st.info("Define your dual-factor shock scenario (Market and Rate-Sensitive) and click 'Run Stress Test' to analyze the scenario.")

st.markdown("---")

# 3. Visualization Section
st.header("3. Portfolio Returns Distribution")

# Create a DataFrame for plotting returns
plot_data = portfolio_returns.to_frame(name="Daily Returns")

# Convert VaR from loss to return (negative)
hist_var_line = -hist_var_return / INITIAL_PORTFOLIO_VALUE / np.sqrt(var_horizon)
param_var_line = -param_var_return / INITIAL_PORTFOLIO_VALUE / np.sqrt(var_horizon)

st.line_chart(
    plot_data,
    y="Daily Returns",
    height=400
)

# Show VaR lines on the chart
if hist_var_line is not None:
    st.markdown(f"***Historical VaR*** (Daily equivalent): **{hist_var_line * 100:.2f}%**")
if param_var_line is not None:
    st.markdown(f"***Parametric VaR*** (Daily equivalent): **{param_var_line * 100:.2f}%**")

st.markdown("---")
st.subheader("Returns Distribution Histogram")
st.bar_chart(plot_data, height=300)

st.markdown("""
***Note on Visualization:***
The VaR thresholds shown are the **daily equivalent** of the calculated T-day VaR, making them comparable to the daily return values plotted on the graph. Any daily return *below* these lines represents an event exceeding the VaR loss estimate for that day.
""")