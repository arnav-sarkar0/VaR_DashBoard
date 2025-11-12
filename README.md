##Portfolio VaR and Stress Testing Dashboard

This Python-based interactive dashboard, built with Streamlit, allows users to perform quantitative risk analysis on a user-defined portfolio. It calculates Value at Risk (VaR) using Historical and Parametric methods and includes a flexible dual-factor stress testing framework (General Market Shock + Rate-Sensitive Asset Shock).

Features

Custom Portfolio: Define asset tickers and weights, including a cash component (risk-free).

VaR Calculation: Calculates T-Day VaR using Non-Parametric (Historical Simulation) and Parametric (Variance-Covariance) methods.

Dual-Factor Stress Testing: Simulate scenarios involving:

A general shock applied to all risky assets.

An additional shock applied only to user-selected rate-sensitive assets (e.g., bonds, REITs).

Interactive Visualization: Displays the portfolio's historical daily returns and highlights the calculated VaR thresholds.

Setup and Installation

Prerequisites

You need Python (3.7+) installed on your system.

1. Clone the Repository

git clone <your-repo-link>
cd portfolio-var-dashboard


2. Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Linux/macOS

OR
.\venv\Scripts\activate   # On Windows


3. Install Dependencies

Install all required Python libraries using the provided requirements.txt file:

pip install -r requirements.txt


Running the Application

Once the dependencies are installed, run the dashboard directly using Streamlit:

streamlit run var_stress_dashboard.py


The application will automatically open in your web browser, usually at http://localhost:8501.

Customization

To analyze a new portfolio, simply update the Tickers, Weights, and Investment Capital in the sidebar of the running dashboard.
