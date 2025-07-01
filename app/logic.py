import numpy as np
import pandas as pd
from scipy.stats import norm
import io
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Tell Matplotlib to use a non-interactive backend suitable for servers
matplotlib.use('Agg')

# Import our own data provider function using a relative import
from .provider import get_historical_data

# Define a dictionary of major historical market stress periods
STRESS_TEST_PERIODS = {
    "2008_financial_crisis": ("2008-09-01", "2009-03-01"),
    "covid_19_crash_2020": ("2020-02-19", "2020-03-23"),
    "dot_com_bubble_2000": ("2000-03-10", "2000-10-10"),
}


def compute_historical_var(daily_returns: pd.Series, initial_value: float):
    var_95 = daily_returns.quantile(0.05)
    return {
        "var_95_percentile": f"{round(var_95 * 100, 2)}%",
        "var_dollar_amount_1_day_95_confidence": round(initial_value * var_95, 2)
    }

def compute_parametric_var(daily_returns: pd.Series, initial_value: float):
    mean_return = daily_returns.mean()
    std_dev = daily_returns.std()
    z_score_95 = norm.ppf(0.05)
    var_95 = z_score_95 * std_dev + mean_return
    return {
        "var_95_parametric_percentile": f"{round(var_95 * 100, 2)}%",
        "var_dollar_amount_1_day_95_confidence": round(initial_value * var_95, 2)
    }

def compute_monte_carlo_var(prices: pd.DataFrame, shares: list[int], initial_value: float, num_simulations: int = 10000):
    daily_returns = prices.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    
    simulation_results = np.zeros(num_simulations)
    try:
        cholesky = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        return {"error": "Covariance matrix not positive-semidefinite."}

    for i in range(num_simulations):
        random_vars = np.random.normal(size=len(shares))
        simulated_returns = mean_returns.values + cholesky @ random_vars
        simulated_value = (prices.iloc[-1].values * (1 + simulated_returns) * shares).sum()
        simulation_results[i] = simulated_value
        
    portfolio_sim_returns = (simulation_results - initial_value) / initial_value
    var_95 = np.percentile(portfolio_sim_returns, 5)
    
    return {
        "var_95_percentile": f"{round(var_95 * 100, 2)}%",
        "var_dollar_amount_1_day_95_confidence": round(initial_value * var_95, 2)
    }

def compute_stress_tests(tickers: list[str], shares: list[int]):
    results = {}
    for test_name, (start, end) in STRESS_TEST_PERIODS.items():
        prices = get_historical_data(tickers, start, end)
        if prices.empty:
            results[test_name] = "Could not fetch sufficient data for this period."
            continue
            
        values = (prices * shares).sum(axis=1)
        initial = values.iloc[0]
        final = values.iloc[-1]
        total_return = (final - initial) / initial
        
        rolling_max = values.cummax()
        daily_drawdown = (values - rolling_max) / rolling_max
        max_drawdown = daily_drawdown.min()
        
        results[test_name] = {
            "total_return": f"{round(total_return * 100, 2)}%",
            "max_drawdown": f"{round(max_drawdown * 100, 2)}%"
        }
    return results

def generate_var_histogram(daily_returns: pd.Series, var_95: float):
    """Generates a histogram plot of daily returns with the VaR cutoff line."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(daily_returns, bins=50, kde=True, ax=ax, color='skyblue', label='Daily Returns')
    ax.axvline(x=var_95, color='red', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.2%}')
    ax.set_title('Distribution of Daily Portfolio Returns with 95% VaR', fontsize=16)
    ax.set_xlabel('Daily Returns', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf