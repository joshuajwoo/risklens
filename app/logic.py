import numpy as np
import pandas as pd
from scipy.stats import norm
import io
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

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

def compute_cornish_fisher_var(daily_returns: pd.Series, initial_value: float, confidence_level: float = 0.05):
    """Calculates the Cornish-Fisher (Modified) VaR."""

    # 1. Calculate skewness and kurtosis
    skew = daily_returns.skew()
    # The .kurt() method in pandas returns "excess" kurtosis, which is what we need.
    kurtosis = daily_returns.kurt() 

    # 2. Get the Z-score for the desired confidence level
    z_score = norm.ppf(confidence_level)

    # 3. Apply the Cornish-Fisher expansion formula to modify the Z-score
    modified_z = (z_score + 
                  (z_score**2 - 1) * skew / 6 + 
                  (z_score**3 - 3 * z_score) * kurtosis / 24 -
                  (2 * z_score**3 - 5 * z_score) * (skew**2) / 36)

    # 4. Calculate VaR using the modified Z-score
    mean_return = daily_returns.mean()
    std_dev = daily_returns.std()

    var = mean_return + modified_z * std_dev

    return {
        "var_95_percentile": f"{round(var * 100, 2)}%",
        "var_dollar_amount_1_day_95_confidence": round(initial_value * var, 2)
    }

def create_volatility_features(daily_returns: pd.Series):
    """Creates features for the volatility prediction model."""

    df = pd.DataFrame(daily_returns)

    # Add existing features
    df['volatility_7d'] = daily_returns.rolling(window=7).std() * np.sqrt(252)
    df['volatility_21d'] = daily_returns.rolling(window=21).std() * np.sqrt(252)

    for i in range(1, 6):
        df[f'return_lag_{i}'] = daily_returns.shift(i)

    # --- New Advanced Features ---
    # Add Relative Strength Index (RSI)
    df['rsi'] = ta.rsi(daily_returns, length=14)

    # Add MACD (Moving Average Convergence Divergence)
    macd = ta.macd(daily_returns, fast=12, slow=26, signal=9)
    # We'll use the MACD histogram, which shows momentum changes
    df['macd_hist'] = macd['MACDh_12_26_9']

    # --- Target Variable ---
    df['target_volatility'] = df['volatility_7d'].shift(-1)

    df.dropna(inplace=True)

    return df

def train_and_predict_volatility(daily_returns: pd.Series):
    """
    Performs a grid search to find the best XGBoost model, evaluates it,
    and predicts the next day's volatility.
    """
    df = create_volatility_features(daily_returns)
    if df.empty or len(df) < 50: # Grid search needs more data
        return None, None, None

    X = df.drop('target_volatility', axis=1)
    y = df['target_volatility']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # --- Hyperparameter Tuning Step ---
    # 1. Define the grid of parameters to search
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1]
    }

    # 2. Create the XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # 3. Set up and run the Grid Search with 3-fold cross-validation
    # n_jobs=-1 uses all available CPU cores to speed up the search
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error', 
        verbose=1, 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # 4. Get the best model found by the search
    best_model = grid_search.best_estimator_

    # --- Model Evaluation ---
    test_predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, test_predictions)

    # --- Final Prediction ---
    prediction_features = X.iloc[[-1]]
    predicted_volatility_annualized = best_model.predict(prediction_features)[0]
    predicted_volatility_daily = predicted_volatility_annualized / np.sqrt(252)

    # Return the prediction, performance score, and best parameters
    return float(predicted_volatility_daily), float(mse), grid_search.best_params_