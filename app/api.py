from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
import pandas as pd
from scipy.stats import norm

# Import our new modules
from .models import Portfolio
from .provider import get_historical_data
from .logic import (
    compute_historical_var,
    compute_parametric_var,
    compute_cornish_fisher_var, # <-- ADD THIS
    compute_monte_carlo_var,
    compute_stress_tests,
    generate_var_histogram,
    train_and_predict_volatility,
)

router = APIRouter()

@router.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to RiskLens API"}

@router.post("/risk/analysis")
def get_full_risk_analysis(portfolio: Portfolio):
    tickers = list(portfolio.stocks.keys())
    shares = list(portfolio.stocks.values())
        
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=5)
    prices = get_historical_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if prices.empty:
        raise HTTPException(status_code=404, detail="Could not fetch sufficient data for the given portfolio tickers.")

    portfolio_daily_values = (prices * shares).sum(axis=1)
    portfolio_daily_returns = portfolio_daily_values.pct_change().dropna()
    initial_portfolio_value = portfolio_daily_values.iloc[-1]
    
    hvar_results = compute_historical_var(portfolio_daily_returns, initial_portfolio_value)
    pvar_results = compute_parametric_var(portfolio_daily_returns, initial_portfolio_value)
    cf_var_results = compute_cornish_fisher_var(portfolio_daily_returns, initial_portfolio_value)
    mc_results = compute_monte_carlo_var(prices, shares, initial_portfolio_value)
    stress_test_results = compute_stress_tests(tickers, shares)

    # 4. Assemble the final response
    return {
        "message": "Full risk analysis successful.",
        "current_portfolio_value": round(initial_portfolio_value, 2),
        "portfolio_analysis": {
            "historical_var": hvar_results,
            "parametric_var": pvar_results,
            "cornish_fisher_var": cf_var_results,
            "monte_carlo_var": mc_results
        },
        "stress_tests": stress_test_results
    }

@router.post("/risk/visualize-var", response_class=StreamingResponse)
def get_var_visualization(portfolio: Portfolio):
    tickers = list(portfolio.stocks.keys())
    shares = list(portfolio.stocks.values())

    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=5)
    prices = get_historical_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if prices.empty:
        raise HTTPException(status_code=404, detail="Could not fetch data for the given portfolio tickers.")

    portfolio_daily_values = (prices * shares).sum(axis=1)
    portfolio_daily_returns = portfolio_daily_values.pct_change().dropna()
    var_95_percentile = portfolio_daily_returns.quantile(0.05)

    plot_buffer = generate_var_histogram(portfolio_daily_returns, var_95_percentile)
    
    return StreamingResponse(plot_buffer, media_type="image/png")

@router.post("/risk/parametric-var-ml")
def get_parametric_var_with_ml(portfolio: Portfolio):
    """
    Calculates Parametric VaR using a hyperparameter-tuned, ML-forecasted volatility.
    """
    # ... (The data fetching and portfolio return calculation part is the same) ...
    tickers = list(portfolio.stocks.keys())
    shares = list(portfolio.stocks.values())

    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=5)
    prices = get_historical_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if prices.empty:
        raise HTTPException(status_code=404, detail="Could not fetch sufficient data.")

    portfolio_daily_values = (prices * shares).sum(axis=1)
    portfolio_daily_returns = portfolio_daily_values.pct_change().dropna()

    # --- ML Integration Step ---
    # Our function now returns three values
    predicted_std_dev, mse_score, best_params = train_and_predict_volatility(portfolio_daily_returns)

    if predicted_std_dev is None:
        raise HTTPException(status_code=500, detail="Could not generate volatility prediction.")

    # --- Use the prediction in the VaR formula ---
    mean_return = portfolio_daily_returns.mean()
    z_score_95 = norm.ppf(0.05)

    var_95 = z_score_95 * predicted_std_dev + mean_return

    initial_portfolio_value = portfolio_daily_values.iloc[-1]
    var_dollar_amount = initial_portfolio_value * var_95

    return {
        "message": "Parametric VaR with ML Volatility Forecast calculated successfully.",
        "model_info": {
            "best_model_params": best_params,
            "model_performance_mse": round(mse_score, 8),
        },
        "predicted_next_day_volatility": f"{round(predicted_std_dev * 100, 4)}%",
        "var_dollar_amount_1_day_95_confidence": round(var_dollar_amount, 2)
    }