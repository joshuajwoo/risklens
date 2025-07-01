from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
import pandas as pd

# Import our new modules
from .models import Portfolio
from .provider import get_historical_data
from .logic import (
    compute_historical_var,
    compute_parametric_var,
    compute_monte_carlo_var,
    compute_stress_tests,
    generate_var_histogram
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
    mc_results = compute_monte_carlo_var(prices, shares, initial_portfolio_value)
    stress_test_results = compute_stress_tests(tickers, shares)
    
    return {
        "message": "Full risk analysis successful.",
        "current_portfolio_value": round(initial_portfolio_value, 2),
        "portfolio_analysis": {
            "historical_var": hvar_results,
            "parametric_var": pvar_results,
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