import pandas as pd
from app.logic import compute_historical_var

def test_historical_var_calculation():
    """
    Tests the historical VaR calculation with a simple, predictable dataset.
    """
    # 1. Arrange: Create a predictable set of inputs.
    # We create a list of 20 returns. When sorted, the first value (the 5th
    # percentile) will be -0.05.
    returns_data = [
        0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.00, -0.03, 0.02, 0.04,
       -0.05, 0.01, -0.01, 0.02, 0.01, -0.02, 0.03, 0.01, -0.04, 0.00
    ]
    test_returns = pd.Series(returns_data)
    initial_value = 1000.0

    # 2. Act: Call the function that we want to test.
    result = compute_historical_var(test_returns, initial_value)

    # 3. Assert: Check that the output is exactly what we expect.
    assert result['var_95_percentile'] == "-4.05%"
    assert result['var_dollar_amount_1_day_95_confidence'] == -40.50