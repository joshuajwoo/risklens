import pandas as pd
import yfinance as yf
from datetime import datetime

def get_historical_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical adjusted closing prices for a list of stock tickers from Yahoo Finance.

    Args:
        tickers (list[str]): A list of stock ticker symbols (e.g., ['AAPL', 'GOOGL']).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A pandas DataFrame with dates as the index and adjusted closing
                      prices for each ticker in columns. Returns an empty DataFrame if
                      data fetch fails.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    try:
        # yfinance downloads the data. We specify auto_adjust=True to get adjusted prices.
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

        # If only one ticker is requested, yf.download returns a Series. We convert it to a DataFrame.
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])

        # Drop rows with any missing values
        data.dropna(inplace=True)

        print("Data fetched successfully.")
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

# --- Test Block ---
# This part of the script will only run when you execute `python data_provider.py` directly.
# It's a useful way to test your module in isolation.
if __name__ == '__main__':
    # Example usage:
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    # Let's get data for the last 5 years
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

    price_data = get_historical_data(test_tickers, start_date=start, end_date=end)

    if not price_data.empty:
        print("\n--- Fetched Data Sample ---")
        print(price_data.head()) # Print the first 5 rows
        print("\n--- Data Info ---")
        price_data.info()