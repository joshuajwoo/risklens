import pandas as pd
import yfinance as yf

def get_historical_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical adjusted closing prices for a list of stock tickers.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
        if data.empty:
            return pd.DataFrame()
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()