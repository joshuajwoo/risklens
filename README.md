# RiskLens - A Portfolio Value at Risk (VaR) & Stress Testing Engine

`RiskLens` is a backend Python service designed to quantify the market risk of a stock portfolio. It moves beyond simple profit and loss tracking to calculate potential loss under various market conditions using both established financial models and predictive machine learning.

## Features

* **Comprehensive VaR Analysis:**
    * **Historical VaR:** Calculates risk based on direct historical performance.
    * **Parametric VaR:** Models risk using a normal distribution.
    * **Cornish-Fisher VaR:** An advanced model that adjusts parametric VaR for real-world skewness and kurtosis in returns.
    * **Monte Carlo VaR:** Simulates thousands of potential future price paths to model complex outcomes.

* **Machine Learning Integration:**
    * **ML-Powered Parametric VaR:** Uses a hyperparameter-tuned **XGBoost** model to forecast next-day volatility based on historical returns and technical indicators (RSI, MACD, Bollinger Bands), providing a dynamic, forward-looking risk assessment.
    * **Model Evaluation:** The ML pipeline includes a train/test split and reports the Mean Squared Error (MSE) to validate model performance.

* **Historical Stress Testing:** Measures portfolio resilience by calculating its performance during specific market crashes (e.g., 2008 crisis, 2020 COVID drop).

* **Data Visualization:** Generates and serves charts of the portfolio's return distribution and VaR cutoff points via a dedicated API endpoint.

## Tech Stack

* **Core Language:** Python 3.12+
* **API Framework:** [FastAPI](https://fastapi.tiangolo.com/)
* **Web Server:** [Uvicorn](https://www.uvicorn.org/)
* **Data & Numerical Libraries:**
    * [Pandas](https://pandas.pydata.org/): For data manipulation and time-series analysis.
    * [NumPy](https://numpy.org/): For efficient numerical operations.
    * [SciPy](https://scipy.org/): For statistical functions.
* **Machine Learning:**
    * [XGBoost](https://xgboost.ai/): For volatility forecasting.
    * [Scikit-learn](https://scikit-learn.org/): For model evaluation (`train_test_split`, `MSE`) and hyperparameter tuning (`GridSearchCV`).
    * [pandas-ta](https://github.com/twopirllc/pandas-ta): For generating technical analysis features.
* **Financial Data Source:** [yfinance](https://pypi.org/project/yfinance/)
* **Visualization:** [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)
* **Deployment:** [Docker](https://www.docker.com/)

## API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/risk/analysis` | Runs a full risk analysis using all statistical models. |
| `POST` | `/risk/parametric-var-ml` | Runs a Parametric VaR analysis using the ML volatility forecast. |
| `POST` | `/risk/visualize-var` | Returns a PNG histogram of portfolio returns and historical VaR. |
| `GET` | `/` | Health check endpoint. |

## Setup and Installation

Follow these steps to set up the project locally.

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd risklens
```

**2. Create and Activate a Virtual Environment**

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

## How to Run

The application can be run locally for development using Docker, which enables live-reloading.

```bash
# 1. Build the Docker image (only necessary when requirements.txt changes)
docker build -t risklens .

# 2. Run the container with a volume to sync your code
docker run -p 8000:8000 -v ./app:/code/app risklens
```

The API will be available at `http://127.0.0.1:8000`.

## How to Use

The easiest way to interact with the API is through the automatically generated documentation.

**1. Open the API Docs**

Navigate to `http://127.0.0.1:8000/docs` in your web browser.

**2. Test an Endpoint**

* Expand the desired endpoint (e.g., `/risk/analysis`).
* Click "Try it out".
* Provide a portfolio in the "Request body".

**Example Request Body:**
