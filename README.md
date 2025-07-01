# RiskLens - A Portfolio Value at Risk (VaR) & Stress Testing Engine

> **Resume Pitch:** Designed and built a Python-based financial risk engine to quantify portfolio market risk. Implemented three distinct Value at Risk (VaR) models: historical, parametric, and a Monte Carlo simulation with 10,000+ paths. The system provides risk metrics and historical stress testing via a consolidated FastAPI endpoint, enabling data-driven risk management decisions.

`RiskLens` is a backend Python service designed to quantify the market risk of a stock portfolio. It moves beyond simple profit and loss tracking to calculate the potential maximum loss under various market conditions using established financial risk models.

## Features

* **Three VaR Models:**
    * **Historical Value at Risk (VaR):** Calculates VaR based on the direct historical performance of the portfolio.
    * **Parametric VaR (Variance-Covariance):** Calculates VaR by modeling portfolio returns using a normal distribution.
    * **Monte Carlo Simulation VaR:** Simulates thousands of potential future price paths for the portfolio to derive a probability distribution of returns and calculate VaR.

* **Historical Stress Testing:** Measures portfolio resilience by calculating its performance during specific historical market crashes (e.g., the 2008 financial crisis, the COVID-19 drop in March 2020).

* **RESTful API:** All risk metrics are exposed through a clean, consolidated API endpoint.

* **Visualization:** Can generate and serve charts of the portfolio's return distribution and VaR cutoff points.

## Tech Stack

* **Core Language:** Python 3.12+
* **API Framework:** [FastAPI](https://fastapi.tiangolo.com/)
* **Web Server:** [Uvicorn](https://www.uvicorn.org/)
* **Data & Numerical Libraries:**
    * [Pandas](https://pandas.pydata.org/): For data manipulation and time-series analysis.
    * [NumPy](https://numpy.org/): For efficient numerical operations.
    * [SciPy](https://scipy.org/): For statistical functions.
* **Financial Data Source:** [yfinance](https://pypi.org/project/yfinance/) library.
* **Visualization:** [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)
* **Deployment:** [Docker](https://www.docker.com/)

## API Endpoint

The primary endpoint consolidates all risk analysis into a single powerful call.

| Method | Path              | Description                               |
| :----- | :---------------- | :---------------------------------------- |
| `POST` | `/risk/analysis`  | Runs a full risk analysis on the portfolio. |

## Setup and Installation

Follow these steps to set up the project locally.

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd RiskLens