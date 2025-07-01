from fastapi import FastAPI
from .api import router

app = FastAPI(
    title="RiskLens API",
    description="A portfolio Value at Risk (VaR) & Stress Testing Engine.",
    version="1.0.0",
)

# Include the routes from our api.py file
app.include_router(router)