from pydantic import BaseModel, Field

class Portfolio(BaseModel):
    stocks: dict[str, int] = Field(
        ...,
        description="A dictionary of stock tickers and their holding amounts.",
        example={"AAPL": 100, "GOOGL": 50}
    )