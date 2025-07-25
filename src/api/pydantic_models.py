from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    Amount: float
    Value: float
    ProductCategory: str
    ChannelId: str
    CountryCode: str
    # Add other features as needed

class PredictionResponse(BaseModel):
    risk_probability: float