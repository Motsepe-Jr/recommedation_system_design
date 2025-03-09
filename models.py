
from pydantic import BaseModel
from typing import List


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    score: float
    
class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[ProductRecommendation]
    model_version: str
    timestamp: str