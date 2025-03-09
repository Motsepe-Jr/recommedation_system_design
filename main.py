
import pandas as pd
import os
import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import redis
import json
from datetime import datetime


from collaborative_filter_recommender import CollaborativeFilteringRecommender
from data_processor import DataProcessor
from models import RecommendationResponse, ProductRecommendation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("recommendation-service")


app = FastAPI(title="Sanlam Recommendation Service")


redis_client = None
if os.environ.get("REDIS_HOST"):
    try:
        redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            password=os.environ.get("REDIS_PASSWORD", ""),
            decode_responses=True
        )
        redis_client.ping() 
        logger.info("Connected to Redis cache")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {str(e)}")
        redis_client = None


recommender_model = None

@app.on_event("startup")
async def load_model():
    """Load the recommendation model on startup"""
    global recommender_model
    
    model_path = os.environ.get("MODEL_PATH", "/app/models/recommender_model.pkl")
    
    try:
        recommender_model = CollaborativeFilteringRecommender.load(model_path)
        logger.info("Recommendation model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
      
        if os.path.exists("app/data/training_data.csv"):
            logger.info("Training new model from available data")
            train_model()
        else:
            logger.error("No model or training data available!")

def train_model():
    """Train a new recommendation model from training data"""
    global recommender_model
    
    try:
        df = pd.read_csv("/app/data/training_data.csv")
        
        processor = DataProcessor()
        processed_df = processor.process_data(df)
        
        user_item_matrix, customer_indices, product_columns = processor.create_user_item_matrix(processed_df)
        
        recommender_model = CollaborativeFilteringRecommender()
        recommender_model.fit(user_item_matrix, customer_indices, product_columns)
        
        os.makedirs("/app/models", exist_ok=True)
        recommender_model.save("/app/models/recommender_model.pkl")
        
        logger.info("New model trained and saved successfully")
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")

@app.get("/recommendations/customer/{customer_id}", response_model=RecommendationResponse)
async def get_recommendations(
    customer_id: str,
    n: int = Query(3, ge=1, le=10),
    exclude_owned: bool = Query(True)
):
    """Get product recommendations for a customer"""
    global recommender_model
    
    if recommender_model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    if redis_client:
        cache_key = f"recommendations:customer:{customer_id}:n:{n}:exclude:{exclude_owned}"
        cached_response = redis_client.get(cache_key)
        
        if cached_response:
            logger.info(f"Returning cached recommendations for customer {customer_id}")
            return json.loads(cached_response)
    
    try:
        recommendations = recommender_model.recommend_products(
            customer_id=int(customer_id),
            n=n,
            exclude_owned=exclude_owned
        )
        
        formatted_recommendations = []
        for product_id, score in recommendations:
            product_name = product_id.replace('prd_sanlam_', '').replace('_', ' ')
            
            formatted_recommendations.append(
                ProductRecommendation(
                    product_id=product_id,
                    product_name=product_name.title(),
                    score=float(score)
                )
            )
        
        response = RecommendationResponse(
            customer_id=customer_id,
            recommendations=formatted_recommendations,
            model_version="1.0.0",
            timestamp=datetime.now().isoformat()
        )
        
        if redis_client:
            redis_client.set(
                cache_key, 
                json.dumps(response.dict()), 
                ex=3600  
            )
        
        return response
        
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found in training data"
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/products/popular", response_model=List[ProductRecommendation])
async def get_popular_products(n: int = Query(5, ge=1, le=10)):
    """Get the most popular products"""
    global recommender_model
    
    if recommender_model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        popular_products = recommender_model._get_popular_products(n)
        
        return [
            ProductRecommendation(
                product_id=product_id,
                product_name=product_id.replace('prd_sanlam_', '').replace('_', ' ').title(),
                score=float(score)
            )
            for product_id, score in popular_products
        ]
    except Exception as e:
        logger.error(f"Error fetching popular products: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching popular products: {str(e)}"
        )

@app.get("/health")
async def health_check():
    
    global recommender_model
    
    status = "healthy" if recommender_model is not None else "model_unavailable"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/admin/retrain")
async def trigger_retraining():
    try:
        train_model()
        return {"status": "success", "message": "Model retrained successfully"}
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retraining model: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)