import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
from scipy.sparse import csr_matrix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("recommendation-service")

class CollaborativeFilteringRecommender:
    
    
    def __init__(self):
        self.similarity_matrix = None
        self.user_item_df = None
        self.customer_indices = None
        self.product_columns = None
        
    def fit(self, user_item_matrix, customer_indices, product_columns):
       
        logger.info("Training collaborative filtering model")
        
        
        self.customer_indices = pd.Index(customer_indices)
        self.product_columns = product_columns
        
      
        if isinstance(user_item_matrix, csr_matrix):
            self.user_item_df = pd.DataFrame.sparse.from_spmatrix(user_item_matrix, index=self.customer_indices, columns=self.product_columns)
        else:
            self.user_item_df = pd.DataFrame(user_item_matrix, index=self.customer_indices, columns=self.product_columns)
    
        self.similarity_matrix = cosine_similarity(user_item_matrix)
        
        logger.info(f"Model trained successfully with {len(customer_indices)} customers and {len(product_columns)} products")
        return self
    
    def get_similar_customers(self, customer_id, n=5):
    
        try:
            
            customer_idx = self.customer_indices.get_loc(customer_id)
            
            similarity_scores = self.similarity_matrix[customer_idx]
            
            similar_indices = np.argsort(similarity_scores)[::-1][1:n+1]
         
            similar_customers = [
                (self.customer_indices[idx], similarity_scores[idx])
                for idx in similar_indices
            ]
            
            return similar_customers
        
        except KeyError:
            logger.warning(f"Customer ID {customer_id} not found in training data")
            return []
    
    def recommend_products(self, customer_id, n=3, exclude_owned=True):
       
        try:
          
            customer_products = self.user_item_df.loc[customer_id].to_dict()
            
            similar_customers = self.get_similar_customers(customer_id, n=10)
            
            if not similar_customers:
                logger.info(f"No similar customers found for {customer_id}, using popularity-based recommendations")
                return self._get_popular_products(n)
            
            recommendation_scores = {}
            
            for sim_customer_id, similarity in similar_customers:
                sim_customer_products = self.user_item_df.loc[sim_customer_id]
                
                for product in self.product_columns:
                    if exclude_owned and customer_products.get(product, 0) > 0:
                        continue
                    
                    if product not in recommendation_scores:
                        recommendation_scores[product] = 0
                    
                    recommendation_scores[product] += similarity * sim_customer_products[product]
    
            sorted_recommendations = sorted(
                recommendation_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return sorted_recommendations[:n]
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._get_popular_products(n)
    
    def _get_popular_products(self, n=3):
        product_popularity = self.user_item_df.sum().sort_values(ascending=False)
        return [(product, score) for product, score in product_popularity.items()][:n]
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'similarity_matrix': self.similarity_matrix,
                'user_item_df': self.user_item_df,
                'customer_indices': self.customer_indices,
                'product_columns': self.product_columns
            }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()
        model.similarity_matrix = model_data['similarity_matrix']
        model.user_item_df = model_data['user_item_df']
        model.customer_indices = model_data['customer_indices']
        model.product_columns = model_data['product_columns']
        
        logger.info(f"Model loaded from {filepath}")
        return model
