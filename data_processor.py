import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("recommendation-service")

# ==========================================================
# Data Processing and Feature Engineering
# ==========================================================

class DataProcessor:
   
    
    def __init__(self):
        self.product_columns = None
        
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
    
        logger.info("Processing data for model training")
        
        
        df.columns = [col.strip() for col in df.columns]
        
        self.product_columns = [col for col in df.columns if col.startswith('prd_')]
       
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found {df.isnull().sum().sum()} missing values in the dataset")
            df = df.fillna(0)
        
        for col in self.product_columns:
            df[col] = df[col].astype(int)
        
        logger.info(f"Data processing complete. Found {len(self.product_columns)} product features")
        return df
    
    def create_user_item_matrix(self, df: pd.DataFrame) -> csr_matrix:
      
        df_matrix = df.set_index('customerId')
        
        user_item_df = df_matrix[self.product_columns]
        
        user_item_matrix = csr_matrix(user_item_df.values)
        
        return user_item_matrix, user_item_df.index, self.product_columns