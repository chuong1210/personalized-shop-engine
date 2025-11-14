"""
cf_engine.py - Collaborative Filtering Engine using ALS (FIXED)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix
import implicit
import logging

logger = logging.getLogger(__name__)


class CollaborativeFilteringEngine:
    """
    Collaborative Filtering using Alternating Least Squares (ALS)
    Recommends products based on user behavior patterns
    """
    
    def __init__(self, factors: int = 64, regularization: float = 0.01, 
                 iterations: int = 15):
        """
        Initialize CF engine
        
        Args:
            factors: Number of latent factors (embedding dimension)
            regularization: L2 regularization parameter
            iterations: Number of training iterations
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=False  # Set True if GPU available
        )
        
        self.user_map = {}  # user_id -> index
        self.product_map = {}  # product_id -> index
        self.reverse_user_map = {}  # index -> user_id
        self.reverse_product_map = {}  # index -> product_id
        self.user_item_matrix = None
        
        logger.info(f"CF Engine initialized: factors={factors}, reg={regularization}, iter={iterations}")
    
    def train(self, interaction_data: pd.DataFrame):
        """
        Train the ALS model
        
        Args:
            interaction_data: DataFrame with columns [user_id, product_id, score]
                             score should include weights and time decay
        
        Expected format:
            user_id  | product_id | score
            ---------|------------|-------
            user123  | prod001    | 8.5
            user123  | prod002    | 3.2
            ...
        """
        if interaction_data.empty:
            logger.warning("No interaction data provided for training!")
            return
        
        logger.info(f"Training CF model with {len(interaction_data)} interactions...")
        
        # Create user and product mappings
        unique_users = interaction_data['user_id'].unique()
        unique_products = interaction_data['product_id'].unique()
        
        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.product_map = {pid: idx for idx, pid in enumerate(unique_products)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_map.items()}
        self.reverse_product_map = {idx: pid for pid, idx in self.product_map.items()}
        
        logger.info(f"Mapped {len(unique_users)} users and {len(unique_products)} products")
        
        # Convert to matrix indices
        rows = interaction_data['user_id'].map(self.user_map)
        cols = interaction_data['product_id'].map(self.product_map)
        data = interaction_data['score'].values
        
        # Create sparse matrix
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_users), len(unique_products))
        )
        
        logger.info(f"Created sparse matrix: {self.user_item_matrix.shape}, "
                   f"density: {self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape) * 100:.4f}%")
        
        # Train model
        try:
            self.model.fit(self.user_item_matrix)
            logger.info("CF model training completed successfully!")
        except Exception as e:
            logger.error(f"CF model training failed: {e}")
            raise
    
    def recommend(self, user_id: str, n: int = 20, 
                  filter_already_liked: bool = True) -> List[Tuple[str, float]]:
        """
        Get product recommendations for a user
        
        Args:
            user_id: User ID
            n: Number of recommendations to return
            filter_already_liked: Whether to exclude products user already interacted with
            
        Returns:
            List of (product_id, score) tuples, sorted by score descending
        """
        if user_id not in self.user_map:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_map[user_id]
        
        try:
            # Get recommendations from model
            product_indices, scores = self.model.recommend(
                userid=user_idx,
                user_items=self.user_item_matrix[user_idx],
                N=n,
                filter_already_liked_items=filter_already_liked
            )
            
            # Convert indices back to product IDs
            recommendations = [
                (self.reverse_product_map[idx], float(score))
                for idx, score in zip(product_indices, scores)
            ]
            
            logger.debug(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation failed for user {user_id}: {e}")
            return []
    
    def similar_products(self, product_id: str, n: int = 20) -> List[Tuple[str, float]]:
        """
        Find products similar to a given product
        
        Args:
            product_id: Product ID
            n: Number of similar products to return
            
        Returns:
            List of (product_id, similarity_score) tuples
        """
        if product_id not in self.product_map:
            logger.warning(f"Product {product_id} not found in training data")
            return []
        
        product_idx = self.product_map[product_id]
        
        try:
            # Get similar items (includes the item itself)
            similar_indices, scores = self.model.similar_items(
                itemid=product_idx,
                N=n + 1  # +1 because it includes the item itself
            )
            
            # Remove the item itself (first result)
            similar_products = [
                (self.reverse_product_map[idx], float(score))
                for idx, score in zip(similar_indices[1:], scores[1:])
            ]
            
            logger.debug(f"Found {len(similar_products)} similar products for {product_id}")
            return similar_products
            
        except Exception as e:
            logger.error(f"Similar products search failed for {product_id}: {e}")
            return []
    
    def batch_recommend(self, user_ids: List[str], n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get recommendations for multiple users at once
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            
        Returns:
            Dict mapping user_id to list of recommendations
        """
        results = {}
        
        for user_id in user_ids:
            results[user_id] = self.recommend(user_id, n)
        
        return results
    
    def get_user_factors(self, user_id: str) -> np.ndarray:
        """
        Get latent factor vector for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Numpy array of user factors (embedding)
        """
        if user_id not in self.user_map:
            return None
        
        user_idx = self.user_map[user_id]
        return self.model.user_factors[user_idx]
    
    def get_product_factors(self, product_id: str) -> np.ndarray:
        """
        Get latent factor vector for a product
        
        Args:
            product_id: Product ID
            
        Returns:
            Numpy array of product factors (embedding)
        """
        if product_id not in self.product_map:
            return None
        
        product_idx = self.product_map[product_id]
        return self.model.item_factors[product_idx]
    
    def get_stats(self) -> Dict:
        """
        Get model statistics
        
        Returns:
            Dictionary with model statistics
        """
        # FIX: Use 'is not None' instead of boolean evaluation
        num_interactions = 0
        matrix_density = 0.0
        
        if self.user_item_matrix is not None:
            num_interactions = self.user_item_matrix.nnz
            matrix_density = (num_interactions / np.prod(self.user_item_matrix.shape) * 100)
        
        return {
            'num_users': len(self.user_map),
            'num_products': len(self.product_map),
            'num_interactions': num_interactions,
            'matrix_density': matrix_density,
            'factors': self.model.factors
        }
    
    def save_model(self, filepath: str):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save model
        """
        import pickle
        
        model_data = {
            'model': self.model,
            'user_map': self.user_map,
            'product_map': self.product_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_product_map': self.reverse_product_map,
            'user_item_matrix': self.user_item_matrix
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        
        Args:
            filepath: Path to load model from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.user_map = model_data['user_map']
        self.product_map = model_data['product_map']
        self.reverse_user_map = model_data['reverse_user_map']
        self.reverse_product_map = model_data['reverse_product_map']
        self.user_item_matrix = model_data['user_item_matrix']
        
        logger.info(f"Model loaded from {filepath}")