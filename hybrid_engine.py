import numpy as np
import pandas as pd
import implicit
from scipy.sparse import csr_matrix
import logging
import pickle

logger = logging.getLogger(__name__)

class HybridEngine:
    """
    Thay thế LightFM bằng Implicit BPR (Bayesian Personalized Ranking).
    Chạy ổn định trên Windows.
    """
    def __init__(self, no_components=64, loss='warp'):
        # Implicit dùng factors thay vì no_components
        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=no_components,
            learning_rate=0.01,
            regularization=0.01,
            iterations=50
        )
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
        
    def train(self, interactions_df, products_df=None):
        # products_df ở đây không dùng trực tiếp để train model BPR thuần,
        # nhưng ta giữ tham số để code không bị lỗi khi gọi.
        
        logger.info("Preparing data for Implicit BPR...")
        
        # 1. Mapping ID
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['product_id'].unique()
        
        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_map = {iid: i for i, iid in enumerate(unique_items)}
        self.reverse_item_map = {i: iid for iid, i in self.item_map.items()}
        
        # 2. Tạo Sparse Matrix
        row_ind = [self.user_map[uid] for uid in interactions_df['user_id']]
        col_ind = [self.item_map[pid] for pid in interactions_df['product_id']]
        data = np.ones(len(interactions_df), dtype=np.float32)
        
        # Matrix shape: (Items, Users) <-- Lưu ý Implicit BPR thích Item-User hơn
        user_item_matrix = csr_matrix(
            (data, (row_ind, col_ind)), 
            shape=(len(unique_users), len(unique_items))
        )
        
        # 3. Train Model
        logger.info("Training Implicit BPR Model...")
        # Implicit fit nhận vào (user_items) matrix
        self.model.fit(user_item_matrix)
        logger.info("Training completed!")

    def get_item_vector(self, product_id):
        """Lấy vector sản phẩm (factors)"""
        if product_id not in self.item_map:
            return None
            
        item_idx = self.item_map[product_id]
        # Trong Implicit, item_factors là mảng numpy
        vector = self.model.item_factors[item_idx]
        return vector.tolist()

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)