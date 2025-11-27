# hybrid_engine.py

import numpy as np
import pandas as pd
import implicit
from scipy.sparse import csr_matrix
import logging
import pickle

logger = logging.getLogger(__name__)

class HybridEngine:
    def __init__(self, no_components=64, loss='bpr'): # loss='bpr' hoặc 'warp'
        self.factors = no_components
        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=0.01,
            regularization=0.01,
            iterations=50
        )
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
        
    def train(self, interactions_df, products_df=None):
        logger.info("Preparing data for Implicit BPR...")
        
        # 1. Mapping ID
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['product_id'].unique()
        
        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_map = {iid: i for i, iid in enumerate(unique_items)}
        self.reverse_item_map = {i: iid for iid, i in self.item_map.items()}
        
        # 2. Tạo Sparse Matrix
        # Implicit yêu cầu matrix dạng (item, user) cho việc train hiệu quả
        user_ids = [self.user_map[uid] for uid in interactions_df['user_id']]
        item_ids = [self.item_map[pid] for pid in interactions_df['product_id']]
        
        # Matrix shape: (Users, Items)
        user_item_matrix = csr_matrix(
            (np.ones(len(interactions_df), dtype=np.float32), (user_ids, item_ids)),
            shape=(len(unique_users), len(unique_items))
        )
        
        # 3. Train Model
        logger.info("Training Implicit BPR Model...")
        self.model.fit(user_item_matrix) # implicit tự chuyển vị bên trong nếu cần
        logger.info("Training completed!")

    def get_item_vector(self, product_id):
        """Lấy vector sản phẩm. Nếu sp mới chưa có vector, trả về vector ngẫu nhiên nhỏ (fallback)"""
        if product_id in self.item_map:
            item_idx = self.item_map[product_id]
            vector = self.model.item_factors[item_idx]
            return vector.tolist()
        else:
            # Fallback: Trả về vector ngẫu nhiên (hoặc vector 0) để không bị lỗi code update DB
            # Tuy nhiên, sp này sẽ không tìm thấy được chính xác.
            # logger.warning(f"Product {product_id} has no interaction history (Cold Start).")
            return np.random.normal(0, 0.01, self.factors).tolist()

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)