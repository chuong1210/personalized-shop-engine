"""
cb_engine.py - Content-Based Filtering Engine (FIXED)
Uses text embeddings to find similar products based on content
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class ContentBasedEngine:
    """
    Content-Based Filtering using Sentence Transformers
    Recommends products based on text similarity (name, description, category, brand)
    """
    
    def __init__(self, model_name: str = 'dangvantuan/vietnamese-embedding', 
                 max_seq_length: int = 256):
        """
        Initialize Content-Based engine
        
        Args:
            model_name: Sentence Transformer model name
                       - dangvantuan/vietnamese-embedding (768 dim, Vietnamese)
                       - paraphrase-multilingual-MiniLM-L12-v2 (384 dim, multilingual)
                       - all-MiniLM-L6-v2 (384 dim, English)
            max_seq_length: Maximum sequence length (default 256, model default is 512)
        """
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        
        # Force CPU để tránh CUDA issues
        import torch
        device = 'cpu'
        if torch.cuda.is_available():
            logger.info("CUDA available but using CPU for stability")
        
        self.model = SentenceTransformer(model_name, device=device)
        
        # FIX: Set max sequence length to avoid index out of range
        # RoBERTa has max 512 tokens, but we use 256 for safety
        self.max_seq_length = max_seq_length
        self.model.max_seq_length = max_seq_length
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.product_embeddings = {}  # product_id -> embedding
        self.product_metadata = {}    # product_id -> metadata dict
        
        logger.info(f"Content-Based engine initialized (dim={self.embedding_dim}, "
                   f"max_seq_length={max_seq_length}, device={device})")
    
    def _truncate_text(self, text: str, max_words: int = 100) -> str:
        """
        Truncate text to avoid exceeding token limit
        
        Args:
            text: Input text
            max_words: Maximum number of words to keep
            
        Returns:
            Truncated text
        """
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for a text
        
        Args:
            text: Input text
            
        Returns:
            Numpy array embedding vector
        """
        # Truncate to prevent index errors
        truncated_text = self._truncate_text(text, max_words=100)
        return self.model.encode(truncated_text, convert_to_numpy=True)
    
    def add_product(self, product_id: str, name: str, description: str = "",
                   category: str = "", brand: str = "", 
                   metadata: Optional[Dict] = None):
        """
        Add a product and compute its embedding
        
        Args:
            product_id: Product ID
            name: Product name
            description: Product description
            category: Category name
            brand: Brand name
            metadata: Additional metadata (price, shop_id, etc.)
        """
        # Combine all text fields
        text_parts = []
        
        if name:
            text_parts.append(f"Tên: {name}")
        if description:
            # Truncate description to avoid token overflow
            desc_truncated = self._truncate_text(description, max_words=50)
            text_parts.append(f"Mô tả: {desc_truncated}")
        if category:
            text_parts.append(f"Danh mục: {category}")
        if brand:
            text_parts.append(f"Thương hiệu: {brand}")
        
        combined_text = ". ".join(text_parts)
        
        # Compute embedding
        embedding = self.compute_embedding(combined_text)
        
        # Store
        self.product_embeddings[product_id] = embedding
        self.product_metadata[product_id] = {
            'name': name,
            'category': category,
            'brand': brand,
            **(metadata or {})
        }
        
        logger.debug(f"Added product {product_id}: {name[:50]}...")
    
    def add_products_batch(self, products: List[Dict]):
        """
        Add multiple products at once (more efficient)
        
        Args:
            products: List of product dicts with keys:
                     product_id, name, description, category, brand, metadata
        """
        logger.info(f"Adding {len(products)} products...")
        
        # Prepare texts with truncation
        texts = []
        product_ids = []
        
        for prod in products:
            text_parts = []
            
            if prod.get('name'):
                text_parts.append(f"Tên: {prod['name']}")
            
            if prod.get('description'):
                # FIX: Truncate description to max 50 words
                desc = str(prod['description'])
                desc_truncated = self._truncate_text(desc, max_words=50)
                text_parts.append(f"Mô tả: {desc_truncated}")
            
            if prod.get('category'):
                text_parts.append(f"Danh mục: {prod['category']}")
            
            if prod.get('brand'):
                text_parts.append(f"Thương hiệu: {prod['brand']}")
            
            combined_text = ". ".join(text_parts)
            
            # Extra safety: Truncate final combined text
            combined_text = self._truncate_text(combined_text, max_words=100)
            
            texts.append(combined_text)
            product_ids.append(prod['product_id'])
        
        # Batch encode with error handling
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=True,
                batch_size=32  # Smaller batch size for stability
            )
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            logger.info("Falling back to sequential encoding...")
            
            # Fallback: Encode one by one
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    emb = self.compute_embedding(text)
                    embeddings.append(emb)
                except Exception as inner_e:
                    logger.error(f"Failed to encode product {product_ids[i]}: {inner_e}")
                    # Use zero vector as fallback
                    embeddings.append(np.zeros(self.embedding_dim))
            
            embeddings = np.array(embeddings)
        
        # Store
        for i, prod in enumerate(products):
            self.product_embeddings[product_ids[i]] = embeddings[i]
            self.product_metadata[product_ids[i]] = {
                'name': prod.get('name', ''),
                'category': prod.get('category', ''),
                'brand': prod.get('brand', ''),
                **(prod.get('metadata', {}))
            }
        
        logger.info(f"Added {len(products)} products successfully")
    
    def find_similar(self, product_id: str, n: int = 20,
                    category_filter: Optional[str] = None,
                    min_score: float = 0.0) -> List[Tuple[str, float]]:
        """
        Find similar products based on content
        
        Args:
            product_id: Query product ID
            n: Number of similar products to return
            category_filter: Only return products from this category
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of (product_id, similarity_score) tuples
        """
        if product_id not in self.product_embeddings:
            logger.warning(f"Product {product_id} not found")
            return []
        
        query_embedding = self.product_embeddings[product_id]
        similarities = []
        
        for pid, emb in self.product_embeddings.items():
            if pid == product_id:
                continue
            
            # Apply category filter
            if category_filter:
                prod_category = self.product_metadata[pid].get('category')
                if prod_category != category_filter:
                    continue
            
            # Compute cosine similarity
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                emb.reshape(1, -1)
            )[0][0]
            
            if sim >= min_score:
                similarities.append((pid, float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n]
    
    def search(self, query: str, n: int = 20,
              category_filter: Optional[str] = None,
              min_score: float = 0.0) -> List[Tuple[str, float]]:
        """
        Search products by text query
        
        Args:
            query: Search query text
            n: Number of results
            category_filter: Only search in this category
            min_score: Minimum similarity score
            
        Returns:
            List of (product_id, score) tuples
        """
        # Encode query with truncation
        query_truncated = self._truncate_text(query, max_words=20)
        query_embedding = self.compute_embedding(query_truncated)
        similarities = []
        
        for pid, emb in self.product_embeddings.items():
            # Apply category filter
            if category_filter:
                prod_category = self.product_metadata[pid].get('category')
                if prod_category != category_filter:
                    continue
            
            # Compute similarity
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                emb.reshape(1, -1)
            )[0][0]
            
            if sim >= min_score:
                similarities.append((pid, float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n]
    
    def get_embedding(self, product_id: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a product
        
        Args:
            product_id: Product ID
            
        Returns:
            Numpy array or None if not found
        """
        return self.product_embeddings.get(product_id)
    
    def get_all_embeddings_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get all embeddings as a matrix
        
        Returns:
            (embeddings_matrix, product_ids) tuple
        """
        product_ids = list(self.product_embeddings.keys())
        embeddings = np.array([self.product_embeddings[pid] for pid in product_ids])
        
        return embeddings, product_ids
    
    def batch_similar_products(self, n: int = 20) -> Dict[str, List[str]]:
        """
        Pre-compute similar products for all products
        Useful for caching
        
        Args:
            n: Number of similar products per product
            
        Returns:
            Dict mapping product_id to list of similar product_ids
        """
        logger.info("Pre-computing similar products for all products...")
        
        result = {}
        product_ids = list(self.product_embeddings.keys())
        
        for pid in product_ids:
            similar = self.find_similar(pid, n=n)
            result[pid] = [p[0] for p in similar]
        
        logger.info(f"Pre-computed similar products for {len(product_ids)} products")
        
        return result
    
    def get_stats(self) -> Dict:
        """
        Get engine statistics
        
        Returns:
            Dictionary with stats
        """
        return {
            'num_products': len(self.product_embeddings),
            'embedding_dim': self.embedding_dim,
            'max_seq_length': self.max_seq_length,
            'model_name': self.model._modules['0'].auto_model.name_or_path
        }
    
    def save_embeddings(self, filepath: str):
        """
        Save embeddings to file
        
        Args:
            filepath: Path to save file
        """
        import pickle
        
        data = {
            'embeddings': self.product_embeddings,
            'metadata': self.product_metadata,
            'embedding_dim': self.embedding_dim,
            'max_seq_length': self.max_seq_length
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """
        Load embeddings from file
        
        Args:
            filepath: Path to load from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.product_embeddings = data['embeddings']
        self.product_metadata = data['metadata']
        
        logger.info(f"Loaded {len(self.product_embeddings)} embeddings from {filepath}")
    
    def update_database_embeddings(self, db):
        """
        Update embeddings in database
        
        Args:
            db: Database instance
        """
        logger.info("Updating embeddings in database...")
        
        data = [
            (emb.tolist(), pid)
            for pid, emb in self.product_embeddings.items()
        ]
        
        db.execute_many("""
            UPDATE product_features
            SET text_embedding = %s, last_updated = NOW()
            WHERE product_id = %s
        """, data)
        
        logger.info(f"Updated {len(data)} embeddings in database")
    
    def clear(self):
        """Clear all stored embeddings"""
        self.product_embeddings.clear()
        self.product_metadata.clear()
        logger.info("Cleared all embeddings")