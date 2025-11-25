import pickle
import numpy as np

# Load model vừa train
with open('models/cf_model_latest.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
user_map = data['user_map']
reverse_product_map = data['reverse_product_map']
train_matrix = data['user_item_matrix']

# Lấy user đầu tiên để test
test_user_id = list(user_map.keys())[0]
user_idx = user_map[test_user_id]

print(f"Testing recommendations for User ID: {test_user_id} (Index: {user_idx})")

# Gọi hàm recommend
ids, scores = model.recommend(user_idx, train_matrix[user_idx], N=5)

print("Recommendations:")
for idx, score in zip(ids, scores):
    real_product_id = reverse_product_map[idx]
    print(f"- Product: {real_product_id}, Score: {score:.4f}")