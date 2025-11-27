import redis
import yaml

# Load config để lấy host/port redis
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

r = redis.Redis(**config['redis'])

# Xóa sạch sành sanh
r.flushall()
print("✅ Đã xóa sạch Cache Redis!")