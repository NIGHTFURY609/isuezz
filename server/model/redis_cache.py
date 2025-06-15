# server/model/redis_cache.py
import os
import redis
import pickle
import hashlib

class RedisCache:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.Redis.from_url(redis_url)
        self.cache_ttl = 3600  # 1 hour TTL

    def get_cache_key(self, data):
        return hashlib.md5(str(data).encode()).hexdigest()

    def get(self, key):
        raw_data = self.client.get(key)
        return pickle.loads(raw_data) if raw_data else None

    def set(self, key, value):
        self.client.setex(key, self.cache_ttl, pickle.dumps(value))
