# server/model/cache.py
import os

USE_REDIS = os.getenv("USE_REDIS_CACHE", "false").lower() == "true"

if USE_REDIS:
    from .redis_cache import RedisCache as Cache
else:
    import hashlib
    import pickle

    class Cache:
        def __init__(self):
            self.store = {}
            self.cache_ttl = 3600

        def get_cache_key(self, data):
            return hashlib.md5(str(data).encode()).hexdigest()

        def get(self, key):
            data = self.store.get(key)
            return pickle.loads(data) if data else None

        def set(self, key, value):
            self.store[key] = pickle.dumps(value)
