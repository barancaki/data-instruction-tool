"""
Simple file-based caching system for scraped data.
Avoids re-scraping recently fetched pages.
"""

import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Any
from ..config.settings import Settings


class ScrapeCache:
    """File-based cache for scrape results."""
    
    def __init__(self, cache_dir: str = None, expiry_days: int = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
            expiry_days: Number of days before cache expires
        """
        self.cache_dir = cache_dir or Settings.CACHE_DIR
        self.expiry_days = expiry_days or Settings.CACHE_EXPIRY_DAYS
        self.enabled = Settings.CACHE_ENABLED
        
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, url: str) -> Optional[Any]:
        """
        Get cached data for URL.
        
        Args:
            url: URL to lookup
            
        Returns:
            Cached data or None if not found/expired
        """
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check expiry
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            expiry_time = cached_time + timedelta(days=self.expiry_days)
            
            if datetime.now() > expiry_time:
                # Expired, delete and return None
                os.remove(cache_path)
                return None
            
            return cache_data['data']
            
        except Exception:
            return None
    
    def set(self, url: str, data: Any):
        """
        Cache data for URL.
        
        Args:
            url: URL to cache
            data: Data to cache
        """
        if not self.enabled:
            return
        
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Silent fail on cache write
    
    def clear(self):
        """Clear all cached data."""
        if not self.enabled or not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except:
                    pass
    
    def clear_expired(self):
        """Remove expired cache entries."""
        if not self.enabled or not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                cache_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    cached_time = datetime.fromisoformat(cache_data['timestamp'])
                    expiry_time = cached_time + timedelta(days=self.expiry_days)
                    
                    if datetime.now() > expiry_time:
                        os.remove(cache_path)
                except:
                    pass
