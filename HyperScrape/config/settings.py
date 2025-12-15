"""
Global settings for HyperScrape
"""

class Settings:
    """Global configuration settings."""
    
    # Selenium settings
    HEADLESS = True
    IMPLICIT_WAIT = 10
    PAGE_LOAD_TIMEOUT = 30
    
    # Performance settings
    MAX_WORKERS = 3  # Number of parallel workers
    REQUEST_DELAY = 0.5  # Delay between requests (seconds)
    PAGE_LOAD_DELAY = 2  # Default page load delay
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    RETRY_BACKOFF = 2.0
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_DIR = ".hyperscrape_cache"
    CACHE_EXPIRY_DAYS = 1
    
    # Email enrichment
    EMAIL_SEARCH_PAGES = ['/contact', '/iletisim', '/kontakt', '/about', '/hakkimizda', '/impressum']
    EMAIL_BLACKLIST = ['noreply', 'no-reply', 'example.com', 'test.com', 'packagingfair.com']
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "hyperscrape.log"
    
    # User agent
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
