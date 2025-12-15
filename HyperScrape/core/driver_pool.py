"""
Selenium WebDriver pool for reusing browser instances.
Improves performance by avoiding repeated driver initialization.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from queue import Queue, Empty
import threading
import time
from ..config.settings import Settings


class DriverPool:
    """Pool of Selenium WebDriver instances for reuse."""
    
    def __init__(self, pool_size: int = None):
        """
        Initialize driver pool.
        
        Args:
            pool_size: Number of drivers to maintain in pool
        """
        self.pool_size = pool_size or Settings.MAX_WORKERS
        self._pool: Queue = Queue(maxsize=self.pool_size)
        self._lock = threading.Lock()
        self._initialized = False
        
    def get_driver(self) -> webdriver.Chrome:
        """
        Get a driver from the pool or create a new one.
        
        Returns:
            Chrome WebDriver instance
        """
        try:
            # Try to get existing driver from pool (non-blocking)
            driver = self._pool.get_nowait()
            return driver
        except Empty:
            # Pool is empty, create new driver
            return self._create_driver()
    
    def return_driver(self, driver: webdriver.Chrome):
        """
        Return a driver to the pool.
        
        Args:
            driver: Driver to return to pool
        """
        try:
            # Clean the driver before returning
            driver.delete_all_cookies()
            self._pool.put_nowait(driver)
        except:
            # Pool is full or driver is invalid, quit it
            try:
                driver.quit()
            except:
                pass
    
    def _create_driver(self) -> webdriver.Chrome:
        """Create a new Chrome WebDriver with standard options."""
        options = Options()
        
        if Settings.HEADLESS:
            options.add_argument("--headless")
        
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"user-agent={Settings.USER_AGENT}")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        
        # Fix SSL errors
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--ignore-ssl-errors")
        options.add_argument("--allow-insecure-localhost")
        
        # Additional stability options
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins-discovery")
        
        # Disable images for faster loading (optional)
        # prefs = {"profile.managed_default_content_settings.images": 2}
        # options.add_experimental_option("prefs", prefs)
        
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        driver.implicitly_wait(Settings.IMPLICIT_WAIT)
        driver.set_page_load_timeout(Settings.PAGE_LOAD_TIMEOUT)
        
        return driver
    
    def close_all(self):
        """Close all drivers in the pool."""
        while not self._pool.empty():
            try:
                driver = self._pool.get_nowait()
                driver.quit()
            except:
                pass
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close_all()
