"""
Base template class for all scraper templates.
Provides common functionality and defines the interface.
"""

from abc import ABC, abstractmethod
from typing import List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import logging
from ..models.fair import Company, FairConfig
from ..core.driver_pool import DriverPool
from ..core.cache import ScrapeCache
from ..config.settings import Settings


logger = logging.getLogger(__name__)


class BaseTemplate(ABC):
    """
    Abstract base class for all scraper templates.
    Implements Template Method pattern.
    """
    
    def __init__(self, config: FairConfig, driver_pool: DriverPool, cache: ScrapeCache):
        """
        Initialize template.
        
        Args:
            config: Fair configuration
            driver_pool: Shared driver pool
            cache: Cache instance
        """
        self.config = config
        self.driver_pool = driver_pool
        self.cache = cache
        self.selectors = config.selectors
        
    @abstractmethod
    def scrape_page(self, url: str) -> List[Company]:
        """
        Scrape a single page and return list of companies.
        Must be implemented by each template.
        
        Args:
            url: URL to scrape
            
        Returns:
            List of Company objects
        """
        pass
    
    def _safe_find_element(self, driver, by: By, selector: str, default: str = ""):
        """
        Safely find an element and return its text.
        
        Args:
            driver: WebDriver instance
            by: Locator strategy
            selector: Selector string
            default: Default value if not found
            
        Returns:
            Element text or default value
        """
        try:
            element = driver.find_element(by, selector)
            return element.text.strip()
        except:
            return default
    
    def _safe_find_elements(self, driver, by: By, selector: str) -> list:
        """
        Safely find elements.
        
        Args:
            driver: WebDriver instance
            by: Locator strategy
            selector: Selector string
            
        Returns:
            List of elements (empty list if not found)
        """
        try:
            return driver.find_elements(by, selector)
        except:
            return []
    
    def _safe_get_attribute(self, driver, by: By, selector: str, attribute: str, default: str = ""):
        """
        Safely get an element's attribute.
        
        Args:
            driver: WebDriver instance
            by: Locator strategy
            selector: Selector string
            attribute: Attribute name
            default: Default value if not found
            
        Returns:
            Attribute value or default
        """
        try:
            element = driver.find_element(by, selector)
            return element.get_attribute(attribute) or default
        except:
            return default
    
    def _wait_for_element(self, driver, by: By, selector: str, timeout: int = 10):
        """
        Wait for an element to be present.
        
        Args:
            driver: WebDriver instance
            by: Locator strategy
            selector: Selector string
            timeout: Maximum wait time in seconds
            
        Returns:
            Element or None if timeout
        """
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return element
        except:
            return None
    
    def _extract_country_from_address(self, address: str) -> str:
        """
        Extract country from address string.
        Usually the last part after / or ,
        
        Args:
            address: Full address string
            
        Returns:
            Country name
        """
        if not address:
            return ""
        
        # Try splitting by /
        if "/" in address:
            parts = address.split("/")
            return parts[-1].strip()
        
        # Try splitting by ,
        if "," in address:
            parts = address.split(",")
            return parts[-1].strip()
        
        return ""
    
    def _clean_phone(self, phone: str) -> str:
        """Clean phone number."""
        if not phone:
            return ""
        
        # Remove common prefixes
        phone = phone.replace("Telefon:", "").replace("Phone:", "").replace("Tel:", "")
        return phone.strip()
    
    def _clean_email(self, email: str) -> str:
        """Clean email address."""
        if not email:
            return ""
        
        # Remove mailto: prefix
        email = email.replace("mailto:", "")
        
        # Remove query parameters
        if "?" in email:
            email = email.split("?")[0]
        
        return email.strip()
    
    def _is_valid_email(self, email: str) -> bool:
        """Check if email is valid and not in blacklist."""
        if not email or "@" not in email:
            return False
        
        email_lower = email.lower()
        
        for blacklisted in Settings.EMAIL_BLACKLIST:
            if blacklisted in email_lower:
                return False
        
        return True
    
    def _load_page(self, driver, url: str):
        """Load a page with error handling."""
        try:
            driver.get(url)
            time.sleep(Settings.PAGE_LOAD_DELAY)
        except Exception as e:
            logger.error(f"Error loading {url}: {e}")
            raise
    
    def _join_url(self, base_url: str, path: str) -> str:
        """Join base URL with a path."""
        if path.startswith("http"):
            return path
        
        if path.startswith("/"):
            return base_url.rstrip("/") + path
        
        return base_url.rstrip("/") + "/" + path.lstrip("/")
    
    def _enrich_company_data(self, company):
        """
        Enrich company data with email and other info if enabled.
        
        Args:
            company: Company object to enrich
            
        Returns:
            Enriched company object
        """
        # Email enrichment
        if self.config.enable_email_enrichment and not company.email and company.website:
            try:
                from ..enrichment.email_finder import EnhancedEmailFinder
                email_finder = EnhancedEmailFinder()
                emails = email_finder.find_emails(company.website, company.name)
                if emails:
                    company.email = emails[0]
                    if len(emails) > 1:
                        company.email2 = emails[1]
            except Exception as e:
                logger.debug(f"Email enrichment failed: {e}")
        
        # Alternate: If no website but has name, try Bing search
        if self.config.enable_email_enrichment and not company.email and not company.website and company.name:
            try:
                from ..enrichment.email_finder import EnhancedEmailFinder
                email_finder = EnhancedEmailFinder()
                emails = email_finder.find_emails("", company.name)
                if emails:
                    company.email = emails[0]
            except Exception as e:
                logger.debug(f"Email enrichment via search failed: {e}")
        
        # Address enrichment
        if self.config.enable_address_enrichment and not company.address and company.website:
            try:
                from ..enrichment.address_finder import AddressFinder
                address_finder = AddressFinder()
                address = address_finder.find_address(company.website)
                if address:
                    company.address = address
            except Exception as e:
                logger.debug(f"Address enrichment failed: {e}")
        
        # Social media enrichment
        if self.config.enable_social_enrichment and company.website:
            try:
                from ..enrichment.address_finder import SocialMediaFinder
                social_finder = SocialMediaFinder()
                social_links = social_finder.find_social_links(company.website)
                company.linkedin = social_links.get('linkedin', '')
                company.facebook = social_links.get('facebook', '')
                company.twitter = social_links.get('twitter', '')
                company.instagram = social_links.get('instagram', '')
                company.youtube = social_links.get('youtube', '')
            except Exception as e:
                logger.debug(f"Social enrichment failed: {e}")
        
        return company
