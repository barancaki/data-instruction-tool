"""
Enhanced Email Finder
Searches for emails from multiple sources including contact pages
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import List, Set
from urllib.parse import urljoin, urlparse
from selenium import webdriver
import time
from ..config.settings import Settings


class EnhancedEmailFinder:
    """Find emails from websites using multiple strategies."""
    
    EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    def __init__(self, driver_pool=None):
        """Initialize email finder."""
        self.driver_pool = driver_pool
        self.contact_pages = Settings.EMAIL_SEARCH_PAGES
        self.blacklist = Settings.EMAIL_BLACKLIST
    
    def find_emails(self, website: str, company_name: str = "") -> List[str]:
        """
        Find emails for a company.
        
        Args:
            website: Company website URL
            company_name: Company name (for fallback search)
            
        Returns:
            List of valid emails
        """
        emails = set()
        
        if not website:
            # No website, try Bing search
            if company_name:
                search_url = self._bing_search(company_name)
                if search_url:
                    emails.update(self._extract_from_page(search_url))
        else:
            # Try main page
            emails.update(self._extract_from_page(website))
            
            # Try contact pages
            for page_path in self.contact_pages:
                contact_url = urljoin(website, page_path)
                emails.update(self._extract_from_page(contact_url))
        
        # Filter and return
        valid_emails = [email for email in emails if self._is_valid_email(email)]
        return valid_emails[:2]  #  Return max 2 emails
    
    def _extract_from_page(self, url: str) -> Set[str]:
        """Extract emails from a page."""
        emails = set()
        
        try:
            headers = {
                'User-Agent': Settings.USER_AGENT
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Extract from HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            emails.update(re.findall(self.EMAIL_PATTERN, text))
            
            # Extract from mailto links
            for a in soup.find_all('a', href=True):
                if a['href'].startswith('mailto:'):
                    email = a['href'].replace('mailto:', '').split('?')[0]
                    emails.add(email)
        
        except:
            pass
        
        return emails
    
    def _bing_search(self, company_name: str) -> str:
        """Search for company website via Bing."""
        try:
            from urllib.parse import quote
            query = quote(f"{company_name} resmi web sitesi")
            url = f"https://www.bing.com/search?q={query}"
            
            headers = {'User-Agent': Settings.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get first result
            for result in soup.select("li.b_algo h2 a"):
                link = result.get("href")
                if link and link.startswith("http"):
                    # Avoid social media
                    if not any(s in link for s in ["facebook", "linkedin", "instagram", "twitter"]):
                        return link
        except:
            pass
        
        return ""
    
    def _is_valid_email(self, email: str) -> bool:
        """Check if email is valid and not blacklisted."""
        if not email or "@" not in email:
            return False
        
        email_lower = email.lower()
        
        for blacklisted in self.blacklist:
            if blacklisted in email_lower:
                return False
        
        return True
