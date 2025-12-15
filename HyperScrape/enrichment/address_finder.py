"""Address finder from company websites"""

import re
import requests
from bs4 import BeautifulSoup
from typing import Optional
from urllib.parse import urljoin


class AddressFinder:
    """Find company addresses from their websites."""
    
    ADDRESS_SELECTORS = [
        '[class*="address"]',
        '[class*="adres"]',
        '[itemtype*="PostalAddress"]',
        'address',
        '[class*="location"]',
        '[class*="contact"]',
    ]
    
    ADDRESS_PATTERNS = [
        r'\d{5}\s+[A-Za-zığüşöçİĞÜŞÖÇ]+',  # Turkish postal code
        r'[A-Z]{1,2}\d{1,2}\s*\d[A-Z]{2}',  # UK postal code
        r'\d{5}(?:-\d{4})?',  # US postal code
    ]
    
    CONTACT_PAGES = ['/contact', '/iletisim', '/about', '/hakkimizda']
    
    def find_address(self, website: str) -> Optional[str]:
        """
        Find address from company website.
        
        Args:
            website: Company website URL
            
        Returns:
            Address string or None
        """
        if not website:
            return None
        
        # Try contact pages first
        for page_path in self.CONTACT_PAGES:
            contact_url = urljoin(website, page_path)
            address = self._extract_address_from_page(contact_url)
            if address:
                return address
        
        # Try main page
        address = self._extract_address_from_page(website)
        return address
    
    def _extract_address_from_page(self, url: str) -> Optional[str]:
        """Extract address from a page."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try CSS selectors
            for selector in self.ADDRESS_SELECTORS:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(strip=True)
                    # Check if looks like an address
                    for pattern in self.ADDRESS_PATTERNS:
                        if re.search(pattern, text):
                            return text
            
        except:
            pass
        
        return None


class SocialMediaFinder:
    """Find social media links from company websites."""
    
    SOCIAL_PATTERNS = {
        'linkedin': r'linkedin\.com/company/[^/\s"]+',
        'facebook': r'facebook\.com/[^/\s"]+',
        'twitter': r'twitter\.com/[^/\s"]+',
        'instagram': r'instagram\.com/[^/\s"]+',
        'youtube': r'youtube\.com/(c|channel|user)/[^/\s"]+',
    }
    
    def find_social_links(self, website: str) -> dict:
        """
        Find social media links from website.
        
        Args:
            website: Company website URL
            
        Returns:
            Dictionary of social media links
        """
        social_links = {
            'linkedin': '',
            'facebook': '',
            'twitter': '',
            'instagram': '',
            'youtube': ''
        }
        
        if not website:
            return social_links
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(website, headers=headers, timeout=10)
            response.raise_for_status()
            
            content = response.text
            
            # Search for each social platform
            for platform, pattern in self.SOCIAL_PATTERNS.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Get first match and ensure it's a full URL
                    url = matches[0]
                    if not url.startswith('http'):
                        url = 'https://' + url
                    social_links[platform] = url
        
        except:
            pass
        
        return social_links
