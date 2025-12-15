"""
Link + Detail Page Template
For sites with brand cards that link to detail pages.
Examples: packaging_fair, plast_eurasia, intermob, woodtech
"""

from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class LinkDetailTemplate(BaseTemplate):
    """Template for sites with card links to detail pages."""
    
    def scrape_page(self, url: str) -> List[Company]:
        """Scrape a single page."""
        driver = self.driver_pool.get_driver()
        companies = []
        
        try:
            self._load_page(driver, url)
            
            # Collect all detail page links first
            detail_links = []
            brand_cards = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("brand_cards"))
            
            base_url = self.config.template_config.get("base_url_for_join", self.config.base_url)
            
            for card in brand_cards:
                try:
                    href = card.get_attribute("href")
                    if href:
                        full_link = href if href.startswith("http") else self._join_url(base_url, href)
                        detail_links.append(full_link)
                except:
                    continue
            
            # Visit each detail page
            for link in detail_links:
                try:
                    self._load_page(driver, link)
                    company = self._extract_from_detail_page(driver)
                    if company.name:
                        # Enrich with email if enabled
                        company = self._enrich_company_data(company)
                        companies.append(company)
                except Exception as e:
                    continue
            
        finally:
            self.driver_pool.return_driver(driver)
        
        return companies
    
    def _extract_from_detail_page(self, driver) -> Company:
        """Extract company data from detail page."""
        company = Company(data_source=self.config.name)
        
        # Company name
        company.name = self._safe_find_element(
            driver, By.CSS_SELECTOR, self.selectors.get("company_name")
        )
        
        # Country (from icon)
        try:
            country_icon = driver.find_element(By.CSS_SELECTOR, self.selectors.get("country_icon"))
            country_parent = country_icon.find_element(By.XPATH, "..")
            company.country = country_parent.text.strip().upper()
        except:
            pass
        
        # Phone, Address, Website from info list
        try:
            info_items = driver.find_elements(By.CSS_SELECTOR, self.selectors.get("info_list"))
            
            for li in info_items:
                try:
                    icon_html = li.get_attribute("innerHTML")
                    
                    if self.selectors.get("phone_icon") in icon_html:
                        company.phone = self._clean_phone(li.text)
                    elif self.selectors.get("address_icon") in icon_html:
                        company.address = li.text.strip()
                    elif self.selectors.get("website_icon") in icon_html:
                        try:
                            website_elem = li.find_element(By.TAG_NAME, "a")
                            company.website = website_elem.get_attribute("href")
                        except:
                            company.website = li.text.strip()
                except:
                    continue
        except:
            pass
        
        return company
