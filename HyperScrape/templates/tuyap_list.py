"""
Tuyap List Template
For sites using .filter-list__item structure with optional detail buttons.
Examples: replast_eurasia, burtarim, pencere, smtech, iplik, maktek
"""

from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class TuyapListTemplate(BaseTemplate):
    """Template for Tuyap-style list layouts."""
    
    def scrape_page(self, url: str) -> List[Company]:
        """Scrape a single page."""
        driver = self.driver_pool.get_driver()
        companies = []
        
        try:
            self._load_page(driver, url)
            
            #Wait for company list to load
            items = self._wait_for_element(driver, By.CSS_SELECTOR, self.selectors.get("company_list"))
            if not items:
                return companies
            
            # Get all company items
            company_items = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("company_list"))
            
            for item in company_items:
                try:
                    company = self._extract_company_from_item(driver, item)
                    if company.name:  # Only add if has a name
                        # Enrich with email/address if enabled
                        company = self._enrich_company_data(company)
                        companies.append(company)
                except Exception as e:
                    continue
            
        finally:
            self.driver_pool.return_driver(driver)
        
        return companies
    
    def _extract_company_from_item(self, driver, item) -> Company:
        """Extract company data from a single list item."""
        company = Company(data_source=self.config.name)
        
        # Company name
        try:
            name_elem = item.find_element(By.CSS_SELECTOR, self.selectors.get("company_name"))
            company.name = name_elem.text.strip()
        except:
            pass
        
        # Address
        try:
            address_elem = item.find_element(By.CSS_SELECTOR, self.selectors.get("address"))
            company.address = address_elem.text.strip()
            company.country = self._extract_country_from_address(company.address)
        except:
            pass
        
        # Phone
        try:
            phone_elem = item.find_element(By.CSS_SELECTOR, self.selectors.get("phone"))
            company.phone = self._clean_phone(phone_elem.text)
        except:
            pass
        
        # Website
        try:
            website_elem = item.find_element(By.CSS_SELECTOR, self.selectors.get("website"))
            company.website = website_elem.get_attribute("href")
        except:
            pass
        
        # Products (if detail button exists)
        if self.selectors.get("detail_button"):
            try:
                detail_button = item.find_element(By.CSS_SELECTOR, self.selectors.get("detail_button"))
                driver.execute_script("arguments[0].click();", detail_button)
                time.sleep(0.3)
                
                product_elems = item.find_elements(By.CSS_SELECTOR, self.selectors.get("product_groups"))
                products = [elem.text.strip() for elem in product_elems if elem.text.strip()]
                company.products = ", ".join(products)
            except:
                pass
        
        return company
