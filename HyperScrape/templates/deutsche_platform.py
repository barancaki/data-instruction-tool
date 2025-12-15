"""
Deutsche Messe Platform Template
For Deutsche Messe platform sites.
Examples: win_eurasia, hub_of_warehouse, sodex, automechanika
"""

from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class DeutschePlatformTemplate(BaseTemplate):
    """Template for Deutsche Messe platform."""
    
    def scrape_page(self, url: str) -> List[Company]:
        """Scrape a single page."""
        driver = self.driver_pool.get_driver()
        companies = []
        
        try:
            self._load_page(driver, url)
            
            # Check if this is automechanika variant
            variant = self.config.template_config.get("variant", "default")
            
            if variant == "automechanika":
                companies = self._scrape_automechanika_style(driver)
            else:
                companies = self._scrape_default_style(driver)
            
        finally:
            self.driver_pool.return_driver(driver)
        
        return companies
    
    def _scrape_default_style(self, driver) -> List[Company]:
        """Scrape default Deutsche Messe style."""
        companies = []
        base_url = self.config.template_config.get("base_url_for_join", self.config.base_url)
        
        # Get all company cards
        cards = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("company_card"))
        
        # Collect detail links
        detail_links = []
        names_and_countries = []
        
        for card in cards:
            try:
                link_elem = card.find_element(By.CSS_SELECTOR, self.selectors.get("detail_link"))
                href = link_elem.get_attribute("href")
                if href:
                    detail_link = href if href.startswith("http") else self._join_url(base_url, href)
                    detail_links.append(detail_link)
                    
                    # Get name and country from card
                    name = self._safe_find_element(card, By.CSS_SELECTOR, self.selectors.get("company_name"))
                    country = self._safe_find_element(card, By.CSS_SELECTOR, self.selectors.get("country"))
                    names_and_countries.append((name, country.upper()))
            except:
                continue
        
        # Visit detail pages
        for idx, link in enumerate(detail_links):
            try:
                self._load_page(driver, link)
                
                company = Company(data_source=self.config.name)
                company.name = names_and_countries[idx][0]
                company.country = names_and_countries[idx][1]
                
                # Address
                try:
                    address_items = driver.find_elements(By.CSS_SELECTOR, self.selectors.get("address_list"))
                    company.address = " ".join([item.text.strip() for item in address_items])
                except:
                    pass
                
                # Phone
                try:
                    phone_links = driver.find_elements(By.CSS_SELECTOR, self.selectors.get("phone"))
                    for link in phone_links:
                        if "Telefon" in link.text:
                            company.phone = self._clean_phone(link.text)
                            break
                except:
                    pass
                
                # Email
                try:
                    email_elem = driver.find_element(By.CSS_SELECTOR, self.selectors.get("email"))
                    company.email = self._clean_email(email_elem.get_attribute("href"))
                except:
                    pass
                
                # Enrich company data
                company = self._enrich_company_data(company)
                companies.append(company)
                
            except:
                continue
        
        return companies
    
    def _scrape_automechanika_style(self, driver) -> List[Company]:
        """Scrape Automechanika variant."""
        companies = []
        base_url = self.config.template_config.get("base_url_for_join", self.config.base_url)
        
        # Get company list items
        items = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("company_list"))
        
        detail_links = []
        for item in items:
            try:
                link_elem = item.find_element(By.CSS_SELECTOR, self.selectors.get("detail_link"))
                href = link_elem.get_attribute("href")
                if href:
                    detail_link = href if href.startswith("http") else self._join_url(base_url, href)
                    detail_links.append((detail_link, link_elem.text.strip()))
            except:
                continue
        
        # Visit detail pages
        for link, name in detail_links:
            try:
                self._load_page(driver, link)
                
                company = Company(data_source=self.config.name, name=name)
                
                # Address
                company.address = self._safe_find_element(driver, By.CSS_SELECTOR, self.selectors.get("address"))
                if company.address:
                    company.country = company.address.split("-")[-1].strip().upper()
                
                # Website
                try:
                    links = driver.find_elements(By.CSS_SELECTOR, self.selectors.get("website_links"))
                    for lnk in links:
                        href = lnk.get_attribute("href") or ""
                        if "www." in href and not any(s in href for s in ["facebook", "instagram", "linkedin", "youtube"]):
                            company.website = href
                            break
                except:
                    pass
                
                # Enrich company data
                company = self._enrich_company_data(company)
                companies.append(company)
                
            except:
                continue
        
        return companies
