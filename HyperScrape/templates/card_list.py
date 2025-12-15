from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class CardListTemplate(BaseTemplate):
    """Card list template (hvacr_world)."""
    def scrape_page(self, url: str) -> List[Company]:
        driver = self.driver_pool.get_driver()
        companies = []
        try:
            self._load_page(driver, url)
            cards = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("company_cards"))
            detail_links = [self._safe_get_attribute(card, By.CSS_SELECTOR, self.selectors.get("detail_link"), "href") for card in cards]
            
            for link in detail_links:
                if not link: continue
                try:
                    self._load_page(driver, link)
                    company = Company(data_source=self.config.name)
                    company.name = self._safe_find_element(driver, By.CSS_SELECTOR, self.selectors.get("company_name"))
                    
                    # Stand and country from h6 elements
                    h6_els = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("stand_info"))
                    for el in h6_els:
                        if "Stand No" in el.text: 
                            company.stand_no = el.text
                        else: 
                            company.country = el.text.strip().upper()
                    
                    # Products
                    company.products = self._safe_find_element(driver, By.CSS_SELECTOR, self.selectors.get("category"))
                    
                    # Extract phone, email, website, address from info elements
                    try:
                        info_divs = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("info_elements"))
                        for div in info_divs:
                            try:
                                icon_html = div.get_attribute("innerHTML")
                                text = div.text.strip()
                                
                                if "fa-phone" in icon_html or "Phone" in text:
                                    company.phone = self._clean_phone(text)
                                elif "fa-envelope" in icon_html or "@" in text:
                                    company.email = self._clean_email(text)
                                elif "fa-globe" in icon_html and "http" in icon_html:
                                    # Extract website from anchor
                                    try:
                                        a_elem = div.find_element(By.TAG_NAME, "a")
                                        company.website = a_elem.get_attribute("href")
                                    except:
                                        pass
                                elif "fa-location" in icon_html or "Address" in text:
                                    company.address = text
                            except:
                                continue
                    except:
                        pass
                    
                    # Enrich if enabled
                    if company.name:
                        company = self._enrich_company_data(company)
                        companies.append(company)
                except: 
                    continue
        finally:
            self.driver_pool.return_driver(driver)
        return companies


