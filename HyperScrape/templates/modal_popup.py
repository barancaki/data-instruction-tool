from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class ModalPopupTemplate(BaseTemplate):
    """Modal popup template (evchargeshow)."""
    def scrape_page(self, url: str) -> List[Company]:
        driver = self.driver_pool.get_driver()
        companies = []
        try:
            self._load_page(driver, url)
            # Scroll to load all (with max iterations to prevent infinite loop)
            last_height = driver.execute_script("return document.body.scrollHeight")
            max_scrolls = 20  # Safety limit
            scroll_count = 0
            
            while scroll_count < max_scrolls:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height: 
                    break
                last_height = new_height
                scroll_count += 1
            
            cards = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("company_cards"))
            for card in cards:
                try:
                    company = Company(data_source=self.config.name)
                    company.name = self._safe_find_element(card, By.CSS_SELECTOR, self.selectors.get("company_name"))
                    country_els = card.find_elements(By.CSS_SELECTOR, self.selectors.get("country"))
                    if len(country_els) > 1:
                        company.country = country_els[1].text.strip()
                    
                    # Click detail button for website
                    try:
                        btn = card.find_element(By.CSS_SELECTOR, self.selectors.get("detail_button"))
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(0.5)
                        company.website = self._safe_get_attribute(driver, By.CSS_SELECTOR, self.selectors.get("website_in_modal"), "href")
                        # Close modal
                        close_btn = driver.find_element(By.CSS_SELECTOR, self.selectors.get("close_button"))
                        driver.execute_script("arguments[0].click();", close_btn)
                        time.sleep(0.3)
                    except: pass
                    
                    if company.name:
                        # Enrich company data if enabled
                        company = self._enrich_company_data(company)
                        companies.append(company)
                except: continue
        finally:
            self.driver_pool.return_driver(driver)
        return companies
