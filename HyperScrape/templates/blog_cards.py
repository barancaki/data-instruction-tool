from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class BlogCardsTemplate(BaseTemplate):
    """Blog cards template (atech_fuari)."""
    def scrape_page(self, url: str) -> List[Company]:
        driver = self.driver_pool.get_driver()
        companies = []
        try:
            self._load_page(driver, url)
            cards = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("company_cards"))
            
            for card in cards:
                try:
                    company = Company(data_source=self.config.name)
                    company.name = self._safe_find_element(card, By.CSS_SELECTOR, self.selectors.get("company_name"))
                    company.website = self._safe_get_attribute(card, By.CSS_SELECTOR, self.selectors.get("website"), "href")
                    if company.name:                         # Enrich company data if enabled
                        company = self._enrich_company_data(company)

                except: continue
        finally:
            self.driver_pool.return_driver(driver)
        return companies
