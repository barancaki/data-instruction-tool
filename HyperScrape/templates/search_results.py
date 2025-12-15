from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class SearchResultsTemplate(BaseTemplate):
    """Search results template (advanced_engineering, mesago)."""
    def scrape_page(self, url: str) -> List[Company]:
        driver = self.driver_pool.get_driver()
        companies = []
        try:
            self._load_page(driver, url)
            cards = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("company_cards"))
            base_url = self.config.template_config.get("base_url_for_join", self.config.base_url)
            
            detail_links = []
            for card in cards:
                try:
                    link = card.get_attribute("href") if card.tag_name == "a" else self._safe_get_attribute(card, By.CSS_SELECTOR, self.selectors.get("detail_link"), "href")
                    if link:
                        detail_links.append(link if link.startswith("http") else self._join_url(base_url, link))
                except: continue
            
            for link in detail_links:
                try:
                    self._load_page(driver, link)
                    company = Company(data_source=self.config.name)
                    company.name = self._safe_find_element(driver, By.CSS_SELECTOR, self.selectors.get("company_name"))
                    company.website = self._safe_get_attribute(driver, By.CSS_SELECTOR, self.selectors.get("website"), "href")
                    company.phone = self._safe_get_attribute(driver, By.CSS_SELECTOR, self.selectors.get("phone"), "href").replace("tel:", "")
                    if self.selectors.get("email"):
                        company.email = self._clean_email(self._safe_get_attribute(driver, By.CSS_SELECTOR, self.selectors.get("email"), "href"))
                    if self.selectors.get("products"):
                        prod_els = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("products"))
                        company.products = ", ".join([el.text.strip() for el in prod_els])
                    if company.name:                         # Enrich company data if enabled
                        company = self._enrich_company_data(company)

                except: continue
        finally:
            self.driver_pool.return_driver(driver)
        return companies


