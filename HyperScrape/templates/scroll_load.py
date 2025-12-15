from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class ScrollLoadTemplate(BaseTemplate):
    """Scroll load template (gitex_africa, bauma)."""
    def scrape_page(self, url: str) -> List[Company]:
        driver = self.driver_pool.get_driver()
        companies = []
        try:
            self._load_page(driver, url)
            
            # Scroll specified times
            scroll_count = self.config.template_config.get("scroll_count", 10)
            for _ in range(scroll_count):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                driver.execute_script("window.scrollBy(0, -200);")
                time.sleep(1)
            
            cards = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("company_cards"))
            detail_links = []
            for card in cards:
                try:
                    btn = card.find_element(By.CSS_SELECTOR, self.selectors.get("detail_button"))
                    href = btn.get_attribute("href")
                    if href and "ExbDetails" in href:
                        detail_links.append(href)
                except: continue
            
            base_url = self.config.template_config.get("base_url_for_join", self.config.base_url)
            for link in detail_links:
                try:
                    self._load_page(driver, link)
                    company = Company(data_source=self.config.name)
                    company.name = self._safe_find_element(driver, By.CSS_SELECTOR, self.selectors.get("company_name"))
                    company.country = self._safe_find_element(driver, By.CSS_SELECTOR, self.selectors.get("country")).upper()
                    company.website = self._safe_get_attribute(driver, By.CSS_SELECTOR, self.selectors.get("website"), "href")
                    if company.name:                         # Enrich company data if enabled
                        company = self._enrich_company_data(company)

                except: continue
        finally:
            self.driver_pool.return_driver(driver)
        return companies


