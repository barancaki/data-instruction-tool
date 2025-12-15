from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class XPathDetailTemplate(BaseTemplate):
    """XPath detail template (texhibitionist)."""
    def scrape_page(self, url: str) -> List[Company]:
        driver = self.driver_pool.get_driver()
        companies = []
        try:
            self._load_page(driver, url)
            table = self._wait_for_element(driver, By.CSS_SELECTOR, self.selectors.get("company_table"))
            if not table: return companies
            
            links = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("company_links"))
            detail_urls = [link.get_attribute("href") for link in links if link.get_attribute("href")]
            base_url = self.config.template_config.get("base_url_for_join", self.config.base_url)
            
            for link in detail_urls:
                try:
                    full_link = link if link.startswith("http") else self._join_url(base_url, link)
                    self._load_page(driver, full_link)
                    company = Company(data_source=self.config.name)
                    company.name = self._safe_find_element(driver, By.CSS_SELECTOR, self.selectors.get("company_name"))
                    company.email = self._safe_find_element(driver, By.XPATH, self.selectors.get("email_xpath"))
                    company.phone = self._safe_find_element(driver, By.XPATH, self.selectors.get("phone_xpath"))
                    try:
                        website_elem = driver.find_element(By.XPATH, self.selectors.get("website_xpath"))
                        try:
                            company.website = website_elem.find_element(By.TAG_NAME, "a").get_attribute("href")
                        except:
                            company.website = website_elem.text.strip()
                    except: pass
                    company.address = self._safe_find_element(driver, By.CSS_SELECTOR, self.selectors.get("address"))
                    if company.name:                         # Enrich company data if enabled
                        company = self._enrich_company_data(company)

                except: continue
        finally:
            self.driver_pool.return_driver(driver)
        return companies


