from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class TextListTemplate(BaseTemplate):
    """Text list template (kalite_fuari)."""
    def scrape_page(self, url: str) -> List[Company]:
        driver = self.driver_pool.get_driver()
        companies = []
        try:
            self._load_page(driver, url)
            container = self._wait_for_element(driver, By.CSS_SELECTOR, self.selectors.get("list_container"))
            if not container: return companies
            
            raw_html = container.get_attribute("innerHTML")
            separator = self.config.template_config.get("separator", "<br>")
            firm_list = [f.strip() for f in raw_html.split(separator) if f.strip()]
            
            for firm_name in firm_list:
                firm_name = firm_name.replace("\n", "").strip()
                if firm_name:
                    company = Company(data_source=self.config.name, name=firm_name)
                    company = self._enrich_company_data(company)
                    companies.append(company)
        finally:
            self.driver_pool.return_driver(driver)
        return companies


