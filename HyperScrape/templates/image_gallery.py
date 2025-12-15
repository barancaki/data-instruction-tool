from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class ImageGalleryTemplate(BaseTemplate):
    """Image gallery template (mobisadimex)."""
    def scrape_page(self, url: str) -> List[Company]:
        driver = self.driver_pool.get_driver()
        companies = []
        try:
            self._load_page(driver, url)
            gallery = self._wait_for_element(driver, By.CSS_SELECTOR, self.selectors.get("gallery_id"))
            if not gallery: return companies
            
            items = gallery.find_elements(By.CSS_SELECTOR, self.selectors.get("gallery_items"))
            
            for item in items:
                try:
                    img = item.find_element(By.CSS_SELECTOR, self.selectors.get("image"))
                    src = img.get_attribute("src")
                    
                    # Extract name from src
                    import re
                    pattern = self.config.template_config.get("name_pattern", r"/([^/]+)-logo")
                    match = re.search(pattern, src)
                    if match:
                        firm_name = match.group(1).replace("-", " ").title()
                        company = Company(data_source=self.config.name, name=firm_name)
                        company = self._enrich_company_data(company)
                        companies.append(company)
                except: continue
        finally:
            self.driver_pool.return_driver(driver)
        return companies


