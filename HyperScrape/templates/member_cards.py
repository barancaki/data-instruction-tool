from typing import List
from selenium.webdriver.common.by import By
import time
from .base import BaseTemplate
from ..models.fair import Company


class MemberCardsTemplate(BaseTemplate):
    """Member cards template (enosad, roboder)."""
    def scrape_page(self, url: str) -> List[Company]:
        driver = self.driver_pool.get_driver()
        companies = []
        try:
            self._load_page(driver, url)
            # Handle load more if needed (with max iterations)
            if self.config.template_config.get("requires_load_more"):
                max_clicks = 50  # Safety limit
                click_count = 0
                
                while click_count < max_clicks:
                    try:
                        btn = driver.find_element(By.CSS_SELECTOR, self.selectors.get("load_more_button"))
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(2)
                        click_count += 1
                    except: 
                        break
            
            cards = self._safe_find_elements(driver, By.CSS_SELECTOR, self.selectors.get("member_cards"))
            links = [card.get_attribute("href") for card in cards]
            
            for link in links:
                if not link: continue
                try:
                    self._load_page(driver, link)
                    company = Company(data_source=self.config.name)
                    
                    # Use XPath selectors
                    company.name = self._safe_find_element(driver, By.XPATH, self.selectors.get("company_name_xpath"))
                    company.phone = self._safe_find_element(driver, By.XPATH, self.selectors.get("phone_xpath"))
                    company.email = self._safe_find_element(driver, By.XPATH, self.selectors.get("email_xpath"))
                    company.address = self._safe_find_element(driver, By.XPATH, self.selectors.get("address_xpath"))
                    company.website = self._safe_get_attribute(driver, By.XPATH, self.selectors.get("website_xpath"), "href")
                    
                    if company.name:                         # Enrich company data if enabled
                        company = self._enrich_company_data(company)

                except: continue
        finally:
            self.driver_pool.return_driver(driver)
        return companies


