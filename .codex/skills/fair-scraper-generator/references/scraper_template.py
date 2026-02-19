import pandas as pd
import time
import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from Scrape.scrape import find_email_advanced, handle_cookie_consent_final
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

def scrape_fair_name(iterations):
    """
    Template for fair scrapers.
    iterations: can be page_count, scroll_count, or click_count.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    base_url = "EXHIBITOR_LIST_URL"
    tablo = []

    try:
        driver.get(base_url)
        time.sleep(3)
        handle_cookie_consent_final(driver)

        for i in range(1, iterations + 1):
            # PAGINATION LOGIC HERE (URL change, Scroll, or Click)

            # Find exhibitor elements
            exhibitors = driver.find_elements(By.CSS_SELECTOR, "EXHIBITOR_SELECTOR")

            for exhibitor in exhibitors:
                try:
                    name = exhibitor.find_element(By.CSS_SELECTOR, "NAME_SELECTOR").text.strip()

                    # Optional: Click to see details or extract from attributes
                    website = ""
                    try:
                        website = exhibitor.find_element(By.CSS_SELECTOR, "WEBSITE_SELECTOR").get_attribute("href")
                    except:
                        pass

                    email = ""
                    if website:
                        # Use advanced helper to find email from website
                        email = find_email_advanced(driver, website)

                    tablo.append({
                        "Data Source/ExhibitionName": "FAIR_NAME",
                        "ExhibitionProductGroup": "",
                        "CompanyName": name,
                        "CompanyWebsite": website,
                        "CompanyMail": email,
                        "CompanyMail2": "",
                        "CompanyPhone": "",
                        "CompanyAddress": "",
                        "CompanyZipCode": "",
                        "CompanyCity": "",
                        "CompanyCountry": "",
                        "CompanyBusinessType": ""
                    })
                except Exception as e:
                    print(f"Error extracting exhibitor: {e}")
                    continue

        if tablo:
            df = pd.DataFrame(tablo)
            st.session_state['data'] = df
            st.write(df)

            # Common download logic
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "exhibitors.csv", "text/csv")

    finally:
        driver.quit()
