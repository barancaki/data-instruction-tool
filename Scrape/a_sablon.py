from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import streamlit as st
import time
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from webdriver_manager.chrome import ChromeDriverManager
from table_to_sql import save_to_sqlite,create_mysql_dump_from_sqlite
from Scrape.scrape import bing_ilk_link_al,site_icerisinden_email_bul

def scrape_enosad_proses_all_members(base_url="https://enosad.org.tr", start_url="https://enosad.org.tr/tr/proses-otomasyonu"):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(start_url)
    time.sleep(3)

    # Ana sayfadaki üye kartlarını bul
    member_cards = driver.find_elements(By.CSS_SELECTOR, "div.grid.grid-cols-1.md\\:grid-cols-2 a[href^='/tr/']")
    member_links = [card.get_attribute("href") for card in member_cards]

    print(f"Toplam {len(member_links)} üye bulundu.")

    tablo = []

    for idx, link in enumerate(member_links, start=1):
        driver.get(link)
        time.sleep(2)
        print(f"{idx}. üye işleniyor: {link}")

        try:
            firma_adi = driver.find_element(By.XPATH, "//h3[text()='Üye Kurumsal Firma Ünvanı']/following-sibling::div").text
        except:
            firma_adi = " "

        try:
            telefon = driver.find_element(By.XPATH, "//h3[contains(text(),'Telefon')]/following-sibling::div").text
        except:
            telefon = " "

        try:
            email = driver.find_element(By.XPATH, "//h3[contains(text(),'Kontakt e-posta')]/following-sibling::div").text
        except:
            email = " "

        try:
            adres = driver.find_element(By.XPATH, "//h3[contains(text(),'Adresi')]/following-sibling::div").text
        except:
            adres = " "

        try:
            site = driver.find_element(By.XPATH, "//h3[contains(text(),'Web sitesi')]/following-sibling::a").get_attribute("href")
        except:
            site = " "

        tablo.append({
            "Firma": firma_adi,
            "Adres": adres,
            "Telefon": telefon,
            "E-posta": email,
            "Web adresi": site
        })

    driver.quit()

    df = pd.DataFrame(tablo)

    # 📌 SQLite formatında kaydet
    save_to_sqlite(df)

    # 📌 MySQL uyumlu dump oluştur
    create_mysql_dump_from_sqlite()

    if st:
        st.dataframe(df)

        # 📥 DB
        with open("fuar_data.db", "rb") as f:
            st.download_button(
                label="📥 Database (.db) İndir",
                data=f,
                file_name="fuar_data.db",
                mime="application/octet-stream"
            )

        # 📥 SQL
        with open("fuar_data.sql", "rb") as f:
            st.download_button(
                label="📥 SQL (.sql) İndir",
                data=f,
                file_name="fuar_data.sql",
                mime="application/sql"
            )
    else:
        print(f"\n🎯 Toplam çekilen üye sayısı: {len(df)}")
def scrape_enosad_fabrika_all_members(base_url="https://enosad.org.tr", start_url="https://enosad.org.tr/tr/fabrika-otomasyonu"):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(start_url)
    time.sleep(3)

    # Ana sayfadaki üye kartlarını bul
    member_cards = driver.find_elements(By.CSS_SELECTOR, "div.grid.grid-cols-1.md\\:grid-cols-2 a[href^='/tr/']")
    member_links = [card.get_attribute("href") for card in member_cards]

    print(f"Toplam {len(member_links)} üye bulundu.")

    tablo = []

    for idx, link in enumerate(member_links, start=1):
        driver.get(link)
        time.sleep(2)
        print(f"{idx}. üye işleniyor: {link}")

        try:
            firma_adi = driver.find_element(By.XPATH, "//h3[text()='Üye Kurumsal Firma Ünvanı']/following-sibling::div").text
        except:
            firma_adi = " "

        try:
            telefon = driver.find_element(By.XPATH, "//h3[contains(text(),'Telefon')]/following-sibling::div").text
        except:
            telefon = " "

        try:
            email = driver.find_element(By.XPATH, "//h3[contains(text(),'Kontakt e-posta')]/following-sibling::div").text
        except:
            email = " "

        try:
            adres = driver.find_element(By.XPATH, "//h3[contains(text(),'Adresi')]/following-sibling::div").text
        except:
            adres = " "

        try:
            site = driver.find_element(By.XPATH, "//h3[contains(text(),'Web sitesi')]/following-sibling::a").get_attribute("href")
        except:
            site = " "

        tablo.append({
            "Data Source/E_Exhibition": "ENOSAD Fabrika Otomasyonu",
            "Product": "",
            "CompanyName": firma_adi,
            "CompanyWebsite": site,
            "CompanyMail": email,
            "CompanyMail2": "",
            "CompanyPhone": telefon,
            "CompanyAddress": adres,
            "CompanyZipCode": "",
            "CompanyCity": "",
            "CompanyCountry": ""
        })

    driver.quit()

    df = pd.DataFrame(tablo)

    # 📌 SQLite formatında kaydet
    save_to_sqlite(df)

    # 📌 MySQL uyumlu dump oluştur
    create_mysql_dump_from_sqlite()

    if st:
        st.dataframe(df)

        # 📥 DB
        with open("fuar_data.db", "rb") as f:
            st.download_button(
                label="📥 Database (.db) İndir",
                data=f,
                file_name="fuar_data.db",
                mime="application/octet-stream"
            )

        # 📥 SQL
        with open("fuar_data.sql", "rb") as f:
            st.download_button(
                label="📥 SQL (.sql) İndir",
                data=f,
                file_name="fuar_data.sql",
                mime="application/sql"
            )
    else:
        print(f"\n🎯 Toplam çekilen üye sayısı: {len(df)}")
def scrape_enosad_robotik_all_members(base_url="https://enosad.org.tr", start_url="https://enosad.org.tr/tr/robotik-ve-mekatronik"):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(start_url)
    time.sleep(3)

    # Ana sayfadaki üye kartlarını bul
    member_cards = driver.find_elements(By.CSS_SELECTOR, "div.grid.grid-cols-1.md\\:grid-cols-2 a[href^='/tr/']")
    member_links = [card.get_attribute("href") for card in member_cards]

    print(f"Toplam {len(member_links)} üye bulundu.")

    tablo = []

    for idx, link in enumerate(member_links, start=1):
        driver.get(link)
        time.sleep(2)
        print(f"{idx}. üye işleniyor: {link}")

        try:
            firma_adi = driver.find_element(By.XPATH, "//h3[text()='Üye Kurumsal Firma Ünvanı']/following-sibling::div").text
        except:
            firma_adi = " "

        try:
            telefon = driver.find_element(By.XPATH, "//h3[contains(text(),'Telefon')]/following-sibling::div").text
        except:
            telefon = " "

        try:
            email = driver.find_element(By.XPATH, "//h3[contains(text(),'Kontakt e-posta')]/following-sibling::div").text
        except:
            email = " "

        try:
            adres = driver.find_element(By.XPATH, "//h3[contains(text(),'Adresi')]/following-sibling::div").text
        except:
            adres = " "

        try:
            site = driver.find_element(By.XPATH, "//h3[contains(text(),'Web sitesi')]/following-sibling::a").get_attribute("href")
        except:
            site = " "

        tablo.append({
            "Data Source/E_Exhibition": "ENOSAD Robotik ve Mekatronik",
            "Product": "",
            "CompanyName": firma_adi,
            "CompanyWebsite": site,
            "CompanyMail": email,
            "CompanyMail2": "",
            "CompanyPhone": telefon,
            "CompanyAddress": adres,
            "CompanyZipCode": "",
            "CompanyCity": "",
            "CompanyCountry": ""
        })

    driver.quit()

    df = pd.DataFrame(tablo)

    # 📌 SQLite formatında kaydet
    save_to_sqlite(df)

    # 📌 MySQL uyumlu dump oluştur
    create_mysql_dump_from_sqlite()

    if st:
        st.dataframe(df)

        # 📥 DB
        with open("fuar_data.db", "rb") as f:
            st.download_button(
                label="📥 Database (.db) İndir",
                data=f,
                file_name="fuar_data.db",
                mime="application/octet-stream"
            )

        # 📥 SQL
        with open("fuar_data.sql", "rb") as f:
            st.download_button(
                label="📥 SQL (.sql) İndir",
                data=f,
                file_name="fuar_data.sql",
                mime="application/sql"
            )
    else:
        print(f"\n🎯 Toplam çekilen üye sayısı: {len(df)}")
def scrape_enosad_sanayi_all_members(base_url="https://enosad.org.tr", start_url="https://enosad.org.tr/tr/sanayide-dijital-donusum"):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(start_url)
    time.sleep(3)

    # Ana sayfadaki üye kartlarını bul
    member_cards = driver.find_elements(By.CSS_SELECTOR, "div.grid.grid-cols-1.md\\:grid-cols-2 a[href^='/tr/']")
    member_links = [card.get_attribute("href") for card in member_cards]

    print(f"Toplam {len(member_links)} üye bulundu.")

    tablo = []

    for idx, link in enumerate(member_links, start=1):
        driver.get(link)
        time.sleep(2)
        print(f"{idx}. üye işleniyor: {link}")

        try:
            firma_adi = driver.find_element(By.XPATH, "//h3[text()='Üye Kurumsal Firma Ünvanı']/following-sibling::div").text
        except:
            firma_adi = " "

        try:
            telefon = driver.find_element(By.XPATH, "//h3[contains(text(),'Telefon')]/following-sibling::div").text
        except:
            telefon = " "

        try:
            email = driver.find_element(By.XPATH, "//h3[contains(text(),'Kontakt e-posta')]/following-sibling::div").text
        except:
            email = " "

        try:
            adres = driver.find_element(By.XPATH, "//h3[contains(text(),'Adresi')]/following-sibling::div").text
        except:
            adres = " "

        try:
            site = driver.find_element(By.XPATH, "//h3[contains(text(),'Web sitesi')]/following-sibling::a").get_attribute("href")
        except:
            site = " "

        tablo.append({
            "Data Source/E_Exhibition": "ENOSAD Sanayide Dijital Dönüşüm",
            "Product": "",
            "CompanyName": firma_adi,
            "CompanyWebsite": site,
            "CompanyMail": email,
            "CompanyMail2": "",
            "CompanyPhone": telefon,
            "CompanyAddress": adres,
            "CompanyZipCode": "",
            "CompanyCity": "",
            "CompanyCountry": ""
        })

    driver.quit()

    df = pd.DataFrame(tablo)

    # 📌 SQLite formatında kaydet
    save_to_sqlite(df)

    # 📌 MySQL uyumlu dump oluştur
    create_mysql_dump_from_sqlite()

    if st:
        st.dataframe(df)

        # 📥 DB
        with open("fuar_data.db", "rb") as f:
            st.download_button(
                label="📥 Database (.db) İndir",
                data=f,
                file_name="fuar_data.db",
                mime="application/octet-stream"
            )

        # 📥 SQL
        with open("fuar_data.sql", "rb") as f:
            st.download_button(
                label="📥 SQL (.sql) İndir",
                data=f,
                file_name="fuar_data.sql",
                mime="application/sql"
            )
    else:
        print(f"\n🎯 Toplam çekilen üye sayısı: {len(df)}")

def scrape_roboder_all_members(base_url="https://uyeler.roboder.org.tr/", start_url="https://uyeler.roboder.org.tr/"):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(start_url)
    time.sleep(3)

    # 🔄 Load More butonuna basarak tüm kartları yükle
    while True:
        try:
            load_more_btn = driver.find_element(By.CSS_SELECTOR, "div.jet-filters-pagination__link")
            driver.execute_script("arguments[0].click();", load_more_btn)
            time.sleep(2)  # Butonun yüklemesi için bekle
        except:
            print("Tüm üyeler yüklendi.")
            break

    # Ana sayfadaki firma kartlarını al
    member_cards = driver.find_elements(By.CSS_SELECTOR, "div.elementor-widget-container a[href^='https://uyeler.roboder.org.tr/firma/']")
    member_links = [card.get_attribute("href") for card in member_cards]

    print(f"Toplam {len(member_links)} üye bulundu.")

    tablo = []

    for idx, link in enumerate(member_links, start=1):
        driver.get(link)
        time.sleep(2)
        print(f"{idx}. üye işleniyor: {link}")

        try:
            firma_adi = driver.find_element(By.CSS_SELECTOR, "h2.elementor-heading-title").text
        except:
            firma_adi = " "

        try:
            site = driver.find_element(By.XPATH, "//span[contains(text(),'.com') or contains(text(),'.net') or contains(text(),'.org')]").text
        except:
            site = " "

        try:
            email = driver.find_element(By.XPATH, "//span[contains(text(),'@')]").text
        except:
            email = " "

        try:
            telefon = driver.find_element(By.XPATH, "//span[contains(text(),'0')]").text
        except:
            telefon = " "

        try:
            adres = driver.find_element(By.XPATH, "//span[contains(text(),'/')]").text
        except:
            adres = " "

        tablo.append({
            "Data Source/E_Exhibition": "ROBODER",
            "Product": "",
            "CompanyName": firma_adi,
            "CompanyWebsite": site,
            "CompanyMail": email,
            "CompanyMail2": "",
            "CompanyPhone": telefon,
            "CompanyAddress": adres,
            "CompanyZipCode": "",
            "CompanyCity": "",
            "CompanyCountry": ""
        })

    driver.quit()

    df = pd.DataFrame(tablo)

    # 📌 SQLite formatında kaydet
    save_to_sqlite(df)

    # 📌 MySQL uyumlu dump oluştur
    create_mysql_dump_from_sqlite()

    if st:
        st.dataframe(df)

        # 📥 DB
        with open("fuar_data.db", "rb") as f:
            st.download_button(
                label="📥 Database (.db) İndir",
                data=f,
                file_name="fuar_data.db",
                mime="application/octet-stream"
            )

        # 📥 SQL
        with open("fuar_data.sql", "rb") as f:
            st.download_button(
                label="📥 SQL (.sql) İndir",
                data=f,
                file_name="fuar_data.sql",
                mime="application/sql"
            )
    else:
        print(f"\n🎯 Toplam çekilen üye sayısı: {len(df)}")
