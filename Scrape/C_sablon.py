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


def scrape_kalitefuari():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = "https://kalitefuari.com/katilimci-listesi/"
    driver.get(url)

    tablo = []

    try:
        # <h4> elementinin yüklenmesini bekle
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.wpb_wrapper h4"))
        )

        h4_element = driver.find_element(By.CSS_SELECTOR, "div.wpb_wrapper h4")
        # <br> tagları ile ayrılmış firma isimlerini al
        raw_html = h4_element.get_attribute("innerHTML")
        firma_listesi = [f.strip() for f in raw_html.split("<br>") if f.strip()]

        for firma_adi in firma_listesi:
            firma_adi = firma_adi.replace("\n", "").strip()
            website = ""
            firma_mail = ""

            try:
                # Firmanın web sitesini bul
                firmanin_url = bing_ilk_link_al(firma_adi)
                if firmanin_url:
                    website = firmanin_url
                    # Maili çek
                    firma_mail = site_icerisinden_email_bul(website)
            except Exception as e:
                print(f"{firma_adi} için hata: {e}")
                firma_mail = ""

            tablo.append({
                "Firma": firma_adi,
                "Web adresi": website,
                "Mail": firma_mail
            })
            time.sleep(0.5)  # sayfanın yavaşlamasını önlemek için

    except Exception as e:
        print(f"Hata oluştu: {e}")

    driver.quit()

    df = pd.DataFrame(tablo)

    # 📌 SQLite formatında kaydet
    save_to_sqlite(df)
    # 📌 MySQL uyumlu dump oluştur
    create_mysql_dump_from_sqlite()

    if st:
        st.dataframe(df)

        with open("fuar_data.db", "rb") as f:
            st.download_button(
                label="📥 Database (.db) İndir",
                data=f,
                file_name="fuar_data.db",
                mime="application/octet-stream"
            )

        with open("fuar_data.sql", "rb") as f:
            st.download_button(
                label="📥 SQL (.sql) İndir",
                data=f,
                file_name="fuar_data.sql",
                mime="application/sql"
            )
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
        print(df.head())
def scrape_mobisadimex():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = "https://www.mobisadimex.com/2024-katilimci-listesi/"
    driver.get(url)

    tablo = []

    try:
        # gallery-1 elementi yüklenene kadar bekle
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "gallery-1"))
        )

        figures = driver.find_elements(By.CSS_SELECTOR, "#gallery-1 figure.gallery-item")

        for fig in figures:
            try:
                img = fig.find_element(By.TAG_NAME, "img")
                src = img.get_attribute("src")
                
                # Dosya adından firma ismini al
                # Örn: https://.../adcom-logo.jpg -> adcom
                firma_adi = src.split("/")[-1].split("-logo")[0]
                # Büyük/küçük harf düzenlemesi
                firma_adi = firma_adi.replace("-", " ").title()

                website = ""
                firma_mail = ""

                try:
                    # Firmanın web sitesini bul
                    firmanin_url = bing_ilk_link_al(firma_adi)
                    if firmanin_url:
                        website = firmanin_url
                        firma_mail = site_icerisinden_email_bul(website)
                except Exception as e:
                    print(f"{firma_adi} için hata: {e}")
                    firma_mail = ""

                tablo.append({
                    "Firma": firma_adi,
                    "Web adresi": website,
                    "Mail": firma_mail
                })

                time.sleep(0.5)

            except Exception as e:
                print(f"Figure işlenirken hata: {e}")
                continue

    except Exception as e:
        print(f"Hata oluştu: {e}")

    driver.quit()

    df = pd.DataFrame(tablo)

    # 📌 SQLite formatında kaydet
    save_to_sqlite(df)
    # 📌 MySQL uyumlu dump oluştur
    create_mysql_dump_from_sqlite()

    if st:
        st.dataframe(df)

        with open("fuar_data.db", "rb") as f:
            st.download_button(
                label="📥 Database (.db) İndir",
                data=f,
                file_name="fuar_data.db",
                mime="application/octet-stream"
            )

        with open("fuar_data.sql", "rb") as f:
            st.download_button(
                label="📥 SQL (.sql) İndir",
                data=f,
                file_name="fuar_data.sql",
                mime="application/sql"
            )
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
        print(df.head())