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

def scrape_win_eurasia_all_pages(url, sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)

    base_url = "https://platform.win-eurasia.com"
    tablo = []

    for page_num in range(1, sayfa_sayisi + 1):
        print(f"🔄 {page_num}. sayfa yükleniyor...")
        driver.get(f"{base_url}/participants?page={page_num}")
        time.sleep(2)

        # Tüm detay linklerini topla
        detay_linkleri = []
        firma_adi_listesi = []
        ulke_listesi = []

        firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.cell.small-12")

        for kart in firma_kartlari:
            try:
                link_element = kart.find_element(By.CSS_SELECTOR, "a.o.link.as-block.fx.dropshadow.for-child")
                href = link_element.get_attribute("href")
                if href:
                    detay_link = href if href.startswith("http") else base_url + href
                else:
                    continue

                firma_adi = kart.find_element(By.CLASS_NAME, "search-snippet-name").get_attribute("innerText").strip()                
                ulke = kart.find_element(By.CLASS_NAME, "search-snippet-description").text.upper().strip()

                detay_linkleri.append(detay_link)
                firma_adi_listesi.append(firma_adi)
                ulke_listesi.append(ulke)

            except Exception as e:
                print(f"❌ Link/firma bilgisi alınamadı: {e}")
                continue

        # Her detay sayfasına gir ve veriyi çek
        for i, detay_link in enumerate(detay_linkleri):
            firma_adi = firma_adi_listesi[i]
            ulke = ulke_listesi[i]

            try:
                driver.get(detay_link)
                time.sleep(2)

                # Adres
                try:
                    adres_listesi = driver.find_elements(By.CSS_SELECTOR, "ul.t.set-250-regular.as-copy li")
                    adres = " ".join([li.text.strip() for li in adres_listesi])
                except:
                    adres = ""

                # Telefon
                try:
                    telefonlar = driver.find_elements(By.CSS_SELECTOR, "ul.t.set-250-regular.as-copy li a")
                    telefon = ""
                    for tel in telefonlar:
                        if "Telefon" in tel.text:
                            telefon = tel.text.replace("Telefon:", "").strip()
                            break
                except:
                    telefon = ""

                # Company Mail
                try:
                    mail_element = driver.find_element(By.CSS_SELECTOR, "a[href^='mailto:']")
                    email = mail_element.get_attribute("href").replace("mailto:", "").strip()
                except:
                    email = ""

                tablo.append({
                    "Firma": firma_adi,
                    "Ülke": ulke,
                    "Adres": adres,
                    "Telefon": telefon,
                    "Company Mail": email
                })

                print(f"✅ {firma_adi} eklendi.")

            except Exception as e:
                print(f"❌ {firma_adi} için detay sayfasına gidilemedi: {e}")
                continue

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

        ulke_sayilari = df["Ülke"].value_counts().reset_index()
        ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
        fig = px.bar(ulke_sayilari, x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
        st.plotly_chart(fig)
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
def scrape_how_all_pages(url, sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)

    base_url = "https://platform.hubofwarehouse.com"
    tablo = []

    for page_num in range(1, sayfa_sayisi + 1):
        print(f"🔄 {page_num}. sayfa yükleniyor...")
        driver.get(f"{base_url}/participants?page={page_num}")
        time.sleep(2)

        # Tüm detay linklerini topla
        detay_linkleri = []
        firma_adi_listesi = []
        ulke_listesi = []

        firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.cell.small-12")

        for kart in firma_kartlari:
            try:
                link_element = kart.find_element(By.CSS_SELECTOR, "a.o.link.as-block.fx.dropshadow.for-child")
                href = link_element.get_attribute("href")
                if href:
                    detay_link = href if href.startswith("http") else base_url + href
                else:
                    continue

                firma_adi = kart.find_element(By.CLASS_NAME, "search-snippet-name").get_attribute("innerText").strip()                
                ulke = kart.find_element(By.CLASS_NAME, "search-snippet-description").text.upper().strip()

                detay_linkleri.append(detay_link)
                firma_adi_listesi.append(firma_adi)
                ulke_listesi.append(ulke)

            except Exception as e:
                print(f"❌ Link/firma bilgisi alınamadı: {e}")
                continue

        # Her detay sayfasına gir ve veriyi çek
        for i, detay_link in enumerate(detay_linkleri):
            firma_adi = firma_adi_listesi[i]
            ulke = ulke_listesi[i]

            try:
                driver.get(detay_link)
                time.sleep(2)

                # Adres
                try:
                    adres_listesi = driver.find_elements(By.CSS_SELECTOR, "ul.t.set-250-regular.as-copy li")
                    adres = " ".join([li.text.strip() for li in adres_listesi])
                except:
                    adres = ""

                # Telefon
                try:
                    telefonlar = driver.find_elements(By.CSS_SELECTOR, "ul.t.set-250-regular.as-copy li a")
                    telefon = ""
                    for tel in telefonlar:
                        if "Telefon" in tel.text:
                            telefon = tel.text.replace("Telefon:", "").strip()
                            break
                except:
                    telefon = ""

                # Company Mail
                try:
                    mail_element = driver.find_element(By.CSS_SELECTOR, "a[href^='mailto:']")
                    email = mail_element.get_attribute("href").replace("mailto:", "").strip()
                except:
                    email = ""

                tablo.append({
                    "Firma": firma_adi,
                    "Ülke": ulke,
                    "Adres": adres,
                    "Telefon": telefon,
                    "Company Mail": email
                })

                print(f"✅ {firma_adi} eklendi.")

            except Exception as e:
                print(f"❌ {firma_adi} için detay sayfasına gidilemedi: {e}")
                continue

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

        ulke_sayilari = df["Ülke"].value_counts().reset_index()
        ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
        fig = px.bar(ulke_sayilari, x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
        st.plotly_chart(fig)
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
def scrape_sodex_all_pages(url, sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)

    base_url = "https://platform.sodex.com.tr"
    tablo = []

    for page_num in range(1, sayfa_sayisi + 1):
        print(f"🔄 {page_num}. sayfa yükleniyor...")
        driver.get(f"{base_url}/participants?page={page_num}")
        time.sleep(2)

        # Tüm detay linklerini topla
        detay_linkleri = []
        firma_adi_listesi = []
        ulke_listesi = []

        firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.cell.small-12")

        for kart in firma_kartlari:
            try:
                link_element = kart.find_element(By.CSS_SELECTOR, "a.o.link.as-block.fx.dropshadow.for-child")
                href = link_element.get_attribute("href")
                if href:
                    detay_link = href if href.startswith("http") else base_url + href
                else:
                    continue

                firma_adi = kart.find_element(By.CLASS_NAME, "search-snippet-name").get_attribute("innerText").strip()                
                ulke = kart.find_element(By.CLASS_NAME, "search-snippet-description").text.upper().strip()

                detay_linkleri.append(detay_link)
                firma_adi_listesi.append(firma_adi)
                ulke_listesi.append(ulke)

            except Exception as e:
                print(f"❌ Link/firma bilgisi alınamadı: {e}")
                continue

        # Her detay sayfasına gir ve veriyi çek
        for i, detay_link in enumerate(detay_linkleri):
            firma_adi = firma_adi_listesi[i]
            ulke = ulke_listesi[i]

            try:
                driver.get(detay_link)
                time.sleep(2)

                # Adres
                try:
                    adres_listesi = driver.find_elements(By.CSS_SELECTOR, "ul.t.set-250-regular.as-copy li")
                    adres = " ".join([li.text.strip() for li in adres_listesi])
                except:
                    adres = ""

                # Telefon
                try:
                    telefonlar = driver.find_elements(By.CSS_SELECTOR, "ul.t.set-250-regular.as-copy li a")
                    telefon = ""
                    for tel in telefonlar:
                        if "Telefon" in tel.text:
                            telefon = tel.text.replace("Telefon:", "").strip()
                            break
                except:
                    telefon = ""

                # Company Mail
                try:
                    mail_element = driver.find_element(By.CSS_SELECTOR, "a[href^='mailto:']")
                    email = mail_element.get_attribute("href").replace("mailto:", "").strip()
                except:
                    email = ""

                tablo.append({
                    "Firma": firma_adi,
                    "Ülke": ulke,
                    "Adres": adres,
                    "Telefon": telefon,
                    "Company Mail": email
                })

                print(f"✅ {firma_adi} eklendi.")

            except Exception as e:
                print(f"❌ {firma_adi} için detay sayfasına gidilemedi: {e}")
                continue

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

        ulke_sayilari = df["Ülke"].value_counts().reset_index()
        ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
        fig = px.bar(ulke_sayilari, x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
        st.plotly_chart(fig)
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
