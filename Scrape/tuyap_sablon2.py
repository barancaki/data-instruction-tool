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


def scrape_packaging_fair(sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    base_url = "https://packagingfair.com/katilimci-listesi"
    full_url_prefix = "https://packagingfair.com"
    tablo = []

    for page_num in range(1, sayfa_sayisi + 1):
        page_url = f"{base_url}?page={page_num}"
        print(f"\n🔄 {page_num}. sayfa yükleniyor: {page_url}")
        driver.get(page_url)
        time.sleep(2)

        try:
            brand_elements = driver.find_elements(By.CSS_SELECTOR, "div.brand-item.mt-30.active a.brand-link")
            firma_linkleri = []
            for a_tag in brand_elements:
                href = a_tag.get_attribute("href")
                if href:
                    full_link = href if href.startswith("http") else full_url_prefix + "/" + href.lstrip("/")
                    firma_linkleri.append(full_link)
                    print(f"🔗 Firma linki bulundu: {full_link}")
        except Exception as e:
            print(f"❌ Sayfa {page_num} linkleri çekilemedi: {e}")
            continue

        # Her firma detayına gir
        for link in firma_linkleri:
            print(f"  🔍 Firma detayına giriliyor: {link}")
            try:
                driver.get(link)
                time.sleep(2)

                # Firma adı
                try:
                    firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.company-title").text.strip()
                except:
                    firma_adi = ""

                # Ülke bilgisi
                try:
                    ulke_ikon = driver.find_element(By.CSS_SELECTOR, "i.fa.fa-globe")
                    ulke = ulke_ikon.find_element(By.XPATH, "..").text.strip()
                except:
                    ulke = ""

                # Telefon, Adres, Web Sitesi
                telefon, adres, website = "", "", ""
                try:
                    bilgiler = driver.find_elements(By.CSS_SELECTOR, "div.schedule-list ul li")
                    for li in bilgiler:
                        icon_html = li.get_attribute("innerHTML")

                        if "fa-phone" in icon_html:
                            telefon = li.text.strip()
                        elif "fa-location-dot" in icon_html:
                            adres = li.text.strip()
                        elif "fa-globe" in icon_html:
                            try:
                                website = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                            except:
                                website = li.text.strip()
                except:
                    pass
                if website:
                    try:
                    # Mail çekmek için şirketin websitesine otomatik giden program
                        firma_mail = site_icerisinden_email_bul(website)
                        if firma_mail == "team@packagingfair.com":
                            firma_mail = "Bu websitesi artık geçerli değildir."
                    except:
                        firma_mail = ""
                else:   
                    try:
                        # Mail çekmek için google üzerinden ilk search şirketin websitesine otomatik giden program
                            firmanin_url = bing_ilk_link_al(firma_adi)
                            firma_mail = site_icerisinden_email_bul(firmanin_url)
                    except:
                            firma_mail = ""

                tablo.append({
                    "Firma Adı": firma_adi,
                    "Ülke": ulke.upper(),
                    "Telefon": telefon,
                    "Adres": adres,
                    "Web Sitesi": website,
                    "Firma Mail": firma_mail                
                    })

                print(f"  ✅ Eklendi: {firma_adi}")

            except Exception as e:
                print(f"  ❌ Firma detay alınamadı: {e}")
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
def scrape_plast_eurasia_all_pages(sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    base_url = "https://plasteurasia.com/katilimci-listesi"
    full_url_prefix = "https://plasteurasia.com"
    tablo = []

    for page_num in range(1, sayfa_sayisi + 1):
        page_url = f"{base_url}?page={page_num}"
        print(f"\n🔄 {page_num}. sayfa yükleniyor: {page_url}")
        driver.get(page_url)
        time.sleep(2)

        try:
            brand_elements = driver.find_elements(By.CSS_SELECTOR, "div.brand-item.mt-30.active a.brand-link")
            firma_linkleri = []
            for a_tag in brand_elements:
                href = a_tag.get_attribute("href")
                if href:
                    full_link = href if href.startswith("http") else full_url_prefix + "/" + href.lstrip("/")
                    firma_linkleri.append(full_link)
                    print(f"🔗 Firma linki bulundu: {full_link}")
        except Exception as e:
            print(f"❌ Sayfa {page_num} linkleri çekilemedi: {e}")
            continue

        # Her firma detayına gir
        for link in firma_linkleri:
            print(f"  🔍 Firma detayına giriliyor: {link}")
            try:
                driver.get(link)
                time.sleep(2)

                # Firma adı
                try:
                    firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.company-title").text.strip()
                except:
                    firma_adi = ""

                # Ülke bilgisi
                try:
                    ulke_ikon = driver.find_element(By.CSS_SELECTOR, "i.fa.fa-globe")
                    ulke = ulke_ikon.find_element(By.XPATH, "..").text.strip()
                except:
                    ulke = ""

                # Telefon, Adres, Web Sitesi
                telefon, adres, website = "", "", ""
                try:
                    bilgiler = driver.find_elements(By.CSS_SELECTOR, "div.schedule-list ul li")
                    for li in bilgiler:
                        icon_html = li.get_attribute("innerHTML")

                        if "fa-phone" in icon_html:
                            telefon = li.text.strip()
                        elif "fa-location-dot" in icon_html:
                            adres = li.text.strip()
                        elif "fa-globe" in icon_html:
                            try:
                                website = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                            except:
                                website = li.text.strip()
                except:
                    pass
                if website:
                    try:
                    # Mail çekmek için şirketin websitesine otomatik giden program
                        firma_mail = site_icerisinden_email_bul(website)
                        if firma_mail == "team@packagingfair.com":
                            firma_mail = "Bu websitesi artık geçerli değildir."
                    except:
                        firma_mail = ""
                else:   
                    try:
                        # Mail çekmek için google üzerinden ilk search şirketin websitesine otomatik giden program
                            firmanin_url = bing_ilk_link_al(firma_adi)
                            firma_mail = site_icerisinden_email_bul(firmanin_url)
                    except:
                            firma_mail = ""

                tablo.append({
                    "Firma Adı": firma_adi,
                    "Ülke": ulke.upper(),
                    "Telefon": telefon,
                    "Adres": adres,
                    "Web Sitesi": website,
                    "Firma Mail": firma_mail                
                    })

                print(f"  ✅ Eklendi: {firma_adi}")

            except Exception as e:
                print(f"  ❌ Firma detay alınamadı: {e}")
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
def scrape_intermob_all_pages(sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    base_url = "https://www.intermobistanbul.com/katilimci-listesi"
    full_url_prefix = "https://www.intermobistanbul.com"
    tablo = []

    for page_num in range(1, sayfa_sayisi + 1):
        page_url = f"{base_url}?page={page_num}"
        print(f"\n🔄 {page_num}. sayfa yükleniyor: {page_url}")
        driver.get(page_url)
        time.sleep(2)

        try:
            brand_elements = driver.find_elements(By.CSS_SELECTOR, "div.brand-item.mt-30.active a.brand-link")
            firma_linkleri = []
            for a_tag in brand_elements:
                href = a_tag.get_attribute("href")
                if href:
                    full_link = href if href.startswith("http") else full_url_prefix + "/" + href.lstrip("/")
                    firma_linkleri.append(full_link)
                    print(f"🔗 Firma linki bulundu: {full_link}")
        except Exception as e:
            print(f"❌ Sayfa {page_num} linkleri çekilemedi: {e}")
            continue

        # Her firma detayına gir
        for link in firma_linkleri:
            print(f"  🔍 Firma detayına giriliyor: {link}")
            try:
                driver.get(link)
                time.sleep(2)

                # Firma adı
                try:
                    firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.company-title").text.strip()
                except:
                    firma_adi = ""

                # Ülke bilgisi
                try:
                    ulke_ikon = driver.find_element(By.CSS_SELECTOR, "i.fa.fa-globe")
                    ulke = ulke_ikon.find_element(By.XPATH, "..").text.strip()
                except:
                    ulke = ""

                # Telefon, Adres, Web Sitesi
                telefon, adres, website = "", "", ""
                try:
                    bilgiler = driver.find_elements(By.CSS_SELECTOR, "div.schedule-list ul li")
                    for li in bilgiler:
                        icon_html = li.get_attribute("innerHTML")

                        if "fa-phone" in icon_html:
                            telefon = li.text.strip()
                        elif "fa-location-dot" in icon_html:
                            adres = li.text.strip()
                        elif "fa-globe" in icon_html:
                            try:
                                website = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                            except:
                                website = li.text.strip()
                except:
                    pass
                if website:
                    try:
                    # Mail çekmek için şirketin websitesine otomatik giden program
                        firma_mail = site_icerisinden_email_bul(website)
                        if firma_mail == "team@packagingfair.com":
                            firma_mail = "Bu websitesi artık geçerli değildir."
                    except:
                        firma_mail = ""
                else:   
                    try:
                        # Mail çekmek için google üzerinden ilk search şirketin websitesine otomatik giden program
                            firmanin_url = bing_ilk_link_al(firma_adi)
                            firma_mail = site_icerisinden_email_bul(firmanin_url)
                    except:
                            firma_mail = ""

                tablo.append({
                    "Firma Adı": firma_adi,
                    "Ülke": ulke.upper(),
                    "Telefon": telefon,
                    "Adres": adres,
                    "Web Sitesi": website,
                    "Firma Mail": firma_mail                
                    })

                print(f"  ✅ Eklendi: {firma_adi}")

            except Exception as e:
                print(f"  ❌ Firma detay alınamadı: {e}")
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