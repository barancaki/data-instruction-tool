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

        if "CompanyCountry" in df.columns:
            ulke_sayilari = df["CompanyCountry"].value_counts().reset_index()
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
                    "Data Source/E_Exhibition": "Plast Eurasia",
                    "Product": "",
                    "CompanyName": firma_adi,
                    "CompanyWebsite": website,
                    "CompanyMail": firma_mail,
                    "CompanyMail2": "",
                    "CompanyPhone": telefon,
                    "CompanyAddress": adres,
                    "CompanyZipCode": "",
                    "CompanyCity": "",
                    "CompanyCountry": ulke.upper()
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

        if "CompanyCountry" in df.columns:
            ulke_sayilari = df["CompanyCountry"].value_counts().reset_index()
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
                    "Data Source/E_Exhibition": "Intermob",
                    "Product": "",
                    "CompanyName": firma_adi,
                    "CompanyWebsite": website,
                    "CompanyMail": firma_mail,
                    "CompanyMail2": "",
                    "CompanyPhone": telefon,
                    "CompanyAddress": adres,
                    "CompanyZipCode": "",
                    "CompanyCity": "",
                    "CompanyCountry": ulke.upper()
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

        if "CompanyCountry" in df.columns:
            ulke_sayilari = df["CompanyCountry"].value_counts().reset_index()
            ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
            fig = px.bar(ulke_sayilari, x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
            st.plotly_chart(fig)
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
def scrape_woodtech_all_pages(sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    base_url = "https://woodtechistanbul.com/katilimci-listesi"
    full_url_prefix = "https://woodtechistanbul.com"
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
                    "Data Source/E_Exhibition": "Woodtech",
                    "Product": "",
                    "CompanyName": firma_adi,
                    "CompanyWebsite": website,
                    "CompanyMail": firma_mail,
                    "CompanyMail2": "",
                    "CompanyPhone": telefon,
                    "CompanyAddress": adres,
                    "CompanyZipCode": "",
                    "CompanyCity": "",
                    "CompanyCountry": ulke.upper()
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

        if "CompanyCountry" in df.columns:
            ulke_sayilari = df["CompanyCountry"].value_counts().reset_index()
            ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
            fig = px.bar(ulke_sayilari, x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
            st.plotly_chart(fig)
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
def scrape_texhibitionist_all_pages(sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    base_url = "https://www.texhibitionist.com/katilimcilar"
    full_url_prefix = "https://www.texhibitionist.com"
    tablo = []

    for page_num in range(1, sayfa_sayisi + 1):
        page_url = f"{base_url}?page={page_num}"
        print(f"\n🔄 {page_num}. sayfa yükleniyor: {page_url}")
        driver.get(page_url)
        time.sleep(2)

        try:
            # Ana tablo elementini bul
            ana_tablo = driver.find_element(By.CSS_SELECTOR, "div.row.row-cols-2.row-cols-lg-3.g-2.g-lg-4.gy-5")
            
            # Tek firma elementlerini bul
            firma_elements = ana_tablo.find_elements(By.CSS_SELECTOR, "div.col a")
            firma_linkleri = []
            
            for a_tag in firma_elements:
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
                    firma_adi = driver.find_element(By.CSS_SELECTOR, "div.title").text.strip()
                except:
                    firma_adi = ""

                # Email
                try:
                    email_key = driver.find_element(By.XPATH, "//div[@class='key'][contains(text(), 'E-mail')]")
                    email = email_key.find_element(By.XPATH, "following-sibling::*[1]").text.strip()
                except:
                    email = ""

                # Telefon
                try:
                    telefon_key = driver.find_element(By.XPATH, "//div[@class='key'][contains(text(), 'Telefon')]")
                    telefon = telefon_key.find_element(By.XPATH, "following-sibling::*[1]").text.strip()
                except:
                    telefon = ""

                # Web Sitesi
                try:
                    website_key = driver.find_element(By.XPATH, "//div[@class='key'][contains(text(), 'Web Site')]")
                    website_element = website_key.find_element(By.XPATH, "following-sibling::*[1]")
                    # Eğer a tagı varsa href'i al, yoksa text'i al
                    try:
                        website = website_element.find_element(By.TAG_NAME, "a").get_attribute("href")
                    except:
                        website = website_element.text.strip()
                except:
                    website = ""

                # Adres
                try:
                    adres_element = driver.find_element(By.CSS_SELECTOR, "div.address")
                    adres = adres_element.text.strip().replace('\n', ' ')
                except:
                    adres = ""

                tablo.append({
                    "Data Source/E_Exhibition": "Texhibitionist",
                    "Product": "",
                    "CompanyName": firma_adi,
                    "CompanyWebsite": website,
                    "CompanyMail": email,
                    "CompanyMail2": "",
                    "CompanyPhone": telefon,
                    "CompanyAddress": adres,
                    "CompanyZipCode": "",
                    "CompanyCity": "",
                    "CompanyCountry": ""
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

        # Email alanına göre firma dağılımı (email olanlar vs olmayanlar)
        email_stats = df['CompanyMail'].apply(lambda x: 'Email Var' if x and x.strip() else 'Email Yok').value_counts().reset_index()
        email_stats.columns = ["Email Durumu", "Firma Sayısı"]
        fig = px.bar(email_stats, x="Email Durumu", y="Firma Sayısı", title="Email Durumuna Göre Firma Dağılımı")
        st.plotly_chart(fig)
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
def scrape_bauma_all_exhibitors(max_load_more_clicks=50, debug_mode=False):
    options = Options()
    if not debug_mode:  # Debug modunda headless kapalı
        options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Ana sayfa URL'i
    base_url = "https://exhibitors.bauma.de/en/exhibitors-and-products/exhibitors-brand-names/"
    full_url_prefix = "https://exhibitors.bauma.de"
    tablo = []

    print(f"\n🔄 Ana sayfa yükleniyor: {base_url}")
    driver.get(base_url)
    time.sleep(5)

    # Load More butonuna sürekli tıkla
    load_more_count = 0
    previous_company_count = 0
    
    while load_more_count < max_load_more_clicks:
        try:
            # Mevcut firma sayısını kontrol et
            current_companies = driver.find_elements(By.CSS_SELECTOR, "td.content_company")
            current_count = len(current_companies)
            
            print(f"📊 Şu anda {current_count} firma görünüyor")
            
            # Load More butonunu bul
            load_more_button = None
            
            # Farklı seçiciler ile Load More butonunu ara
            selectors = [
                "tr.lazymore td",
                "tr[class*='lazymore'] td", 
                "td[class*='text-center']:contains('Load more')",
                "tr:has(td:contains('Load more'))"
            ]
            
            # XPath ile de dene
            xpath_selectors = [
                "//tr[@class='lazymore']//td",
                "//td[contains(text(), 'Load more')]",
                "//tr[contains(@class, 'lazymore')]",
                "//td[contains(text(), 'Load more') or contains(text(), 'load more')]"
            ]
            
            # CSS selectors dene
            for selector in selectors:
                try:
                    if ":contains" in selector:
                        continue  # CSS contains desteklenmiyor, XPath'e geç
                    load_more_button = driver.find_element(By.CSS_SELECTOR, selector)
                    if load_more_button and load_more_button.is_displayed():
                        print(f"✅ Load More butonu bulundu: {selector}")
                        break
                except:
                    continue
            
            # XPath selectors dene
            if not load_more_button:
                for xpath in xpath_selectors:
                    try:
                        load_more_button = driver.find_element(By.XPATH, xpath)
                        if load_more_button and load_more_button.is_displayed():
                            print(f"✅ Load More butonu bulundu (XPath): {xpath}")
                            break
                    except:
                        continue
            
            if not load_more_button:
                print(f"✅ Load More butonu bulunamadı veya tükendi")
                break
            
            # Sayfanın en altına scroll yap
            driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
            time.sleep(1)
            
            # Butona tıkla - farklı yöntemler dene
            click_success = False
            try:
                # Normal click
                load_more_button.click()
                click_success = True
                print(f"🔄 Load More butonuna tıklandı (normal click) ({load_more_count + 1}. kez)")
            except:
                try:
                    # JavaScript click
                    driver.execute_script("arguments[0].click();", load_more_button)
                    click_success = True
                    print(f"🔄 Load More butonuna tıklandı (JS click) ({load_more_count + 1}. kez)")
                except:
                    try:
                        # ActionChains ile click
                        from selenium.webdriver.common.action_chains import ActionChains
                        ActionChains(driver).move_to_element(load_more_button).click().perform()
                        click_success = True
                        print(f"🔄 Load More butonuna tıklandı (ActionChains) ({load_more_count + 1}. kez)")
                    except:
                        print(f"❌ Load More butonuna tıklanamadı")
                        break
            
            if not click_success:
                break
            
            # Yeni içeriğin yüklenmesi için bekle
            time.sleep(5)
            
            # Yeni firmalar yüklenip yüklenmediğini kontrol et
            new_companies = driver.find_elements(By.CSS_SELECTOR, "td.content_company")
            new_count = len(new_companies)
            
            if new_count <= current_count:
                print(f"⚠️ Yeni firma yüklenmedi, bir kez daha deneniyor...")
                time.sleep(3)
                # Tekrar kontrol
                final_companies = driver.find_elements(By.CSS_SELECTOR, "td.content_company")
                if len(final_companies) <= current_count:
                    print(f"❌ Yeni firma yüklenemedi, Load More işlemi sonlandırılıyor")
                    break
            else:
                print(f"✅ {new_count - current_count} yeni firma yüklendi")
            
            load_more_count += 1
            previous_company_count = new_count
            
        except Exception as e:
            print(f"❌ Load More işleminde hata: {e}")
            break

    print(f"🏁 Load More işlemi tamamlandı. Toplam {load_more_count} kez tıklandı")

    # Tüm firma linklerini topla
    try:
        firma_linkleri = []
        
        # Ana firma kartlarını bul
        firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "td.content_company")
        print(f"🔍 {len(firma_kartlari)} firma kartı bulundu")
        
        for kart in firma_kartlari:
            try:
                # Her kartın içindeki linki bul (ilk div.col-sm-2 a veya treffer-titel a)
                link_selectors = [
                    "div.col-sm-2 a[href*='exhibitorDetail']",
                    "div.treffer-titel a[href*='exhibitorDetail']"
                ]
                
                for selector in link_selectors:
                    try:
                        link_element = kart.find_element(By.CSS_SELECTOR, selector)
                        href = link_element.get_attribute("href")
                        if href and "exhibitorDetail" in href:
                            firma_linkleri.append(href)
                            break
                    except:
                        continue
            except:
                continue
                
        # Tekrar eden linkleri kaldır
        firma_linkleri = list(set(firma_linkleri))
        print(f"🔗 Toplam {len(firma_linkleri)} unique firma linki bulundu")
        
        if len(firma_linkleri) == 0:
            print("❌ Hiç firma linki bulunamadı, sayfa yapısını kontrol et")
            # Debug için sayfa kaynağını yazdır
            try:
                all_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='exhibitor']")
                print(f"📋 Sayfada bulunan exhibitor linkleri: {len(all_links)}")
                for i, link in enumerate(all_links[:3]):  # İlk 3 linki göster
                    print(f"  Link {i+1}: {link.get_attribute('href')}")
            except:
                pass
            driver.quit()
            return
        
    except Exception as e:
        print(f"❌ Firma linkleri çekilemedi: {e}")
        driver.quit()
        return

    # Her firma detayına gir
    for i, link in enumerate(firma_linkleri, 1):
        print(f"  🔍 ({i}/{len(firma_linkleri)}) Firma detayına giriliyor: {link}")
        try:
            driver.get(link)
            time.sleep(2)

            # Firma adı
            try:
                firma_adi = driver.find_element(By.CSS_SELECTOR, "div.contentblock_firma h1").text.strip()
            except:
                firma_adi = ""

            # Adres bilgisi
            try:
                adres_element = driver.find_element(By.CSS_SELECTOR, "div.exhibitordetails-locationinfo p")
                adres = adres_element.text.strip().replace('\n', ' ')
            except:
                adres = ""

            # İletişim bilgileri
            email, telefon, website = "", "", ""
            try:
                contact_list = driver.find_elements(By.CSS_SELECTOR, "div.exhibitordetails-contactinfo ul.exhibitordetails-contactinfo-list li")
                
                for li in contact_list:
                    try:
                        label = li.find_element(By.CSS_SELECTOR, "div:first-child").text.strip().lower()
                        value_div = li.find_element(By.CSS_SELECTOR, "div:last-child")
                        
                        if "phone" in label:
                            telefon = value_div.text.strip()
                        elif "e-mail" in label or "email" in label:
                            try:
                                # E-mail linkini bul
                                email_link = value_div.find_element(By.TAG_NAME, "a")
                                email = email_link.get_attribute("href").replace("mailto:", "")
                                # URL decode işlemi
                                import urllib.parse
                                email = urllib.parse.unquote(email)
                            except:
                                email = value_div.text.strip()
                        elif "website" in label:
                            try:
                                # Website linkini bul
                                website_link = value_div.find_element(By.TAG_NAME, "a")
                                website = website_link.get_attribute("href")
                            except:
                                website = value_div.text.strip()
                    except:
                        continue
                        
            except Exception as e:
                print(f"    ⚠️ İletişim bilgileri alınamadı: {e}")

            tablo.append({
                "Data Source/E_Exhibition": "Bauma",
                "Product": "",
                "CompanyName": firma_adi,
                "CompanyWebsite": website,
                "CompanyMail": email,
                "CompanyMail2": "",
                "CompanyPhone": telefon,
                "CompanyAddress": adres,
                "CompanyZipCode": "",
                "CompanyCity": "",
                "CompanyCountry": ""
            })

            print(f"  ✅ Eklendi: {firma_adi}")

        except Exception as e:
            print(f"  ❌ Firma detay alınamadı: {e}")
            continue

    driver.quit()

    df = pd.DataFrame(tablo)
    
    # Boş DataFrame kontrolü
    if df.empty:
        print("❌ Hiç veri bulunamadı, işlem sonlandırılıyor")
        return df

    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # 📌 SQLite formatında kaydet
    try:
        save_to_sqlite(df)
        print("✅ SQLite kaydı tamamlandı")
    except Exception as e:
        print(f"❌ SQLite kayıt hatası: {e}")

    # 📌 MySQL uyumlu dump oluştur
    try:
        create_mysql_dump_from_sqlite()
        print("✅ MySQL dump oluşturuldu")
    except Exception as e:
        print(f"❌ MySQL dump hatası: {e}")

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

        # Email alanına göre firma dağılımı
        email_stats = df['CompanyMail'].apply(lambda x: 'Email Var' if x and x.strip() else 'Email Yok').value_counts().reset_index()
        email_stats.columns = ["Email Durumu", "Firma Sayısı"]
        fig = px.bar(email_stats, x="Email Durumu", y="Firma Sayısı", title="Email Durumuna Göre Firma Dağılımı")
        st.plotly_chart(fig)
        
        # Ülke dağılımı (adresten çıkararak)
        try:
            df['Ülke'] = df['Adres'].str.split('\n').str[-1].str.strip()
            ulke_sayilari = df['Ülke'].value_counts().head(10).reset_index()
            ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
            fig2 = px.bar(ulke_sayilari, x="Ülke", y="Firma Sayısı", title="En Çok Firma Bulunan 10 Ülke")
            st.plotly_chart(fig2)
        except:
            pass
            
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")