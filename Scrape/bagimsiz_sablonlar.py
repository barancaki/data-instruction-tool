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

def scrape_evchargeshow():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    url = "https://www.evchargeshow.com/exhibitor"
    driver.get(url)

    tablo = []
    # Tüm firmaları yüklemek için scroll yap
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)  # yüklenme süresi
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    try:
        # Tüm firma kartlarını bekle
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#exhibitorsList .col-lg-4.col-md-6.col-sm-12"))
        )

        cards = driver.find_elements(By.CSS_SELECTOR, "#exhibitorsList .col-lg-4.col-md-6.col-sm-12")

        for card in cards:
            try:
                firma_adi = card.find_element(By.TAG_NAME, "h5").text.strip()
            except:
                firma_adi = " "

            try:
                ulke = card.find_elements(By.CSS_SELECTOR, ".text-muted")[1].text.strip()
            except:
                ulke = " "

            # Detay butonuna tıkla
            try:
                detay_btn = card.find_element(By.TAG_NAME, "button")
                driver.execute_script("arguments[0].click();", detay_btn)

                # Modal açılmasını bekle
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".modal.show"))
                )

                try:
                    website = driver.find_element(By.CSS_SELECTOR, ".modal.show a[href^='http']").get_attribute("href")
                except:
                    website = " "
                
                # Modal kapatma
                try:
                    close_btn = driver.find_element(By.CSS_SELECTOR, ".modal.show button.btn-close")
                    driver.execute_script("arguments[0].click();", close_btn)
                    WebDriverWait(driver, 3).until(
                        EC.invisibility_of_element((By.CSS_SELECTOR, ".modal.show"))
                    )
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

            except:
                website = " "

            tablo.append({
                "Firma": firma_adi,
                "Ülke": ulke,
                "Web adresi": website,
                "Mail": firma_mail
            })

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
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
def scrape_atechfuari():
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    base_url = "https://atechfuari.com/firmalar/"
    driver.get(base_url)
    time.sleep(2)  # Sayfanın yüklenmesi için küçük bir bekleme

    tablo = []

    # Ana sayfadaki tüm firma kartlarını bul
    firms = driver.find_elements(By.CSS_SELECTOR, ".blog-standard-content.row .news-block-one.col-lg-4")

    for firm in firms:
        try:
            # Firma adı
            firma_adi = firm.find_element(By.CSS_SELECTOR, "h3 a").text.strip()
        except Exception as e:
            firma_adi = " "

        try:
            # Firma web sitesi (btn-secondary linkinden alıyoruz)
            website = firm.find_element(By.CSS_SELECTOR, "a.btn.btn-sm.btn-secondary").get_attribute("href")
        except:
            website = ""

        # Mail bilgisi (websitesine gidip bulmaya çalış)
        if website:
            try:
                firma_mail = site_icerisinden_email_bul(website)
                if firma_mail == "team@packagingfair.com":
                    firma_mail = "Bu websitesi artık geçerli değildir."
            except:
                firma_mail = ""
        else:
            try:
                # Google/Bing üzerinden firma web sitesi arama fallback
                firmanin_url = bing_ilk_link_al(firma_adi)
                firma_mail = site_icerisinden_email_bul(firmanin_url)
            except:
                firma_mail = ""

        tablo.append({
            "Firma": firma_adi,
            "Web adresi": website,
            "Mail": firma_mail
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
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

