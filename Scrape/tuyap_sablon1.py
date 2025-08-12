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

def scrape_replast_all_pages(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    tablo = []
    page_num = 1

    while True:
        url = f"https://replasteurasia.com/katilimci-listesi?page={page_num}"
        driver.get(url)
        time.sleep(3)

        # Sayfadaki firma bloklarını al
        table = driver.find_elements(By.CLASS_NAME, "filter-list__item")

        # Sayfa boşsa döngüyü bitir
        if len(table) == 0:
            print(f"Veri bitti. Son sayfa: {page_num-1}")
            break

        print(f"{page_num}. sayfa işleniyor...")

        for item in table:
            try:
                firma_adi = item.find_element(By.XPATH, ".//div[@class='table-block-content'][1]").text
            except:
                firma_adi = " "

            try:
                adres = item.find_element(By.XPATH, ".//div[@class='table-block-content'][2]").text
                parcalar = adres.split("/")
                ulke = parcalar[-1].strip()
            except:
                adres = " "
                ulke = " "
            try:
                telefon = item.find_element(By.XPATH, ".//a[starts-with(@href, 'tel:')]").text
            except:
                telefon = " "
            try:
                site = item.find_element(By.XPATH, ".//a[starts-with(@href, 'http')]").get_attribute("href")
            except:
                site = " "
            try:
                # Butona tıkla
                detay_buton = item.find_element(By.CLASS_NAME, "js-open-table-detail")
                driver.execute_script("arguments[0].click();", detay_buton)
                time.sleep(0.5)  # açılma süresi

                # Ürün gruplarını listele
                urun_gruplari_liste = item.find_elements(By.CLASS_NAME, "table-detail-wrapper__list-item")
                urun_gruplari = ", ".join([li.text for li in urun_gruplari_liste])
            except:
                urun_gruplari = " "

            tablo.append({
                "Firma": firma_adi,
                "Adres": adres,
                "Ülke":ulke,
                "Telefon": telefon,
                "Web adresi": site,
                "Ürün Grupları": urun_gruplari,
                "Company Mail":"",
                "Company Zip-Code":""            
            })

        page_num += 1

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
def scrape_burtarim_fair(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    tablo = []
    page_num = 1

    while True:
        url = f"https://www.burtarim.com/katilimci-listesi?page={page_num}"
        driver.get(url)
        time.sleep(3)

        # Sayfadaki firma bloklarını al
        table = driver.find_elements(By.CLASS_NAME, "filter-list__item")

        # Sayfa boşsa döngüyü bitir
        if len(table) == 0:
            print(f"Veri bitti. Son sayfa: {page_num-1}")
            break

        print(f"{page_num}. sayfa işleniyor...")

        for item in table:
            try:
                firma_adi = item.find_element(By.XPATH, ".//div[@class='table-block-content'][1]").text
            except:
                firma_adi = " "

            try:
                adres = item.find_element(By.XPATH, ".//div[@class='table-block-content'][2]").text
                parcalar = adres.split("/")
                ulke = parcalar[-1].strip()
            except:
                adres = " "
                ulke = " "
            try:
                telefon = item.find_element(By.XPATH, ".//a[starts-with(@href, 'tel:')]").text
            except:
                telefon = " "
            try:
                site = item.find_element(By.XPATH, ".//a[starts-with(@href, 'http')]").get_attribute("href")
            except:
                site = " "
            try:
                # Butona tıkla
                detay_buton = item.find_element(By.CLASS_NAME, "js-open-table-detail")
                driver.execute_script("arguments[0].click();", detay_buton)
                time.sleep(0.5)  # açılma süresi

                # Ürün gruplarını listele
                urun_gruplari_liste = item.find_elements(By.CLASS_NAME, "table-detail-wrapper__list-item")
                urun_gruplari = ", ".join([li.text for li in urun_gruplari_liste])
            except:
                urun_gruplari = " "

            tablo.append({
                "Firma": firma_adi,
                "Adres": adres,
                "Ülke":ulke,
                "Telefon": telefon,
                "Web adresi": site,
                "Ürün Grupları": urun_gruplari,
                "Company Mail":"",
                "Company Zip-Code":"",          
            })

        page_num += 1

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
def scrape_pencere_kapi_cam_all_pages(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    tablo = []
    page_num = 1

    while True:
        url = f"https://www.avrasyapencerefuari.com/katilimci-listesi?page={page_num}"
        driver.get(url)
        time.sleep(3)

        # Sayfadaki firma bloklarını al
        table = driver.find_elements(By.CLASS_NAME, "filter-list__item")

        # Sayfa boşsa döngüyü bitir
        if len(table) == 0:
            print(f"Veri bitti. Son sayfa: {page_num-1}")
            break

        print(f"{page_num}. sayfa işleniyor...")

        for item in table:
            try:
                firma_adi = item.find_element(By.XPATH, ".//div[@class='table-block-content'][1]").text
            except:
                firma_adi = " "

            try:
                adres = item.find_element(By.XPATH, ".//div[@class='table-block-content'][2]").text
                parcalar = adres.split("/")
                ulke = parcalar[-1].strip()
            except:
                adres = " "
                ulke = " "
            try:
                telefon = item.find_element(By.XPATH, ".//a[starts-with(@href, 'tel:')]").text
            except:
                telefon = " "
            try:
                site = item.find_element(By.XPATH, ".//a[starts-with(@href, 'http')]").get_attribute("href")
            except:
                site = " "
            try:
                # Butona tıkla
                detay_buton = item.find_element(By.CLASS_NAME, "js-open-table-detail")
                driver.execute_script("arguments[0].click();", detay_buton)
                time.sleep(0.5)  # açılma süresi

                # Ürün gruplarını listele
                urun_gruplari_liste = item.find_elements(By.CLASS_NAME, "table-detail-wrapper__list-item")
                urun_gruplari = ", ".join([li.text for li in urun_gruplari_liste])
            except:
                urun_gruplari = " "

            tablo.append({
                "Firma": firma_adi,
                "Adres": adres,
                "Ülke":ulke,
                "Telefon": telefon,
                "Web adresi": site,
                "Ürün Grupları": urun_gruplari,
                "Company Mail":"",
                "Company Zip-Code":""            
            })

        page_num += 1

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
def scrape_smtech_eurasia_all_pages(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    tablo = []
    page_num = 1

    while True:
        url = f"https://smtech-eurasia.com/katilimci-listesi?page={page_num}"
        driver.get(url)
        time.sleep(3)

        # Sayfadaki firma bloklarını al
        table = driver.find_elements(By.CLASS_NAME, "filter-list__item")

        # Sayfa boşsa döngüyü bitir
        if len(table) == 0:
            print(f"Veri bitti. Son sayfa: {page_num-1}")
            break

        print(f"{page_num}. sayfa işleniyor...")

        for item in table:
            try:
                firma_adi = item.find_element(By.XPATH, ".//div[@class='table-block-content'][1]").text
            except:
                firma_adi = " "

            try:
                adres = item.find_element(By.XPATH, ".//div[@class='table-block-content'][2]").text
                parcalar = adres.split("/")
                ulke = parcalar[-1].strip()
            except:
                adres = " "
                ulke = " "
            try:
                telefon = item.find_element(By.XPATH, ".//a[starts-with(@href, 'tel:')]").text
            except:
                telefon = " "
            try:
                site = item.find_element(By.XPATH, ".//a[starts-with(@href, 'http')]").get_attribute("href")
            except:
                site = " "
            try:
                # Butona tıkla
                detay_buton = item.find_element(By.CLASS_NAME, "js-open-table-detail")
                driver.execute_script("arguments[0].click();", detay_buton)
                time.sleep(0.5)  # açılma süresi

                # Ürün gruplarını listele
                urun_gruplari_liste = item.find_elements(By.CLASS_NAME, "table-detail-wrapper__list-item")
                urun_gruplari = ", ".join([li.text for li in urun_gruplari_liste])
            except:
                urun_gruplari = " "

            tablo.append({
                "Firma": firma_adi,
                "Adres": adres,
                "Ülke":ulke,
                "Telefon": telefon,
                "Web adresi": site,
                "Ürün Grupları": urun_gruplari,
                "Company Mail":"",
                "Company Zip-Code":""            
            })

        page_num += 1

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
def scrape_expomed_all_pages(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    tablo = []
    page_num = 1

    while True:
        url = f"https://expomedistanbul.com/katilimci-listesi?page={page_num}"
        driver.get(url)
        time.sleep(3)

        # Sayfadaki firma bloklarını al
        table = driver.find_elements(By.CLASS_NAME, "filter-list__item")

        # Sayfa boşsa döngüyü bitir
        if len(table) == 0:
            print(f"Veri bitti. Son sayfa: {page_num-1}")
            break

        print(f"{page_num}. sayfa işleniyor...")

        for item in table:
            try:
                firma_adi = item.find_element(By.XPATH, ".//div[@class='table-block-content'][1]").text
            except:
                firma_adi = " "

            try:
                adres = item.find_element(By.XPATH, ".//div[@class='table-block-content'][2]").text
                parcalar = adres.split("/")
                ulke = parcalar[-1].strip()
            except:
                adres = " "
                ulke = " "
            try:
                telefon = item.find_element(By.XPATH, ".//a[starts-with(@href, 'tel:')]").text
            except:
                telefon = " "
            try:
                site = item.find_element(By.XPATH, ".//a[starts-with(@href, 'http')]").get_attribute("href")
            except:
                site = " "
            try:
                # Butona tıkla
                detay_buton = item.find_element(By.CLASS_NAME, "js-open-table-detail")
                driver.execute_script("arguments[0].click();", detay_buton)
                time.sleep(0.5)  # açılma süresi

                # Ürün gruplarını listele
                urun_gruplari_liste = item.find_elements(By.CLASS_NAME, "table-detail-wrapper__list-item")
                urun_gruplari = ", ".join([li.text for li in urun_gruplari_liste])
            except:
                urun_gruplari = " "

            tablo.append({
                "Firma": firma_adi,
                "Adres": adres,
                "Ülke":ulke,
                "Telefon": telefon,
                "Web adresi": site,
                "Ürün Grupları": urun_gruplari,
                "Company Mail":"",
                "Company Zip-Code":""            
            })

        page_num += 1

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
