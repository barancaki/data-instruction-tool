import io
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
def scrape_hvacr_world(sayfa_sayisi):
    """
    HVACR World 2025 fuarı katılımcı listesini çeker
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://exhibitors.hvacr-world.com/hvacr-world-2025/Exhibitor"
    tablo = []

    try:
        # İlk sayfayı yükle
        print(f"🔄 Ana sayfa yükleniyor...")
        driver.get(base_url)
        time.sleep(3)

        for page_num in range(1, sayfa_sayisi + 1):
            print(f"\n🔄 {page_num}. sayfa işleniyor...")

            try:
                # Firma kartlarını bul
                firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.card.h-100")
                
                if not firma_kartlari:
                    print(f"⚠️ {page_num}. sayfada firma bulunamadı.")
                    break

                print(f"📊 {len(firma_kartlari)} firma bulundu")

                # Firma linklerini topla (detaya gitmeden önce)
                firma_linkleri = []
                for kart in firma_kartlari:
                    try:
                        detay_link = kart.find_element(By.CSS_SELECTOR, "h5.card-title a").get_attribute("href")
                        if detay_link:
                            firma_linkleri.append(detay_link)
                    except:
                        continue

                print(f"🔗 {len(firma_linkleri)} firma linki toplandı")

                # Her firma linkine git
                for idx, detay_link in enumerate(firma_linkleri, 1):
                    try:
                        # Detay sayfasına git
                        print(f"  {idx}/{len(firma_linkleri)}. 🔍 Detay sayfası açılıyor...")
                        driver.get(detay_link)
                        time.sleep(2)

                        # Firma adı
                        try:
                            firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.company-title").text.strip()
                        except:
                            firma_adi = ""

                        # Stand numarası
                        try:
                            stand_elements = driver.find_elements(By.CSS_SELECTOR, "h6")
                            stand_no = ""
                            for elem in stand_elements:
                                text = elem.text.strip()
                                if "Stand No" in text:
                                    stand_no = text
                                    break
                        except:
                            stand_no = ""

                        # Ülke
                        try:
                            h6_elements = driver.find_elements(By.CSS_SELECTOR, "h6")
                            ulke = ""
                            for i, elem in enumerate(h6_elements):
                                text = elem.text.strip()
                                if "Stand No" in text and i + 1 < len(h6_elements):
                                    ulke = h6_elements[i + 1].text.strip()
                                    break
                            if not ulke and len(h6_elements) > 1:
                                ulke = h6_elements[1].text.strip()
                        except:
                            ulke = ""

                        # Kategori/Sektör
                        try:
                            kategori_elem = driver.find_element(By.CSS_SELECTOR, "span.badge.bg-secondary")
                            kategori = kategori_elem.text.strip()
                        except:
                            kategori = ""

                        # İletişim bilgileri
                        telefon, email, website, adres = "", "", "", ""
                        
                        try:
                            bilgi_elemanlari = driver.find_elements(By.CSS_SELECTOR, "div.company-info div.col-md-12, div.company-info div[class*='col-md']")
                            
                            for elem in bilgi_elemanlari:
                                try:
                                    icon_html = elem.get_attribute("innerHTML")
                                    text = elem.text.strip()
                                    
                                    # Website
                                    if "fa-globe" in icon_html:
                                        try:
                                            website_link = elem.find_element(By.TAG_NAME, "a")
                                            website = website_link.get_attribute("href")
                                            if not website:
                                                website = website_link.text.strip()
                                        except:
                                            website = text.replace("🌐", "").strip()
                                    
                                    # Email
                                    elif "fa-envelope" in icon_html:
                                        try:
                                            email_link = elem.find_element(By.CSS_SELECTOR, "a[href^='mailto:']")
                                            email = email_link.get_attribute("href").replace("mailto:", "")
                                        except:
                                            if "Send Enquiry" not in text:
                                                email = text.replace("📧", "").strip()
                                    
                                    # Telefon
                                    elif "fa-phone" in icon_html:
                                        telefon = text.replace("📞", "").strip()
                                    
                                    # Adres
                                    elif "fa-location-dot" in icon_html or "fa-map-marker" in icon_html:
                                        adres = text.replace("📍", "").strip()
                                        
                                except:
                                    continue
                        except:
                            pass

                        # Sosyal medya linkleri
                        facebook, linkedin, instagram, youtube = "", "", "", ""
                        
                        try:
                            sosyal_links = driver.find_elements(By.CSS_SELECTOR, "div.social-links a")
                            
                            for link in sosyal_links:
                                href = link.get_attribute("href")
                                
                                if href:
                                    if "facebook.com" in href:
                                        facebook = href
                                    elif "linkedin.com" in href:
                                        linkedin = href
                                    elif "instagram.com" in href:
                                        instagram = href
                                    elif "youtube.com" in href:
                                        youtube = href
                        except:
                            pass

                        # Email bulunamadıysa sadece website içinden ara
                        if not email and website:
                            try:
                                print(f"     🔎 Website'den email aranıyor...")
                                firma_mail = site_icerisinden_email_bul(website)
                                if firma_mail and "packagingfair.com" not in firma_mail:
                                    email = firma_mail
                            except:
                                pass

                        print(f"  ✅ {firma_adi} - {ulke}")

                        # Verileri tabloya ekle
                        tablo.append({
                            "Firma Adı": firma_adi,
                            "Ülke": ulke.upper() if ulke else "",
                            "Stand No": stand_no,
                            "Kategori": kategori,
                            "Telefon": telefon,
                            "Email": email,
                            "Adres": adres,
                            "Web Sitesi": website,
                            "Facebook": facebook,
                            "LinkedIn": linkedin,
                            "Instagram": instagram,
                            "YouTube": youtube,
                            "Detay Link": detay_link
                        })

                    except Exception as e:
                        print(f"  ❌ Firma detayı işlenirken hata: {e}")
                        continue

                # Bir sonraki sayfaya geç (son sayfa değilse)
                if page_num < sayfa_sayisi:
                    print(f"\n⏭️ {page_num + 1}. sayfaya geçiliyor...")
                    
                    # Liste sayfasına geri dön
                    driver.get(base_url)
                    time.sleep(2)
                    
                    # Pagination butonunu bul ve tıkla
                    try:
                        # Sayfa numarasına göre offset hesapla
                        offset = page_num * 24
                        
                        # JavaScript ile pagination fonksiyonunu çağır
                        driver.execute_script(f"searchFilter({offset});")
                        time.sleep(3)
                        
                        print(f"✅ {page_num + 1}. sayfa yüklendi")
                    except Exception as e:
                        print(f"⚠️ Sayfa değiştirme hatası: {e}")
                        # Alternatif yöntem: Doğrudan URL ile
                        try:
                            driver.get(f"{base_url}?offset={offset}")
                            time.sleep(3)
                        except:
                            print(f"❌ {page_num + 1}. sayfaya geçilemedi")
                            break

            except Exception as e:
                print(f"❌ Sayfa {page_num} işlenirken hata: {e}")
                break

    finally:
        driver.quit()

    # DataFrame oluştur
    df = pd.DataFrame(tablo)
    
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # Streamlit gösterimi
    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Excel (.xlsx) İndir",
            data=excel_buffer,
            file_name="hvacr_world_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name="hvacr_world_data.csv",
            mime="text/csv"
        )

        # Ülkelere göre dağılım grafiği
        if not df.empty:
            ulke_sayilari = df["Ülke"].value_counts().reset_index()
            ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
            fig = px.bar(ulke_sayilari.head(20), x="Ülke", y="Firma Sayısı", 
                         title="Ülkelere Göre Firma Dağılımı - HVACR World 2025")
            st.plotly_chart(fig)
            
            # Kategori dağılımı
            if "Kategori" in df.columns and not df["Kategori"].isna().all():
                kategori_dagilim = df[df["Kategori"] != ""]["Kategori"].value_counts().reset_index()
                kategori_dagilim.columns = ["Kategori", "Sayı"]
                fig2 = px.pie(kategori_dagilim, values="Sayı", names="Kategori",
                              title="Kategori Dağılımı")
                st.plotly_chart(fig2)
    else:
        print(f"\n📊 İstatistikler:")
        if not df.empty:
            print(df["Ülke"].value_counts())
        
    return df