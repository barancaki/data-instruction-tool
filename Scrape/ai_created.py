import pandas as pd
import time
import io
import json
import re
import html
import base64
import requests
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from Scrape.scrape import site_icerisinden_email_bul, bing_ilk_link_al, google_ilk_link_al, duckduckgo_search_selenium, extract_emails_from_source, find_email_advanced, handle_cookie_consent_final
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException


def scrape_advanced_engineering(sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://www.advancedengineeringuk.com/exhibitors/"
    tablo = []

    try:
        print(f"🔄 Ana sayfa yükleniyor...")
        driver.get(base_url)
        time.sleep(3)

        for page_num in range(1, sayfa_sayisi + 1):
            print(f"\n🔄 {page_num}. sayfa işleniyor...")

            try:
                # Sayfa URL'ini oluştur
                if page_num == 1:
                    current_url = base_url
                else:
                    current_url = f"{base_url}?stands%5Bpage%5D={page_num}"
                
                driver.get(current_url)
                time.sleep(3)

                # Liste sayfasındaki firma kartlarını bul
                firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "li.ais-Hits-item")
                if not firma_kartlari:
                    print(f"⚠️ {page_num}. sayfada firma bulunamadı.")
                    break
                
                print(f"📊 {len(firma_kartlari)} firma bulundu")

                # Detay linklerini topla
                firma_linkleri = []
                for kart in firma_kartlari:
                    try:
                        detay_link_elem = kart.find_element(By.CSS_SELECTOR, "a.card__link")
                        detay_link = detay_link_elem.get_attribute("href")
                        if detay_link:
                            # Eğer relative URL ise, base URL ile birleştir
                            if detay_link.startswith("/"):
                                detay_link = "https://www.advancedengineeringuk.com" + detay_link
                            firma_linkleri.append(detay_link)
                    except:
                        continue
                
                print(f"🔗 {len(firma_linkleri)} firma linki toplandı")

                # Her firmanın detay sayfasına git
                for idx, detay_link in enumerate(firma_linkleri, 1):
                    try:
                        print(f"  {idx}/{len(firma_linkleri)}. 🔍 Detay sayfası açılıyor...")
                        driver.get(detay_link)
                        time.sleep(2)

                        # Firma adı - detay sayfasındaki h1 elementinden
                        try:
                            firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.stand-details__title").text.strip()
                        except:
                            firma_adi = ""
                        
                        # Website
                        website = ""
                        try:
                            website_elem = driver.find_element(By.CSS_SELECTOR, ".stand-details__info-line-content a")
                            website = website_elem.get_attribute("href")
                        except:
                            website = ""
                        
                        # Adres
                        adres = ""
                        try:
                            adres_elem = driver.find_element(By.CSS_SELECTOR, ".contact-info-card__info-line-content")
                            adres = adres_elem.text.strip()
                        except:
                            adres = ""
                        
                        # Email - website varsa websitesinden çek
                        email = ""
                        if website:
                            try:
                                print(f"     🔎 Website'den email aranıyor...")
                                email_list = site_icerisinden_email_bul(website)
                                if email_list and len(email_list) > 0:
                                    # İlk geçerli email'i al
                                    for mail in email_list:
                                        if mail and "@" in mail:
                                            email = mail
                                            break
                            except:
                                pass
                        
                        # Ürün Grupları
                        urun_gruplari = ""
                        try:
                            urun_div_list = driver.find_elements(By.CSS_SELECTOR, "div.stand-details__info-line-content div.stand-details__category-pill")
                            urun_gruplari = ", ".join([div.text.strip() for div in urun_div_list if div.text.strip()])
                        except:
                            urun_gruplari = ""

                        print(f"  ✅ {firma_adi}")

                        tablo.append({
                            "Data Source/E_Exhibition": "Advanced Engineering UK",
                            "Product": urun_gruplari,
                            "CompanyName": firma_adi,
                            "CompanyWebsite": website,
                            "CompanyMail": email,
                            "CompanyMail2": "",
                            "CompanyPhone": "",
                            "CompanyAddress": adres,
                            "CompanyZipCode": "",
                            "CompanyCity": "",
                            "CompanyCountry": "",
                            "Detay Link": detay_link
                        })

                    except Exception as e:
                        print(f"  ❌ Firma detayı işlenirken hata: {e}")
                        continue

            except Exception as e:
                print(f"❌ Sayfa {page_num} işlenirken hata: {e}")
                break

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # !!! TÜM YENİ FONKSİYONLAR BU BLOĞU İÇERMELİ !!!
    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Excel (.xlsx) İndir",
            data=excel_buffer,
            file_name=f"{st.session_state.get('function_name', 'advanced_engineering')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'advanced_engineering')}.csv",
            mime="text/csv"
        )
        
        # Grafikler (bu site için ülke bilgisi olmadığından grafik gösterilmiyor)
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty:
            print(f"Toplam firma: {len(df)}")
            
    return df

def scrape_mesago(sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://sps.mesago.com/nuernberg/en/exhibitor-search.html"
    tablo = []

    try:
        for page_num in range(1, sayfa_sayisi + 1):
            print(f"\n🔄 {page_num}. sayfa işleniyor...")

            try:
                current_url = f"{base_url}?page={page_num}&pagesize=30"
                driver.get(current_url)
                time.sleep(3)

                # Liste sayfasındaki firma kartlarını bul
                firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.ex-exhibitor-search-results-container a.a-link--no-focus")
                if not firma_kartlari:
                    print(f"⚠️ {page_num}. sayfada firma bulunamadı.")
                    break
                
                print(f"📊 {len(firma_kartlari)} firma bulundu")

                # Detay linklerini topla
                firma_linkleri = []
                for kart in firma_kartlari:
                    try:
                        detay_link = kart.get_attribute("href")
                        if detay_link:
                            # Eğer relative URL ise, base URL ile birleştir
                            if detay_link.startswith("/"):
                                detay_link = "https://sps.mesago.com" + detay_link
                            firma_linkleri.append(detay_link)
                    except:
                        continue
                
                print(f"🔗 {len(firma_linkleri)} firma linki toplandı")

                # Her firmanın detay sayfasına git
                for idx, detay_link in enumerate(firma_linkleri, 1):
                    try:
                        print(f"  {idx}/{len(firma_linkleri)}. 🔍 Detay sayfası açılıyor...")
                        driver.get(detay_link)
                        time.sleep(2)

                        # Firma adı
                        try:
                            firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.ex-exhibitor-detail__title-headline").text.strip()
                        except:
                            firma_adi = ""
                        
                        # Telefon
                        telefon = ""
                        try:
                            telefon_elem = driver.find_element(By.CSS_SELECTOR, "a.ex-contact-box__address-field-tel-number")
                            telefon_href = telefon_elem.get_attribute("href")
                            if telefon_href and telefon_href.startswith("tel:"):
                                telefon = telefon_href.replace("tel:", "").strip()
                        except:
                            telefon = ""
                        
                        # Website
                        website = ""
                        try:
                            website_elem = driver.find_element(By.CSS_SELECTOR, "a.ex-contact-box__website-link")
                            website = website_elem.get_attribute("href")
                        except:
                            website = ""
                        
                        # Email - mailto linkinden veya website'den
                        email = ""
                        try:
                            contact_btn = driver.find_element(By.CSS_SELECTOR, "a.ex-contact-box__contact-btn")
                            mailto_href = contact_btn.get_attribute("href")
                            if mailto_href and mailto_href.startswith("mailto:"):
                                email = mailto_href.replace("mailto:", "").split("?")[0].strip()
                        except:
                            pass
                        
                        # Eğer email bulunamadıysa ve website varsa, website'den ara
                        if not email and website:
                            try:
                                print(f"     🔎 Website'den email aranıyor...")
                                email_list = site_icerisinden_email_bul(website)
                                if email_list and len(email_list) > 0:
                                    for mail in email_list:
                                        if mail and "@" in mail:
                                            email = mail
                                            break
                            except:
                                pass
                        
                        # Ürün Grupları (Product Groups)
                        urun_gruplari = ""
                        try:
                            # "Our product groups" bölümünden ürün gruplarını al
                            urun_li_list = driver.find_elements(By.CSS_SELECTOR, "div.ex-exhibitor-detail-categories li.ex-list-toggle__list-item span")
                            urun_gruplari = ", ".join([span.text.strip() for span in urun_li_list if span.text.strip()])
                        except:
                            urun_gruplari = ""
                        
                        # Adres Bilgileri
                        adres = ""
                        posta_kodu = ""
                        sehir = ""
                        ulke = ""
                        
                        try:
                            # Sokak/Adres
                            adres_elem = driver.find_element(By.CSS_SELECTOR, "p.ex-contact-box__address-field-street")
                            adres = adres_elem.text.strip()
                        except:
                            adres = ""
                        
                        try:
                            # Posta Kodu
                            posta_elem = driver.find_element(By.CSS_SELECTOR, "span.ex-contact-box__address-field-zip")
                            posta_kodu = posta_elem.text.strip()
                        except:
                            posta_kodu = ""
                        
                        try:
                            # Şehir
                            sehir_elem = driver.find_element(By.CSS_SELECTOR, "span.ex-contact-box__address-field-city")
                            sehir = sehir_elem.text.strip()
                        except:
                            sehir = ""
                        
                        try:
                            # Ülke
                            ulke_elem = driver.find_element(By.CSS_SELECTOR, "span.ex-contact-box__address-field-country")
                            ulke = ulke_elem.text.strip()
                        except:
                            ulke = ""
                        
                        print(f"  ✅ {firma_adi}")

                        tablo.append({
                            "Data Source/E_Exhibition": "SPS Mesago",
                            "Product": urun_gruplari,
                            "CompanyName": firma_adi,
                            "CompanyWebsite": website,
                            "CompanyMail": email,
                            "CompanyMail2": "",
                            "CompanyPhone": telefon,
                            "CompanyAddress": adres,
                            "CompanyZipCode": posta_kodu,
                            "CompanyCity": sehir,
                            "CompanyCountry": ulke,
                            "Detay Link": detay_link
                        })

                    except Exception as e:
                        print(f"  ❌ Firma detayı işlenirken hata: {e}")
                        continue

            except Exception as e:
                print(f"❌ Sayfa {page_num} işlenirken hata: {e}")
                break

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # !!! TÜM YENİ FONKSİYONLAR BU BLOĞU İÇERMELİ !!!
    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Excel (.xlsx) İndir",
            data=excel_buffer,
            file_name=f"{st.session_state.get('function_name', 'mesago')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'mesago')}.csv",
            mime="text/csv"
        )
        
        # Grafikler (ülke bilgisi olmadığından grafik gösterilmiyor)
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty:
            print(f"Toplam firma: {len(df)}")
            
    return df

def scrape_gitex_africa_morocco(scroll_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://exhibitors-dwtc.exhibitoronlinemanual.com/gitex-africa-2025/Exhibitor"
    tablo = []

    try:
        print(f"🔄 Ana sayfa yükleniyor...")
        driver.get(base_url)
        time.sleep(3)

        # Scroll yaparak içerik yükleme
        print(f"📜 Sayfa {scroll_sayisi} kez scroll ediliyor...")
        for scroll_num in range(scroll_sayisi):
            # Sayfanın sonuna scroll yap
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Yeni içeriğin yüklenmesi için bekle
            
            # Biraz yukarı scroll yap (bazı siteler için gerekli)
            driver.execute_script("window.scrollBy(0, -200);")
            time.sleep(1)
            
            print(f"  📜 {scroll_num + 1}/{scroll_sayisi} scroll tamamlandı")

        print(f"✅ Scroll işlemi tamamlandı. Firma kartları toplanıyor...")

        # Liste sayfasındaki firma kartlarını bul
        firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.item.col-12.list-group-item")
        if not firma_kartlari:
            print(f"⚠️ Firma bulunamadı.")
        else:
            print(f"📊 {len(firma_kartlari)} firma bulundu")

            # Detay linklerini topla
            firma_linkleri = []
            for kart in firma_kartlari:
                try:
                    # "VIEW PROFILE" butonunu bul (div.button_block içindeki a.btn)
                    detay_link_elem = kart.find_element(By.CSS_SELECTOR, "div.button_block a.btn")
                    detay_link = detay_link_elem.get_attribute("href")
                    if detay_link:
                        # Eğer relative URL ise, base URL ile birleştir
                        if detay_link.startswith("/"):
                            detay_link = "https://exhibitors-dwtc.exhibitoronlinemanual.com" + detay_link
                        # Sadece ExbDetails içeren linkleri al (harita linklerini filtrele)
                        if "ExbDetails" in detay_link:
                            firma_linkleri.append(detay_link)
                except:
                    continue
            
            print(f"🔗 {len(firma_linkleri)} firma linki toplandı")

            # Her firmanın detay sayfasına git
            for idx, detay_link in enumerate(firma_linkleri, 1):
                try:
                    print(f"  {idx}/{len(firma_linkleri)}. 🔍 Detay sayfası açılıyor...")
                    driver.get(detay_link)
                    time.sleep(2)

                    # Firma adı
                    try:
                        firma_adi = driver.find_element(By.CSS_SELECTOR, "h4.group.card-title.inner.list-group-item-heading").text.strip()
                    except:
                        try:
                            firma_adi = driver.find_element(By.TAG_NAME, "h4").text.strip()
                        except:
                            firma_adi = ""

                    # Stand No
                    stand_no = ""
                    try:
                        # Sayfadaki tüm metinleri kontrol edip Stand No içeren satırı bulalım
                        p_elements = driver.find_elements(By.CSS_SELECTOR, "p")
                        for p_elem in p_elements:
                            text = p_elem.text.strip()
                            if "Stand No" in text:
                                stand_no = text
                                break
                    except:
                        stand_no = ""

                    # Ülke
                    ulke = ""
                    try:
                        # Eski yöntem: span[style*='float:left']
                        span_elem = driver.find_element(By.CSS_SELECTOR, "span[style*='float:left']")
                        ulke = span_elem.text.strip()
                    except:
                        ulke = ""

                    # Website
                    website = ""
                    try:
                        # Önce sosyal linklerden bak
                        try:
                            website_elem = driver.find_element(By.CSS_SELECTOR, "li.social_website a")
                            website = website_elem.get_attribute("href")
                        except:
                            # Bulamazsa genel linklerde "VISIT WEBSITE" ara
                            links = driver.find_elements(By.TAG_NAME, "a")
                            for link in links:
                                if "VISIT WEBSITE" in link.text.upper():
                                    website = link.get_attribute("href")
                                    break
                    except:
                        website = ""

                    # Email - website varsa websitesinden çek
                    email = ""
                    if website:
                        try:
                            print(f"     🔎 Website'den email aranıyor...")
                            email_list = site_icerisinden_email_bul(website)
                            if email_list and len(email_list) > 0:
                                # İlk geçerli email'i al
                                for mail in email_list:
                                    if mail and "@" in mail:
                                        email = mail
                                        break
                        except:
                            pass

                    # Ürün Grupları
                    urun_gruplari = ""
                    try:
                        urun_li_list = driver.find_elements(By.CSS_SELECTOR, "ul.sector_block li")
                        urun_gruplari = ", ".join([li.text.strip() for li in urun_li_list if li.text.strip()])
                    except:
                        urun_gruplari = ""

                    print(f"  ✅ {firma_adi} - {ulke}")

                    tablo.append({
                        "Data Source/E_Exhibition": "GITEX Africa Morocco",
                        "Product": urun_gruplari,
                        "CompanyName": firma_adi,
                        "CompanyWebsite": website,
                        "CompanyMail": email,
                        "CompanyMail2": "",
                        "CompanyPhone": "",
                        "CompanyAddress": "",
                        "CompanyZipCode": "",
                        "CompanyCity": "",
                        "CompanyCountry": ulke.upper() if ulke else "",
                        "Stand No": stand_no,
                        "Detay Link": detay_link
                    })

                except Exception as e:
                    print(f"  ❌ Firma detayı işlenirken hata: {e}")
                    continue

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # !!! TÜM YENİ FONKSİYONLAR BU BLOĞU İÇERMELİ !!!
    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Excel (.xlsx) İndir",
            data=excel_buffer,
            file_name=f"{st.session_state.get('function_name', 'gitex_africa_morocco')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'gitex_africa_morocco')}.csv",
            mime="text/csv"
        )
        
        # Grafikler
        if not df.empty:
            try:
                if "CompanyCountry" in df.columns:
                    ulke_sayilari = df["CompanyCountry"].value_counts().reset_index()
                    ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
                    fig = px.bar(ulke_sayilari.head(20), x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Grafik çizilirken hata: {e}")
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty and "CompanyCountry" in df.columns:
            print(df["CompanyCountry"].value_counts())
            
    return df

def scrape_yasad_uyeler(sayfa_sayisi):
    """
    YASAD (Yazılım Sanayicileri Derneği) üye firmalarını çeker.
    https://www.yasad.org.tr/uyelerimiz/
    Pagination: /page/2/, /page/3/ formatında
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://www.yasad.org.tr/uyelerimiz/"
    tablo = []
    firma_bilgileri_tum = []  # Tüm sayfalardan toplanan firma bilgileri

    try:
        # Her sayfa için döngü
        for page_num in range(1, sayfa_sayisi + 1):
            print(f"\n🔄 {page_num}. sayfa işleniyor...")
            
            # Sayfa URL'ini oluştur
            if page_num == 1:
                current_url = base_url
            else:
                current_url = f"{base_url}page/{page_num}/"
            
            print(f"📄 URL: {current_url}")
            driver.get(current_url)
            time.sleep(3)

            # Scroll yaparak lazy loading içeriği yükle
            print(f"📜 Sayfa içeriği yükleniyor...")
            for scroll_num in range(3):  # Her sayfada 3 kez scroll yap
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                driver.execute_script("window.scrollBy(0, -200);")
                time.sleep(0.5)

            # Liste sayfasındaki firma kartlarını bul
            firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "li.post-card.fusion-grid-column")
            if not firma_kartlari:
                # Alternatif selector dene
                firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "li.fusion-post-cards-grid-column")
            
            if not firma_kartlari:
                print(f"⚠️ {page_num}. sayfada firma bulunamadı. Pagination sona ermiş olabilir.")
                break
            else:
                print(f"📊 {len(firma_kartlari)} firma bulundu")

                # Detay linklerini ve firma isimlerini topla
                for kart in firma_kartlari:
                    try:
                        # Firma adı ve detay linki h2.fusion-title-heading içindeki a etiketinden
                        baslik_elem = kart.find_element(By.CSS_SELECTOR, "h2.fusion-title-heading a")
                        firma_adi = baslik_elem.text.strip()
                        detay_link = baslik_elem.get_attribute("href")
                        
                        if detay_link and firma_adi:
                            firma_bilgileri_tum.append({
                                "firma_adi": firma_adi,
                                "detay_link": detay_link
                            })
                    except Exception as e:
                        continue
                
                print(f"🔗 Bu sayfadan {len(firma_kartlari)} firma linki toplandı")

        print(f"\n✅ Toplam {len(firma_bilgileri_tum)} firma linki toplandı")
        
        # Her firmanın detay sayfasına git
        for idx, bilgi in enumerate(firma_bilgileri_tum, 1):
            try:
                detay_link = bilgi["detay_link"]
                firma_adi = bilgi["firma_adi"]
                
                print(f"  {idx}/{len(firma_bilgileri_tum)}. 🔍 {firma_adi} - Detay sayfası açılıyor...")
                driver.get(detay_link)
                time.sleep(2)

                # Website
                website = ""
                try:
                    # Website genellikle sosyal linkler arasında veya iletişim bölümünde
                    website_elems = driver.find_elements(By.CSS_SELECTOR, "a[href*='http']")
                    for elem in website_elems:
                        href = elem.get_attribute("href")
                        text = elem.text.strip().lower()
                        # Firma websitesi olabilecek linkleri filtrele
                        if href and "yasad.org.tr" not in href:
                            if "web" in text or "site" in text or "www" in href:
                                website = href
                                break
                            # LinkedIn, Facebook vb. değilse ve http ile başlıyorsa
                            if not any(x in href.lower() for x in ["linkedin", "facebook", "twitter", "instagram", "youtube", "mailto:", "tel:"]):
                                if not website:  # İlk uygun linki al
                                    website = href
                except:
                    website = ""

                # Email - sayfadan çek
                email = ""
                try:
                    # Önce fusion-li-item-content div'lerinde email ara
                    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                    
                    content_divs = driver.find_elements(By.CSS_SELECTOR, "div.fusion-li-item-content")
                    for div in content_divs:
                        text = div.text.strip()
                        if '@' in text:
                            # Regex ile email'i çıkar
                            match = re.search(email_pattern, text)
                            if match:
                                email = match.group(0)
                                break
                    
                    # Bulamazsa mailto: linklerde ara
                    if not email:
                        mailto_elems = driver.find_elements(By.CSS_SELECTOR, "a[href^='mailto:']")
                        for elem in mailto_elems:
                            mailto_href = elem.get_attribute("href")
                            if mailto_href:
                                email = mailto_href.replace("mailto:", "").split("?")[0].strip()
                                break
                except:
                    pass
                
                # Eğer email bulunamadıysa ve website varsa, website'den ara
                if not email and website:
                    try:
                        print(f"     🔎 Website'den email aranıyor...")
                        email_list = site_icerisinden_email_bul(website)
                        if email_list and len(email_list) > 0:
                            for mail in email_list:
                                if mail and "@" in mail:
                                    email = mail
                                    break
                    except:
                        pass

                # Telefon
                telefon = ""
                try:
                    tel_elems = driver.find_elements(By.CSS_SELECTOR, "a[href^='tel:']")
                    for elem in tel_elems:
                        tel_href = elem.get_attribute("href")
                        if tel_href:
                            telefon = tel_href.replace("tel:", "").strip()
                            break
                except:
                    telefon = ""

                # Ürün/Hizmet açıklaması (varsa)
                urun_gruplari = ""
                try:
                    # Açıklama metni varsa al
                    aciklama_elems = driver.find_elements(By.CSS_SELECTOR, "div.fusion-text p")
                    aciklama_list = []
                    for elem in aciklama_elems:
                        text = elem.text.strip()
                        if text and len(text) > 20:  # Sadece anlamlı uzunluktaki metinleri al
                            aciklama_list.append(text)
                    if aciklama_list:
                        urun_gruplari = " | ".join(aciklama_list[:3])  # İlk 3 paragrafı al
                except:
                    urun_gruplari = ""

                print(f"  ✅ {firma_adi}")

                tablo.append({
                    "Data Source/E_Exhibition": "YASAD Üyeleri",
                    "Product": urun_gruplari,
                    "CompanyName": firma_adi,
                    "CompanyWebsite": website,
                    "CompanyMail": email,
                    "CompanyMail2": "",
                    "CompanyPhone": telefon,
                    "CompanyAddress": "",
                    "CompanyZipCode": "",
                    "CompanyCity": "",
                    "CompanyCountry": "Türkiye",
                    "Detay Link": detay_link
                })

            except Exception as e:
                print(f"  ❌ Firma detayı işlenirken hata: {e}")
                continue

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # !!! TÜM YENİ FONKSİYONLAR BU BLOĞU İÇERMELİ !!!
    if st:
        st.dataframe(df)
        
        
        # 📥 Excel İndir
        try:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Excel (.xlsx) İndir",
                data=excel_buffer,
                file_name=f"{st.session_state.get('function_name', 'yasad_uyeler')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.warning("⚠️ Excel (.xlsx) indirme için 'openpyxl' modülü gerekli. Lütfen `pip install openpyxl` komutunu çalıştırın.")
        except Exception as e:
            st.error(f"❌ Excel oluşturulurken hata: {e}")

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'yasad_uyeler')}.csv",
            mime="text/csv"
        )
        
        # İstatistikler
        if not df.empty:
            st.info(f"📊 Toplam {len(df)} firma bilgisi çekildi.")
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty:
            print(f"Toplam firma: {len(df)}")
            
    return df

def scrape_logimat(show_more_count):
    """
    LogiMAT 2026 Scraping Fonksiyonu - Düzeltilmiş Show More Mantığı
    """
    
    # --- Tarayıcı Ayarları ---
    options = Options()
    options.add_argument("--headless")  # Hata ayıklarken headless'ı kapalı tutmak iyidir, üretimde açabilirsiniz.
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://www.logimat-messe.de/en/fair/exhibitor-directory#/search/f=h-entity_orga;v_sg=0;v_fg=0;v_fpa=FUTURE"
    tablo = []
    
    try:
        print(f"🔄 Ana sayfa yükleniyor...")
        driver.get(base_url)
        
        # İlk yükleme beklemesi
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.EWP5KKC-w-J"))
            )
            time.sleep(3) # Ekstra stabilizasyon beklemesi
        except:
            print("⚠️ Sayfa yüklenirken zaman aşımı, devam ediliyor...")

        # --- Show More Döngüsü (DÜZELTİLEN KISIM) ---
        if show_more_count > 0:
            print(f"\n🔄 'Show More' butonuna {show_more_count} kez basılacak...")
            
            for i in range(show_more_count):
                try:
                    # 1. Adım: Sayfanın en altına in (Butonun görünür alana girmesi için)
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1) # Kaydırma animasyonu için kısa bekleme

                    # 2. Adım: Tüm "Show more" buton adaylarını bul
                    # HTML'e göre buton classları: EWP5KKC-u-a ve EWP5KKC-u-d
                    buttons = driver.find_elements(By.CSS_SELECTOR, "div.EWP5KKC-u-a.EWP5KKC-u-d")
                    
                    clicked = False
                    target_btn = None

                    # 3. Adım: Sadece GÖRÜNÜR (displayed) olan butonu bul
                    for btn in buttons:
                        if btn.is_displayed() and "Show more" in btn.text:
                            target_btn = btn
                            break
                    
                    if target_btn:
                        # Butonu ortalayarak emin ol
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_btn)
                        time.sleep(1)
                        
                        # JavaScript ile tıkla (Selenium click bazen element overlay hatası verebilir)
                        driver.execute_script("arguments[0].click();", target_btn)
                        
                        print(f"   ✅ {i+1}. sayfa yüklendi.")
                        
                        # 4. Adım: Yeni içeriğin yüklenmesini bekle
                        # Yükleme sonrası kart sayısının artmasını veya bir süre beklemeyi tercih edebiliriz.
                        # Basitlik adına statik bekleme:
                        time.sleep(4) 
                    else:
                        print("   ⚠️ Görünür 'Show More' butonu bulunamadı (Liste sonu olabilir).")
                        break
                        
                except Exception as e:
                    print(f"   ❌ Show more döngüsünde hata: {e}")
                    break

        # --- Link Toplama ---
        print("\n📋 Firma linkleri toplanıyor...")
        # Önce tüm firma kartlarını bul
        cards = driver.find_elements(By.CSS_SELECTOR, "div.EWP5KKC-w-J.EWP5KKC-w-U")
        
        detail_links = []
        for card in cards:
            try:
                # Kartın içindeki detay linkini al
                link_elem = card.find_element(By.CSS_SELECTOR, "a.gwt-Anchor.EWP5KKC-d-m[href*='#/detail/']")
                url = link_elem.get_attribute("href")
                
                # Karttan ön-bilgileri al (Yedek olarak)
                try:
                    title = card.find_element(By.CSS_SELECTOR, "div.gwt-Label.EWP5KKC-w-Q").text
                except: title = "Bilinmiyor"
                
                detail_links.append({"url": url, "backup_name": title})
            except:
                continue

        print(f"📊 Toplam {len(detail_links)} firma detayı gezilecek.")

        # --- Detayları Gezme (Mevcut kodunuzdaki mantık korunmuştur) ---
        main_window = driver.current_window_handle
        
        for idx, item in enumerate(detail_links, 1):
            target_url = item['url']
            firma_adi = item['backup_name']
            
            # Konsol kirliliğini önlemek için her 5 firmada bir veya hata durumunda yazdırabilirsiniz
            print(f"➡️ {idx}/{len(detail_links)} İşleniyor: {firma_adi}")
            
            try:
                driver.switch_to.new_window('tab')
                driver.get(target_url)
                
                # Detay sayfası yükleme beklemesi
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[itemprop='legalName']"))
                    )
                except:
                    pass

                # --- Veri Çekme (Aynı seçiciler) ---
                # 1. Firma Adı
                try:
                    firma_adi = driver.find_element(By.CSS_SELECTOR, "[itemprop='legalName']").text.strip()
                except: pass

                # 2. Adres
                adres, posta_kodu, sehir, ulke = "", "", "", ""
                try:
                    address_cont = driver.find_element(By.CSS_SELECTOR, "[itemprop='address']")
                    try: adres = address_cont.find_element(By.CSS_SELECTOR, "[itemprop='streetAddress']").text.strip()
                    except: pass
                    try: posta_kodu = address_cont.find_element(By.CSS_SELECTOR, "[itemprop='postalCode']").text.strip()
                    except: pass
                    try: sehir = address_cont.find_element(By.CSS_SELECTOR, "[itemprop='addressLocality']").text.strip()
                    except: pass
                    try: ulke = address_cont.find_element(By.CSS_SELECTOR, "[itemprop='addressCountry']").text.strip()
                    except: pass
                except: pass

                # 3. İletişim
                telefon, website, email = "", "", ""
                try: telefon = driver.find_element(By.CSS_SELECTOR, "[itemprop='telephone']").text.strip()
                except: pass
                try: website = driver.find_element(By.CSS_SELECTOR, "a[itemprop='url']").get_attribute("href")
                except: pass
                
                try:
                    email_links = driver.find_elements(By.XPATH, "//div[contains(@class, 'EWP5KKC-y-nb')]//a")
                    for link in email_links:
                        if "@" in link.get_attribute("textContent"):
                            email = link.get_attribute("textContent").strip().replace('\u200b', '')
                            break
                except: pass

                # 4. Ürün Grupları & Stand
                urun_gruplari, stand_no = "", ""
                try:
                    cats = [c.text.strip() for c in driver.find_elements(By.CSS_SELECTOR, "div.EWP5KKC-y-n .gwt-Label.EWP5KKC-y-H") if c.text.strip()]
                    urun_gruplari = ", ".join([c for c in set(cats) if "Categories" not in c and "Product" not in c])
                except: pass
                
                try:
                    stand_text = driver.find_element(By.CSS_SELECTOR, "a[href*='#/hallplan/']").text.strip()
                    stand_no = stand_text.split("|")[0].strip() if "|" in stand_text else stand_text
                except: pass

                tablo.append({
                    "Company Name": firma_adi,
                    "Product Categories": urun_gruplari,
                    "Email": email,
                    "Phone": telefon,
                    "Website": website,
                    "Address": adres,
                    "Zip Code": posta_kodu,
                    "City": sehir,
                    "Country": ulke,
                    "Stand No": stand_no,
                    "Detail Link": target_url
                })

            except Exception as e:
                print(f"   ❌ Detay hatası: {str(e)}")
            
            finally:
                driver.close()
                driver.switch_to.window(main_window)

    except Exception as main_e:
        print(f"🚨 Genel Hata: {main_e}")
        
    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🏁 İşlem Tamamlandı. Toplam {len(df)} firma çekildi.")
    
    # Streamlit entegrasyonu (varsa)
    if st:
        if not df.empty:
            st.success(f"Başarıyla {len(df)} firma çekildi!")
            st.dataframe(df)

            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(label="📥 Excel İndir", data=excel_buffer, file_name="logimat_2026.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("Veri çekilemedi.")

    return df

def scrape_acrex_india():
    """
    Acrex India 2026 Exhibitor List Scraping Fonksiyonu (DDG + Advanced Email Search)
    """
    print("🚀 Tarayıcı başlatılıyor...")
    options = Options()
    options.add_argument("--headless") # Hata ayıklarken kapalı tutun, çalıştığını görünce açabilirsiniz.
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized") # Pencereyi tam ekran yap
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # Sayfa yükleme stratejisini değiştir (daha hızlı olması için 'eager' denenebilir ama 'normal' daha güvenli)
    options.page_load_strategy = 'normal'

    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    except Exception as e:
        print(f"❌ Driver başlatılamadı: {e}")
        return pd.DataFrame()

    base_url = "https://acrex.in/ExhibitorList-2026"
    
    # --- 1. ADIM: Temel Verileri Çek ---
    ham_veriler = []
    try:
        print(f"🔄 Ana sayfa yükleniyor: {base_url}")
        driver.get(base_url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.exhibit_card"))
        )
        time.sleep(3) # Ekstra bekleme
        
        firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.exhibit_card")
        print(f"📊 {len(firma_kartlari)} firma kartı bulundu. Veriler hafızaya alınıyor...")
        
        for kart in firma_kartlari:
            try:
                temp_data = {}
                temp_data['name'] = kart.find_element(By.CSS_SELECTOR, "h4").text.strip()
                temp_data['hall'] = ""
                temp_data['stall'] = ""
                
                h5_elements = kart.find_elements(By.CSS_SELECTOR, "h5")
                for h5 in h5_elements:
                    text = h5.text
                    if "Hall No:" in text:
                        temp_data['hall'] = text.replace("Hall No:", "").strip()
                    if "Stall No:" in text:
                        temp_data['stall'] = text.replace("Stall No:", "").strip()
                
                if temp_data['name']:
                    ham_veriler.append(temp_data)
            except:
                continue
    except Exception as e:
        print(f"❌ Liste çekme aşamasında hata: {e}")
        driver.quit()
        return pd.DataFrame()

    print(f"✅ {len(ham_veriler)} firmanın temel bilgileri alındı. Detaylı tarama başlıyor...")
    print("-" * 50)
    
    # --- 2. ADIM: Detaylı Arama Döngüsü ---
    tablo = []
    
    # Test için ilk 5 firmada deneyelim, çalışırsa [:5] kısmını kaldırın
    # for idx, firma in enumerate(ham_veriler[:5], 1): 
    for idx, firma in enumerate(ham_veriler, 1):
        firma_adi = firma['name']
        print(f"\nProcessing [{idx}/{len(ham_veriler)}]: {firma_adi}")
        
        # DuckDuckGo ile Website Bul
        website = duckduckgo_search_selenium(driver, firma_adi)
        
        # Website bulunduysa Email Ara (Ana sayfa + İletişim sayfası)
        email = ""
        if website:
            email = find_email_advanced(driver, website)
        
        tablo.append({
            "Data Source": "Acrex India",
            "CompanyName": firma_adi,
            "CompanyWebsite": website if website else "Not Found",
            "CompanyMail": email if email else "Not Found",
            "Hall No": firma['hall'],
            "Stall No": firma['stall'],
            "CompanyCountry": "India",
        })
        
        # DDG'yi çok seri sorgulamamak için kısa bir bekleme
        time.sleep(2)

    print("-" * 50)
    print("🏁 Tüm işlemler tamamlandı. Tarayıcı kapatılıyor.")
    driver.quit()

    # DataFrame Oluşturma
    df = pd.DataFrame(tablo)

    # Streamlit Kontrolü (Eğer bu kod Streamlit içinde çalışıyorsa butonları göster)
    import sys
    if 'streamlit' in sys.modules:
        import streamlit as st
        st.success(f"✅ Tarama Tamamlandı! Toplam {len(df)} firma işlendi.")
        st.dataframe(df)
        
        col1, col2 = st.columns(2)
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        
        col1.download_button(
            label="📥 Excel Olarak İndir",
            data=excel_buffer.getvalue(),
            file_name="acrex_india_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        csv = df.to_csv(index=False).encode('utf-8-sig')
        col2.download_button(
            label="📥 CSV Olarak İndir",
            data=csv,
            file_name="acrex_india_results.csv",
            mime="text/csv"
        )

    return df

def scrape_aquatherm_tashkent(page_limit):
    """
    Aquatherm Tashkent - Sayfa limitli ve otomatik sonlanan scraper.
    """
    
    # --- Tarayıcı Ayarları ---
    options = Options()
    options.add_argument("--headless") # Hata ayıklarken kapalı tutun
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 15)
    
    base_url = "https://aquatherm-tashkent.uz/en/exhibitors-list/year/454"
    tablo = []
    
    # Streamlit Progress Bar (Opsiyonel görselleştirme)
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        print(f"🔄 Ana sayfa yükleniyor: {base_url}")
        driver.get(base_url)
        
        # Tablonun yüklenmesini bekle
        wait.until(EC.presence_of_element_located((By.ID, "ERADataTable")))
        time.sleep(3)

        current_page = 1
        
        # Döngü: Kullanıcı limiti VEYA sayfa sonu gelene kadar
        while current_page <= page_limit:
            status_text.text(f"Processing Page {current_page}/{page_limit}...")
            print(f"\n📄 Sayfa {current_page} işleniyor...")
            
            # --- Satırları Bul ve İşle ---
            rows = driver.find_elements(By.CSS_SELECTOR, "#ERADataTable tbody tr")
            row_count = len(rows)
            
            for i in range(row_count):
                try:
                    # Stale Element hatasını önlemek için listeyi tazeleyin
                    rows = driver.find_elements(By.CSS_SELECTOR, "#ERADataTable tbody tr")
                    if i >= len(rows): break
                    
                    row = rows[i]
                    cols = row.find_elements(By.TAG_NAME, "td")
                    country = cols[1].text.strip() if len(cols) > 1 else ""
                    
                    # Modal Tetikleyici
                    trigger_div = row.find_element(By.CSS_SELECTOR, "div[data-bs-toggle='modal']")
                    company_name = trigger_div.text.strip()
                    print(f"➡️  Firma: {company_name}")

                    # Modalı Aç
                    driver.execute_script("arguments[0].click();", trigger_div)
                    
                    # Modalın yüklenmesini bekle
                    modal_content = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".modal.show .modal-content")))
                    time.sleep(0.5) 

                    # Verileri Çek
                    website = ""
                    product_categories = ""
                    
                    try:
                        # Website
                        try:
                            website_elem = modal_content.find_element(By.XPATH, ".//b[contains(text(),'Website')]/parent::div/following-sibling::div[1]/a")
                            website = website_elem.get_attribute("href")
                        except:
                            pass # Bulunamazsa boş kalsın
                        
                        # Kategoriler
                        try:
                            body_text = modal_content.find_element(By.CLASS_NAME, "modal-body").text
                            if "Product Categories:" in body_text:
                                parts = body_text.split("Product Categories:")
                                if len(parts) > 1:
                                    product_categories = parts[1].split("\n")[0].strip() or parts[1].split("\n")[1].strip()
                        except: pass

                    except Exception: pass

                    # Email Bulma (Yeni Sekmede)
                    email = ""
                    if website:
                        main_window = driver.current_window_handle
                        driver.switch_to.new_window('tab')
                        # NOT: find_email_advanced global fonksiyon olarak tanımlı olmalı
                        email = find_email_advanced(driver, website) 
                        driver.close()
                        driver.switch_to.window(main_window)
                    
                    # Tabloya Ekle
                    tablo.append({
                        "Data Source": "Aquatherm Tashkent Exhibitors List",
                        "ExhibitionName": "Aquatherm Tashkent 2025",
                        "ExhibitionProductGroup": product_categories,
                        "CompanyName": company_name,
                        "CompanyWebsite": website,
                        "CompanyMail": email,
                        "CompanyMail2": "",
                        "CompanyPhone": "", 
                        "CompanyAddress": "", 
                        "CompanyZipCode": "",
                        "CompanyCity": "",
                        "CompanyCountry": country,
                        "CompanyBusinessType": ""
                    })

                    # Modalı Kapat
                    try:
                        close_btn = modal_content.find_element(By.CSS_SELECTOR, "button.btn-close")
                        driver.execute_script("arguments[0].click();", close_btn)
                        wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".modal-backdrop")))
                    except:
                        webdriver.ActionChains(driver).send_keys(webdriver.Keys.ESCAPE).perform()
                        time.sleep(1)

                except Exception as e:
                    print(f"   ❌ Satır hatası: {e}")
                    webdriver.ActionChains(driver).send_keys(webdriver.Keys.ESCAPE).perform()
                    continue

            # Progress Bar Güncelle
            progress_bar.progress(min(current_page / page_limit, 1.0))

            # --- Sayfalama Kontrolü (Burası Otomatik Kapanmayı Sağlar) ---
            if current_page >= page_limit:
                print("🛑 Kullanıcı limitine ulaşıldı.")
                break
            
            try:
                # Next butonu kontrolü
                next_btn = driver.find_element(By.ID, "ERADataTable_next")
                
                # Eğer class'ında "disabled" varsa liste bitmiştir -> DÖNGÜYÜ KIR
                if "disabled" in next_btn.get_attribute("class"):
                    print("🏁 Liste sonuna gelindi (Next butonu pasif).")
                    break
                
                # Sonraki sayfaya tıkla
                driver.execute_script("arguments[0].scrollIntoView();", next_btn)
                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(3) # Sayfa yüklenmesi için bekle
                current_page += 1
                
            except NoSuchElementException:
                print("⚠️ Pagination butonu bulunamadı, işlem bitiriliyor.")
                break

    except Exception as main_e:
        print(f"🚨 Genel Hata: {main_e}")
        st.error(f"Bir hata oluştu: {main_e}")
        
    finally:
        driver.quit()
        status_text.empty()
        progress_bar.empty()

    # Sonuçları Döndür
    df = pd.DataFrame(tablo)
    
    # Otomatik İndirme Butonu Oluşturma
    if not df.empty:
        st.success(f"Successfully scraped {len(df)} companies!")
        st.dataframe(df)
        
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Excel Result",
            data=excel_buffer,
            file_name="aquatherm_tashkent_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No data found.")

    return df

def scrape_ifat_exhibitors(load_more_count):
    
    options = Options()
    options.add_argument("--headless") # Test ederken kapalı kalsın
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 20)
    
    base_url = "https://exhibitors.ifat.de/en/exhibitors-products/exhibitors-brands"
    exhibitor_links = []
    tablo = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)

    try:
        # 1. Siteye Git
        print(f"🔄 Ana sayfa yükleniyor: {base_url}")
        status_text.text("Main page is loading...")
        driver.get(base_url)
        
        # 2. Çerezleri Geç (YENİ FONKSİYON)
        handle_cookie_consent_final(driver)
        
        # 3. Listenin Yüklenmesini Bekle
        status_text.text("Waiting for list...")
        try:
            # IFAT sitesinde firmalar tablo satırlarında (tr.hitrow) gösteriliyor
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tr.hitrow, div.treffer-titel, table.table tbody tr")))
            print("✅ Liste yüklendi.")
        except TimeoutException:
            print("⚠️ Liste yüklenemedi. Sayfa yenilenip tekrar deneniyor...")
            driver.refresh()
            handle_cookie_consent_final(driver)
            time.sleep(5)

        # 4. Load More İşlemleri
        if load_more_count > 0:
            print(f"🔄 'Load more' satırına {load_more_count} kez basılacak...")
            
            for i in range(load_more_count):
                status_text.text(f"Clicking 'Load More': {i+1}/{load_more_count}")
                progress_bar.progress((i+1) / load_more_count)
                
                # 3 deneme mekanizması
                button_clicked = False
                for attempt in range(3):
                    try:
                        # IFAT'ta Load More bir tablo satırı (tr.lazymore)
                        load_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "tr.lazymore")))
                        
                        # Görünür yap ve tıkla
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_btn)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", load_btn)
                        
                        # Yükleme beklemesi
                        time.sleep(4)
                        button_clicked = True
                        print(f"✅ Load More başarılı (Deneme {attempt + 1})")
                        break
                        
                    except TimeoutException:
                        if attempt < 2:  # Son deneme değilse
                            print(f"⚠️ Load More bulunamadı, yeniden deneniyor... ({attempt + 1}/3)")
                            time.sleep(1)
                        else:
                            print(f"⚠️ Load More bitti (3 deneme başarısız).")
                    except Exception as e:
                        if attempt < 2:
                            print(f"⚠️ Hata: {e}, yeniden deneniyor... ({attempt + 1}/3)")
                            time.sleep(1)
                        else:
                            print(f"⚠️ Load More hatası: {e}")
                
                # Hiçbir denemede başarılı olamadıysa döngüyü kır
                if not button_clicked:
                    break
        
        # 5. Linkleri Topla
        status_text.text("Collecting links...")
        print("\n📋 Linkler toplanıyor...")
        
        # IFAT'ta firmalar tr.hitrow içinde, linkler div.treffer-titel a içinde
        rows = driver.find_elements(By.CSS_SELECTOR, "tr.hitrow")
        if not rows:
            # Alternatif: Doğrudan exhibitorDetail linklerini bul
            rows = driver.find_elements(By.CSS_SELECTOR, "a[href*='exhibitorDetail']")

        for row in rows:
            try:
                # tr.hitrow içinden link al
                if row.tag_name == "tr":
                    link_elem = row.find_element(By.CSS_SELECTOR, "div.treffer-titel a, a[href*='exhibitorDetail']")
                else:
                    link_elem = row  # Doğrudan a elementi
                    
                url = link_elem.get_attribute("href")
                name = link_elem.text.strip()
                
                if url and "exhibitorDetail" in url:
                    if not any(d['url'] == url for d in exhibitor_links):
                        exhibitor_links.append({"name": name if name else "Unknown", "url": url})
            except:
                continue

        total_links = len(exhibitor_links)
        print(f"📊 Toplam {total_links} firma bulundu.")
        st.info(f"Total {total_links} exhibitors found.")
        
        # 6. Detayları Gez
        progress_bar.progress(0)
        main_window = driver.current_window_handle
        
        for idx, item in enumerate(exhibitor_links):
            status_text.text(f"Scraping {idx+1}/{total_links}: {item['name']}")
            progress_bar.progress((idx+1) / total_links)
            
            try:
                driver.get(item['url'])
                time.sleep(2)  # Sayfanın yüklenmesi için bekle
                
                # İsim
                firma_adi = ""
                try: 
                    firma_adi = driver.find_element(By.CSS_SELECTOR, "div.contentblock_firma h1, h1").text.strip()
                except: 
                    firma_adi = item['name']
                
                # Adres bilgileri - exhibitordetails-locationinfo içinden
                street_address = ""
                zip_code = ""
                city = ""
                country = ""
                full_address = ""
                
                try:
                    address_block = driver.find_element(By.CSS_SELECTOR, "div.exhibitordetails-locationinfo p")
                    address_html = address_block.get_attribute("innerHTML")
                    # <br> ile ayrılmış satırları al
                    address_lines = [line.strip() for line in address_html.replace("<br>", "\n").split("\n") if line.strip()]
                    
                    if len(address_lines) >= 1:
                        street_address = address_lines[0]
                    if len(address_lines) >= 2:
                        # İkinci satır: "31855 Aerzen" formatında olabilir
                        second_line = address_lines[1]
                        parts = second_line.split(" ", 1)
                        if len(parts) >= 1 and parts[0].isdigit():
                            zip_code = parts[0]
                            city = parts[1] if len(parts) > 1 else ""
                        else:
                            city = second_line
                    if len(address_lines) >= 3:
                        country = address_lines[2]
                    
                    full_address = ", ".join([l for l in address_lines if l])
                except Exception as e:
                    print(f"   Adres hatası: {e}")
                
                # Telefon - Contact info listesinden
                phone = ""
                try:
                    contact_items = driver.find_elements(By.CSS_SELECTOR, "ul.exhibitordetails-contactinfo-list li")
                    for item_li in contact_items:
                        try:
                            label = item_li.find_element(By.CSS_SELECTOR, "div:first-child").text.strip()
                            if "Phone" in label:
                                phone = item_li.find_element(By.CSS_SELECTOR, "div:last-child").text.strip()
                                break
                        except:
                            continue
                except Exception as e:
                    print(f"   Telefon hatası: {e}")
                
                # E-mail - Contact info listesinden
                email_from_site = ""
                try:
                    email_elem = driver.find_element(By.CSS_SELECTOR, "ul.exhibitordetails-contactinfo-list a[href^='mailto:']")
                    email_from_site = email_elem.text.strip()
                except:
                    pass
                
                # Website - Contact info listesinden
                website = ""
                try:
                    # Website linki: target="_blank" olan ve ifat.de içermeyen link
                    website_elems = driver.find_elements(By.CSS_SELECTOR, "ul.exhibitordetails-contactinfo-list a[target='_blank']")
                    for w_elem in website_elems:
                        href = w_elem.get_attribute("href")
                        if href and "ifat.de" not in href:
                            website = href
                            break
                except:
                    pass

                # Ürün grupları - contentblock_nomen içindeki targetTag linklerinden
                product_groups = ""
                try:
                    product_tags = driver.find_elements(By.CSS_SELECTOR, "div.contentblock_nomen div.targetTag a")
                    product_list = list(set([p.text.strip() for p in product_tags if p.text.strip()]))
                    product_groups = ", ".join(product_list)
                except:
                    pass

                # Email arama (website varsa ve siteden email bulunamadıysa)
                email = email_from_site
                if website and not email:
                    try:
                        driver.switch_to.new_window('tab')
                        email = find_email_advanced(driver, website)
                        driver.close()
                        driver.switch_to.window(main_window)
                    except:
                        if len(driver.window_handles) > 1:
                            driver.close()
                            driver.switch_to.window(main_window)

                tablo.append({
                    "Data Source": "IFAT Munich 2026",
                    "ExhibitionProductGroup": product_groups,
                    "CompanyName": firma_adi,
                    "CompanyWebsite": website,
                    "CompanyMail": email,
                    "CompanyMail2": "",
                    "CompanyPhone": phone,
                    "CompanyAddress": street_address,
                    "CompanyZipCode": zip_code,
                    "CompanyCity": city,
                    "CompanyCountry": country,
                    "CompanyBusinessType": "",
                    "DetailUrl": item['url']
                })

            except Exception as e:
                print(f"Hata ({item['name']}): {e}")
                continue

    except Exception as main_e:
        print(f"Genel Hata: {main_e}")
        st.error(str(main_e))
        
    finally:
        driver.quit()
        status_text.empty()
        progress_bar.empty()

    df = pd.DataFrame(tablo)
    
    if not df.empty:
        st.success(f"Tamamlandı! {len(df)} firma çekildi.")
        st.dataframe(df)
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button("📥 Excel İndir", data=excel_buffer, file_name="ifat_exhibitors.xlsx")
        
    return df

def scrape_ahri_members():
    """
    AHRI Members scraping function.
    https://www.ahrinet.org/get-involved/ahri-members
    """
    options = Options()
    # "headless=new" is more stable for recent Chrome versions
    options.add_argument("--headless=new") 
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    # Trying without explicit ChromeDriverManager first, letting Selenium Manager handle it if available
    try:
        driver = webdriver.Chrome(service=Service(), options=options)
    except:
        # Fallback to ChromeDriverManager if default fails
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://www.ahrinet.org/get-involved/ahri-members"
    tablo = []

    try:
        print(f"🔄 AHRI Members sayfası yükleniyor...")
        driver.get(base_url)
        time.sleep(5)

        member_types = [
            ("Full Members", "Full Member"),
            ("International Members", "International Member"),
            ("Affiliate Members", "Affiliate Member")
        ]

        for link_text, group_name in member_types:
            try:
                print(f"📂 {link_text} açılıyor...")
                
                # Başlık elementini bul (div içinde a tagi)
                # XPath: //div[contains(@class, 'coh-accordion-title')]//a[contains(text(), 'Full Members')]
                try:
                    accordion_link = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, f"//div[contains(@class, 'coh-accordion-title')]//a[contains(text(), '{link_text}')]"))
                    )
                except TimeoutException:
                     print(f"⚠️ {link_text} başlığı bulunamadı.")
                     continue

                # Eğer zaten açık değilse tıkla (aria-expanded kontrolü)
                is_expanded = accordion_link.get_attribute("aria-expanded")
                if is_expanded != "true":
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", accordion_link)
                    time.sleep(1)
                    driver.execute_script("arguments[0].click();", accordion_link)
                    time.sleep(2)
                else:
                    print(f"   ℹ️ {link_text} zaten açık.")

                # İçerik div'ini bul
                # href="#id" şeklindedir
                href = accordion_link.get_attribute("href")
                if href and "#" in href:
                    content_id = href.split("#")[-1]
                    
                    try:
                        # İçeriğin görünür olmasını bekle
                        content_div = WebDriverWait(driver, 10).until(
                            EC.visibility_of_element_located((By.ID, content_id))
                        )
                    except TimeoutException:
                        print(f"⚠️ {link_text} içeriği görünür olmadı, DOM'dan çekiliyor...")
                        content_div = driver.find_element(By.ID, content_id)
                    
                    # Firmaları bul
                    links = content_div.find_elements(By.TAG_NAME, "a")
                    print(f"📊 {len(links)} firma bulundu")

                    for link in links:
                        try:
                            name = link.text.strip()
                            website = link.get_attribute("href")
                            
                            if not name or not website or website.startswith("#") or "ahrinet.org" in website:
                                continue
                            
                            # Email arama - Gelişmiş Yöntem (Yeni Tab)
                            email = ""
                            try:
                                # Ana pencereyi kaydet
                                main_window = driver.current_window_handle
                                
                                # Yeni sekme aç
                                driver.switch_to.new_window('tab')
                                
                                # Gelişmiş email arama fonksiyonunu çağır
                                # Not: find_email_advanced içinde driver.get(url) yapılıyor
                                email = find_email_advanced(driver, website)
                                
                                # Sekmeyi kapat ve ana pencereye dön
                                driver.close()
                                driver.switch_to.window(main_window)
                                
                            except Exception as email_e:
                                print(f"     ❌ Email arama hatası ({name}): {email_e}")
                                # Hata durumunda pencere kontrolü yap
                                try:
                                    if len(driver.window_handles) > 1:
                                        driver.close()
                                    driver.switch_to.window(main_window)
                                except:
                                    pass

                            tablo.append({
                                "Data Source/ExhibitionName": "AHRI Members",
                                "ExhibitionProductGroup": group_name,
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
                        except Exception as inner_e:
                            continue

                else:
                    print(f"⚠️ {link_text} için ID bulunamadı.")
                    continue

            except Exception as e:
                print(f"  ❌ {link_text} işlenirken hata: {e}")
                import traceback
                traceback.print_exc()
                continue

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Excel (.xlsx) İndir",
            data=excel_buffer,
            file_name=f"ahri_members.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"ahri_members.csv",
            mime="text/csv"
        )
            
    return df

def scrape_warsaw_hvac_expo(load_more_count):
    """
    Warsaw HVAC Expo Exhibitors Scraper
    https://warsawhvacexpo.com/en/exhibitors-catalog/
    Uses pagination via 'Load More' button
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://warsawhvacexpo.com/en/exhibitors-catalog/"
    tablo = []

    try:
        print(f"🔄 Ana sayfa yükleniyor...")
        driver.get(base_url)
        time.sleep(4)

        # Load More butonuna tıklama döngüsü
        if load_more_count > 0:
            print(f"\n🔄 'Load More' butonuna {load_more_count} kez basılacak...")
            
            for i in range(load_more_count):
                try:
                    # Sayfanın en altına kaydır
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)
                    
                    # Load More butonunu bul
                    try:
                        load_more_btn = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.exhibitor-catalog__pagination-btn"))
                        )
                        
                        # Butonu görünür alana getir
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_btn)
                        time.sleep(0.5)
                        
                        # Butona tıkla
                        driver.execute_script("arguments[0].click();", load_more_btn)
                        print(f"  ✅ {i+1}/{load_more_count} 'Load More' tıklandı")
                        time.sleep(3)  # Yeni içeriğin yüklenmesi için bekle
                        
                    except TimeoutException:
                        print(f"  ⚠️ 'Load More' butonu bulunamadı, tüm firmalar yüklenmiş olabilir.")
                        break
                        
                except Exception as e:
                    print(f"  ❌ Load More tıklanırken hata: {e}")
                    break

        print(f"\n✅ Tüm içerik yüklendi. Firma kartları toplanıyor...")

        # Firma kartlarını bul
        firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.exhibitor-catalog__exh-card")
        
        if not firma_kartlari:
            print(f"⚠️ Firma bulunamadı.")
        else:
            print(f"📊 {len(firma_kartlari)} firma bulundu")

            # Her firma kartından bilgileri çek
            for idx, kart in enumerate(firma_kartlari, 1):
                try:
                    # Firma adı
                    firma_adi = ""
                    try:
                        firma_adi_elem = kart.find_element(By.CSS_SELECTOR, "h3.exhibitor-catalog__exh-card-title")
                        firma_adi = firma_adi_elem.text.strip()
                    except:
                        firma_adi = ""
                    
                    # Website
                    website = ""
                    try:
                        contact_links = kart.find_elements(By.CSS_SELECTOR, "a.exhibitor-catalog__exh-card-info-contanct-single")
                        for link in contact_links:
                            link_text = link.find_element(By.CSS_SELECTOR, "p.exhibitor-catalog__exh-card-info-contanct-single-text").text.strip().lower()
                            if "website" in link_text:
                                website = link.get_attribute("href")
                                break
                    except:
                        website = ""
                    
                    # Email
                    email = ""
                    try:
                        contact_links = kart.find_elements(By.CSS_SELECTOR, "a.exhibitor-catalog__exh-card-info-contanct-single")
                        for link in contact_links:
                            href = link.get_attribute("href")
                            if href and href.startswith("mailto:"):
                                email = href.replace("mailto:", "").split("?")[0].strip()
                                break
                    except:
                        email = ""
                    
                    # Telefon
                    telefon = ""
                    try:
                        contact_links = kart.find_elements(By.CSS_SELECTOR, "a.exhibitor-catalog__exh-card-info-contanct-single")
                        for link in contact_links:
                            href = link.get_attribute("href")
                            if href and href.startswith("tel:"):
                                telefon = href.replace("tel:", "").strip()
                                break
                    except:
                        telefon = ""
                    
                    # Stand numarası
                    stand_no = ""
                    try:
                        stand_elem = kart.find_element(By.CSS_SELECTOR, "p.exhibitor-catalog__exh-card-info-stand-number")
                        stand_no = stand_elem.text.strip()
                    except:
                        stand_no = ""
                    
                    # Ürün grupları / Brands
                    urun_gruplari = ""
                    try:
                        brand_elems = kart.find_elements(By.CSS_SELECTOR, "p.exhibitor-catalog__exh-card-brand-single")
                        urun_gruplari = ", ".join([b.text.strip() for b in brand_elems if b.text.strip()])
                    except:
                        urun_gruplari = ""
                    
                    # Email yoksa website'den bulmayı dene
                    if not email and website:
                        try:
                            print(f"  {idx}/{len(firma_kartlari)}. 🔎 {firma_adi} - Website'den email aranıyor...")
                            email_list = site_icerisinden_email_bul(website)
                            if email_list and len(email_list) > 0:
                                for mail in email_list:
                                    if mail and "@" in mail:
                                        email = mail
                                        break
                        except:
                            pass

                    print(f"  ✅ {idx}/{len(firma_kartlari)} - {firma_adi}")

                    tablo.append({
                        "Data Source/ExhibitionName": "Warsaw HVAC Expo",
                        "ExhibitionProductGroup": urun_gruplari,
                        "CompanyName": firma_adi,
                        "CompanyWebsite": website,
                        "CompanyMail": email,
                        "CompanyMail2": "",
                        "CompanyPhone": telefon,
                        "CompanyAddress": "",
                        "CompanyZipCode": "",
                        "CompanyCity": "",
                        "CompanyCountry": "Poland",
                        "CompanyBusinessType": "",
                        "Stand No": stand_no
                    })

                except Exception as e:
                    print(f"  ❌ Firma bilgisi işlenirken hata: {e}")
                    continue

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # !!! TÜM YENİ FONKSİYONLAR BU BLOĞU İÇERMELİ !!!
    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        try:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Excel (.xlsx) İndir",
                data=excel_buffer,
                file_name=f"{st.session_state.get('function_name', 'warsaw_hvac_expo')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.warning("⚠️ Excel (.xlsx) indirme için 'openpyxl' modülü gerekli. Lütfen `pip install openpyxl` komutunu çalıştırın.")
        except Exception as e:
            st.error(f"❌ Excel oluşturulurken hata: {e}")

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'warsaw_hvac_expo')}.csv",
            mime="text/csv"
        )
        
        # İstatistikler
        if not df.empty:
            st.info(f"📊 Toplam {len(df)} firma bilgisi çekildi.")
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty:
            print(f"Toplam firma: {len(df)}")
            
    return df

def scrape_ptc_asia(page_count):
    """
    PTC Asia (Power Transmission and Control) Exhibitors Scraper
    https://service.ptc-asia.com/VSCENTER2/visitor/PTC25/match/exhibitor?lang=en-US&page=1
    Uses URL-based pagination
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://service.ptc-asia.com/VSCENTER2/visitor/PTC25/match/exhibitor?lang=en-US&page="
    tablo = []

    try:
        # Sayfa sayfa dolaş
        for page in range(1, page_count + 1):
            print(f"\n🔄 Sayfa {page}/{page_count} yükleniyor...")
            driver.get(f"{base_url}{page}")
            time.sleep(3)

            # Firma kartlarını bul
            firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.exh-list-item")
            
            if not firma_kartlari:
                print(f"⚠️ Sayfa {page}'de firma bulunamadı, tarama sonlandırılıyor.")
                break
            else:
                print(f"📊 Sayfa {page}'de {len(firma_kartlari)} firma bulundu")

            # Her firma için detay linklerini topla
            detay_linkleri = []
            for kart in firma_kartlari:
                try:
                    # Firma adını ve detay linkini al
                    firma_link_elem = kart.find_element(By.CSS_SELECTOR, "div.exh-list-title a")
                    firma_adi = firma_link_elem.text.strip()
                    detay_link = firma_link_elem.get_attribute("href")
                    
                    # Hall ve Stand bilgisini al
                    hall = ""
                    stand = ""
                    try:
                        text_spans = kart.find_elements(By.CSS_SELECTOR, "div.exh-list-text-span")
                        for span in text_spans:
                            span_text = span.text
                            if "Hall:" in span_text:
                                hall = span.find_element(By.TAG_NAME, "span").text.strip()
                            elif "Stand:" in span_text:
                                stand = span.find_element(By.TAG_NAME, "span").text.strip()
                    except:
                        pass
                    
                    detay_linkleri.append({
                        "firma_adi": firma_adi,
                        "detay_link": detay_link,
                        "hall": hall,
                        "stand": stand
                    })
                except Exception as e:
                    print(f"  ❌ Firma linki alınırken hata: {e}")
                    continue

            # Her firma detay sayfasına git ve iletişim bilgilerini al
            for idx, firma_bilgi in enumerate(detay_linkleri, 1):
                try:
                    firma_adi = firma_bilgi["firma_adi"]
                    detay_link = firma_bilgi["detay_link"]
                    hall = firma_bilgi["hall"]
                    stand = firma_bilgi["stand"]
                    
                    print(f"  {idx}/{len(detay_linkleri)} - {firma_adi} detay sayfası açılıyor...")
                    
                    # Detay sayfasına git
                    driver.get(detay_link)
                    time.sleep(2)
                    
                    # İletişim bilgilerini al
                    adres = ""
                    email = ""
                    telefon = ""
                    website = ""
                    
                    try:
                        contact_div = driver.find_element(By.CSS_SELECTOR, "div.exh-contact")
                        contact_paragraphs = contact_div.find_elements(By.TAG_NAME, "p")
                        
                        for p in contact_paragraphs:
                            p_text = p.text.strip()
                            if p_text.startswith("Address") or p_text.startswith("地址"):
                                adres = p_text.split("：", 1)[-1].split(":", 1)[-1].strip()
                            elif p_text.startswith("Email") or p_text.startswith("邮箱"):
                                email = p_text.split("：", 1)[-1].split(":", 1)[-1].strip()
                            elif p_text.startswith("Tel") or p_text.startswith("电话"):
                                telefon = p_text.split("：", 1)[-1].split(":", 1)[-1].strip()
                            elif p_text.startswith("Website") or p_text.startswith("网址"):
                                website = p_text.split("：", 1)[-1].split(":", 1)[-1].strip()
                    except Exception as e:
                        print(f"    ⚠️ İletişim bilgisi alınamadı: {e}")
                    
                    # Ürün gruplarını al (Products Recommendation)
                    urun_gruplari = ""
                    try:
                        product_items = driver.find_elements(By.CSS_SELECTOR, "div.product-item h3")
                        if product_items:
                            urun_gruplari = ", ".join([p.text.strip() for p in product_items if p.text.strip()][:5])
                    except:
                        pass
                    
                    # Email yoksa website'den bulmayı dene
                    if not email and website:
                        try:
                            # Website'i düzelt
                            if website and not website.startswith("http"):
                                website = "http://" + website
                            
                            print(f"    🔎 Website'den email aranıyor...")
                            email_list = site_icerisinden_email_bul(website)
                            if email_list and len(email_list) > 0:
                                for mail in email_list:
                                    if mail and "@" in mail:
                                        email = mail
                                        break
                        except:
                            pass

                    print(f"  ✅ {idx}/{len(detay_linkleri)} - {firma_adi}")

                    # Stand No formatla
                    stand_no = f"{hall}-{stand}" if hall and stand else (stand or hall)

                    tablo.append({
                        "Data Source/ExhibitionName": "Power Transmission and Control",
                        "ExhibitionProductGroup": urun_gruplari,
                        "CompanyName": firma_adi,
                        "CompanyWebsite": website,
                        "CompanyMail": email,
                        "CompanyMail2": "",
                        "CompanyPhone": telefon,
                        "CompanyAddress": adres,
                        "CompanyZipCode": "",
                        "CompanyCity": "",
                        "CompanyCountry": "China",
                        "CompanyBusinessType": "",
                        "Stand No": stand_no
                    })

                except Exception as e:
                    print(f"  ❌ Firma bilgisi işlenirken hata: {e}")
                    continue

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # !!! TÜM YENİ FONKSİYONLAR BU BLOĞU İÇERMELİ !!!
    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        try:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Excel (.xlsx) İndir",
                data=excel_buffer,
                file_name=f"{st.session_state.get('function_name', 'ptc_asia')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.warning("⚠️ Excel (.xlsx) indirme için 'openpyxl' modülü gerekli. Lütfen `pip install openpyxl` komutunu çalıştırın.")
        except Exception as e:
            st.error(f"❌ Excel oluşturulurken hata: {e}")

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'ptc_asia')}.csv",
            mime="text/csv"
        )
        
        # İstatistikler
        if not df.empty:
            st.info(f"📊 Toplam {len(df)} firma bilgisi çekildi.")
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty:
            print(f"Toplam firma: {len(df)}")
            
    return df

def scrape_mca_world_fair():
    """
    MCA World Fair Exhibitors Scraper
    https://www.mcaworldfair.com/katilimcilar/
    Extracts company name from img alt and website from href.
    Scrapes company websites for emails.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://www.mcaworldfair.com/katilimcilar/"
    tablo = []

    try:
        print(f"🔄 Ana sayfa yükleniyor: {base_url}")
        driver.get(base_url)
        time.sleep(5)

        # Firma kutularını bul
        zoom_boxes = driver.find_elements(By.CSS_SELECTOR, "div.zoom_box")
        
        if not zoom_boxes:
            print("⚠️ Hiç firma kutusu bulunamadı.")
            # Belki farklı bir selector lazımdır (Örn: .column_zoom_box)
            zoom_boxes = driver.find_elements(By.CSS_SELECTOR, ".column_zoom_box")
            
        print(f"📊 {len(zoom_boxes)} firma kutusu bulundu.")

        temp_data = []
        for box in zoom_boxes:
            try:
                link_elem = box.find_element(By.TAG_NAME, "a")
                website = link_elem.get_attribute("href")
                
                img_elem = link_elem.find_element(By.TAG_NAME, "img")
                alt_text = img_elem.get_attribute("alt")
                
                # Firma adını temizle
                # "AKSFLOW LOGO" -> "AKSFLOW"
                # "mca-logo-auma" -> "AUMA"
                # "ege-endustriyel" -> "EGE ENDUSTRIYEL"
                
                firma_adi = alt_text
                if firma_adi:
                    # Bilinen kalıpları çıkar
                    firma_adi = firma_adi.replace("LOGO", "").replace("logo", "").replace("mca-", "").replace("MCA-", "")
                    # Tireleri boşluğa çevir
                    firma_adi = firma_adi.replace("-", " ")
                    # Gereksiz boşlukları temizle ve büyük harf yap
                    firma_adi = " ".join(firma_adi.split()).upper()
                
                if not firma_adi and website:
                    # Eğer alt text yoksa linkten çıkarmayı dene
                    domain = website.split("//")[-1].split("/")[0]
                    firma_adi = domain.split(".")[0].upper()

                temp_data.append({
                    "firma_adi": firma_adi,
                    "website": website
                })
            except Exception as e:
                continue

        print(f"🔗 {len(temp_data)} firma bilgisi işlenmek üzere hazırlandı.")

        # Her firma için email ara
        for idx, item in enumerate(temp_data, 1):
            try:
                firma_adi = item["firma_adi"]
                website = item["website"]
                email = ""
                
                print(f"  {idx}/{len(temp_data)}. 🔎 {firma_adi} - Website'den email aranıyor: {website}")
                
                if website and "mcaworldfair.com" not in website: # Kendi sitelerini taramasın
                    try:
                        email_list = site_icerisinden_email_bul(website)
                        if email_list and len(email_list) > 0:
                            # Geçerli bir email seç
                            for mail in email_list:
                                if mail and "@" in mail and "." in mail:
                                    email = mail
                                    break
                    except:
                        pass
                
                tablo.append({
                    "Data Source/ExhibitionName": "MCA World Fair",
                    "ExhibitionProductGroup": "",
                    "CompanyName": firma_adi,
                    "CompanyWebsite": website,
                    "CompanyMail": email,
                    "CompanyMail2": "",
                    "CompanyPhone": "",
                    "CompanyAddress": "",
                    "CompanyZipCode": "",
                    "CompanyCity": "",
                    "CompanyCountry": "Turkey",
                    "CompanyBusinessType": "",
                    "Stand No": ""
                })
                
                print(f"    ✅ Tamamlandı: {email if email else 'Email bulunamadı'}")

            except Exception as e:
                print(f"    ❌ Hata: {e}")
                continue

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # !!! TÜM YENİ FONKSİYONLAR BU BLOĞU İÇERMELİ !!!
    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        try:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Excel (.xlsx) İndir",
                data=excel_buffer,
                file_name=f"{st.session_state.get('function_name', 'mca_world_fair')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.warning("⚠️ Excel (.xlsx) indirme için 'openpyxl' modülü gerekli. Lütfen `pip install openpyxl` komutunu çalıştırın.")
        except Exception as e:
            st.error(f"❌ Excel oluşturulurken hata: {e}")

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'mca_world_fair')}.csv",
            mime="text/csv"
        )
        
        # İstatistikler
        if not df.empty:
            st.info(f"📊 Toplam {len(df)} firma bilgisi çekildi.")
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty:
            print(f"Toplam firma: {len(df)}")
            
    return df


def scrape_logimotion(sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://logimotion.ae.messefrankfurt.com/dubai/en/exhibitor-search.html"
    tablo = []

    try:
        for page_num in range(1, sayfa_sayisi + 1):
            print(f"\n🔄 {page_num}. sayfa işleniyor...")

            try:
                # Sayfa URL'i
                current_url = f"{base_url}?page={page_num}&pagesize=30"
                print(f"📄 URL: {current_url}")
                driver.get(current_url)
                time.sleep(3)

                # Liste sayfasındaki firma kartlarını bul
                firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.ex-exhibitor-search-results-container a.a-link--no-focus")
                
                if not firma_kartlari:
                    print(f"⚠️ {page_num}. sayfada firma bulunamadı.")
                    break
                
                print(f"📊 {len(firma_kartlari)} firma bulundu")

                # Detay linklerini topla
                firma_linkleri = []
                for kart in firma_kartlari:
                    try:
                        detay_link = kart.get_attribute("href")
                        if detay_link:
                            # Eğer relative URL ise, base domain ile birleştir
                            if detay_link.startswith("/"):
                                detay_link = "https://logimotion.ae.messefrankfurt.com" + detay_link
                            firma_linkleri.append(detay_link)
                    except:
                        continue
                
                # Linkleri deduplicate et (sırayı koruyarak)
                firma_linkleri = list(dict.fromkeys(firma_linkleri))
                print(f"🔗 {len(firma_linkleri)} firma linki toplandı (Tekrarlar temizlendi)")

                # Her firmanın detay sayfasına git
                for idx, detay_link in enumerate(firma_linkleri, 1):
                    try:
                        print(f"  {idx}/{len(firma_linkleri)}. 🔍 Detay sayfası açılıyor...")
                        driver.get(detay_link)
                        time.sleep(2)

                        # Firma adı
                        try:
                            # Önce h1 headline dene
                            try:
                                firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.ex-exhibitor-detail__title-headline").text.strip()
                            except:
                                firma_adi = driver.find_element(By.TAG_NAME, "h1").text.strip()
                        except:
                            firma_adi = ""
                        
                        # Adres, Ülke, Şehir, Zip Parsing
                        adres_text = ""
                        ulke = ""
                        posta_kodu = ""
                        sehir = ""

                        try:
                            # Adres elementini bul
                            try:
                                adres_elem = WebDriverWait(driver, 5).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, "p.ex-contact-box__address-field-full-address"))
                                )
                            except:
                                # Element yoksa direk geç
                                adres_elem = None

                            if adres_elem:
                                # innerHTML ile alıp <br> leri \n yapalım
                                raw_html = adres_elem.get_attribute("innerHTML")
                                if raw_html:
                                    # <br> taglerini newline ile değiştir
                                    adres_text_raw = raw_html.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
                                    # Diğer HTML taglerini regex ile temizle
                                    adres_text = re.sub(r'<[^>]+>', '', adres_text_raw).strip()
                                else:
                                    adres_text = adres_elem.text.strip() # Fallback
                                
                                # Satırlara böl
                                adres_satirlari = [satir.strip() for satir in adres_text.split('\n') if satir.strip()]
                                
                                if len(adres_satirlari) > 0:
                                    # Son satır ülke
                                    ulke = adres_satirlari[-1]
                                    
                                    # Eğer sondan bir önceki satır varsa, posta kodu ve şehir olabilir
                                    if len(adres_satirlari) > 1:
                                        sondan_bir_onceki = adres_satirlari[-2]
                                        # Zip kodu tespiti (basitçe sayı içeriyor mu veya başta sayı var mı)
                                        # Örnek: "35010 Padova" -> 35010 zip, Padova şehir
                                        parts = sondan_bir_onceki.split(' ', 1)
                                        if len(parts) > 1 and any(c.isdigit() for c in parts[0]):
                                            posta_kodu = parts[0]
                                            sehir = parts[1]
                                        else:
                                            # Zip yoksa tüm satırı şehir kabul et
                                            sehir = sondan_bir_onceki
                        except Exception as e:
                            # print(f"Adres parse hatası: {e}")
                            pass
                            
                        # Telefon
                        telefon = ""
                        try:
                            telefon_elem = driver.find_element(By.CSS_SELECTOR, "a.ex-contact-box__address-field-tel-number")
                            telefon_href = telefon_elem.get_attribute("href")
                            if telefon_href and "tel:" in telefon_href:
                                telefon = telefon_href.replace("tel:", "").strip()
                        except:
                            telefon = ""

                        # Website
                        website = ""
                        try:
                            website_elem = driver.find_element(By.CSS_SELECTOR, "a.ex-contact-box__website-link")
                            website = website_elem.get_attribute("href")
                        except:
                            website = ""
                        
                        # Email
                        email = ""
                        try:
                            email_btn = driver.find_element(By.CSS_SELECTOR, "a.ex-contact-box__contact-btn")
                            mailto_href = email_btn.get_attribute("href")
                            if mailto_href and "mailto:" in mailto_href:
                                email = mailto_href.replace("mailto:", "").split("?")[0].strip()
                        except:
                            email = ""
                        
                        # Eğer email yoksa ve website varsa, siteden ara
                        if not email and website:
                            try:
                                print(f"     🔎 Website'den email aranıyor...")
                                email_list = site_icerisinden_email_bul(website)
                                if email_list and len(email_list) > 0:
                                    for mail in email_list:
                                        if mail and "@" in mail:
                                            email = mail
                                            break
                            except:
                                pass
                        
                        # Ürün Grupları
                        urun_gruplari = ""
                        try:
                            urun_listesi = driver.find_elements(By.CSS_SELECTOR, "div.ex-exhibitor-detail-categories li.ex-list-toggle__list-item span")
                            urun_gruplari = ", ".join([item.text.strip() for item in urun_listesi if item.text.strip()])
                        except:
                            urun_gruplari = ""

                        print(f"  ✅ {firma_adi} - {ulke}")

                        tablo.append({
                            "Data Source/ExhibitionName": "Logimotion Dubai",
                            "ExhibitionProductGroup": urun_gruplari,
                            "CompanyName": firma_adi,
                            "CompanyWebsite": website,
                            "CompanyMail": email,
                            "CompanyMail2": "",
                            "CompanyPhone": telefon,
                            "CompanyAddress": adres_text.replace("\n", " "),
                            "CompanyZipCode": posta_kodu,
                            "CompanyCity": sehir,
                            "CompanyCountry": ulke,
                            "CompanyBusinessType": "",
                            "Detay Link": detay_link
                        })

                    except Exception as e:
                        print(f"  ❌ Firma detayı işlenirken hata: {e}")
                        continue

            except Exception as e:
                print(f"❌ Sayfa {page_num} işlenirken hata: {e}")
                break

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Excel (.xlsx) İndir",
            data=excel_buffer,
            file_name=f"{st.session_state.get('function_name', 'logimotion')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'logimotion')}.csv",
            mime="text/csv"
        )
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty:
            print(f"Toplam firma: {len(df)}")
            
    return df

def scrape_gitex(scroll_count):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://exhibitors.gitex.com/gitex-global-2025/Exhibitor"
    tablo = []

    try:
        print(f"🔄 Gitex Global anasayfası yükleniyor...")
        driver.get(base_url)
        time.sleep(5)

        # Scrolling logic
        print(f"🔄 {scroll_count} kez aşağı kaydırılacak...")
        for i in range(scroll_count):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            print(f"  ⬇️ Scoll {i+1}/{scroll_count} tamamlandı. Yükleniyor...")
            time.sleep(4) # Wait for content to load
        
        # Linkleri topla
        print("🔍 Linkler toplanıyor...")
        try:
            # "VIEW PROFILE" button links
            # Based on inspection: href="/gitex-global-2025/Exhibitor/ExbDetails/..."
            link_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/Exhibitor/ExbDetails/')]")
            
            # Deduplicate links while keeping order
            seen_links = set()
            firma_linkleri = []
            for elem in link_elements:
                url = elem.get_attribute("href")
                if url and url not in seen_links:
                    seen_links.add(url)
                    firma_linkleri.append(url)
                    
            print(f"🔗 {len(firma_linkleri)} benzersiz firma linki bulundu.")
            
        except Exception as e:
            print(f"❌ Linkler alınırken hata: {e}")
            firma_linkleri = []

        # Her firmayı ziyaret et
        for idx, detay_link in enumerate(firma_linkleri, 1):
            try:
                print(f"  {idx}/{len(firma_linkleri)}. 🔍 Detay sayfası: {detay_link}")
                driver.get(detay_link)
                time.sleep(2)
                
                # --- VERİ ÇEKME ---
                
                # 1. Company Name
                # Selector: h4.group.card-title.inner.list-group-item-heading
                try:
                    firma_adi = driver.find_element(By.CSS_SELECTOR, "h4.group.card-title").text.strip()
                except:
                    try: 
                        firma_adi = driver.find_element(By.TAG_NAME, "h4").text.strip()
                    except:
                        firma_adi = ""
                
                # 2. Country
                # Selector: span with float:left
                # Alternatively, check existing HTML logic: <span style="float:left;">Romania</span>
                try:
                    country_elem = driver.find_element(By.XPATH, "//span[contains(@style, 'float:left')]")
                    country = country_elem.text.strip()
                except:
                    country = ""
                
                # 3. Product Groups
                # Selector: ul.sector_block li
                try:
                    products_elems = driver.find_elements(By.CSS_SELECTOR, "ul.sector_block li")
                    products = ", ".join([p.text.strip() for p in products_elems if p.text.strip()])
                except:
                    products = ""
                
                # 4. Website
                # Selector: a containing "VISIT WEBSITE" or img
                website = ""
                try:
                    # Look for 'VISIT WEBSITE' text
                    website_elem = driver.find_element(By.XPATH, "//a[contains(., 'VISIT WEBSITE')]")
                    website = website_elem.get_attribute("href")
                except:
                    try:
                        # Fallback: finding any external link that is not the current domain might be risky, 
                        # so let's stick to specific structure if possible.
                        # Sometimes it is an icon.
                        pass
                    except:
                        pass

                # 5. Email (Search on website)
                email = ""
                if website:
                    try:
                        print(f"     🔎 Website ({website}) taranıyor...")
                        email_list = site_icerisinden_email_bul(website)
                        if email_list:
                             for mail in email_list:
                                if mail and "@" in mail:
                                    email = mail
                                    break
                    except:
                        pass

                # Add row
                tablo.append({
                    "Data Source/ExhibitionName": "Gitex Global",
                    "ExhibitionProductGroup": products,
                    "CompanyName": firma_adi,
                    "CompanyWebsite": website,
                    "CompanyMail": email,
                    "CompanyMail2": "",
                    "CompanyPhone": "",
                    "CompanyAddress": "", # Address logic not strictly defined, leaving empty or could assume Country is partial address
                    "CompanyZipCode": "",
                    "CompanyCity": "",
                    "CompanyCountry": country,
                    "CompanyBusinessType": ""
                })
                
                print(f"     ✅ {firma_adi} | {country}")

            except Exception as e:
                print(f"  ❌ Firma işlenirken hata: {e}")
                continue

    except Exception as e:
        print(f"❌ Genel Hata: {e}")
    
    finally:
        driver.quit()
        
    df = pd.DataFrame(tablo)
    
    # Streamlit Output
    if st:
        st.dataframe(df)
        
        # Excel
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Excel (.xlsx) İndir",
            data=excel_buffer,
            file_name=f"gitex_global_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # CSV
        csv_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_csv,
            file_name="gitex_global_export.csv",
            mime="text/csv"
        )
    
    return df

def scrape_mostra_convegno(scroll_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://www.mcexpocomfort.it/en-gb/exhibitor-directory.html#/"
    tablo = []

    try:
        print(f"🔄 Ana sayfa yükleniyor...")
        driver.get(base_url)
        time.sleep(5) # İlk yükleme için biraz bekle
        
        # Scroll işlemi
        print(f"📜 Sayfa {scroll_sayisi} kez scroll ediliyor...")
        for scroll_num in range(scroll_sayisi):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3) # Yükleme için bekle
            
            # Bazen yukarı aşağı yapmak tetikleyebilir
            driver.execute_script("window.scrollBy(0, -200);")
            time.sleep(1)
            
            print(f"  📜 {scroll_num + 1}/{scroll_sayisi} scroll tamamlandı")

        print(f"✅ Scroll işlemi tamamlandı. Firma kartları toplanıyor...")

        # Firma kartlarını bul
        # Selector: div.directory-item
        firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.directory-item")
        
        if not firma_kartlari:
            print(f"⚠️ Firma bulunamadı.")
        else:
            print(f"📊 {len(firma_kartlari)} firma bulundu")

            # Detay linklerini topla
            firma_linkleri = []
            for kart in firma_kartlari:
                try:
                    # Link genellikle h3 başlığında veya data-dtm*="exhibitorName" olan elementte
                    # main_table.html analizine göre: div[data-testid="name-control"] a
                    link_elem = kart.find_element(By.CSS_SELECTOR, "div[data-testid='name-control'] a")
                    link = link_elem.get_attribute("href")
                    if link:
                        firma_linkleri.append(link)
                except:
                    continue
            
            # Tekil linkleri al (set kullanarak)
            firma_linkleri = list(set(firma_linkleri))
            print(f"🔗 {len(firma_linkleri)} tekil firma linki toplandı")

            # Her firmanın detay sayfasına git
            for idx, detay_link in enumerate(firma_linkleri, 1):
                try:
                    print(f"  {idx}/{len(firma_linkleri)}. 🔍 Detay sayfası açılıyor...")
                    driver.get(detay_link)
                    time.sleep(3)

                    # Firma Adı
                    try:
                        firma_adi = driver.find_element(By.CSS_SELECTOR, "div.details-header h1.wrap-word").text.strip()
                    except:
                        firma_adi = ""
                    
                    # Ürün Grupları
                    urun_gruplari = ""
                    try:
                        tags = driver.find_elements(By.CSS_SELECTOR, "div.filter-tags span.tag-item")
                        urun_gruplari = ", ".join([tag.text.strip() for tag in tags if tag.text.strip()])
                    except:
                        urun_gruplari = ""

                    # İletişim Bilgileri (Container: div.exhibitor-details-contact-us-container)
                    website = ""
                    email = ""
                    telefon = ""
                    
                    try:
                        # Website
                        try:
                            web_elem = driver.find_element(By.CSS_SELECTOR, "a[data-dtm='exhibitorDetails_externalLink']")
                            website = web_elem.get_attribute("href")
                        except:
                            pass
                        
                        # Email
                        try:
                            mail_elem = driver.find_element(By.CSS_SELECTOR, "a[data-dtm='exhibitorDetails_emailLink']")
                            href = mail_elem.get_attribute("href")
                            if href and "mailto:" in href:
                                email = href.replace("mailto:", "").split("?")[0].strip()
                        except:
                            pass
                            
                        # Telefon
                        try:
                            tel_elem = driver.find_element(By.CSS_SELECTOR, "a[data-dtm='exhibitorDetails_phoneLink']")
                            href = tel_elem.get_attribute("href")
                            if href and "tel:" in href:
                                telefon = href.replace("tel:", "").strip()
                        except:
                            pass
                    except:
                        pass

                    # Email advanced search (eğer sayfada yoksa)
                    if not email and website:
                        try:
                            print(f"     🔎 Website'den email aranıyor...")
                            email_list = site_icerisinden_email_bul(website)
                            if email_list and len(email_list) > 0:
                                for mail in email_list:
                                    if mail and "@" in mail:
                                        email = mail
                                        break
                        except:
                            pass

                    # Adres ve Ülke
                    adres = ""
                    ulke = ""
                    sehir = "" # Genelde adres içinde, ayrıştırmak zor olabilir
                    
                    try:
                        adres_div = driver.find_element(By.ID, "exhibitor_details_address")
                        p_elem = adres_div.find_element(By.TAG_NAME, "p")
                        spans = p_elem.find_elements(By.TAG_NAME, "span")
                        
                        adres_parts = [s.text.strip() for s in spans if s.text.strip()]
                        adres = " ".join(adres_parts)
                        
                        if adres_parts:
                            # Son parça genellikle ülkedir
                            ulke = adres_parts[-1]
                    except:
                        adres = ""

                    print(f"  ✅ {firma_adi}")

                    tablo.append({
                        "Data Source/E_Exhibition": "Mostra Convegno Expocomfort",
                        "ExhibitionProductGroup": urun_gruplari,
                        "CompanyName": firma_adi,
                        "CompanyWebsite": website,
                        "CompanyMail": email,
                        "CompanyMail2": "",
                        "CompanyPhone": telefon,
                        "CompanyAddress": adres,
                        "CompanyZipCode": "",
                        "CompanyCity": sehir,
                        "CompanyCountry": ulke,
                        "CompanyBusinessType": "",
                        "Detay Link": detay_link
                    })

                except Exception as e:
                    print(f"  ❌ Firma detayı işlenirken hata: {e}")
                    continue

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # !!! TÜM YENİ FONKSİYONLAR BU BLOĞU İÇERMELİ !!!
    if st:
        st.dataframe(df)
        
        # 📥 Excel İndir
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Excel (.xlsx) İndir",
            data=excel_buffer,
            file_name=f"{st.session_state.get('function_name', 'mostra_convegno')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'mostra_convegno')}.csv",
            mime="text/csv"
        )

        # Grafikler
        if not df.empty:
            try:
                if "CompanyCountry" in df.columns:
                    ulke_sayilari = df["CompanyCountry"].value_counts().reset_index()
                    ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
                    fig = px.bar(ulke_sayilari.head(20), x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Grafik çizilirken hata: {e}")
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty and "CompanyCountry" in df.columns:
            print(df["CompanyCountry"].value_counts())

    return df

def scrape_lopec(iterations):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    base_url = "https://exhibitors.lopec.com/industry-directory/2026/list-of-exhibitors/"
    tablo = []

    try:
        print(f"🔄 LOPEC 2026 ana sayfa yükleniyor...")
        driver.get(base_url)
        time.sleep(3)
        handle_cookie_consent_final(driver)

        for i in range(iterations):
            print(f"\n🔄 Sayfa {i+1} işleniyor...")
            try:
                exhibitor_elements = driver.find_elements(By.CSS_SELECTOR, ".pb_ce .ce_cntnt h2 a")
                exhibitor_links = [el.get_attribute("href") for el in exhibitor_elements if el.get_attribute("href")]

                for link in exhibitor_links:
                    try:
                        driver.execute_script("window.open(arguments[0], '_blank');", link)
                        driver.switch_to.window(driver.window_handles[1])
                        time.sleep(2)

                        company_name = driver.find_element(By.CSS_SELECTOR, ".ce_cntct .ce_head").text.strip() if driver.find_elements(By.CSS_SELECTOR, ".ce_cntct .ce_head") else ""
                        address = driver.find_element(By.CSS_SELECTOR, ".ce_addr").text.strip() if driver.find_elements(By.CSS_SELECTOR, ".ce_addr") else ""
                        phone = driver.find_element(By.CSS_SELECTOR, '.ce_phone a[href^="tel:"]').text.strip() if driver.find_elements(By.CSS_SELECTOR, '.ce_phone a[href^="tel:"]') else ""
                        email = driver.find_element(By.CSS_SELECTOR, '.ce_email a[href^="mailto:"]').text.strip() if driver.find_elements(By.CSS_SELECTOR, '.ce_email a[href^="mailto:"]') else ""
                        website = driver.find_element(By.CSS_SELECTOR, ".ce_website a.vam").get_attribute("href") if driver.find_elements(By.CSS_SELECTOR, ".ce_website a.vam") else ""

                        if not email and website:
                            email = find_email_advanced(driver, website)

                        tablo.append({
                            "Data Source/ExhibitionName": "LOPEC 2026",
                            "ExhibitionProductGroup": "Exhibitor",
                            "CompanyName": company_name,
                            "CompanyWebsite": website,
                            "CompanyMail": email,
                            "CompanyPhone": phone,
                            "CompanyAddress": address,
                            "CompanyMail2": "", "CompanyZipCode": "", "CompanyCity": "", "CompanyCountry": "", "CompanyBusinessType": ""
                        })
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                    except Exception:
                        if len(driver.window_handles) > 1:
                            driver.close()
                            driver.switch_to.window(driver.window_handles[0])

                if i < iterations - 1:
                    try:
                        next_btn = driver.find_element(By.CSS_SELECTOR, 'input[name="SRField_next"]')
                        next_btn.click()
                        time.sleep(4)
                    except Exception:
                        break
            except Exception:
                break
    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    if st.session_state.get('function_name') == 'lopec_2026':
        st.write(f"### Scanned Data ({len(df)} companies)")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "lopec_2026_exhibitors.csv", "text/csv")
    return df


def scrape_wam_morocco(scroll_count):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    base_url = "https://exhibitors.wammorocco.com/wam-morocco-2026/Exhibitor"
    tablo = []

    try:
        driver.get(base_url)
        time.sleep(5)
        # Cookie consent removed as requested

        # Infinite Scroll
        last_height = driver.execute_script("return document.body.scrollHeight")
        for i in range(scroll_count):
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(3)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Extract exhibitor links - Updated Selectors
        links = []
        view_profile_links = driver.find_elements(By.XPATH, "//a[contains(@href, 'ExbDetails')]")
        for link_elem in view_profile_links:
            url_attr = link_elem.get_attribute("href")
            if url_attr and url_attr not in links:
                links.append(url_attr)

        if not links:
            st.warning("No exhibitor links found. Please check the selectors or scroll count.")
            return pd.DataFrame()

        st.info(f"Found {len(links)} exhibitors. Starting detailed extraction...")

        for link in links:
            try:
                driver.get(link)
                time.sleep(2)

                # Name is in h4
                company_name = driver.find_element(By.TAG_NAME, "h4").text.strip() if driver.find_elements(By.TAG_NAME, "h4") else ""

                stand_no = ""
                country = ""
                product_groups = ""
                website = ""

                # Extracting details from the page
                try:
                    all_text = driver.find_element(By.TAG_NAME, "body").text
                    if "Stand No" in all_text:
                        # Extract stand no from text
                        lines = all_text.split("\n")
                        for idx, line in enumerate(lines):
                            if "Stand No" in line:
                                stand_no = line.strip()
                                # Country is often the next line
                                if idx + 1 < len(lines):
                                    country = lines[idx+1].strip()
                                break
                except: pass

                # Product groups - Updated with specific sector_block selector
                try:
                    pg_elements = driver.find_elements(By.CSS_SELECTOR, "ul.sector_block li")
                    product_groups = ", ".join([el.text.strip() for el in pg_elements if el.text.strip()])

                    if not product_groups:
                        pg_elements = driver.find_elements(By.CSS_SELECTOR, ".company_description ul li")
                        product_groups = ", ".join([el.text.strip() for el in pg_elements if el.text.strip() and "VISIT" not in el.text.upper()])
                except:
                    product_groups = ""

                # Website - look for "VISIT WEBSITE"
                try:
                    ws_element = driver.find_element(By.XPATH, "//a[contains(text(), 'VISIT WEBSITE')]")
                    website = ws_element.get_attribute("href")
                except: pass

                email = ""
                if website:
                    email = find_email_advanced(driver, website)

                tablo.append({
                    "Data Source/ExhibitionName": "WAM Morocco 2026",
                    "ExhibitionProductGroup": product_groups,
                    "CompanyName": company_name,
                    "CompanyWebsite": website,
                    "CompanyMail": email,
                    "CompanyPhone": "",
                    "CompanyAddress": stand_no,
                    "CompanyMail2": "",
                    "CompanyZipCode": "",
                    "CompanyCity": "",
                    "CompanyCountry": country,
                    "CompanyBusinessType": ""
                })
            except Exception as e:
                print(f"Error processing {link}: {e}")
                continue

    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    if st.session_state.get('function_name') == 'wam_morocco':
        st.write(f"### Scanned Data ({len(df)} companies)")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "wam_morocco_2026_exhibitors.csv", "text/csv")
    return df

def scrape_chillventa(iterations):
    options = Options()
    options.add_argument("--headless") # Headless modunu kapattık, sorunları görmek için.
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 15)

    base_url = "https://www.chillventa.de/en/exhibitors-products/find-exhibitors"
    tablo = []

    try:
        st.info(f"Opening {base_url}...")
        driver.get(base_url)
        time.sleep(8) # Sayfanın tam yüklenmesi için biraz daha uzun bekle

        # Cookie Consent & Overlay Removal (Fast & Silent)
        driver.execute_script("""
            var ids = ['cmpbox', 'onetrust-consent-sdk', 'onetrust-banner-sdk'];
            ids.forEach(id => {
                var el = document.getElementById(id);
                if (el) el.remove();
            });
            var overlays = document.querySelectorAll('.cmpboxoverlay, .onetrust-pc-dark-filter');
            overlays.forEach(ol => ol.remove());
            document.body.style.overflow = 'auto';
        """)
        time.sleep(2)

        # Iterations for 'Show more'
        for i in range(iterations):
            try:
                # Scroll to bottom to ensure button is in view
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)

                # Try finding the button using provided testid and text content
                try:
                    show_more_xpath = "//button[@data-testid='default-button' and (contains(., 'Show more') or contains(., 'show more'))]"
                    show_more = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, show_more_xpath)))

                    # Scroll and click via JS
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_more)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", show_more)

                    # Print to terminal instead of Streamlit
                    print(f"Clicked 'Show more' ({i+1}/{iterations})")
                    time.sleep(3) # Wait for content to load
                except:
                    # If button not found, maybe all items are loaded
                    break

            except Exception as e:
                break

        # Collect Links
        st.info("Collecting exhibitor links...")
        links = set()
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/en/exhibitors/"]')
            for el in elements:
                href = el.get_attribute("href")
                if href and "/exhibitors/" in href and "favorites" not in href:
                    links.add(href)
        except Exception as e:
            st.error(f"Error collecting links: {e}")

        link_list = list(links)
        st.success(f"Found {len(link_list)} exhibitors. Starting details extraction...")

        progress_bar = st.progress(0)

        for idx, link in enumerate(link_list):
            try:
                driver.get(link)
                time.sleep(1.0)

                name = ""
                address = ""
                phone = ""
                website = ""
                email = ""
                products = ""
                country = ""

                # Name
                try:
                    name = driver.find_element(By.TAG_NAME, "h1").text.strip()
                except:
                    try:
                        name = driver.find_element(By.TAG_NAME, "h2").text.strip()
                    except: pass

                # Address & Country
                try:
                    # Updated selector: p.text-copy-l
                    addr_elems = driver.find_elements(By.CSS_SELECTOR, "p.text-copy-l")
                    addr_lines = [el.text.strip() for el in addr_elems if el.text.strip()]
                    if addr_lines:
                        address = ", ".join(addr_lines)
                        country = addr_lines[-1]
                except: pass

                # Phone
                try:
                    # Updated selector: a[data-testid="company-details-contacts-phone"]
                    phone_el = driver.find_element(By.CSS_SELECTOR, 'a[data-testid="company-details-contacts-phone"]')
                    phone = phone_el.text.strip()
                    if not phone:
                        phone = phone_el.get_attribute("href").replace("tel:", "")
                except:
                    pass

                # Website
                try:
                    # Updated selector: a[data-testid="company-details-contacts-website"]
                    web_el = driver.find_element(By.CSS_SELECTOR, 'a[data-testid="company-details-contacts-website"]')
                    website = web_el.get_attribute("href")
                except:
                    # Fallback
                    try:
                        web_els = driver.find_elements(By.CSS_SELECTOR, 'a[href^="http"]')
                        for we in web_els:
                            w_href = we.get_attribute("href")
                            if "chillventa.de" not in w_href and "linkedin" not in w_href and "facebook" not in w_href and "twitter" not in w_href:
                                website = w_href
                                break
                    except: pass

                # Products
                try:
                    # Updated selector: span[data-testid="item-X"]
                    prod_elems = driver.find_elements(By.CSS_SELECTOR, 'span[data-testid^="item-"]')
                    products = ", ".join([p.text.strip() for p in prod_elems if p.text.strip()])
                except: pass

                # Email (Advanced Search)
                if website and website != "N/A":
                    try:
                        found_emails = site_icerisinden_email_bul(website)
                        if found_emails:
                            for mail in found_emails:
                                if "@" in mail:
                                    email = mail
                                    break
                    except: pass

                item = {
                    "Data Source/ExhibitionName": "Chillventa",
                    "ExhibitionProductGroup": products,
                    "CompanyName": name,
                    "CompanyWebsite": website,
                    "CompanyMail": email,
                    "CompanyMail2": "",
                    "CompanyPhone": phone,
                    "CompanyAddress": address,
                    "CompanyZipCode": "",
                    "CompanyCity": "",
                    "CompanyCountry": country,
                    "CompanyBusinessType": "",
                    "Detay Link": link
                }
                tablo.append(item)

            except Exception as e:
                # print(f"Error processing {link}: {e}")
                pass

            progress_bar.progress((idx + 1) / len(link_list))

    except Exception as e:
        st.error(f"Critical Error: {e}")

    finally:
        driver.quit()

    if tablo:
        df = pd.DataFrame(tablo)
        st.success(f"Extraction complete! {len(df)} companies found.")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='chillventa_exhibitors.csv',
            mime='text/csv',
        )
    else:
        st.warning("No data extracted.")


def scrape_euroshop_2026(sayfa_sayisi):
    search_url = "https://www.euroshop-tradefair.com/vis-api/vis/v3/en/search"
    detail_url_template = "https://www.euroshop-tradefair.com/vis-api/vis/v1/en/exhibitors/{}/json"
    public_profile_url = "https://www.euroshop-tradefair.com/vis/v1/en/exhprofiles/{}"
    rows_per_page = 30
    tablo = []

    headers = {
        "x-vis-domain": "https://www.euroshop-tradefair.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    max_retry_wait = 60

    def request_with_retry(session, url, request_label, params=None, timeout=45, max_retries=7, base_wait=2):
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(url, headers=headers, params=params, timeout=timeout)
                status_code = response.status_code

                if status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_seconds = min(max_retry_wait, base_wait * (2 ** (attempt - 1)))
                    if retry_after:
                        try:
                            wait_seconds = max(wait_seconds, int(float(retry_after)))
                        except (TypeError, ValueError):
                            pass
                    print(
                        f"{request_label} rate limited (429). "
                        f"Retry {attempt}/{max_retries} after {wait_seconds}s"
                    )
                    time.sleep(wait_seconds)
                    continue

                if 500 <= status_code < 600 and attempt < max_retries:
                    wait_seconds = min(max_retry_wait, base_wait * (2 ** (attempt - 1)))
                    print(
                        f"{request_label} server error ({status_code}). "
                        f"Retry {attempt}/{max_retries} after {wait_seconds}s"
                    )
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                return response

            except requests.RequestException as e:
                if attempt >= max_retries:
                    print(f"{request_label} failed after {max_retries} retries: {e}")
                    return None
                wait_seconds = min(max_retry_wait, base_wait * (2 ** (attempt - 1)))
                print(
                    f"{request_label} request error ({e}). "
                    f"Retry {attempt}/{max_retries} after {wait_seconds}s"
                )
                time.sleep(wait_seconds)

        return None

    def normalize_text(value):
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("E-mail:", "").replace("Email:", "").replace("Phone:", "").strip()
        return text

    def unique_join(values):
        seen = set()
        cleaned = []
        for val in values:
            norm = normalize_text(val)
            if not norm:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(norm)
        return ", ".join(cleaned)

    try:
        with requests.Session() as session:
            for page_num in range(1, sayfa_sayisi + 1):
                page_start = (page_num - 1) * rows_per_page
                print(f"\nProcessing page {page_num} (_start={page_start})")

                params = {
                    "_query": "",
                    "_start": page_start,
                    "_rows": rows_per_page,
                    "_sort": "score_desc",
                    "f_type": "profile",
                }

                try:
                    response = request_with_retry(
                        session,
                        search_url,
                        request_label=f"Page {page_num}",
                        params=params,
                        timeout=45,
                        max_retries=10,
                        base_wait=3,
                    )
                    if response is None:
                        print(f"Skipping page {page_num} after all retries.")
                        continue
                    search_data = response.json()
                except Exception as e:
                    print(f"Failed to fetch page {page_num}: {e}")
                    continue

                docs = search_data.get("docs", [])
                if not docs:
                    print(f"No results on page {page_num}.")
                    break

                print(f"Found {len(docs)} profiles")

                for idx, doc in enumerate(docs, 1):
                    exh_id = normalize_text(doc.get("exh"))
                    if not exh_id:
                        continue

                    try:
                        print(f"  {idx}/{len(docs)} Fetching detail: {exh_id}")

                        detail_url = detail_url_template.format(exh_id)
                        detail_response = request_with_retry(
                            session,
                            detail_url,
                            request_label=f"Detail {exh_id}",
                            timeout=45,
                            max_retries=7,
                            base_wait=2,
                        )
                        if detail_response is None:
                            print(f"  Skipping profile after retries ({exh_id})")
                            continue
                        detail_data = detail_response.json().get("result", {})

                        profile = detail_data.get("profile", {}) or {}
                        profile_address = profile.get("profileAddress", {}) or {}
                        links = profile.get("links", []) or []
                        categories = detail_data.get("categories", []) or []
                        business_data = profile.get("businessData", []) or []
                        contacts = detail_data.get("contacts", []) or []

                        company_name = normalize_text(profile.get("name")) or normalize_text(doc.get("exhName"))
                        website = ""
                        for link_item in links:
                            website = normalize_text(link_item.get("link"))
                            if website:
                                break

                        primary_email = normalize_text(profile.get("email"))
                        profile_phone = normalize_text((profile.get("phone") or {}).get("phone", ""))

                        contact_emails = []
                        contact_phones = []
                        for contact in contacts:
                            for field in contact.get("fields", []) or []:
                                field_id = normalize_text(field.get("id")).lower()
                                field_label = normalize_text(field.get("label")).lower()
                                values = [normalize_text(v) for v in (field.get("values") or [])]
                                values = [v for v in values if v]
                                if not values:
                                    continue

                                if field_id == "email" or "mail" in field_label:
                                    for val in values:
                                        if "@" in val and val.lower() not in [m.lower() for m in contact_emails]:
                                            contact_emails.append(val)

                                if field_id == "phone" or "phone" in field_label:
                                    for val in values:
                                        if val.lower() not in [p.lower() for p in contact_phones]:
                                            contact_phones.append(val)

                        email_candidates = []
                        for mail in [primary_email] + contact_emails:
                            if mail and mail.lower() not in [m.lower() for m in email_candidates]:
                                email_candidates.append(mail)

                        phone_candidates = []
                        for phone in [profile_phone] + contact_phones:
                            if phone and phone.lower() not in [p.lower() for p in phone_candidates]:
                                phone_candidates.append(phone)

                        address_lines = profile_address.get("address", [])
                        if isinstance(address_lines, str):
                            address_lines = [address_lines]

                        product_groups = unique_join([cat.get("label", "") for cat in categories])
                        business_types = []
                        for entry in business_data:
                            for value in entry.get("values", []) or []:
                                if isinstance(value, dict):
                                    business_types.append(value.get("label") or value.get("value") or "")
                                else:
                                    business_types.append(value)

                        exh_seo_id = normalize_text(detail_data.get("exhSeoId")) or normalize_text(doc.get("exhSeoId"))
                        detail_link = public_profile_url.format(exh_seo_id) if exh_seo_id else ""

                        tablo.append({
                            "Data Source/ExhibitionName": "EuroShop 2026",
                            "ExhibitionProductGroup": product_groups,
                            "CompanyName": company_name,
                            "CompanyWebsite": website,
                            "CompanyMail": email_candidates[0] if len(email_candidates) > 0 else "",
                            "CompanyMail2": email_candidates[1] if len(email_candidates) > 1 else "",
                            "CompanyPhone": phone_candidates[0] if len(phone_candidates) > 0 else "",
                            "CompanyAddress": unique_join(address_lines),
                            "CompanyZipCode": normalize_text(profile_address.get("zip", "")),
                            "CompanyCity": normalize_text(profile_address.get("city", "")),
                            "CompanyCountry": normalize_text(profile_address.get("country", "")) or normalize_text(profile_address.get("countryCode", "")),
                            "CompanyBusinessType": unique_join(business_types),
                            "Detay Link": detail_link,
                        })

                    except Exception as e:
                        print(f"  Error while processing profile ({exh_id}): {e}")
                        continue

    except Exception as e:
        print(f"EuroShop scraper error: {e}")

    required_columns = [
        "Data Source/ExhibitionName",
        "ExhibitionProductGroup",
        "CompanyName",
        "CompanyWebsite",
        "CompanyMail",
        "CompanyMail2",
        "CompanyPhone",
        "CompanyAddress",
        "CompanyZipCode",
        "CompanyCity",
        "CompanyCountry",
        "CompanyBusinessType",
    ]

    df = pd.DataFrame(tablo)
    for col in required_columns + ["Detay Link"]:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns + ["Detay Link"]]

    print(f"\nTotal companies scraped: {len(df)}")

    if st:
        st.dataframe(df)

        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="Download Excel (.xlsx)",
            data=excel_buffer,
            file_name=f"{st.session_state.get('function_name', 'euroshop_2026')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'euroshop_2026')}.csv",
            mime="text/csv"
        )
    else:
        print("\nStats (non-Streamlit mode):")
        print(f"Total companies: {len(df)}")

    return df


def scrape_perpa_firmalar(page_count):
    base_domain = "https://www.perpa.com"
    list_url = f"{base_domain}/perpa-firmalar"
    tablo = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    required_columns = [
        "Data Source/ExhibitionName",
        "ExhibitionProductGroup",
        "CompanyName",
        "CompanyWebsite",
        "CompanyMail",
        "CompanyMail2",
        "CompanyPhone",
        "CompanyAddress",
        "CompanyZipCode",
        "CompanyCity",
        "CompanyCountry",
        "CompanyBusinessType",
    ]

    def normalize_text(value):
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        text = html.unescape(html.unescape(text))
        return re.sub(r"\s+", " ", text).strip()

    def unique_join(values):
        seen = set()
        cleaned = []
        for value in values:
            value = normalize_text(value)
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(value)
        return ", ".join(cleaned)

    def parse_zip_city_country(address):
        zip_code = ""
        city = ""
        country = "Turkey"
        match = re.search(r"\b(\d{5})\b", address or "")
        if match:
            zip_code = match.group(1)
            tail = normalize_text((address or "")[match.end():])
            parts = [p for p in re.split(r"[,\s/]+", tail) if p]
            if parts:
                city = parts[-1].title()
        return zip_code, city, country

    def parse_list_card(card):
        company_anchor = card.select_one("a.lnk[href]")
        if not company_anchor:
            return None

        detail_href = normalize_text(company_anchor.get("href", ""))
        if not detail_href:
            return None

        company_name = normalize_text(company_anchor.get_text(" ", strip=True))
        detail_link = urljoin(base_domain, detail_href)
        business_type = ""
        list_address = ""

        for span in card.select("div.firm-txt span"):
            span_text = normalize_text(span.get_text(" ", strip=True))
            if not span_text:
                continue

            span_lower = span_text.lower()
            if "guncellenme" in span_lower or "güncellenme" in span_lower:
                continue
            if "sektor" in span_lower or "sektör" in span_lower:
                business_type = normalize_text(span_text.split(":", 1)[1] if ":" in span_text else span_text)
                continue

            if not list_address:
                list_address = span_text

        return {
            "name": company_name,
            "detail_link": detail_link,
            "business_type": business_type,
            "list_address": list_address,
        }

    try:
        with requests.Session() as session:
            session.headers.update(headers)

            for page_num in range(1, page_count + 1):
                page_url = list_url if page_num == 1 else f"{list_url}?Page={page_num}"
                print(f"Processing page {page_num}: {page_url}")

                try:
                    list_response = session.get(page_url, timeout=30)
                    list_response.raise_for_status()
                except Exception as e:
                    print(f"Failed to fetch list page {page_num}: {e}")
                    continue

                list_soup = BeautifulSoup(list_response.text, "html.parser")
                cards = list_soup.select("div.page-firm-list ul li")
                if not cards:
                    print(f"No company cards on page {page_num}.")
                    break

                print(f"Found {len(cards)} companies on page {page_num}")

                for index, card in enumerate(cards, 1):
                    parsed_card = parse_list_card(card)
                    if not parsed_card:
                        continue

                    company_name = parsed_card["name"]
                    detail_link = parsed_card["detail_link"]
                    business_type = parsed_card["business_type"]
                    list_address = parsed_card["list_address"]

                    print(f"  {index}/{len(cards)} -> {company_name}")

                    website = ""
                    email_values = []
                    phone_values = []
                    address = list_address
                    product_group = ""

                    try:
                        detail_response = session.get(detail_link, timeout=30)
                        detail_response.raise_for_status()
                        detail_soup = BeautifulSoup(detail_response.text, "html.parser")

                        address_el = detail_soup.select_one("div.cmpy-1-txt span")
                        if address_el:
                            address = normalize_text(address_el.get_text(" ", strip=True))

                        for info_block in detail_soup.select("div.cmpy-1-phone"):
                            label_el = info_block.select_one("span")
                            label = normalize_text(label_el.get_text(" ", strip=True)).lower() if label_el else ""
                            block_text = normalize_text(info_block.get_text(" ", strip=True))
                            link_el = info_block.select_one("a[href]")
                            href_value = normalize_text(link_el.get("href", "")) if link_el else ""

                            if "web" in label:
                                candidate_website = href_value or (
                                    normalize_text(link_el.get_text(" ", strip=True)) if link_el else ""
                                )
                                if candidate_website:
                                    website = candidate_website
                                continue

                            if "mail" in label:
                                candidate_mail = ""
                                if href_value.lower().startswith("mailto:"):
                                    candidate_mail = normalize_text(href_value.replace("mailto:", "", 1))
                                elif href_value:
                                    candidate_mail = href_value
                                elif link_el:
                                    candidate_mail = normalize_text(link_el.get_text(" ", strip=True))

                                if not candidate_mail:
                                    mail_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", block_text)
                                    candidate_mail = normalize_text(mail_match.group(0)) if mail_match else ""

                                if candidate_mail and "@" in candidate_mail:
                                    email_values.append(candidate_mail)
                                continue

                            if "tel" in label or "gsm" in label or "fax" in label:
                                phone_value = ""
                                span_values = info_block.select("span")
                                if len(span_values) > 1:
                                    phone_value = normalize_text(span_values[-1].get_text(" ", strip=True))
                                if not phone_value and ":" in block_text:
                                    phone_value = normalize_text(block_text.split(":", 1)[1])
                                if phone_value:
                                    phone_values.append(phone_value)

                        if not website:
                            website_el = detail_soup.select_one("div.cmpy-1-phone a[href^='http']")
                            if website_el:
                                website = normalize_text(website_el.get("href", ""))

                        if not email_values:
                            mail_el = detail_soup.select_one("a[href^='mailto:']")
                            if mail_el:
                                mail_href = normalize_text(mail_el.get("href", ""))
                                if mail_href.lower().startswith("mailto:"):
                                    email_values.append(normalize_text(mail_href.replace("mailto:", "", 1)))

                        product_list = [
                            normalize_text(li.get_text(" ", strip=True))
                            for li in detail_soup.select("div.company-2-inner li")
                        ]
                        product_group = unique_join(product_list)

                        if not product_group:
                            product_wrapper = detail_soup.select_one("div.company-2-inner")
                            if product_wrapper:
                                product_group = normalize_text(product_wrapper.get_text(" ", strip=True))

                    except Exception as e:
                        print(f"Detail parse failed for {detail_link}: {e}")

                    if website and not email_values:
                        try:
                            extra_mails = site_icerisinden_email_bul(website)
                            if extra_mails:
                                for extra_mail in extra_mails:
                                    cleaned_mail = normalize_text(extra_mail)
                                    if cleaned_mail and "@" in cleaned_mail:
                                        email_values.append(cleaned_mail)
                        except Exception:
                            pass

                    email_values = [mail for mail in email_values if "@" in mail]
                    unique_emails = []
                    for mail in email_values:
                        if mail.lower() not in [m.lower() for m in unique_emails]:
                            unique_emails.append(mail)

                    phone_text = unique_join(phone_values)
                    address = normalize_text(address)
                    zip_code, city, country = parse_zip_city_country(address)

                    tablo.append({
                        "Data Source/ExhibitionName": "Perpa Firmalar",
                        "ExhibitionProductGroup": product_group,
                        "CompanyName": company_name,
                        "CompanyWebsite": website,
                        "CompanyMail": unique_emails[0] if len(unique_emails) > 0 else "",
                        "CompanyMail2": unique_emails[1] if len(unique_emails) > 1 else "",
                        "CompanyPhone": phone_text,
                        "CompanyAddress": address,
                        "CompanyZipCode": zip_code,
                        "CompanyCity": city,
                        "CompanyCountry": country,
                        "CompanyBusinessType": business_type,
                        "Detay Link": detail_link,
                    })

    except Exception as e:
        print(f"Perpa scraper error: {e}")

    df = pd.DataFrame(tablo)
    for col in required_columns + ["Detay Link"]:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns + ["Detay Link"]]

    print(f"Total companies scraped: {len(df)}")

    if st:
        st.dataframe(df)

        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="Download Excel (.xlsx)",
            data=excel_buffer,
            file_name=f"{st.session_state.get('function_name', 'perpa_firmalar')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'perpa_firmalar')}.csv",
            mime="text/csv"
        )
    else:
        print(f"Total companies: {len(df)}")

    return df


def scrape_embedded_world(page_count):
    base_domain = "https://www.embedded-world.de"
    algolia_url = "https://4EB6G0V1NT-dsn.algolia.net/1/indexes/prod_website_companies_en/query"
    tablo = []

    required_columns = [
        "Data Source/ExhibitionName",
        "ExhibitionProductGroup",
        "CompanyName",
        "CompanyWebsite",
        "CompanyMail",
        "CompanyMail2",
        "CompanyPhone",
        "CompanyAddress",
        "CompanyZipCode",
        "CompanyCity",
        "CompanyCountry",
        "CompanyBusinessType",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    algolia_headers = {
        "X-Algolia-API-Key": "f0416e3d1b38ae3aa789c8750e12bfe5",
        "X-Algolia-Application-Id": "4EB6G0V1NT",
        "Content-Type": "application/json",
    }

    def normalize_text(value):
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return re.sub(r"\s+", " ", html.unescape(text)).strip()

    def unique_join(values):
        seen = set()
        output = []
        for value in values:
            cleaned = normalize_text(value)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(cleaned)
        return ", ".join(output)

    def decode_mailto_value(value):
        mail_value = normalize_text(value)
        if not mail_value:
            return ""
        if "@" in mail_value:
            return mail_value

        try:
            padded = mail_value + ("=" * (-len(mail_value) % 4))
            decoded = base64.b64decode(padded).decode("utf-8", errors="ignore")
            decoded = normalize_text(decoded)
            if "@" in decoded:
                return decoded
        except Exception:
            pass
        return ""

    def extract_detail_company_data(detail_html):
        soup = BeautifulSoup(detail_html, "html.parser")
        company_data = {}

        next_data_script = soup.select_one("script#__NEXT_DATA__")
        if next_data_script and next_data_script.string:
            try:
                next_data_json = json.loads(next_data_script.string)
                company_data = (
                    next_data_json.get("props", {})
                    .get("pageProps", {})
                    .get("companyData", {})
                    or {}
                )
            except Exception:
                company_data = {}

        return soup, company_data

    def extract_zip_city_from_line(line):
        line_text = normalize_text(line)
        if not line_text:
            return "", ""

        match = re.match(r"^([A-Za-z0-9\-]{3,12})\s+(.+)$", line_text)
        if not match:
            return "", ""

        return normalize_text(match.group(1)), normalize_text(match.group(2))

    seen_links = set()

    try:
        with requests.Session() as session:
            session.headers.update(headers)

            max_pages = max(1, int(page_count))

            for page_idx in range(max_pages):
                params = (
                    f"query=&hitsPerPage=100&page={page_idx}"
                    "&filters=site:embwld AND isExhibitor:Yes"
                )

                try:
                    response = session.post(
                        algolia_url,
                        headers=algolia_headers,
                        json={"params": params},
                        timeout=45,
                    )
                    response.raise_for_status()
                    search_data = response.json()
                except Exception as e:
                    print(f"Algolia page request failed ({page_idx + 1}): {e}")
                    continue

                if page_idx == 0:
                    print(
                        f"Embedded World total companies: {search_data.get('nbHits', 0)} "
                        f"(Algolia pages: {search_data.get('nbPages', 0)})"
                    )

                hits = search_data.get("hits", []) or []
                if not hits:
                    print(f"No records on Algolia page {page_idx + 1}, stopping.")
                    break

                print(f"Processing Algolia page {page_idx + 1}: {len(hits)} companies")

                for hit_index, hit in enumerate(hits, 1):
                    detail_path = normalize_text(hit.get("url"))
                    if not detail_path:
                        continue

                    detail_link = urljoin(base_domain, detail_path)
                    if detail_link in seen_links:
                        continue
                    seen_links.add(detail_link)

                    company_name = normalize_text(hit.get("companyName"))
                    website = ""
                    company_mail = normalize_text(hit.get("email"))
                    company_mail2 = ""
                    phone = ""
                    address = normalize_text(hit.get("streetno"))
                    zip_code = normalize_text(hit.get("postcode"))
                    city = normalize_text(hit.get("city"))
                    country = normalize_text(hit.get("country"))
                    business_type = normalize_text(hit.get("companyType"))
                    product_group = unique_join(hit.get("keyword", []) or [])

                    if not product_group:
                        product_group = unique_join(hit.get("products", []) or [])

                    try:
                        detail_response = session.get(detail_link, timeout=45)
                        detail_response.raise_for_status()
                        detail_soup, company_data = extract_detail_company_data(detail_response.text)

                        company_name = (
                            normalize_text(company_data.get("displayname_company"))
                            or normalize_text(company_data.get("companyprofilename"))
                            or company_name
                        )

                        website = normalize_text(company_data.get("url")) or website
                        company_mail = normalize_text(company_data.get("email")) or company_mail
                        phone = normalize_text(company_data.get("telephonenumber")) or phone
                        address = normalize_text(company_data.get("streetno")) or address
                        zip_code = normalize_text(company_data.get("postcode")) or zip_code
                        city = normalize_text(company_data.get("city")) or city
                        country = normalize_text(company_data.get("country")) or country
                        business_type = normalize_text(company_data.get("companytype")) or business_type

                        keyword_values = []
                        for item in company_data.get("keywords", []) or []:
                            if isinstance(item, dict):
                                keyword_values.append(item.get("keyword", ""))
                            else:
                                keyword_values.append(item)
                        if keyword_values:
                            product_group = unique_join(keyword_values)

                        if not product_group:
                            def_values = []
                            for nomenclature in company_data.get("nomenclatures", []) or []:
                                if not isinstance(nomenclature, dict):
                                    continue
                                nomenclature_type = normalize_text(
                                    nomenclature.get("nomenclaturetype")
                                ).upper()
                                if nomenclature_type == "DEF":
                                    def_values.append(
                                        nomenclature.get("nomenclaturedisplay")
                                        or nomenclature.get("nomenclatures")
                                    )
                            product_group = unique_join(def_values)

                        website_el = detail_soup.select_one(
                            "[data-testid='company-details-contacts-website'][href]"
                        )
                        if website_el and not website:
                            website = normalize_text(website_el.get("href", ""))

                        phone_el = detail_soup.select_one(
                            "[data-testid='company-details-contacts-phone'][href]"
                        )
                        if phone_el and not phone:
                            phone_href = normalize_text(phone_el.get("href", ""))
                            if phone_href.lower().startswith("tel:"):
                                phone = normalize_text(phone_href.replace("tel:", "", 1))
                            if not phone:
                                phone = normalize_text(phone_el.get_text(" ", strip=True))

                        email_el = detail_soup.select_one(
                            "[data-testid='company-details-contacts-email'][href]"
                        )
                        if email_el and not company_mail:
                            email_href = normalize_text(email_el.get("href", ""))
                            if email_href.lower().startswith("mailto:"):
                                raw_email = normalize_text(email_href.replace("mailto:", "", 1))
                                company_mail = decode_mailto_value(raw_email)

                        address_lines = [
                            normalize_text(p.get_text(" ", strip=True))
                            for p in detail_soup.select(
                                "h4[data-testid='company-details-contacts-headline'] + div section p.text-copy-l"
                            )
                            if normalize_text(p.get_text(" ", strip=True))
                        ]

                        if address_lines:
                            if not address:
                                address = address_lines[0]
                            if len(address_lines) >= 2:
                                parsed_zip, parsed_city = extract_zip_city_from_line(address_lines[1])
                                if not zip_code:
                                    zip_code = parsed_zip
                                if not city:
                                    city = parsed_city
                            if not country and len(address_lines) >= 3:
                                country = address_lines[-1]

                        if not product_group:
                            we_offer_headline = detail_soup.select_one(
                                "h4[data-testid='company-keywords-1-headline']"
                            )
                            if we_offer_headline and "offer" in normalize_text(
                                we_offer_headline.get_text(" ", strip=True)
                            ).lower():
                                keyword_container = we_offer_headline.find_parent("div")
                                if keyword_container:
                                    keyword_spans = [
                                        normalize_text(span.get_text(" ", strip=True))
                                        for span in keyword_container.select("span.pure-tag")
                                    ]
                                    product_group = unique_join(keyword_spans)

                    except Exception as e:
                        print(f"Detail parse failed for {detail_link}: {e}")

                    print(
                        f"  {hit_index}/{len(hits)} -> {company_name or '[No Name]'} "
                        f"(page {page_idx + 1})"
                    )

                    tablo.append({
                        "Data Source/ExhibitionName": "Embedded World",
                        "ExhibitionProductGroup": product_group,
                        "CompanyName": company_name,
                        "CompanyWebsite": website,
                        "CompanyMail": company_mail,
                        "CompanyMail2": company_mail2,
                        "CompanyPhone": phone,
                        "CompanyAddress": address,
                        "CompanyZipCode": zip_code,
                        "CompanyCity": city,
                        "CompanyCountry": country,
                        "CompanyBusinessType": business_type,
                        "Detay Link": detail_link,
                    })

    except Exception as e:
        print(f"Embedded World scraper error: {e}")

    df = pd.DataFrame(tablo)
    for col in required_columns + ["Detay Link"]:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns + ["Detay Link"]]

    print(f"Total companies scraped: {len(df)}")

    if st:
        st.dataframe(df)

        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label="Download Excel (.xlsx)",
            data=excel_buffer,
            file_name=f"{st.session_state.get('function_name', 'embedded_world')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'embedded_world')}.csv",
            mime="text/csv"
        )
    else:
        print(f"Total companies: {len(df)}")

    return df


def scrape_global_industrie_exhibitors(load_more_count):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 20)

    base_url = "https://www.global-industrie.com/en/exhibitors-list"
    required_columns = [
        "Data Source/ExhibitionName",
        "ExhibitionProductGroup",
        "CompanyName",
        "CompanyWebsite",
        "CompanyMail",
        "CompanyMail2",
        "CompanyPhone",
        "CompanyAddress",
        "CompanyZipCode",
        "CompanyCity",
        "CompanyCountry",
        "CompanyBusinessType",
    ]

    def normalize_text(value):
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def pick_value_from_labels(label_map, candidates):
        for key in candidates:
            value = label_map.get(key, "")
            if value:
                return value
        return ""

    status_text = st.empty()
    progress_bar = st.progress(0)
    tablo = []
    email_cache = {}

    try:
        status_text.text("Opening exhibitor list...")
        driver.get(base_url)
        time.sleep(3)
        handle_cookie_consent_final(driver)

        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/en/exhibitor/']")))

        max_clicks = max(0, int(load_more_count))
        for click_idx in range(max_clicks):
            status_text.text(f"Clicking 'Load more' ({click_idx + 1}/{max_clicks})")
            progress_bar.progress((click_idx + 1) / max_clicks if max_clicks else 0)

            previous_count = len(driver.find_elements(By.CSS_SELECTOR, "a[href*='/en/exhibitor/']"))
            button_clicked = False

            for _ in range(3):
                try:
                    load_more_btn = None
                    candidate_buttons = driver.find_elements(By.CSS_SELECTOR, "div.sc-fmiMXH.ckTNWo button")
                    for candidate in candidate_buttons:
                        if candidate.is_displayed() and candidate.is_enabled():
                            load_more_btn = candidate
                            break

                    if load_more_btn is None:
                        load_more_btn = wait.until(
                            EC.element_to_be_clickable(
                                (By.XPATH, "//button[.//span[contains(normalize-space(), 'Load more')] or contains(normalize-space(), 'Load more')]")
                            )
                        )

                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_btn)
                    time.sleep(0.8)
                    driver.execute_script("arguments[0].click();", load_more_btn)

                    wait.until(
                        lambda d: len(d.find_elements(By.CSS_SELECTOR, "a[href*='/en/exhibitor/']")) > previous_count
                    )
                    time.sleep(1.2)
                    button_clicked = True
                    break
                except Exception:
                    time.sleep(1.2)

            if not button_clicked:
                print("Load more button not found/clickable anymore. Continuing with collected records.")
                break

        status_text.text("Collecting exhibitor links...")
        raw_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/en/exhibitor/']")
        unique_links = []
        seen_links = set()

        for element in raw_links:
            href = normalize_text(element.get_attribute("href"))
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(base_url, href)
            if "/en/exhibitor/" not in href:
                continue
            if href in seen_links:
                continue
            seen_links.add(href)
            unique_links.append(href)

        total_links = len(unique_links)
        st.info(f"Total {total_links} exhibitor detail links found.")

        progress_bar.progress(0)
        for idx, detail_url in enumerate(unique_links, 1):
            status_text.text(f"Scraping exhibitor {idx}/{total_links}")
            progress_bar.progress(idx / total_links if total_links else 1)

            company_name = ""
            website = ""
            company_mail = ""
            company_phone = ""
            company_address = ""
            company_zipcode = ""
            company_city = ""
            company_country = ""
            company_business_type = ""
            product_group = ""

            try:
                driver.get(detail_url)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1.hero__title, h1")))
                time.sleep(1.2)

                try:
                    company_name = normalize_text(driver.find_element(By.CSS_SELECTOR, "h1.hero__title, h1").text)
                except Exception:
                    company_name = ""

                soup = BeautifulSoup(driver.page_source, "html.parser")

                if not company_name:
                    h1 = soup.find("h1")
                    company_name = normalize_text(h1.get_text(" ", strip=True)) if h1 else ""

                website_candidates = []
                for anchor in soup.select("div.sc-kMribo.gYPcRV a[target='_blank'][href]"):
                    href = normalize_text(anchor.get("href"))
                    if href:
                        website_candidates.append(href)

                if not website_candidates:
                    for anchor in soup.select("a[target='_blank'][href]"):
                        href = normalize_text(anchor.get("href"))
                        if href:
                            website_candidates.append(href)

                for href in website_candidates:
                    href_lower = href.lower()
                    if not href_lower.startswith("http"):
                        continue
                    if "google.com/maps" in href_lower:
                        continue
                    if "global-industrie.com" in href_lower:
                        continue
                    website = href
                    break

                label_map = {}
                label_nodes = {}
                for block in soup.select("div.sc-cYeeTI.gCvAOK"):
                    label_el = block.select_one("div.sc-eEWMrZ.llANtp")
                    value_el = block.select_one("div.sc-cgCHwa.eALaUR")
                    if not label_el or not value_el:
                        continue

                    label = normalize_text(label_el.get_text(" ", strip=True)).rstrip(":").lower()
                    value = normalize_text(value_el.get_text(" ", strip=True))
                    if label and value:
                        label_map[label] = value
                        label_nodes[label] = value_el

                company_country = pick_value_from_labels(label_map, ["country"])
                company_phone = pick_value_from_labels(label_map, ["phone", "telephone", "mobile"])
                company_address = pick_value_from_labels(label_map, ["address", "adress", "street"])
                company_zipcode = pick_value_from_labels(label_map, ["zip code", "zipcode", "postal code", "postcode", "zip"])
                company_city = pick_value_from_labels(label_map, ["city", "town"])
                company_business_type = pick_value_from_labels(
                    label_map,
                    ["business type", "company type", "organization type", "organisation type", "type of business"]
                )

                activity_node = label_nodes.get("activity nomenclature")
                if activity_node is not None:
                    product_items = []
                    seen_items = set()
                    for node in activity_node.select("div.sc-cVOTOZ.VUOsa"):
                        item = normalize_text(node.get_text(" ", strip=True))
                        if not item or item.lower() == "activity":
                            continue
                        key = item.lower()
                        if key in seen_items:
                            continue
                        seen_items.add(key)
                        product_items.append(item)
                    if product_items:
                        product_group = ", ".join(product_items)

                if not product_group:
                    product_group = pick_value_from_labels(label_map, ["activity nomenclature"])

                if website:
                    if website in email_cache:
                        company_mail = email_cache[website]
                    else:
                        main_window = driver.current_window_handle
                        try:
                            driver.switch_to.new_window("tab")
                            company_mail = normalize_text(find_email_advanced(driver, website))
                        except Exception:
                            company_mail = ""
                        finally:
                            if len(driver.window_handles) > 1:
                                driver.close()
                                driver.switch_to.window(main_window)
                        email_cache[website] = company_mail

            except Exception as e:
                print(f"Global Industrie detail parse error ({detail_url}): {e}")

            tablo.append({
                "Data Source/ExhibitionName": "Global Industrie Exhibitors",
                "ExhibitionProductGroup": product_group,
                "CompanyName": company_name,
                "CompanyWebsite": website,
                "CompanyMail": company_mail,
                "CompanyMail2": "",
                "CompanyPhone": company_phone,
                "CompanyAddress": company_address,
                "CompanyZipCode": company_zipcode,
                "CompanyCity": company_city,
                "CompanyCountry": company_country,
                "CompanyBusinessType": company_business_type,
                "DetailUrl": detail_url,
            })

    except Exception as e:
        print(f"Global Industrie scraper error: {e}")
        st.error(str(e))
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        status_text.empty()
        progress_bar.empty()

    df = pd.DataFrame(tablo)
    for col in required_columns + ["DetailUrl"]:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns + ["DetailUrl"]]

    st.dataframe(df)

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)
    st.download_button(
        label="Download Excel (.xlsx)",
        data=excel_buffer,
        file_name=f"{st.session_state.get('function_name', 'global_industrie')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{st.session_state.get('function_name', 'global_industrie')}.csv",
        mime="text/csv",
    )

    return df


def scrape_industryweek_exhibitors():
    base_url = "https://industryweek.pl/en/exhibitors-catalog/"
    fallback_data_url = "https://industryweek.pl/wp-content/uploads/exhibitor-catalogs/pwe-exhibitors.json"
    required_columns = [
        "Data Source/ExhibitionName",
        "ExhibitionProductGroup",
        "CompanyName",
        "CompanyWebsite",
        "CompanyMail",
        "CompanyMail2",
        "CompanyPhone",
        "CompanyAddress",
        "CompanyZipCode",
        "CompanyCity",
        "CompanyCountry",
        "CompanyBusinessType",
    ]

    def normalize_text(value):
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def extract_items(value):
        if value is None:
            return []

        if isinstance(value, dict):
            collected = []
            for nested in value.values():
                collected.extend(extract_items(nested))
            return collected

        if isinstance(value, (list, tuple, set)):
            collected = []
            for nested in value:
                collected.extend(extract_items(nested))
            return collected

        text = normalize_text(value)
        if not text:
            return []

        pieces = [normalize_text(part) for part in re.split(r"[,\n;|]+", text)]
        return [part for part in pieces if part]

    def unique_join(values):
        unique_values = []
        seen = set()
        for raw in values:
            normalized = normalize_text(raw)
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique_values.append(normalized)
        return ", ".join(unique_values)

    tablo = []
    status_text = st.empty()
    progress_bar = st.progress(0)

    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            )
        })

        status_text.text("Loading exhibitor catalog page...")
        page_response = session.get(base_url, timeout=45)
        page_response.raise_for_status()

        soup = BeautifulSoup(page_response.text, "html.parser")
        data_url = fallback_data_url

        config_script = soup.select_one("script#vue-catalog-js-before")
        if config_script:
            config_match = re.search(
                r"window\.VUE_CATALOG_CONFIG\s*=\s*(\{.*?\});",
                config_script.get_text(" ", strip=True),
            )
            if config_match:
                try:
                    config_data = json.loads(html.unescape(config_match.group(1)))
                    configured_url = normalize_text(config_data.get("dataUrl"))
                    if configured_url:
                        data_url = configured_url
                except json.JSONDecodeError:
                    print("VUE_CATALOG_CONFIG could not be parsed. Falling back to default data URL.")

        status_text.text("Downloading exhibitor dataset...")
        data_response = session.get(data_url, timeout=90)
        data_response.raise_for_status()
        raw_payload = data_response.json()

        if isinstance(raw_payload, list):
            records = raw_payload
        elif isinstance(raw_payload, dict):
            records = [row for row in raw_payload.values() if isinstance(row, dict)]
        else:
            records = []

        total_records = len(records)
        st.info(f"Total {total_records} exhibitors found in Industry Week dataset.")

        for idx, item in enumerate(records, start=1):
            status_text.text(f"Processing exhibitor {idx}/{total_records}")
            progress_bar.progress(idx / total_records if total_records else 1)

            if not isinstance(item, dict):
                continue

            company_info = item.get("companyInfo") or {}
            exhibitor = item.get("exhibitor") or {}
            product_list = item.get("products") or []
            if not isinstance(product_list, list):
                product_list = []

            product_group_values = []
            product_group_values.extend(extract_items(company_info.get("industries")))
            product_group_values.extend(extract_items(company_info.get("catalogTags")))
            for product in product_list:
                if isinstance(product, dict):
                    product_group_values.extend(extract_items(product.get("name")))
                    product_group_values.extend(extract_items(product.get("tags")))
                    product_group_values.extend(extract_items(product.get("tabList")))
                else:
                    product_group_values.extend(extract_items(product))

            company_name = normalize_text(company_info.get("displayName")) or normalize_text(
                company_info.get("name")
            )
            company_website = normalize_text(company_info.get("website"))
            company_mail = normalize_text(company_info.get("contactEmail"))
            company_phone = normalize_text(company_info.get("contactPhone"))
            company_address = normalize_text(exhibitor.get("address"))
            company_zipcode = normalize_text(exhibitor.get("postalCode"))
            company_city = normalize_text(exhibitor.get("city"))
            company_country = normalize_text(exhibitor.get("country"))
            company_business_type = unique_join(extract_items(company_info.get("types")))

            tablo.append({
                "Data Source/ExhibitionName": "Warsaw Industry Week",
                "ExhibitionProductGroup": unique_join(product_group_values),
                "CompanyName": company_name,
                "CompanyWebsite": company_website,
                "CompanyMail": company_mail,
                "CompanyMail2": "",
                "CompanyPhone": company_phone,
                "CompanyAddress": company_address,
                "CompanyZipCode": company_zipcode,
                "CompanyCity": company_city,
                "CompanyCountry": company_country,
                "CompanyBusinessType": company_business_type,
            })

    except Exception as e:
        print(f"Industry Week scraper error: {e}")
        st.error(str(e))
    finally:
        status_text.empty()
        progress_bar.empty()

    df = pd.DataFrame(tablo)
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns]

    st.dataframe(df)

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)
    st.download_button(
        label="Download Excel (.xlsx)",
        data=excel_buffer,
        file_name=f"{st.session_state.get('function_name', 'industryweek_exhibitors')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{st.session_state.get('function_name', 'industryweek_exhibitors')}.csv",
        mime="text/csv",
    )

    return df


def scrape_electronica_2026_exhibitors(page_count, email_lookup_limit=30):
    base_url = "https://exhibitors.electronica.de/exhibitor-portal/2026/list-of-exhibitors/"
    required_columns = [
        "Data Source/ExhibitionName",
        "ExhibitionProductGroup",
        "CompanyName",
        "CompanyWebsite",
        "CompanyMail",
        "CompanyMail2",
        "CompanyPhone",
        "CompanyAddress",
        "CompanyZipCode",
        "CompanyCity",
        "CompanyCountry",
        "CompanyBusinessType",
    ]

    def normalize_text(value):
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def extract_first_valid_email(email_values):
        if not email_values:
            return ""
        for item in email_values:
            candidate = normalize_text(item)
            if "@" in candidate:
                return candidate
        return ""

    def parse_location(topic_value):
        raw_topic = normalize_text(topic_value)
        zip_code = ""
        city = ""
        country = ""
        address = raw_topic

        if not raw_topic:
            return address, zip_code, city, country

        left_part = raw_topic
        if "," in raw_topic:
            left_part, right_part = raw_topic.rsplit(",", 1)
            left_part = normalize_text(left_part)
            country = normalize_text(right_part)

        start_zip_pattern = re.compile(r"^\s*([A-Za-z]{0,2}\s*\d[\dA-Za-z\- ]{1,10})\s+(.+)$")
        end_zip_pattern = re.compile(r"^\s*(.+?)\s+([A-Za-z]{0,2}\s*\d[\dA-Za-z\- ]{1,10})\s*$")

        match_start = start_zip_pattern.match(left_part)
        match_end = end_zip_pattern.match(left_part)

        if match_start:
            zip_code = normalize_text(match_start.group(1))
            city = normalize_text(match_start.group(2))
        elif match_end:
            city = normalize_text(match_end.group(1))
            zip_code = normalize_text(match_end.group(2))
        else:
            city = normalize_text(left_part)

        return address, zip_code, city, country

    def extract_page_rows(soup):
        rows = []
        for card in soup.select("div.pb_ce div.ct_le"):
            company_name = ""
            company_address = ""
            company_zipcode = ""
            company_city = ""
            company_country = ""
            company_business_type = ""

            name_el = card.select_one("div.ce_head h2")
            topic_el = card.select_one("div.ce_topic")
            business_type_el = card.select_one("div.ce_exTy img[alt]")

            if name_el:
                company_name = normalize_text(name_el.get_text(" ", strip=True))
            if topic_el:
                parsed = parse_location(topic_el.get_text(" ", strip=True))
                company_address, company_zipcode, company_city, company_country = parsed
            if business_type_el:
                company_business_type = normalize_text(business_type_el.get("alt", ""))

            if not company_name:
                continue

            rows.append({
                "Data Source/ExhibitionName": "electronica 2026",
                "ExhibitionProductGroup": "",
                "CompanyName": company_name,
                "CompanyWebsite": "",
                "CompanyMail": "",
                "CompanyMail2": "",
                "CompanyPhone": "",
                "CompanyAddress": company_address,
                "CompanyZipCode": company_zipcode,
                "CompanyCity": company_city,
                "CompanyCountry": company_country,
                "CompanyBusinessType": company_business_type,
            })

        return rows

    def get_total_pages(soup):
        submit_el = soup.select_one("form#paging_1 input[name='SRField'][type='submit']")
        if submit_el:
            raw_value = normalize_text(submit_el.get("value", ""))
            if raw_value.isdigit():
                return int(raw_value)
        return None

    def build_next_payload(form):
        if form is None:
            return None

        next_button = form.select_one("input[name='SRField_next']")
        if not next_button or next_button.has_attr("disabled"):
            return None

        payload = {}
        for input_el in form.select("input[name]"):
            name = normalize_text(input_el.get("name", ""))
            if not name:
                continue

            input_type = normalize_text(input_el.get("type", "")).lower()
            if input_type == "image":
                continue
            if input_type == "submit" and name not in {"SRField_next"}:
                continue

            payload[name] = input_el.get("value", "")

        payload["SRField_next"] = "next"
        payload.pop("SRField", None)
        return payload

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    max_pages = max(1, int(page_count))
    max_email_lookup = max(0, int(email_lookup_limit))

    session = requests.Session()
    session.headers.update(headers)

    status_text = st.empty()
    progress_bar = st.progress(0)
    tablo = []
    website_cache = {}
    email_cache = {}

    try:
        status_text.text("Opening electronica 2026 exhibitor list...")
        response = session.get(base_url, timeout=45)
        response.raise_for_status()
        current_html = response.text
        current_url = response.url

        detected_total_pages = None
        for page_no in range(1, max_pages + 1):
            status_text.text(f"Parsing list page {page_no}/{max_pages}...")
            progress_bar.progress(page_no / max_pages)

            soup = BeautifulSoup(current_html, "html.parser")
            if detected_total_pages is None:
                detected_total_pages = get_total_pages(soup)
                if detected_total_pages:
                    st.info(f"Detected approximately {detected_total_pages} pages on the portal.")

            page_rows = extract_page_rows(soup)
            if not page_rows:
                print(f"No exhibitors found on page {page_no}; stopping.")
                break

            tablo.extend(page_rows)

            if page_no >= max_pages:
                break

            form = soup.select_one("form#paging_1") or soup.select_one("form[id^='paging_']")
            payload = build_next_payload(form)
            if not payload:
                print("Pagination next payload not available; stopping.")
                break

            action_url = urljoin(current_url, form.get("action", "") if form else "")
            next_response = session.post(action_url, data=payload, timeout=45)
            next_response.raise_for_status()
            current_html = next_response.text
            current_url = next_response.url or action_url

        if tablo:
            target_count = len(tablo) if max_email_lookup == 0 else min(len(tablo), max_email_lookup)
            for idx, row in enumerate(tablo[:target_count], start=1):
                status_text.text(f"Enriching website/email {idx}/{target_count}...")
                progress_bar.progress(idx / target_count if target_count else 1)

                company_name = normalize_text(row.get("CompanyName", ""))
                if not company_name:
                    continue

                website = website_cache.get(company_name, "")
                if not website:
                    try:
                        website = normalize_text(google_ilk_link_al(company_name))
                        if not website:
                            website = normalize_text(bing_ilk_link_al(company_name))
                    except Exception:
                        website = ""
                    website_cache[company_name] = website

                if website:
                    row["CompanyWebsite"] = website

                if website:
                    email = email_cache.get(website, "")
                    if not email:
                        try:
                            email_values = site_icerisinden_email_bul(website)
                            email = extract_first_valid_email(email_values)
                        except Exception:
                            email = ""
                        email_cache[website] = email
                    row["CompanyMail"] = email

    except Exception as e:
        print(f"electronica 2026 scraper error: {e}")
        st.error(str(e))
    finally:
        status_text.empty()
        progress_bar.empty()

    df = pd.DataFrame(tablo)
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns]

    st.dataframe(df)

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)
    st.download_button(
        label="Download Excel (.xlsx)",
        data=excel_buffer,
        file_name=f"{st.session_state.get('function_name', 'electronica_2026')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{st.session_state.get('function_name', 'electronica_2026')}.csv",
        mime="text/csv",
    )

    return df


def scrape_ifema_matelec(page_count, email_lookup_limit=30):
    catalogue_url = "https://www.ifema.es/en/matelec/exhibitors/catalogue?page=1"
    required_columns = [
        "Data Source/ExhibitionName",
        "ExhibitionProductGroup",
        "CompanyName",
        "CompanyWebsite",
        "CompanyMail",
        "CompanyMail2",
        "CompanyPhone",
        "CompanyAddress",
        "CompanyZipCode",
        "CompanyCity",
        "CompanyCountry",
        "CompanyBusinessType",
    ]

    def normalize_text(value):
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def normalize_url(value):
        url_value = normalize_text(value)
        if not url_value:
            return ""
        if url_value.startswith("//"):
            return f"https:{url_value}"
        if url_value.startswith(("http://", "https://")):
            return url_value
        return ""

    def strip_html_text(value):
        cleaned = normalize_text(value)
        if not cleaned:
            return ""
        return normalize_text(BeautifulSoup(html.unescape(cleaned), "html.parser").get_text(" ", strip=True))

    def unique_join(values):
        unique_values = []
        seen = set()
        for raw in values:
            text = normalize_text(raw)
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique_values.append(text)
        return ", ".join(unique_values)

    def extract_first_valid_email(values):
        if not values:
            return ""

        if isinstance(values, str):
            source_values = [values]
        else:
            source_values = values

        for raw in source_values:
            candidate = normalize_text(raw)
            if not candidate:
                continue
            emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", candidate)
            for email_value in emails:
                normalized_email = normalize_text(email_value)
                if normalized_email:
                    return normalized_email
        return ""

    def dynamic_field_values(field_obj):
        values = []
        if not isinstance(field_obj, dict):
            return values

        for item in field_obj.get("values") or []:
            if isinstance(item, dict):
                values.append(item.get("value"))
            else:
                values.append(item)

        raw_value = field_obj.get("value")
        if isinstance(raw_value, list):
            values.extend(raw_value)
        elif raw_value is not None:
            values.append(raw_value)

        normalized_values = []
        for value in values:
            if isinstance(value, dict):
                normalized_values.append(normalize_text(value.get("value") or value.get("name")))
            else:
                normalized_values.append(normalize_text(value))

        return [value for value in normalized_values if value]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    max_pages = max(1, int(page_count))
    max_email_lookup = max(0, int(email_lookup_limit))

    status_text = st.empty()
    progress_bar = st.progress(0)
    tablo = []
    website_cache = {}
    email_cache = {}
    email_lookup_attempts = 0

    session = requests.Session()
    session.headers.update(headers)

    try:
        status_text.text("Opening IFEMA MATELEC catalogue...")
        landing_response = session.get(catalogue_url, timeout=45)
        landing_response.raise_for_status()

        landing_soup = BeautifulSoup(landing_response.text, "html.parser")
        container = landing_soup.select_one("section.live-connect-exhibitors")
        if container is None:
            raise ValueError("Could not find live-connect-exhibitors section on catalogue page.")

        api_base_url = normalize_text(container.get("data-base-url"))
        if not api_base_url:
            raise ValueError("API base URL is missing in page configuration.")

        page_size_raw = normalize_text(container.get("data-page-size"))
        page_size = int(page_size_raw) if page_size_raw.isdigit() else 18
        max_records = max_pages * page_size

        status_text.text("Downloading exhibitor list from API...")
        search_payload = {
            "page": 0,
            "pageSize": 10000,
            "search": "",
            "dynamicFields": [],
            "countryIds": [],
        }
        search_response = session.post(
            f"{api_base_url}/exhibitors/search?language=en",
            json=search_payload,
            timeout=60,
        )
        search_response.raise_for_status()
        search_data = search_response.json().get("data") or []

        if not isinstance(search_data, list):
            search_data = []

        exhibitors = search_data[:max_records]
        st.info(
            f"API returned {len(search_data)} exhibitors. "
            f"Processing first {len(exhibitors)} records ({max_pages} page(s) x {page_size})."
        )

        total = len(exhibitors)
        for idx, list_item in enumerate(exhibitors, start=1):
            status_text.text(f"Processing exhibitor {idx}/{total}")
            progress_bar.progress(idx / total if total else 1)

            if not isinstance(list_item, dict):
                continue

            exhibitor_id = normalize_text(list_item.get("id"))
            detail_url = f"{api_base_url}/exhibitors/{exhibitor_id}?language=en" if exhibitor_id else ""

            detail_data = {}
            if detail_url:
                try:
                    detail_response = session.get(detail_url, timeout=45)
                    detail_response.raise_for_status()
                    parsed_detail = detail_response.json()
                    if isinstance(parsed_detail, dict):
                        detail_data = parsed_detail
                except Exception as detail_error:
                    print(f"IFEMA detail parse failed for {exhibitor_id}: {detail_error}")

            company_name = normalize_text(detail_data.get("name")) or normalize_text(list_item.get("name"))
            company_website = (
                normalize_url(detail_data.get("link"))
                or normalize_url(list_item.get("link"))
            )
            company_mail = extract_first_valid_email([
                detail_data.get("email"),
                list_item.get("email"),
            ])
            company_phone = ""
            company_address = ""
            company_zip_code = ""
            company_city = ""
            company_country = ""
            product_group_values = []
            business_type_values = []

            location = detail_data.get("location")
            if isinstance(location, dict):
                company_address = normalize_text(location.get("address"))
                company_zip_code = normalize_text(location.get("postalCode"))
                company_city = normalize_text(location.get("city"))
                company_country = normalize_text(location.get("countryCode"))

            categories = detail_data.get("categories") or []
            category_values = []
            for category in categories:
                if isinstance(category, dict):
                    category_values.append(category.get("name") or category.get("value"))
                else:
                    category_values.append(category)

            dynamic_fields = detail_data.get("dynamicFields") or []
            for field in dynamic_fields:
                if not isinstance(field, dict):
                    continue

                field_name = normalize_text(field.get("name"))
                field_name_lower = field_name.casefold()
                values = dynamic_field_values(field)
                if not values:
                    continue

                if any(token in field_name_lower for token in ("sector", "producto", "product", "category")):
                    product_group_values.extend(values)

                if any(token in field_name_lower for token in ("actividad", "activity", "business", "company")):
                    business_type_values.extend(values)

                if any(token in field_name_lower for token in ("phone", "tel", "telephone")) and not company_phone:
                    company_phone = unique_join(values)

                if any(token in field_name_lower for token in ("mail", "email")) and not company_mail:
                    company_mail = extract_first_valid_email(values)

                if any(token in field_name_lower for token in ("web", "website", "site", "url")) and not company_website:
                    for value in values:
                        candidate_website = normalize_url(value)
                        if candidate_website:
                            company_website = candidate_website
                            break

            product_group = unique_join(product_group_values) or unique_join(category_values)
            company_business_type = unique_join(business_type_values)

            if (not company_mail) and (max_email_lookup == 0 or email_lookup_attempts < max_email_lookup):
                email_lookup_attempts += 1

                if not company_website and company_name:
                    if company_name in website_cache:
                        company_website = website_cache[company_name]
                    else:
                        website_guess = ""
                        try:
                            website_guess = normalize_url(google_ilk_link_al(company_name))
                            if not website_guess:
                                website_guess = normalize_url(bing_ilk_link_al(company_name))
                        except Exception:
                            website_guess = ""
                        website_cache[company_name] = website_guess
                        company_website = website_guess

                if company_website:
                    if company_website in email_cache:
                        fallback_email = email_cache[company_website]
                    else:
                        fallback_email = ""
                        try:
                            email_values = site_icerisinden_email_bul(company_website)
                            fallback_email = extract_first_valid_email(email_values)
                        except Exception:
                            fallback_email = ""
                        email_cache[company_website] = fallback_email

                    if fallback_email:
                        company_mail = fallback_email

            tablo.append({
                "Data Source/ExhibitionName": "IFEMA MATELEC",
                "ExhibitionProductGroup": product_group,
                "CompanyName": company_name,
                "CompanyWebsite": company_website,
                "CompanyMail": company_mail,
                "CompanyMail2": "",
                "CompanyPhone": company_phone,
                "CompanyAddress": company_address,
                "CompanyZipCode": company_zip_code,
                "CompanyCity": company_city,
                "CompanyCountry": company_country,
                "CompanyBusinessType": company_business_type,
                "DetailUrl": detail_url,
                "CompanyDescription": strip_html_text(detail_data.get("description")),
            })

    except Exception as e:
        print(f"IFEMA MATELEC scraper error: {e}")
        st.error(str(e))
    finally:
        status_text.empty()
        progress_bar.empty()

    df = pd.DataFrame(tablo)
    for col in required_columns + ["DetailUrl", "CompanyDescription"]:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns + ["DetailUrl", "CompanyDescription"]]

    st.dataframe(df)

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)
    st.download_button(
        label="Download Excel (.xlsx)",
        data=excel_buffer,
        file_name=f"{st.session_state.get('function_name', 'ifema_matelec')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{st.session_state.get('function_name', 'ifema_matelec')}.csv",
        mime="text/csv",
    )

    return df


def scrape_maintenance_dortmund_exhibitors(page_count):
    base_url = "https://www.maintenance-dortmund.de/en/exhibitor/"
    required_columns = [
        "Data Source/ExhibitionName",
        "ExhibitionProductGroup",
        "CompanyName",
        "CompanyWebsite",
        "CompanyMail",
        "CompanyMail2",
        "CompanyPhone",
        "CompanyAddress",
        "CompanyZipCode",
        "CompanyCity",
        "CompanyCountry",
        "CompanyBusinessType",
    ]

    def normalize_text(value):
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def extract_stand_number(raw_text):
        text = normalize_text(raw_text)
        if not text:
            return ""
        match = re.search(r"stand\s*[:\-]?\s*([a-z0-9\-\/]+)", text, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(1)).upper()
        return ""

    def parse_total_pages(soup):
        pages = set()
        for anchor in soup.select("a[href]"):
            href = normalize_text(anchor.get("href"))
            page_data = normalize_text(anchor.get("data-page"))
            if page_data.isdigit():
                pages.add(int(page_data))

            match_path = re.search(r"/en/exhibitor/page/(\d+)/", href)
            if match_path:
                pages.add(int(match_path.group(1)))

            match_query = re.search(r"[?&]_paged=(\d+)", href)
            if match_query:
                pages.add(int(match_query.group(1)))

        for script in soup.select("script"):
            script_text = script.string or script.get_text(" ", strip=True)
            if not script_text:
                continue
            for match in re.finditer(r'"total_pages"\s*:\s*(\d+)', script_text):
                try:
                    pages.add(int(match.group(1)))
                except Exception:
                    continue

        return max(pages) if pages else 1

    def parse_product_groups_from_article(article):
        groups = []
        for cls in article.get("class", []):
            if not cls.startswith("exhibitor-category-"):
                continue
            group_slug = re.sub(r"-\d+$", "", cls.replace("exhibitor-category-", ""))
            group_name = normalize_text(group_slug.replace("-", " ").title())
            if group_name and group_name not in groups:
                groups.append(group_name)
        return ", ".join(groups)

    def parse_exhibitors_from_list_page(soup):
        exhibitors = []
        for article in soup.select("article[role='listitem']"):
            name_el = article.select_one("h3.elementor-post__title a, h3 a")
            if not name_el:
                continue

            company_name = normalize_text(name_el.get_text(" ", strip=True))
            detail_url = urljoin(base_url, normalize_text(name_el.get("href")))
            stand_number = extract_stand_number(
                article.select_one("div.elementor-post__badge").get_text(" ", strip=True)
                if article.select_one("div.elementor-post__badge")
                else ""
            )
            description = normalize_text(
                article.select_one("a.elementor-post__excerpt p").get_text(" ", strip=True)
                if article.select_one("a.elementor-post__excerpt p")
                else ""
            )
            product_groups = parse_product_groups_from_article(article)

            exhibitors.append({
                "name": company_name,
                "detail_url": detail_url,
                "stand_number": stand_number,
                "description": description,
                "product_groups": product_groups,
            })
        return exhibitors

    def parse_website_from_detail(detail_soup):
        blocked_domains = (
            "maintenance-dortmund.de",
            "easyfairs.com",
            "easyfairsassets.com",
            "linkedin.com",
            "facebook.com",
            "instagram.com",
            "youtube.com",
            "twitter.com",
            "x.com",
            "xing.com",
            "tiktok.com",
        )
        for anchor in detail_soup.select("div.elementor-widget-post-info a[href]"):
            href = normalize_text(anchor.get("href"))
            if not href.startswith(("http://", "https://")):
                continue
            href_lower = href.lower()
            if any(domain in href_lower for domain in blocked_domains):
                continue
            return href
        return ""

    def extract_first_valid_email(values):
        if not values:
            return ""

        if isinstance(values, str):
            source_values = [values]
        else:
            source_values = values

        for raw in source_values:
            candidate = normalize_text(raw)
            if not candidate:
                continue
            emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", candidate)
            for email_value in emails:
                normalized_email = normalize_text(email_value)
                if normalized_email:
                    return normalized_email
        return ""

    def normalize_website_url(raw_url):
        url_value = normalize_text(raw_url)
        if not url_value:
            return ""
        if url_value.startswith("//"):
            return f"https:{url_value}"
        if url_value.startswith(("http://", "https://")):
            return url_value
        return f"https://{url_value}"

    def extract_mailto_from_soup(soup):
        for anchor in soup.select("a[href^='mailto:']"):
            href = normalize_text(anchor.get("href"))
            candidate = href.replace("mailto:", "").split("?")[0].strip()
            if candidate and extract_first_valid_email(candidate):
                return candidate
        return ""

    def fetch_html_with_requests(target_url):
        try:
            response = session.get(target_url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            return response.text
        except Exception:
            return ""

    def collect_contact_links(soup, page_url, max_links=6):
        keywords = (
            "contact",
            "kontakt",
            "about",
            "imprint",
            "impressum",
            "privacy",
            "datenschutz",
            "company",
            "about-us",
        )
        parsed_page = urlparse(page_url)
        root_host = normalize_text(parsed_page.netloc).lower().replace("www.", "")
        links = []

        for anchor in soup.select("a[href]"):
            href = normalize_text(anchor.get("href"))
            if not href:
                continue
            if href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue

            text_value = normalize_text(anchor.get_text(" ", strip=True)).lower()
            href_value = href.lower()
            if not any(token in text_value or token in href_value for token in keywords):
                continue

            full_url = urljoin(page_url, href)
            parsed_full = urlparse(full_url)
            if parsed_full.scheme not in {"http", "https"}:
                continue

            link_host = normalize_text(parsed_full.netloc).lower().replace("www.", "")
            if root_host and link_host and root_host != link_host:
                continue

            if full_url not in links:
                links.append(full_url)
            if len(links) >= max_links:
                break

        return links

    def find_email_via_requests(website_url):
        normalized_url = normalize_website_url(website_url)
        if not normalized_url:
            return ""

        main_html = fetch_html_with_requests(normalized_url)
        if not main_html:
            return ""

        main_soup = BeautifulSoup(main_html, "html.parser")
        main_mailto = extract_mailto_from_soup(main_soup)
        if main_mailto:
            return main_mailto

        direct_email = extract_first_valid_email(extract_emails_from_source(main_html))
        if direct_email:
            return direct_email

        for contact_url in collect_contact_links(main_soup, normalized_url):
            contact_html = fetch_html_with_requests(contact_url)
            if not contact_html:
                continue

            contact_soup = BeautifulSoup(contact_html, "html.parser")
            contact_mailto = extract_mailto_from_soup(contact_soup)
            if contact_mailto:
                return contact_mailto

            contact_email = extract_first_valid_email(extract_emails_from_source(contact_html))
            if contact_email:
                return contact_email

        return ""

    def parse_location_from_detail(detail_soup):
        for ul in detail_soup.select("ul.elementor-icon-list-items"):
            text_value = normalize_text(ul.get_text(" ", strip=True))
            if not text_value:
                continue
            if "STAND" in text_value.upper():
                continue
            if ul.select_one("i.fa-map-marker-alt") is None:
                continue

            pieces = []
            for li in ul.select("li.elementor-icon-list-item"):
                piece = normalize_text(li.get_text(" ", strip=True)).strip(",")
                if piece:
                    pieces.append(piece)

            if len(pieces) >= 3:
                zip_code = normalize_text(pieces[0]).strip(",")
                city = normalize_text(pieces[1]).strip(",")
                country = normalize_text(pieces[2]).strip(",")
                return "", zip_code, city, country

        return "", "", "", ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    max_pages = max(1, int(page_count))
    session = requests.Session()
    session.headers.update(headers)

    status_text = st.empty()
    progress_bar = st.progress(0)
    tablo = []
    seen_detail_urls = set()
    email_cache = {}

    try:
        status_text.text("Opening maintenance Dortmund exhibitor list...")
        first_page_url = f"{base_url}?_paged=1"
        first_response = session.get(first_page_url, timeout=45)
        first_response.raise_for_status()
        first_soup = BeautifulSoup(first_response.text, "html.parser")

        detected_total_pages = parse_total_pages(first_soup)
        pages_to_scan = max_pages
        st.info(
            f"Detected approximately {detected_total_pages} pages on the portal. "
            f"Scanning requested {pages_to_scan} page(s) with '?_paged=' pagination."
        )

        for page_no in range(1, pages_to_scan + 1):
            progress_bar.progress(page_no / pages_to_scan if pages_to_scan else 1)
            status_text.text(f"Scanning list page {page_no}/{pages_to_scan}...")

            page_url = f"{base_url}?_paged={page_no}"
            page_response = first_response if page_no == 1 else session.get(page_url, timeout=45)
            if page_no != 1:
                page_response.raise_for_status()

            page_soup = first_soup if page_no == 1 else BeautifulSoup(page_response.text, "html.parser")
            exhibitors = parse_exhibitors_from_list_page(page_soup)
            if not exhibitors:
                print(f"No exhibitors found on page {page_no}; stopping.")
                break

            for idx, exhibitor in enumerate(exhibitors, start=1):
                detail_url = normalize_text(exhibitor.get("detail_url"))
                if not detail_url or detail_url in seen_detail_urls:
                    continue
                seen_detail_urls.add(detail_url)

                status_text.text(
                    f"Reading detail pages (page {page_no}/{pages_to_scan}, "
                    f"company {idx}/{len(exhibitors)})..."
                )

                company_name = normalize_text(exhibitor.get("name"))
                company_stand = normalize_text(exhibitor.get("stand_number"))
                company_description = normalize_text(exhibitor.get("description"))
                exhibition_product_group = normalize_text(exhibitor.get("product_groups"))

                company_website = ""
                company_mail = ""
                company_zip_code = ""
                company_city = ""
                company_country = ""
                company_address = ""

                try:
                    detail_response = session.get(detail_url, timeout=45)
                    detail_response.raise_for_status()
                    detail_soup = BeautifulSoup(detail_response.text, "html.parser")

                    detail_h1 = detail_soup.select_one("h1.elementor-heading-title, h1")
                    if detail_h1:
                        company_name = normalize_text(detail_h1.get_text(" ", strip=True)) or company_name

                    company_website = parse_website_from_detail(detail_soup)
                    company_address, company_zip_code, company_city, company_country = parse_location_from_detail(detail_soup)

                    if company_website:
                        if company_website in email_cache:
                            company_mail = email_cache[company_website]
                        else:
                            status_text.text(
                                f"Searching email on website (page {page_no}/{pages_to_scan}, "
                                f"company {idx}/{len(exhibitors)})..."
                            )
                            try:
                                company_mail = find_email_via_requests(company_website)
                            except Exception:
                                company_mail = ""
                            email_cache[company_website] = company_mail
                except Exception as detail_error:
                    print(f"Detail parsing failed for {detail_url}: {detail_error}")

                tablo.append({
                    "Data Source/ExhibitionName": "maintenance Dortmund",
                    "ExhibitionProductGroup": exhibition_product_group,
                    "CompanyName": company_name,
                    "CompanyWebsite": company_website,
                    "CompanyMail": company_mail,
                    "CompanyMail2": "",
                    "CompanyPhone": "",
                    "CompanyAddress": company_address,
                    "CompanyZipCode": company_zip_code,
                    "CompanyCity": company_city,
                    "CompanyCountry": company_country,
                    "CompanyBusinessType": "",
                    "StandNumber": company_stand,
                    "DetailUrl": detail_url,
                    "CompanyDescription": company_description,
                })

    except Exception as e:
        print(f"maintenance Dortmund scraper error: {e}")
        st.error(str(e))
    finally:
        status_text.empty()
        progress_bar.empty()

    df = pd.DataFrame(tablo)
    extra_columns = ["StandNumber", "DetailUrl", "CompanyDescription"]
    for col in required_columns + extra_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns + extra_columns]

    st.dataframe(df)

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)
    st.download_button(
        label="Download Excel (.xlsx)",
        data=excel_buffer,
        file_name=f"{st.session_state.get('function_name', 'maintenance_dortmund')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{st.session_state.get('function_name', 'maintenance_dortmund')}.csv",
        mime="text/csv",
    )

    return df


def scrape_intralogistik_dortmund_exhibitors(page_count):
    base_url = "https://www.dortmund.intralogistik-messen.de/exhibitors/"
    list_api_url = "https://my.easyfairs.com/widgets/api/stands/?language=en"
    detail_api_template = "https://my.easyfairs.com/widgets/api/stands/detail/{stand_id}/?language=en"
    required_columns = [
        "Data Source/ExhibitionName",
        "ExhibitionProductGroup",
        "CompanyName",
        "CompanyWebsite",
        "CompanyMail",
        "CompanyMail2",
        "CompanyPhone",
        "CompanyAddress",
        "CompanyZipCode",
        "CompanyCity",
        "CompanyCountry",
        "CompanyBusinessType",
    ]

    def normalize_text(value):
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def normalize_website_url(raw_url):
        url_value = normalize_text(raw_url)
        if not url_value:
            return ""
        if url_value.startswith("//"):
            return f"https:{url_value}"
        if url_value.startswith(("http://", "https://")):
            return url_value
        return f"https://{url_value}"

    def slugify_text(value):
        text_value = normalize_text(value).lower()
        if not text_value:
            return ""
        text_value = text_value.replace("&", " and ")
        text_value = re.sub(r"[^a-z0-9]+", "-", text_value)
        return text_value.strip("-")

    def get_localized_text(value, preferred_language="en"):
        if isinstance(value, dict):
            preferred = normalize_text(value.get(preferred_language))
            if preferred:
                return preferred
            for nested_value in value.values():
                candidate = normalize_text(nested_value)
                if candidate:
                    return candidate
            return ""
        if isinstance(value, list):
            for item in value:
                candidate = get_localized_text(item, preferred_language)
                if candidate:
                    return candidate
            return ""
        return normalize_text(value)

    def extract_first_valid_email(values):
        if not values:
            return ""
        source_values = [values] if isinstance(values, str) else values
        for raw in source_values:
            candidate = normalize_text(raw)
            if not candidate:
                continue
            emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", candidate)
            for email_value in emails:
                normalized_email = normalize_text(email_value)
                if normalized_email:
                    return normalized_email
        return ""

    def extract_mailto_from_soup(soup):
        for anchor in soup.select("a[href^='mailto:']"):
            href = normalize_text(anchor.get("href"))
            candidate = href.replace("mailto:", "").split("?")[0].strip()
            if candidate and extract_first_valid_email(candidate):
                return candidate
        return ""

    def fetch_html_with_requests(target_url):
        try:
            response = session.get(target_url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            return response.text
        except Exception:
            return ""

    def collect_contact_links(soup, page_url, max_links=6):
        keywords = (
            "contact",
            "kontakt",
            "about",
            "imprint",
            "impressum",
            "privacy",
            "datenschutz",
            "company",
            "about-us",
        )
        parsed_page = urlparse(page_url)
        root_host = normalize_text(parsed_page.netloc).lower().replace("www.", "")
        links = []

        for anchor in soup.select("a[href]"):
            href = normalize_text(anchor.get("href"))
            if not href:
                continue
            if href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue

            text_value = normalize_text(anchor.get_text(" ", strip=True)).lower()
            href_value = href.lower()
            if not any(token in text_value or token in href_value for token in keywords):
                continue

            full_url = urljoin(page_url, href)
            parsed_full = urlparse(full_url)
            if parsed_full.scheme not in {"http", "https"}:
                continue

            link_host = normalize_text(parsed_full.netloc).lower().replace("www.", "")
            if root_host and link_host and root_host != link_host:
                continue

            if full_url not in links:
                links.append(full_url)
            if len(links) >= max_links:
                break

        return links

    def find_email_via_requests(website_url):
        normalized_url = normalize_website_url(website_url)
        if not normalized_url:
            return ""

        main_html = fetch_html_with_requests(normalized_url)
        if not main_html:
            return ""

        main_soup = BeautifulSoup(main_html, "html.parser")
        main_mailto = extract_mailto_from_soup(main_soup)
        if main_mailto:
            return main_mailto

        direct_email = extract_first_valid_email(extract_emails_from_source(main_html))
        if direct_email:
            return direct_email

        for contact_url in collect_contact_links(main_soup, normalized_url):
            contact_html = fetch_html_with_requests(contact_url)
            if not contact_html:
                continue

            contact_soup = BeautifulSoup(contact_html, "html.parser")
            contact_mailto = extract_mailto_from_soup(contact_soup)
            if contact_mailto:
                return contact_mailto

            contact_email = extract_first_valid_email(extract_emails_from_source(contact_html))
            if contact_email:
                return contact_email

        return ""

    def parse_categories(category_list):
        if not isinstance(category_list, list):
            return ""
        values = []
        for category in category_list:
            if isinstance(category, dict):
                category_name = get_localized_text(category.get("name"))
            else:
                category_name = normalize_text(category)
            if category_name and category_name not in values:
                values.append(category_name)
        return ", ".join(values)

    def parse_widget_state(detail_soup):
        script_tag = detail_soup.select_one("script#widget-state")
        if not script_tag:
            return {}

        script_text = script_tag.get_text("\n", strip=False)
        match = re.search(
            r"window\.__WIDGET_STATE__\s*=\s*(\{.*\})\s*;?\s*$",
            script_text,
            flags=re.DOTALL,
        )
        if not match:
            return {}

        raw_json = html.unescape(match.group(1))
        try:
            return json.loads(raw_json)
        except Exception:
            return {}

    def build_public_detail_url(company_name, stand_id):
        stand_slug = slugify_text(company_name)
        stand_id_text = normalize_text(stand_id)
        if stand_slug and stand_id_text:
            return f"https://www.dortmund.intralogistik-messen.de/en/exhibitors/{stand_slug}-{stand_id_text}/"
        return ""

    common_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    api_headers = {
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.dortmund.intralogistik-messen.de",
        "Referer": base_url,
    }
    html_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Origin": "https://www.dortmund.intralogistik-messen.de",
        "Referer": base_url,
    }

    max_pages = max(1, int(page_count))
    session = requests.Session()
    session.headers.update(common_headers)

    status_text = st.empty()
    progress_bar = st.progress(0)
    tablo = []
    seen_stand_ids = set()
    email_cache = {}
    detected_total_pages = None

    try:
        st.info(
            f"Scanning requested {max_pages} page(s) with '?stands%5Bpage%5D=' pagination."
        )

        for page_no in range(1, max_pages + 1):
            progress_bar.progress(page_no / max_pages if max_pages else 1)
            status_text.text(f"Scanning list page {page_no}/{max_pages}...")

            list_page_url = f"{base_url}?stands%5Bpage%5D={page_no}"
            payload = [
                {
                    "indexName": "stands",
                    "params": {
                        "query": "",
                        "page": page_no - 1,
                        "hitsPerPage": 24,
                        "filters": "(containerId: 2591)",
                    },
                }
            ]

            list_response = session.post(
                list_api_url,
                json=payload,
                headers=api_headers,
                timeout=45,
            )
            list_response.raise_for_status()
            list_json = list_response.json() if list_response.text else {}

            results = list_json.get("results", []) if isinstance(list_json, dict) else []
            first_result = results[0] if results else {}
            hits = first_result.get("hits", []) if isinstance(first_result, dict) else []

            if page_no == 1 and isinstance(first_result, dict):
                raw_nb_pages = first_result.get("nbPages")
                try:
                    detected_total_pages = int(raw_nb_pages)
                except Exception:
                    detected_total_pages = None
                if detected_total_pages:
                    st.info(
                        f"Detected approximately {detected_total_pages} pages on the portal. "
                        f"Scanning requested {max_pages} page(s)."
                    )

            if not hits:
                print(f"No exhibitors found on page {page_no}; stopping.")
                break

            for idx, hit in enumerate(hits, start=1):
                stand_id = normalize_text(hit.get("objectID"))
                if not stand_id or stand_id in seen_stand_ids:
                    continue
                seen_stand_ids.add(stand_id)

                status_text.text(
                    f"Reading detail pages (page {page_no}/{max_pages}, "
                    f"company {idx}/{len(hits)})..."
                )

                company_name = normalize_text(hit.get("name"))
                company_stand = normalize_text(hit.get("standNumber"))
                company_description = normalize_text(hit.get("description"))
                exhibition_product_group = parse_categories(hit.get("categories"))

                company_website = ""
                company_mail = ""
                company_phone = ""
                company_address = ""
                company_zip_code = ""
                company_city = ""
                company_country = ""
                detail_url = build_public_detail_url(company_name, stand_id)

                try:
                    detail_response = session.get(
                        detail_api_template.format(stand_id=stand_id),
                        headers=html_headers,
                        timeout=45,
                    )
                    detail_response.raise_for_status()

                    detail_soup = BeautifulSoup(detail_response.text, "html.parser")
                    widget_state = parse_widget_state(detail_soup)
                    stand_data = widget_state.get("stand", {}) if isinstance(widget_state, dict) else {}

                    if isinstance(stand_data, dict) and stand_data:
                        company_name = normalize_text(stand_data.get("name")) or company_name
                        company_stand = normalize_text(stand_data.get("standNumber")) or company_stand
                        company_description = (
                            get_localized_text(stand_data.get("description")) or company_description
                        )

                        detail_categories = parse_categories(stand_data.get("categories"))
                        if detail_categories:
                            exhibition_product_group = detail_categories

                        company_website = normalize_website_url(stand_data.get("websiteUrl"))
                        company_phone = normalize_text(stand_data.get("phone"))
                        company_mail = normalize_text(stand_data.get("email"))
                        company_zip_code = normalize_text(stand_data.get("zipCode"))
                        company_city = normalize_text(stand_data.get("town"))
                        company_country = normalize_text(stand_data.get("country"))
                        company_address = normalize_text(stand_data.get("address"))

                        detail_url = build_public_detail_url(company_name, stand_id) or detail_url

                    if not company_website:
                        website_anchor = detail_soup.select_one(
                            ".stand-details__info-line-content a[href^='http']"
                        )
                        if website_anchor:
                            company_website = normalize_website_url(website_anchor.get("href"))

                    if company_website and not company_mail:
                        if company_website in email_cache:
                            company_mail = email_cache[company_website]
                        else:
                            status_text.text(
                                f"Searching email on website (page {page_no}/{max_pages}, "
                                f"company {idx}/{len(hits)})..."
                            )
                            try:
                                company_mail = find_email_via_requests(company_website)
                            except Exception:
                                company_mail = ""
                            email_cache[company_website] = company_mail

                except Exception as detail_error:
                    print(f"Detail parsing failed for stand {stand_id}: {detail_error}")

                tablo.append({
                    "Data Source/ExhibitionName": "Intralogistik Dortmund",
                    "ExhibitionProductGroup": exhibition_product_group,
                    "CompanyName": company_name,
                    "CompanyWebsite": company_website,
                    "CompanyMail": company_mail,
                    "CompanyMail2": "",
                    "CompanyPhone": company_phone,
                    "CompanyAddress": company_address,
                    "CompanyZipCode": company_zip_code,
                    "CompanyCity": company_city,
                    "CompanyCountry": company_country,
                    "CompanyBusinessType": "",
                    "StandNumber": company_stand,
                    "DetailUrl": detail_url,
                    "CompanyDescription": company_description,
                    "ListPageUrl": list_page_url,
                })

    except Exception as e:
        print(f"Intralogistik Dortmund scraper error: {e}")
        st.error(str(e))
    finally:
        status_text.empty()
        progress_bar.empty()

    df = pd.DataFrame(tablo)
    extra_columns = ["StandNumber", "DetailUrl", "CompanyDescription", "ListPageUrl"]
    for col in required_columns + extra_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns + extra_columns]

    st.dataframe(df)

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)
    st.download_button(
        label="Download Excel (.xlsx)",
        data=excel_buffer,
        file_name=f"{st.session_state.get('function_name', 'intralogistik_dortmund')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{st.session_state.get('function_name', 'intralogistik_dortmund')}.csv",
        mime="text/csv",
    )

    return df
