import pandas as pd
import time
import io
import re
import streamlit as st
import plotly.express as px
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

