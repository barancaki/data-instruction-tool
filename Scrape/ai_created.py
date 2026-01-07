import pandas as pd
import time
import io
import streamlit as st
import plotly.express as px
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from Scrape.scrape import site_icerisinden_email_bul

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

