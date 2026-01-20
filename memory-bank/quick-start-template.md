# 🚀 Hızlı Başlangıç - Yeni Website Scraping Fonksiyonu

## Kullanıcıya Sorulacak Sorular (Checklist)

```
📝 KULLANICI BİLGİLERİ

Website URL: _________________________________

Sayfa Sayısı: ________________________________

Organizasyon/Fuar Adı: _______________________

Pagination Tipi:
  [ ] ?page=2 formatı
  [ ] /page/2/ formatı  
  [ ] Özel format: _________________________
  [ ] Scroll (lazy loading)
  [ ] Tek sayfa (pagination yok)

Detay Sayfası:
  [ ] Var (liste → detay)
  [ ] Yok (tüm bilgi listede)

Liste Sayfası CSS Selectors:
  Firma Kartı: _____________________________
  Detay Linki: _____________________________

Detay Sayfası CSS Selectors:
  ✓ Firma Adı: _____________________________
  ✓ Website: _______________________________
  ☐ Email: _________________________________
  ☐ Telefon: _______________________________
  ☐ Adres: _________________________________
  ☐ Posta Kodu: ____________________________
  ☐ Şehir: _________________________________
  ☐ Ülke: __________________________________
  ☐ Ürün Grupları: _________________________
  ☐ Stand No: ______________________________
  ☐ Diğer: _________________________________

Email Durumu:
  [ ] Sayfada açıkça var (selector: _______)
  [ ] mailto: linkinde
  [ ] Website'den çekilmeli

Özel Durumlar:
  [ ] JavaScript/Selenium gerekli
  [ ] Scroll gerekli (kaç kez: _____)
  [ ] Bekleme süresi uzun olmalı (DDoS koruması)
```

---

## Kod Şablonu (Kopyala-Yapıştır)

### 1. Import Bloğu (Her Zaman Aynı)

```python
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
from Scrape.scrape import site_icerisinden_email_bul
```

### 2. Fonksiyon İmzası

```python
def scrape_FUNCTION_NAME(sayfa_sayisi):
    """
    [WEBSITE ADI] için scraping fonksiyonu
    """
```

### 3. Selenium Setup (Her Zaman Aynı)

```python
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "URL_BURAYA"
    tablo = []

    try:
```

### 4. Sayfa Döngüsü (Pagination Tipine Göre)

**Tip A: ?page=2 formatı**
```python
        for page_num in range(1, sayfa_sayisi + 1):
            print(f"\n🔄 {page_num}. sayfa işleniyor...")
            try:
                if page_num == 1:
                    current_url = base_url
                else:
                    current_url = f"{base_url}?page={page_num}"
                
                driver.get(current_url)
                time.sleep(3)
```

**Tip B: /page/2/ formatı**
```python
                if page_num == 1:
                    current_url = base_url
                else:
                    current_url = f"{base_url}page/{page_num}/"
```

**Tip C: Scroll (Tek sayfa)**
```python
        driver.get(base_url)
        time.sleep(3)
        
        for scroll_num in range(sayfa_sayisi):  # sayfa_sayisi = scroll sayısı
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            driver.execute_script("window.scrollBy(0, -200);")
            time.sleep(1)
```

### 5. Firma Kartlarını Bul

```python
                firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "FIRMA_KARTI_SELECTOR")
                if not firma_kartlari:
                    print(f"⚠️ {page_num}. sayfada firma bulunamadı.")
                    break
                
                print(f"📊 {len(firma_kartlari)} firma bulundu")
```

### 6. Detay Linkleri Topla (Detay sayfa varsa)

```python
                firma_linkleri = []
                for kart in firma_kartlari:
                    try:
                        detay_link_elem = kart.find_element(By.CSS_SELECTOR, "DETAY_LINK_SELECTOR")
                        detay_link = detay_link_elem.get_attribute("href")
                        if detay_link:
                            if detay_link.startswith("/"):
                                detay_link = "https://DOMAIN.com" + detay_link
                            firma_linkleri.append(detay_link)
                    except:
                        continue
                
                print(f"🔗 {len(firma_linkleri)} firma linki toplandı")
```

### 7. Detay Sayfaları İşle

```python
                for idx, detay_link in enumerate(firma_linkleri, 1):
                    try:
                        print(f"  {idx}/{len(firma_linkleri)}. 🔍 Detay sayfası açılıyor...")
                        driver.get(detay_link)
                        time.sleep(2)

                        # Firma Adı
                        try:
                            firma_adi = driver.find_element(By.CSS_SELECTOR, "SELECTOR").text.strip()
                        except:
                            firma_adi = ""
                        
                        # Website
                        try:
                            website_elem = driver.find_element(By.CSS_SELECTOR, "SELECTOR")
                            website = website_elem.get_attribute("href")
                        except:
                            website = ""
                        
                        # Email - 3 Aşamalı
                        email = ""
                        try:
                            # Yöntem 1: Direkt selector (varsa)
                            email = driver.find_element(By.CSS_SELECTOR, "SELECTOR").text.strip()
                        except:
                            pass
                        
                        if not email:
                            try:
                                # Yöntem 2: mailto: link
                                email_elem = driver.find_element(By.CSS_SELECTOR, "a[href^='mailto:']")
                                mailto_href = email_elem.get_attribute("href")
                                email = mailto_href.replace("mailto:", "").split("?")[0].strip()
                            except:
                                pass
                        
                        if not email and website:
                            try:
                                # Yöntem 3: Website'den ara
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
                        try:
                            telefon_elem = driver.find_element(By.CSS_SELECTOR, "a[href^='tel:']")
                            telefon_href = telefon_elem.get_attribute("href")
                            telefon = telefon_href.replace("tel:", "").strip()
                        except:
                            telefon = ""
                        
                        # Adres
                        try:
                            adres = driver.find_element(By.CSS_SELECTOR, "SELECTOR").text.strip()
                        except:
                            adres = ""
                        
                        # Posta Kodu
                        try:
                            posta_kodu = driver.find_element(By.CSS_SELECTOR, "SELECTOR").text.strip()
                        except:
                            posta_kodu = ""
                        
                        # Şehir
                        try:
                            sehir = driver.find_element(By.CSS_SELECTOR, "SELECTOR").text.strip()
                        except:
                            sehir = ""
                        
                        # Ülke
                        try:
                            ulke = driver.find_element(By.CSS_SELECTOR, "SELECTOR").text.strip()
                        except:
                            ulke = ""
                        
                        # Ürün Grupları (çoklu element)
                        try:
                            urun_div_list = driver.find_elements(By.CSS_SELECTOR, "SELECTOR")
                            urun_gruplari = ", ".join([div.text.strip() for div in urun_div_list if div.text.strip()])
                        except:
                            urun_gruplari = ""

                        print(f"  ✅ {firma_adi}")

                        tablo.append({
                            "Data Source/E_Exhibition": "WEBSITE_ADI",
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
```

### 8. Cleanup ve DataFrame (Her Zaman Aynı)

```python
    finally:
        driver.quit()

    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")
```

### 9. Streamlit Çıktıları (ZORUNLU - Her Zaman Aynı)

```python
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
            file_name=f"{st.session_state.get('function_name', 'FUNCTION_NAME')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📥 CSV İndir
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV İndir",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.get('function_name', 'FUNCTION_NAME')}.csv",
            mime="text/csv"
        )
        
        # 📊 Grafikler (Ülke bilgisi varsa)
        if not df.empty and "CompanyCountry" in df.columns:
            try:
                ulke_sayilari = df["CompanyCountry"].value_counts().reset_index()
                ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
                fig = px.bar(ulke_sayilari.head(20), x="Ülke", y="Firma Sayısı", 
                            title="Ülkelere Göre Firma Dağılımı")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Grafik çizilirken hata: {e}")
    else:
        print("\n📊 İstatistikler (Streamlit dışı çalıştırma):")
        if not df.empty:
            print(f"Toplam firma: {len(df)}")
            if "CompanyCountry" in df.columns:
                print(df["CompanyCountry"].value_counts())
            
    return df
```

---

## 2_Fair_Scraper.py Entegrasyon Şablonu

### Adım 1: Import ekle (dosya başına)

```python
from Scrape.ai_created import scrape_advanced_engineering, scrape_mesago, ..., scrape_FUNCTION_NAME
```

### Adım 2: URL kontrolü ekle (dosya sonuna)

**Normal Pagination:**
```python
if url in ["https://example.com/exhibitors", "https://example.com/exhibitors?page=1"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                st.session_state['function_name'] = 'FUNCTION_NAME'
                scrape_FUNCTION_NAME(int(sayfa_sayisi))
            st.success("The scan is complete!")
```

**Scroll Tipi:**
```python
if url in ["https://example.com/exhibitors"]:
    scroll_sayisi = st.text_input("How many times do you want to scroll?")
    if scroll_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                st.session_state['function_name'] = 'FUNCTION_NAME'
                scrape_FUNCTION_NAME(int(scroll_sayisi))
            st.success("The scan is complete!")
```

**Tek Sayfa (parametre yok):**
```python
if url in ["https://example.com/members", "https://example.com/members/"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_FUNCTION_NAME()
        st.success("The scan is complete!")
```

---

## Son Kontrol Listesi

Kodu yazdıktan sonra:

```
✓ FUNCTION_NAME tüm yerlerde tutarlı mı?
✓ WEBSITE_ADI "Data Source/E_Exhibition" alanında doğru mu?
✓ CSS selector'lar kullanıcıdan alınan bilgiler mi?
✓ Email 3 yöntemle aranıyor mu?
✓ Try-except blokları her veri çekiminde var mı?
✓ Streamlit çıktı bloğu eksiksiz kopyalandı mı?
✓ st.session_state['function_name'] set ediliyor mu?
✓ driver.quit() finally bloğunda mı?
✓ 2_Fair_Scraper.py'ye import eklendi mi?
✓ 2_Fair_Scraper.py'ye URL kontrolü eklendi mi?
```

---

## CSS Selector Hızlı Referans

```css
/* Class */
.company-name
div.card
h3.title

/* ID */
#main-content

/* Tag */
h1
p
div
span
a

/* Attribute */
a[href^="mailto:"]      /* mailto: ile başlayan */
a[href^="tel:"]         /* tel: ile başlayan */
a[href*="website"]      /* içinde website geçen */

/* Çoklu */
h1, h2, h3              /* herhangi biri */

/* İç içe */
div.card h3             /* div.card içindeki h3 */
ul > li                 /* ul'nin direkt çocuğu li */
```

---

**İPUCU:** Tüm detaylar için `website-scraping-guide.md` dosyasına bakın!
