# 🎯 Website Scraping Fonksiyonu Oluşturma Kılavuzu

Bu doküman, yeni bir website için scraping fonksiyonu oluştururken AI coding araçlarının kullanıcıya sorması gereken soruları ve fonksiyonun nasıl yazılması gerektiğini detaylı şekilde açıklar.

## 📋 İÇİNDEKİLER
1. [Ön Hazırlık ve Analiz Aşaması](#1-ön-hazırlık-ve-analiz-aşaması)
2. [Kullanıcıya Sorulacak Kritik Sorular](#2-kullanıcıya-sorulacak-kritik-sorular)
3. [Fonksiyon Yapısı ve Şablon](#3-fonksiyon-yapısı-ve-şablon)
4. [Standart Çıktı Formatı](#4-standart-çıktı-formatı)
5. [Test ve Entegrasyon](#5-test-ve-entegrasyon)
6. [Hata Yönetimi ve Best Practices](#6-hata-yönetimi-ve-best-practices)

---

## 1. ÖN HAZIRLIK VE ANALİZ AŞAMASI

### 1.1 Website Analizi Checklist

Kullanıcıdan URL aldıktan sonra, aşağıdaki adımları manuel olarak gerçekleştir:

```
✓ Website'i tarayıcıda aç
✓ Firma listesi sayfasının yapısını incele
✓ Pagination (sayfalama) sistemini kontrol et
✓ Firma detay sayfalarının varlığını kontrol et
✓ JavaScript ile dinamik yükleme olup olmadığını kontrol et
✓ Lazy loading/infinite scroll olup olmadığını kontrol et
```

### 1.2 HTML Yapı Analizi

Developer Tools kullanarak:
- Liste sayfasındaki firma kartlarının CSS selectorlarını belirle
- Detay sayfasındaki veri alanlarının selectorlarını belirle
- Pagination butonlarının/linklerinin yapısını incele

---

## 2. KULLANICIYA SORULACAK KRİTİK SORULAR

### 📌 SORU 1: Website URL ve Sayfa Yapısı
```
Q: Website'in tam URL'si nedir?
   Örnek: https://example.com/exhibitors
   
Q: Website'de kaç sayfa var veya kaç sayfa taramak istiyorsunuz?
   - Biliniyorsa toplam sayfa sayısı
   - Bilinmiyorsa taranacak sayfa sayısı
   
Q: Sayfalama nasıl çalışıyor?
   A) URL parametresi ile (örn: ?page=2, ?p=2)
   B) URL path ile (örn: /page/2/)
   C) Buton tıklama ile
   D) Infinite scroll/lazy loading
   E) Sayfalama yok, tek sayfa
```

### 📌 SORU 2: Firma Listesi Yapısı
```
Q: Firma listesi ana sayfada mı yoksa her firmanın detay sayfası var mı?
   A) Tüm bilgiler liste sayfasında (detay sayfası yok)
   B) Detay sayfaları var (liste → detay)
   
Q: Liste sayfasında firma kartlarının CSS selector'u nedir?
   Örnek: div.exhibitor-card, li.company-item, a.firm-link
   
Q: [Detay sayfa varsa] Detay linkinin CSS selector'u nedir?
   Örnek: a.detail-link, button.view-profile
```

### 📌 SORU 3: Çekilecek Veri Alanları
```
Q: Hangi bilgiler çekilecek? (Her biri için CSS selector gerekli)

Standart Alanlar:
□ Firma Adı (CompanyName) - Selector: _______________
□ Website (CompanyWebsite) - Selector: _______________
□ Email (CompanyMail) - Selector: _______________
□ Telefon (CompanyPhone) - Selector: _______________
□ Adres (CompanyAddress) - Selector: _______________
□ Posta Kodu (CompanyZipCode) - Selector: _______________
□ Şehir (CompanyCity) - Selector: _______________
□ Ülke (CompanyCountry) - Selector: _______________
□ Ürün Grupları (Product) - Selector: _______________

Ek Alanlar:
□ Stand No - Selector: _______________
□ Katılımcı Kategorisi - Selector: _______________
□ Diğer: _______________ - Selector: _______________
```

### 📌 SORU 4: Özel Durumlar
```
Q: Email bilgisi sayfada açıkça var mı?
   A) Evet, açıkça görünüyor (Selector: _______)
   B) Hayır, mailto: linkinde (Selector: a[href^='mailto:'])
   C) Hayır, firma websitesinden çekilmeli
   
Q: Website dinamik yükleme (JavaScript) kullanıyor mu?
   A) Evet, Selenium gerekli
   B) Hayır, BeautifulSoup yeterli
   
Q: Scroll yaparak içerik yükleme var mı?
   A) Evet (Kaç kez scroll: _____)
   B) Hayır
   
Q: Sayfalar arası bekleme süresi gerekli mi?
   Önerilen: 2-3 saniye (DDoS koruması varsa daha fazla)
```

### 📌 SORU 5: Veri Kaynağı Adı
```
Q: Bu website/fuar/organizasyon adı nedir?
   Örnek: "Advanced Engineering UK", "SPS Mesago", "GITEX Africa"
   
   Bu isim "Data Source/E_Exhibition" alanında kullanılacak.
```

---

## 3. FONKSİYON YAPISI VE ŞABLON

### 3.1 Temel Fonksiyon Şablonu

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


def scrape_FUNCTION_NAME(sayfa_sayisi):
    """
    [WEBSITE ADI] için scraping fonksiyonu
    
    Args:
        sayfa_sayisi (int): Taranacak sayfa sayısı
        
    Returns:
        pd.DataFrame: Çekilen firma bilgileri
    """
    
    # ============================================
    # ADIM 1: SELENIUM SETUP
    # ============================================
    options = Options()
    options.add_argument("--headless")  # Arka planda çalıştır
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # ============================================
    # ADIM 2: URL VE VERİ YAPISINI TANIMLA
    # ============================================
    base_url = "https://example.com/exhibitors"
    tablo = []

    try:
        # ============================================
        # ADIM 3: SAYFA DÖNGÜSÜ
        # ============================================
        for page_num in range(1, sayfa_sayisi + 1):
            print(f"\n🔄 {page_num}. sayfa işleniyor...")

            try:
                # Sayfa URL'ini oluştur (pagination tipine göre)
                if page_num == 1:
                    current_url = base_url
                else:
                    # ÖRN 1: URL parametresi (?page=2)
                    current_url = f"{base_url}?page={page_num}"
                    
                    # ÖRN 2: URL path (/page/2/)
                    # current_url = f"{base_url}page/{page_num}/"
                    
                    # ÖRN 3: Özel format
                    # current_url = f"{base_url}?stands%5Bpage%5D={page_num}"
                
                driver.get(current_url)
                time.sleep(3)  # Sayfa yüklenmesini bekle

                # ============================================
                # ADIM 4: SCROLL (Gerekirse)
                # ============================================
                # Eğer lazy loading varsa:
                # for scroll_num in range(3):
                #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                #     time.sleep(2)

                # ============================================
                # ADIM 5: FİRMA KARTLARINI BUL
                # ============================================
                firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.exhibitor-card")
                if not firma_kartlari:
                    print(f"⚠️ {page_num}. sayfada firma bulunamadı.")
                    break
                
                print(f"📊 {len(firma_kartlari)} firma bulundu")

                # ============================================
                # ADIM 6A: DETAY LİNKLERİ TOPLA (Detay sayfa varsa)
                # ============================================
                firma_linkleri = []
                for kart in firma_kartlari:
                    try:
                        detay_link_elem = kart.find_element(By.CSS_SELECTOR, "a.detail-link")
                        detay_link = detay_link_elem.get_attribute("href")
                        if detay_link:
                            # Relative URL'yi absolute yap
                            if detay_link.startswith("/"):
                                detay_link = "https://example.com" + detay_link
                            firma_linkleri.append(detay_link)
                    except:
                        continue
                
                print(f"🔗 {len(firma_linkleri)} firma linki toplandı")

                # ============================================
                # ADIM 6B: DETAY SAYFALARI İŞLE
                # ============================================
                for idx, detay_link in enumerate(firma_linkleri, 1):
                    try:
                        print(f"  {idx}/{len(firma_linkleri)}. 🔍 Detay sayfası açılıyor...")
                        driver.get(detay_link)
                        time.sleep(2)

                        # ============================================
                        # ADIM 7: VERİ ÇEK
                        # ============================================
                        
                        # Firma Adı
                        try:
                            firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.company-name").text.strip()
                        except:
                            firma_adi = ""
                        
                        # Website
                        try:
                            website_elem = driver.find_element(By.CSS_SELECTOR, "a.website-link")
                            website = website_elem.get_attribute("href")
                        except:
                            website = ""
                        
                        # Email - Önce sayfada ara
                        email = ""
                        try:
                            # Yöntem 1: mailto: linkinden
                            email_elem = driver.find_element(By.CSS_SELECTOR, "a[href^='mailto:']")
                            mailto_href = email_elem.get_attribute("href")
                            email = mailto_href.replace("mailto:", "").split("?")[0].strip()
                        except:
                            pass
                        
                        # Yöntem 2: Email bulunamadıysa websiteden çek
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
                        try:
                            telefon_elem = driver.find_element(By.CSS_SELECTOR, "a[href^='tel:']")
                            telefon_href = telefon_elem.get_attribute("href")
                            telefon = telefon_href.replace("tel:", "").strip()
                        except:
                            telefon = ""
                        
                        # Adres
                        try:
                            adres = driver.find_element(By.CSS_SELECTOR, "p.address").text.strip()
                        except:
                            adres = ""
                        
                        # Posta Kodu
                        try:
                            posta_kodu = driver.find_element(By.CSS_SELECTOR, "span.zip-code").text.strip()
                        except:
                            posta_kodu = ""
                        
                        # Şehir
                        try:
                            sehir = driver.find_element(By.CSS_SELECTOR, "span.city").text.strip()
                        except:
                            sehir = ""
                        
                        # Ülke
                        try:
                            ulke = driver.find_element(By.CSS_SELECTOR, "span.country").text.strip()
                        except:
                            ulke = ""
                        
                        # Ürün Grupları (çoklu element varsa)
                        try:
                            urun_div_list = driver.find_elements(By.CSS_SELECTOR, "div.product-category")
                            urun_gruplari = ", ".join([div.text.strip() for div in urun_div_list if div.text.strip()])
                        except:
                            urun_gruplari = ""

                        print(f"  ✅ {firma_adi}")

                        # ============================================
                        # ADIM 8: TABLOYA EKLE
                        # ============================================
                        tablo.append({
                            "Data Source/E_Exhibition": "WEBSITE ADI",
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

    # ============================================
    # ADIM 9: DATAFRAME OLUŞTUR
    # ============================================
    df = pd.DataFrame(tablo)
    print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")

    # ============================================
    # ADIM 10: STREAMLIT ÇIKTILARI (ZORUNLU!)
    # ============================================
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

## 4. STANDART ÇIKTI FORMATI

### 4.1 Zorunlu Kolonlar

Her fonksiyon **mutlaka** aşağıdaki kolonları içermeli:

```python
{
    "Data Source/E_Exhibition": str,  # Website/Fuar adı
    "Product": str,                    # Ürün grupları (virgülle ayrılmış)
    "CompanyName": str,                # Firma adı
    "CompanyWebsite": str,             # Website URL
    "CompanyMail": str,                # Email (birincil)
    "CompanyMail2": str,               # Email (ikincil) - boş olabilir
    "CompanyPhone": str,               # Telefon
    "CompanyAddress": str,             # Adres
    "CompanyZipCode": str,             # Posta kodu
    "CompanyCity": str,                # Şehir
    "CompanyCountry": str,             # Ülke
    "Detay Link": str                  # Detay sayfa URL
}
```

### 4.2 Opsiyonel Kolonlar

İhtiyaca göre eklenebilir:

```python
{
    "Stand No": str,                   # Stand numarası
    "Category": str,                   # Katılımcı kategorisi
    "Hall": str,                       # Salon/Hall bilgisi
    "LinkedIn": str,                   # LinkedIn URL
    "Facebook": str,                   # Facebook URL
    # ... diğer sosyal medya
}
```

---

## 5. TEST VE ENTEGRASYON

### 5.1 Fonksiyon Testi

Fonksiyon oluşturulduktan sonra:

```python
# Test kodu (ai_created.py dosyasına eklenmeden önce)
if __name__ == "__main__":
    # Küçük bir sayfa sayısı ile test et
    df = scrape_FUNCTION_NAME(sayfa_sayisi=2)
    print(df.head())
    print(f"\nToplam: {len(df)} firma")
```

### 5.2 2_Fair_Scraper.py'ye Entegrasyon

Fonksiyon test edildikten sonra:

**ADIM 1:** `ai_created.py` dosyasının import satırına ekle:

```python
from Scrape.ai_created import scrape_advanced_engineering, scrape_mesago, scrape_gitex_africa_morocco, scrape_yasad_uyeler, scrape_FUNCTION_NAME
```

**ADIM 2:** `2_Fair_Scraper.py` dosyasına URL kontrolü ekle:

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

**ÖNEMLİ:** `st.session_state['function_name']` değeri, indirme dosya adı için kullanılır!

### 5.3 Scroll Varyasyonu (Lazy Loading için)

Eğer infinite scroll varsa:

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

---

## 6. HATA YÖNETİMİ VE BEST PRACTICES

### 6.1 Try-Except Blokları

Her veri çekme işleminde try-except kullan:

```python
# ✅ DOĞRU
try:
    firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.company-name").text.strip()
except:
    firma_adi = ""

# ❌ YANLIŞ (hata durumunda script durur)
firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.company-name").text.strip()
```

### 6.2 Bekleme Süreleri

```python
time.sleep(2)   # Detay sayfalar için
time.sleep(3)   # Liste sayfalar için
time.sleep(5)   # Ağır/korumalı siteler için
```

### 6.3 Email Bulma Öncelik Sırası

1. Sayfada açıkça görünüyor mu? → CSS selector ile çek
2. `mailto:` linkinde mi? → Link'ten parse et
3. Yoksa → `site_icerisinden_email_bul(website)` kullan

```python
# Öncelik 1: Direkt selector
try:
    email = driver.find_element(By.CSS_SELECTOR, "span.email").text.strip()
except:
    email = ""

# Öncelik 2: mailto: link
if not email:
    try:
        mailto_elem = driver.find_element(By.CSS_SELECTOR, "a[href^='mailto:']")
        email = mailto_elem.get_attribute("href").replace("mailto:", "").split("?")[0]
    except:
        pass

# Öncelik 3: Website'den ara
if not email and website:
    try:
        email_list = site_icerisinden_email_bul(website)
        if email_list:
            email = email_list[0]
    except:
        pass
```

### 6.4 URL Normalizasyonu

```python
# Relative URL'leri absolute yap
if detay_link.startswith("/"):
    detay_link = "https://example.com" + detay_link

# HTTP → HTTPS
if website.startswith("http://"):
    website = website.replace("http://", "https://")
```

### 6.5 Boş Değer Kontrolü

```python
# Liste birleştirme (virgülle ayrılmış)
urun_gruplari = ", ".join([item.text.strip() for item in items if item.text.strip()])

# Boş string yerine tutarlı format
telefon = telefon if telefon else ""
```

### 6.6 Print Mesajları (Debugging)

```python
print(f"\n🔄 {page_num}. sayfa işleniyor...")          # Sayfa başlangıcı
print(f"📊 {len(firma_kartlari)} firma bulundu")       # Bulgu
print(f"🔗 {len(firma_linkleri)} firma linki toplandı") # Link toplama
print(f"  {idx}/{total}. 🔍 Detay sayfası açılıyor...") # Detay işleme
print(f"  ✅ {firma_adi}")                              # Başarılı
print(f"  ❌ Firma detayı işlenirken hata: {e}")       # Hata
```

---

## 7. ÖZEL SENARYOLAR

### 7.1 Pagination Yok - Tek Sayfa

```python
def scrape_FUNCTION_NAME():
    # sayfa_sayisi parametresi yok
    driver.get(base_url)
    # ... veri çekme
```

### 7.2 Detay Sayfa Yok - Liste Sayfasından Veri

```python
# Detay linki toplama adımını atla
for kart in firma_kartlari:
    # Direkt kart içinden veri çek
    firma_adi = kart.find_element(By.CSS_SELECTOR, "h3").text.strip()
    website = kart.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
    # ...
```

### 7.3 İki Aşamalı Toplama (YASAD örneği)

```python
# Önce tüm linkleri topla
firma_bilgileri_tum = []
for page_num in range(1, sayfa_sayisi + 1):
    # Linkleri topla
    firma_bilgileri_tum.append({"firma_adi": ..., "detay_link": ...})

# Sonra detayları çek
for bilgi in firma_bilgileri_tum:
    driver.get(bilgi["detay_link"])
    # ... veri çek
```

### 7.4 Regex ile Email/Telefon Parse

```python
import re

# Email pattern
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
text = "İletişim: info@example.com veya destek@example.com"
emails = re.findall(email_pattern, text)
# ['info@example.com', 'destek@example.com']

# Telefon pattern (Türkiye)
phone_pattern = r'(\+90|0)?[\s]?(\d{3})[\s]?(\d{3})[\s]?(\d{2})[\s]?(\d{2})'
```

---

## 8. SON KONTROL LİSTESİ

Fonksiyon tamamlandığında:

```
✓ Tüm import'lar mevcut mu?
✓ Selenium options doğru ayarlanmış mı?
✓ Pagination doğru çalışıyor mu?
✓ Tüm veri alanları için try-except var mı?
✓ Email bulma 3 yöntem ile denenmiş mi?
✓ Streamlit çıktı bloğu (Excel/CSV indirme) var mı?
✓ st.session_state['function_name'] set ediliyor mu?
✓ Print mesajları informatif mi?
✓ finally bloğunda driver.quit() var mı?
✓ Standart kolon isimleri kullanılmış mı?
✓ 2_Fair_Scraper.py'ye entegrasyon yapıldı mı?
✓ Test ile 2-3 sayfa çekilip doğrulandı mı?
```

---

## 9. ÖRNEK KULLANICI DİYALOGU

### Kullanıcı:
"https://example-fair.com/exhibitors için scraping fonksiyonu yaz"

### AI Asistan:

"Tamamdır! Önce birkaç soru sorayım:

1. **Sayfalama nasıl çalışıyor?** Website'e baktım, URL'de ?page=2 şeklinde mi ilerliyor yoksa farklı bir format mı kullanıyor?

2. **Firma detay sayfaları var mı?** Liste sayfasında sadece firma isimleri mi var, yoksa her firmaya tıkladığımızda detaylı bilgi sayfası mı açılıyor?

3. **Hangi bilgileri çekmek istiyorsunuz?**
   - Firma adı ✓ (her zaman)
   - Website URL ✓ (her zaman)
   - Email (Sayfada var mı yoksa website'den mi bulalım?)
   - Telefon (Var mı?)
   - Adres, Şehir, Ülke (Var mı?)
   - Ürün kategorileri (Var mı?)

4. **Website dinamik içerik yüklüyor mu?** (Scroll yapınca yeni firmalar yükleniyor mu?)

5. **Bu fuar/organizasyonun tam adı nedir?** (Excel dosyasında 'Data Source' olarak yazılacak)

Bu soruları yanıtladıktan sonra tam olarak ihtiyacınıza uygun fonksiyonu yazacağım."

---

## 10. DOSYA YAPISI

```
data-instruction-tool/
├── Scrape/
│   ├── ai_created.py          ← Yeni fonksiyonlar buraya
│   ├── scrape.py              ← Helper fonksiyonlar (site_icerisinden_email_bul)
│   ├── tuyap_sablon1.py
│   ├── tuyap_sablon2.py
│   └── ...
├── pages/
│   └── 2_Fair_Scraper.py      ← UI ve URL kontrolü
└── memory-bank/
    └── website-scraping-guide.md  ← Bu doküman
```

---

## 📌 HIZLI REFERANS

### CSS Selector Örnekleri
```css
/* Tag */
h1, p, div, span, a

/* Class */
.company-name, .btn-primary

/* ID */
#main-content

/* Attribute */
a[href^="mailto:"]
a[href^="tel:"]
input[type="email"]

/* Descendant */
div.card h3.title

/* Child */
ul > li

/* Multiple */
h1.title, h2.title
```

### Selenium Find Element
```python
# Tek element
driver.find_element(By.CSS_SELECTOR, "h1")
driver.find_element(By.TAG_NAME, "h1")
driver.find_element(By.CLASS_NAME, "title")
driver.find_element(By.ID, "main")

# Çoklu element
driver.find_elements(By.CSS_SELECTOR, "div.card")
```

---

**Son Güncelleme:** 2025-01-20  
**Versiyon:** 1.0  
**Hazırlayan:** OpenCode AI
