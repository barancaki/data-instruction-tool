# 📚 Memory Bank - Web Scraping Fonksiyon Kılavuzları

Bu klasör, yeni website scraping fonksiyonları oluştururken AI coding araçlarının kullanması gereken kılavuzları içerir.

## 📁 Dosyalar

### 1. `website-scraping-guide.md` (Ana Kılavuz)
**Amaç:** Detaylı, adım adım scraping fonksiyonu oluşturma kılavuzu

**İçerik:**
- Ön hazırlık ve website analizi
- Kullanıcıya sorulacak tüm sorular (detaylı)
- Fonksiyon yapısı ve şablon kodu
- Standart çıktı formatı
- Test ve entegrasyon adımları
- Hata yönetimi ve best practices
- Özel senaryolar (scroll, tek sayfa, vb.)
- Son kontrol listesi

**Kullanım:** Yeni bir scraping fonksiyonu yazarken tüm süreci takip etmek için

---

### 2. `quick-start-template.md` (Hızlı Başlangıç)
**Amaç:** Hızlı referans ve kopyala-yapıştır kod şablonları

**İçerik:**
- Checklist formatında soru listesi
- Hazır kod blokları (import, setup, vb.)
- Pagination tipleri için örnek kodlar
- 2_Fair_Scraper.py entegrasyon şablonları
- CSS selector hızlı referansı
- Son kontrol listesi

**Kullanım:** Hızlıca yeni fonksiyon oluşturmak için

---

## 🎯 Kullanım Senaryoları

### Senaryo 1: İlk Kez Scraping Fonksiyonu Yazıyorsanız

1. `website-scraping-guide.md` dosyasını baştan sona okuyun
2. "Kullanıcıya Sorulacak Sorular" bölümündeki tüm soruları sorun
3. Fonksiyon şablonunu kullanarak adım adım kodlayın
4. Test ve entegrasyon bölümünü takip edin

### Senaryo 2: Deneyimliyseniz ve Hızlı Çalışmak İstiyorsanız

1. `quick-start-template.md` dosyasını açın
2. Checklist'i kullanarak gerekli bilgileri toplayın
3. İlgili kod bloklarını kopyala-yapıştır yapın
4. CSS selector'ları ve değişken isimlerini düzenleyin
5. 2_Fair_Scraper.py entegrasyon şablonunu kullanın

### Senaryo 3: Özel Durum (Scroll, Tek Sayfa, vb.)

1. `website-scraping-guide.md` → "7. Özel Senaryolar" bölümüne bakın
2. İlgili senaryonun kod örneğini kullanın
3. `quick-start-template.md` → İlgili entegrasyon şablonunu kullanın

---

## 🔄 İş Akışı Özeti

```
1. KULLANICI: Website URL'i verir
   ↓
2. AI: quick-start-template.md checklist ile soruları sorar
   ↓
3. KULLANICI: Soruları cevaplar (CSS selector'lar vb.)
   ↓
4. AI: quick-start-template.md kod bloklarını kullanarak fonksiyon yazar
   ↓
5. AI: ai_created.py dosyasına ekler
   ↓
6. AI: 2_Fair_Scraper.py'ye entegre eder (import + URL kontrolü)
   ↓
7. AI: Test için 2-3 sayfa çekerek doğrular
   ↓
8. BİTTİ: Kullanıcı URL'i girip "Scan" butonuna basabilir!
```

---

## 📋 Standart Çıktı Formatı

Tüm scraping fonksiyonları aşağıdaki kolonları içeren DataFrame döndürmelidir:

| Kolon Adı | Tip | Zorunlu | Açıklama |
|-----------|-----|---------|----------|
| Data Source/E_Exhibition | str | ✅ | Website/Fuar adı |
| Product | str | ✅ | Ürün grupları (virgülle ayrılmış) |
| CompanyName | str | ✅ | Firma adı |
| CompanyWebsite | str | ✅ | Website URL |
| CompanyMail | str | ✅ | Email (birincil) |
| CompanyMail2 | str | ✅ | Email (ikincil) |
| CompanyPhone | str | ✅ | Telefon |
| CompanyAddress | str | ✅ | Adres |
| CompanyZipCode | str | ✅ | Posta kodu |
| CompanyCity | str | ✅ | Şehir |
| CompanyCountry | str | ✅ | Ülke |
| Detay Link | str | ✅ | Detay sayfa URL |
| Stand No | str | ☐ | Stand numarası (opsiyonel) |
| Category | str | ☐ | Kategori (opsiyonel) |

**NOT:** Zorunlu kolonlar boş olabilir (`""`) ama mutlaka bulunmalıdır!

---

## ⚙️ Teknik Gereksinimler

### Python Kütüphaneleri
```python
pandas
selenium
webdriver_manager
streamlit
plotly
beautifulsoup4 (bazı fonksiyonlar için)
requests (email bulma helper için)
```

### Selenium WebDriver
- Chrome WebDriver (otomatik yüklenir: webdriver_manager)
- Headless mode (arka plan çalışma)

### Helper Fonksiyonlar
- `site_icerisinden_email_bul(website_url)` - Email arama (Scrape/scrape.py)

---

## 📝 Önemli Notlar

1. **Email Bulma Önceliği:**
   - Önce sayfada açıkça aranır
   - Sonra `mailto:` linklerinde
   - Son olarak `site_icerisinden_email_bul()` ile website'den

2. **Streamlit Çıktı Bloğu:**
   - Her fonksiyonda **mutlaka** bulunmalı
   - Excel ve CSV indirme butonları
   - DataFrame görüntüleme
   - Ülke bazlı grafik (varsa)

3. **session_state['function_name']:**
   - 2_Fair_Scraper.py'de set edilmeli
   - İndirilen dosyanın adını belirler

4. **Error Handling:**
   - Her veri çekiminde try-except
   - Boş string `""` döndür (None değil!)
   - Print mesajları ile debug kolaylığı

5. **Timing:**
   - Liste sayfalar: 3 saniye
   - Detay sayfalar: 2 saniye
   - Ağır siteler: 5+ saniye

---

## 🐛 Sık Karşılaşılan Sorunlar

### Sorun 1: Firma bulunamıyor
**Çözüm:** CSS selector yanlış olabilir. Developer Tools ile tekrar kontrol edin.

### Sorun 2: Email bulunamıyor
**Çözüm:** 3 yöntem de denenmiş mi kontrol edin. Website'den email bulma en yavaş ama en garantili yöntemdir.

### Sorun 3: Sayfa yüklenmiyor
**Çözüm:** `time.sleep()` süresini artırın. DDoS koruması varsa User-Agent'ı değiştirin.

### Sorun 4: Relative URL hatası
**Çözüm:** 
```python
if detay_link.startswith("/"):
    detay_link = "https://domain.com" + detay_link
```

### Sorun 5: Excel indirme çalışmıyor
**Çözüm:** `st.session_state['function_name']` set edilmiş mi kontrol edin.

---

## 📞 Destek

Sorularınız için:
- `website-scraping-guide.md` → Detaylı açıklamalar
- `quick-start-template.md` → Hızlı referans
- Mevcut fonksiyonlar → `Scrape/ai_created.py` (örnek olarak incelenebilir)

---

**Son Güncelleme:** 2025-01-20  
**Versiyon:** 1.0
