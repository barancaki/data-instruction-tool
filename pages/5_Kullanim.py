import streamlit as st

st.set_page_config(page_title="Kullanım Kılavuzu", layout="centered")

st.title("📘 Kullanım Kılavuzu")

st.markdown('''
Bu araç, fuarcılık sektöründeki web sitelerinden firma ve katılımcı bilgilerini toplamak ve bu bilgileri bir yapay zeka modeline analiz ettirerek anlamlı sonuçlar çıkarmak amacıyla geliştirilmiştir.
 
📌 Yapı modülerdir ve ileride farklı fuar siteleri de kolayca entegre edilebilir.

---

## 🔍 1. Katılımcı Listesini Tarama

### ➤ Adımlar:

1. Sol menüden **"Fuar Scraper"** sayfasına gidin.
2. Aşağıdaki örnek URL’yi kutuya yapıştırın:

https://replasteurasia.com/katilimci-listesi

3.	“Tara” butonuna tıklayın.
4.	Sistem otomatik olarak tüm sayfalardaki katılımcı bilgilerini toplar.

⚠️ Not: Taratılan sitede sayfa sayısı 15'ten fazla ise taratılcak site sayısını sizin girmeniz gerekmektedir.

✔️ Çekilen Bilgiler:
	•	Firma adı
	•	Firma adresi
	•	Firma ülkesi
	•	Firma telefonu
    •   Firma websitesi
	•   Firma ürün grupları(her site için olmayabilir)
            
✔️ Çekilen Tabloda:
    İndirme arama ve indeksleme gibi bir sürü özellikten faydalanabilirsiniz.
✔️ Dilerseniz grafik çıktısı oluşturma talebinde de bulunabilirsiniz. Örnek grafik : Ülke - Firma sayısı

---

## 🤖 2. AI ile Web Sitesi Analizi

Yapay zeka desteği sayesinde fuarcılık web sayfasından doğal dilde istediğiniz bilgiyi çıkartabilirsiniz.

🔧 Kullanılan Yapay Zeka Modeli:
•	Model: llama3:8b
            
•	Altyapı: Ollama üzerinden yerel (local) olarak çalışır
            
•	Çalışma Prensibi: Sayfa içeriği parçalara bölünerek analiz edilir. Prompt’a göre filtreli bilgi döner.

➤ Adımlar:
1.	Sol menüden “AI Analiz” sayfasına geçin.
            
2.	İncelemek istediğiniz fuarcılık web sayfasının tam URL’sini yazın.
            
3.	“Yapay zekaya ne sormak istiyorsunuz?” kısmına doğal dilde bir istek girin:
            
4.	“Analiz Et” butonuna tıklayın.
            
5.	AI modeli sayfayı analiz eder ve sonuçları aşağıda listeler.            

            
Örnekler:      
Sayfadaki tüm firma isimlerini ve ülkelerini listele.
Tüm iletişim e-posta adreslerini çıkar.
Ürün kategorilerini özetle.


ℹ️ Bu prompt özelliği sadece ingilizce dilinde yazılmalıdır !

---
## ❓ SSS

### Q: Bazı websitelerini yazdığımda Tara butonu gelmiyor, sebebi nedir ?
A: Scraper alanı özel fonksiyonlar için yazılmıştır. Kullanılacak websitesi isteklerine göre fonksiyonlar arttırılacak ve model geliştirilecektir.

### Q: AI neden bazı bilgileri kaçırıyor?
A: Web sayfalarının içeriği dinamik (JavaScript ile) yükleniyorsa, veya içerik çok uzun ve karışıksa, AI bazı bölümleri atlayabilir. Prompt’u daha net yazmak genellikle daha iyi sonuç verir.

### Q: AI ne kadar güvenilir?
A: LLM’ler metin çıkarımında oldukça başarılıdır, ancak %100 doğruluk beklenmemelidir. Kritik bilgiler için elle kontrol önerilir.

### Q: Çok yavaş çalışıyor , ne yapmalıyım?
A: Linkini girdiğiniz websitesinin sayfa sayısını daha az girebilirsiniz , fakat bu işlemci-gpu ilişkisi ile ilgilidir.Bilgisayar hızlıysa otomasyon hızlıdır.

## 🧑‍💻 Geliştirici Notu

Bu araç veri toplamak ve anlamlandırmak isteyen sektör profesyonellerine zaman kazandırmak için geliştirildi.
Açık kaynaklıdır ve geri bildirimlerinize göre sürekli geliştirilecektir.

İyi analizler! 🚀''')
st.sidebar.text('© Baran Çakı 2025')
st.text('© Baran Çakı 2025')