import streamlit as st
from ai_scripts.ollama_parser import get_clean_text_from_url, parse_with_ollama
from auth_helper import check_authentication,get_user_info,show_user_info_sidebar
# Authentication kontrolü
check_authentication()

# Kullanıcı bilgilerini al
user_info = get_user_info()

# Sidebar'da kullanıcı bilgilerini göster
show_user_info_sidebar()

st.set_page_config(page_title="AI Scraper", layout="centered")
st.sidebar.header("AI Scraper aracina hos geldiniz !")
st.sidebar.markdown('''## 🤖 AI ile Web Sitesi Analizi

Yapay zeka desteği sayesinde fuarcılık web sayfasından doğal dilde istediğiniz bilgiyi çıkartabilirsiniz.

🔧 Kullanılan Yapay Zeka Modeli:
•	Model: llama3:8b
            
•	Altyapı: Ollama üzerinden yerel (local) olarak çalışır
            
•	Çalışma Prensibi: Sayfa içeriği parçalara bölünerek analiz edilir. Prompt’a göre filtreli bilgi döner.

➤ Adımlar:
1.	Sol menüden “AI Analiz” sayfasına geçin.
            
2.	İncelemek istediğiniz web sayfasının tam URL’sini yazın.
            
3.	“Yapay zekaya ne sormak istiyorsunuz?” kısmına doğal dilde bir istek girin:
            
4.	“Analiz Et” butonuna tıklayın.
            
5.	AI modeli sayfayı analiz eder ve sonuçları aşağıda listeler.            

            
Örnekler:      
Sayfadaki tüm firma isimlerini ve ülkelerini listele.
Tüm iletişim e-posta adreslerini çıkar.
Ürün kategorilerini özetle.


ℹ️ Bu prompt özelliği sadece ingilizce dilinde yazılmalıdır ! ''')
st.sidebar.text('© Baran Çakı 2025')
st.header("AI ile Web Sitesi Analizi")

with st.expander("AI ile analiz yapmak için tıklayın"):
    ai_url = st.text_input("İncelemek istediğiniz web sitesi URL’sini girin:", key="ai_url")
    parse_description = st.text_area("AI’ya neyi analiz etmesini istiyorsunuz? (örneğin: 'Sayfadaki tüm şirket isimlerini listele')", key="parse_description")

    if st.button("AI ile Analiz Et"):
        if not ai_url or not parse_description:
            st.warning("Lütfen hem URL hem de analiz isteğini girin.")
        else:
            with st.spinner("AI ile analiz ediliyor..."):
                cleaned_text = get_clean_text_from_url(ai_url)
                result = parse_with_ollama(cleaned_text, parse_description)

            st.success("Analiz tamamlandı!")
            st.subheader("Sonuç")
            st.text_area("AI Cevabı:", result, height=300)

st.text('© Baran Çakı 2025')