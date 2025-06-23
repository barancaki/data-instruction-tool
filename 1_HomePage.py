import streamlit as st

st.set_page_config(page_title="Home Page")

st.header(":red[Scrape] , :blue[AI] ve :green[Excel Tool] Uygulamasına Hoş Geldin!")

st.markdown("""
Hoş geldiniz! Bu uygulama üç temel işlev sunar:

1.  Web sitelerinden katılımcı listesini otomatik tarama
2.  AI destekli analiz yaparak web sitelerinden bilgi çıkarımı
3.  Excel formatındaki dosyaları karşılaştırma ve eşleştirme

Sol menüden app sayfasına geçerek başlayabilir veya Kullanım'a geçerek bilgi edinebilirsiniz.

""")

st.link_button("Go to github", "https://github.com/barancaki")
st.markdown('---')
st.link_button('Fuar Scraper için',"Fuar_Scraper")
st.link_button('AI Scraper için',"AI_Scraper")
st.link_button('Kullanım için',"Kullanim")
st.link_button('Excel Tool için',"Excel_Tool")

st.text('© Baran Çakı 2025')