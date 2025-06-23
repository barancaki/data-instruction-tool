from scrape import scrape_replast_all_pages,scrape_win_eurasia_all_pages,scrape_packaging_fair
import streamlit as st

st.set_page_config(page_title="Fuar Scraper", layout="centered")
st.sidebar.header("Fuar Scraper Aracina hos geldiniz !")
st.sidebar.markdown('''Aracımız iki kısımdan oluşmaktadır.
                    
İlk kısımda istediğiniz bir fuarcılık sitesinden web scraping ile tablo elde etmenizi sağlar.
                    
İkinci kısımda ise istediğiniz bir url girerek o urldeki herhangi bir websitesi hakkında bilgi edinmenizi sağlar.
                    
Daha fazla bilgi edinmek için Kullanım sayfasına geçebilirsiniz.''')
st.sidebar.text('© Baran Çakı 2025')

st.header("_Fuar_ scrape :blue[aracı] ✅")

url = st.text_input("Tarama yapmak istediginiz fuarcılık URL’sini giriniz :")

if url in ["https://replasteurasia.com/katilimci-listesi", "https://replasteurasia.com/katilimci-listesi?page=1"]:
    if st.button("Tara"):
        with st.spinner("Sayfalar taranıyor..."):
            scrape_replast_all_pages(url)
        st.success("Tarama tamamlandı!")

if url in ["https://platform.win-eurasia.com/participants?page=1", "https://platform.win-eurasia.com/participants","https://platform.win-eurasia.com/participants?new","https://platform.win-eurasia.com/participants?new&lang=tr"]:
    sayfa_sayisi = st.text_input("Kaçıncı sayfaya kadar scrape etmek istiyorsunuz ?")
    if sayfa_sayisi:
        if st.button("Tara"):
            with st.spinner("Sayfalar taranıyor..."):
                scrape_win_eurasia_all_pages(url,int(sayfa_sayisi))
            st.success("Tarama tamamlandı!")

if url in ["https://packagingfair.com/katilimci-listesi"]:
    sayfa_sayisi = st.text_input("Kaçıncı sayfaya kadar scrape etmek istiyorsunuz ?")
    if sayfa_sayisi:
        if st.button("Tara"):
            with st.spinner("Sayfalar taranıyor..."):
                scrape_packaging_fair(int(sayfa_sayisi))
            st.success("Tarama tamamlandı!")

st.text('© Baran Çakı 2025')