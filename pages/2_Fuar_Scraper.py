from Scrape.tuyap_sablon1 import scrape_burtarim_fair,scrape_replast_all_pages,scrape_pencere_kapi_cam_all_pages,scrape_smtech_eurasia_all_pages,scrape_iplikfuari_all_pages,scrape_maktek_all_pages
from Scrape.deustche_messe_sablon import scrape_win_eurasia_all_pages,scrape_how_all_pages,scrape_sodex_all_pages,scrape_automechanika_all_pages
from Scrape.tuyap_sablon2 import scrape_packaging_fair,scrape_plast_eurasia_all_pages,scrape_intermob_all_pages,scrape_woodtech_all_pages
import streamlit as st

st.set_page_config(page_title="Fuar Scraper", layout="centered")
st.sidebar.header("Fuar Scraper Aracina hos geldiniz !")
st.sidebar.markdown('''Fuar Scraper Tool Kitinde Aktif Olan Siteler :

- Replast Eurasia
                    
- Win Eurasia
                    
- Ambalaj Fuarı (Packaging Fair)
                    
- Burtarım Fuarı
                    
- Teknopark İstanbul (Aktif fakat Cloudflare korumalı site!)
                    
- Hub Of Warehouse
                    
- SODEX
                    
- AUTOMECHANİKA
                    
- Pencere Kapı Cam Fuarı
                    
- BUMATECH (SMTECH)
                    
- PLAST Eurasia
                    
- İNTERMOB Fuarı
                    
- Woodtech İstanbul Fuarı
                    
- Expomed Fuarı
                    
- İPLİK Fuarı
                    
- MAKTEK Fuarı''')
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

if url in ["https://www.burtarim.com/katilimci-listesi", "https://www.burtarim.com/katilimci-listesi?page=1"]:
    if st.button("Tara"):
        with st.spinner("Sayfalar taranıyor..."):
            scrape_burtarim_fair(url)
        st.success("Tarama tamamlandı!")

if url in ["https://www.teknoparkistanbul.com.tr/firmalar"]:
    if st.button("Tara"):
        with st.spinner("Sayfalar taranıyor..."):
            st.success("Tarama tamamlandı!")

if url in ["https://platform.hubofwarehouse.com/participants?page=1", "https://platform.hubofwarehouse.com/participants","https://platform.hubofwarehouse.com/participants?new","https://platform.hubofwarehouse.com/participants?new&lang=tr"]:
    sayfa_sayisi = st.text_input("Kaçıncı sayfaya kadar scrape etmek istiyorsunuz ?")
    if sayfa_sayisi:
        if st.button("Tara"):
            with st.spinner("Sayfalar taranıyor..."):
                scrape_how_all_pages(url,int(sayfa_sayisi))
            st.success("Tarama tamamlandı!")

if url in ["https://platform.sodex.com.tr/participants?new&lang=en", "https://platform.sodex.com.tr/participants","https://platform.sodex.com.tr/participants?new","https://platform.sodex.com.tr/participants?page=1"]:
    sayfa_sayisi = st.text_input("Kaçıncı sayfaya kadar scrape etmek istiyorsunuz ?")
    if sayfa_sayisi:
        if st.button("Tara"):
            with st.spinner("Sayfalar taranıyor..."):
                scrape_sodex_all_pages(url,int(sayfa_sayisi))
            st.success("Tarama tamamlandı!")

if url in ["https://automechanikaistanbulplus.com/participants?new&lang=en", "https://automechanikaistanbulplus.com/participants","https://automechanikaistanbulplus.com/participants?new","https://automechanikaistanbulplus.com/participants?page=1"]:
    sayfa_sayisi = st.text_input("Kaçıncı sayfaya kadar scrape etmek istiyorsunuz ?")
    if sayfa_sayisi:
        if st.button("Tara"):
            with st.spinner("Sayfalar taranıyor..."):
                scrape_automechanika_all_pages(int(sayfa_sayisi))
            st.success("Tarama tamamlandı!")

if url in ["https://www.avrasyapencerefuari.com/katilimci-listesi", "https://www.avrasyapencerefuari.com/katilimci-listesi?page=1"]:
    if st.button("Tara"):
        with st.spinner("Sayfalar taranıyor..."):
            scrape_pencere_kapi_cam_all_pages(url)
        st.success("Tarama tamamlandı!")

if url in ["https://smtech-eurasia.com/katilimci-listesi", "https://smtech-eurasia.com/katilimci-listesi?page=1"]:
    if st.button("Tara"):
        with st.spinner("Sayfalar taranıyor..."):
            scrape_smtech_eurasia_all_pages(url)
        st.success("Tarama tamamlandı!")

if url in ["https://plasteurasia.com/katilimci-listesi"]:
    sayfa_sayisi = st.text_input("Kaçıncı sayfaya kadar scrape etmek istiyorsunuz ?")
    if sayfa_sayisi:
        if st.button("Tara"):
            with st.spinner("Sayfalar taranıyor..."):
                scrape_plast_eurasia_all_pages(int(sayfa_sayisi))
            st.success("Tarama tamamlandı!")

if url in ["https://www.intermobistanbul.com/katilimci-listesi"]:
    sayfa_sayisi = st.text_input("Kaçıncı sayfaya kadar scrape etmek istiyorsunuz ?")
    if sayfa_sayisi:
        if st.button("Tara"):
            with st.spinner("Sayfalar taranıyor..."):
                scrape_intermob_all_pages(int(sayfa_sayisi))
            st.success("Tarama tamamlandı!")

if url in ["https://www.woodtechistanbul.com/katilimci-listesi"]:
    sayfa_sayisi = st.text_input("Kaçıncı sayfaya kadar scrape etmek istiyorsunuz ?")
    if sayfa_sayisi:
        if st.button("Tara"):
            with st.spinner("Sayfalar taranıyor..."):
                scrape_woodtech_all_pages(int(sayfa_sayisi))
            st.success("Tarama tamamlandı!")

if url in ["https://expomedistanbul.com/katilimci-listesi", "https://expomedistanbul.com/katilimci-listesi?page=1"]:
    if st.button("Tara"):
        with st.spinner("Sayfalar taranıyor..."):
            scrape_replast_all_pages(url)
        st.success("Tarama tamamlandı!")

if url in ["https://iplikfuari.com/katilimci-listesi", "https://iplikfuari.com/katilimci-listesi?page=1"]:
    if st.button("Tara"):
        with st.spinner("Sayfalar taranıyor..."):
            scrape_iplikfuari_all_pages(url)
        st.success("Tarama tamamlandı!")

if url in ["https://maktekfuari.com/katilimci-listesi", "https://maktekfuari.com/katilimci-listesi?page=1"]:
    if st.button("Tara"):
        with st.spinner("Sayfalar taranıyor..."):
            scrape_maktek_all_pages(url)
        st.success("Tarama tamamlandı!")
st.text('© Baran Çakı 2025')