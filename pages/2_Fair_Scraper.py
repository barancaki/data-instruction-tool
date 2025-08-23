from Scrape.tuyap_sablon1 import scrape_burtarim_fair,scrape_replast_all_pages,scrape_pencere_all_pages,scrape_smtech_eurasia_all_pages,scrape_iplikfuari_all_pages,scrape_maktek_all_pages
from Scrape.deustche_messe_sablon import scrape_win_eurasia_all_pages,scrape_how_all_pages,scrape_sodex_all_pages,scrape_automechanika_all_pages
from Scrape.tuyap_sablon2 import scrape_packaging_fair,scrape_plast_eurasia_all_pages,scrape_intermob_all_pages,scrape_woodtech_all_pages
from Scrape.bagimsiz_sablonlar import scrape_evchargeshow,scrape_atechfuari
from Scrape.C_sablon import scrape_kalitefuari,scrape_mobisadimex
import streamlit as st
from auth_helper import check_authentication,get_user_info,show_user_info_sidebar

st.set_page_config(page_title="Fair Scraper", layout="centered")
# Authentication kontrolü
check_authentication()

# Kullanıcı bilgilerini al
user_info = get_user_info()

# Sidebar'da kullanıcı bilgilerini göster
show_user_info_sidebar()
st.sidebar.header("Welcome to the Fair Scraper tool!")

st.sidebar.text('© Baran Çakı 2025')

st.header("_Fair_ scrape :blue[tool] ✅")

url = st.text_input("Enter the URL you want to scan:")

if url in ["https://replasteurasia.com/katilimci-listesi", "https://replasteurasia.com/katilimci-listesi?page=1"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_replast_all_pages(url)
        st.success("The scan is complete!")

if url in ["https://platform.win-eurasia.com/participants?page=1", "https://platform.win-eurasia.com/participants","https://platform.win-eurasia.com/participants?new","https://platform.win-eurasia.com/participants?new&lang=tr"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_win_eurasia_all_pages(url,int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://packagingfair.com/katilimci-listesi"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_packaging_fair(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://www.burtarim.com/katilimci-listesi", "https://www.burtarim.com/katilimci-listesi?page=1"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_burtarim_fair(url)
        st.success("The scan is complete!")

if url in ["https://www.teknoparkistanbul.com.tr/firmalar"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            st.success("The scan is complete!")

if url in ["https://platform.hubofwarehouse.com/participants?page=1", "https://platform.hubofwarehouse.com/participants","https://platform.hubofwarehouse.com/participants?new","https://platform.hubofwarehouse.com/participants?new&lang=tr"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_how_all_pages(url,int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://platform.sodex.com.tr/participants?new&lang=en", "https://platform.sodex.com.tr/participants","https://platform.sodex.com.tr/participants?new","https://platform.sodex.com.tr/participants?page=1"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_sodex_all_pages(url,int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://automechanikaistanbulplus.com/participants?new&lang=en", "https://automechanikaistanbulplus.com/participants","https://automechanikaistanbulplus.com/participants?new","https://automechanikaistanbulplus.com/participants?page=1"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_automechanika_all_pages(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://www.avrasyapencerefuari.com/katilimci-listesi", "https://www.avrasyapencerefuari.com/katilimci-listesi?page=1"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_pencere_all_pages(url)
        st.success("The scan is complete!")

if url in ["https://smtech-eurasia.com/katilimci-listesi", "https://smtech-eurasia.com/katilimci-listesi?page=1"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_smtech_eurasia_all_pages(url)
        st.success("The scan is complete!")

if url in ["https://plasteurasia.com/katilimci-listesi"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_plast_eurasia_all_pages(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://www.intermobistanbul.com/katilimci-listesi"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_intermob_all_pages(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://www.woodtechistanbul.com/katilimci-listesi"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_woodtech_all_pages(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://expomedistanbul.com/katilimci-listesi", "https://expomedistanbul.com/katilimci-listesi?page=1"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_replast_all_pages(url)
        st.success("The scan is complete!")

if url in ["https://iplikfuari.com/katilimci-listesi", "https://iplikfuari.com/katilimci-listesi?page=1"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_iplikfuari_all_pages(url)
        st.success("The scan is complete!")

if url in ["https://maktekfuari.com/katilimci-listesi", "https://maktekfuari.com/katilimci-listesi?page=1"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_maktek_all_pages(url)
        st.success("The scan is complete!")

if url in ["https://www.evchargeshow.com/exhibitor"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_evchargeshow()
        st.success("The scan is complete!")

if url in ["https://atechfuari.com/firmalar/" , "https://atechfuari.com/firmalar"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_atechfuari()
        st.success("The scan is complete!")

if url in ["https://kalitefuari.com/katilimci-listesi/","https://kalitefuari.com/katilimci-listesi"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_kalitefuari()
        st.success("The scan is complete!")

if url in ["https://www.mobisadimex.com/2024-katilimci-listesi","https://www.mobisadimex.com/2024-katilimci-listesi/"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            scrape_mobisadimex()
        st.success("The scan is complete!")
st.text('© Baran Çakı 2025')