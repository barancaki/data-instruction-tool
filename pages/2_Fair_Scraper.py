from Scrape.tuyap_sablon1 import scrape_burtarim_fair,scrape_replast_all_pages,scrape_pencere_all_pages,scrape_smtech_eurasia_all_pages,scrape_iplikfuari_all_pages,scrape_maktek_all_pages
from Scrape.deustche_messe_sablon import scrape_win_eurasia_all_pages,scrape_how_all_pages,scrape_sodex_all_pages,scrape_automechanika_all_pages
from Scrape.tuyap_sablon2 import scrape_packaging_fair,scrape_plast_eurasia_all_pages,scrape_intermob_all_pages,scrape_woodtech_all_pages,scrape_texhibitionist_all_pages,scrape_bauma_all_exhibitors
from Scrape.bagimsiz_sablonlar import scrape_evchargeshow,scrape_atechfuari,scrape_hvacr_world
from Scrape.ai_created import scrape_advanced_engineering, scrape_mesago, scrape_gitex_africa_morocco, scrape_yasad_uyeler
from Scrape.C_sablon import scrape_kalitefuari,scrape_mobisadimex
from Scrape.scrape_innotrans import scrape_innotrans,save_to_sqlite,create_mysql_dump_from_sqlite
from Scrape.a_sablon import scrape_enosad_proses_all_members,scrape_enosad_fabrika_all_members,scrape_enosad_robotik_all_members,scrape_enosad_sanayi_all_members,scrape_roboder_all_members
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

if url in ["https://www.konyatarimfuari.com/katilimci-listesi", "https://www.konyatarimfuari.com/katilimci-listesi?page=1"]:
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

if url in ["https://www.texhibitionist.com/katilimcilar"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_texhibitionist_all_pages(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://exhibitors.bauma.de/en/exhibitors-and-products/exhibitors-brand-names"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_bauma_all_exhibitors(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://enosad.org.tr/tr/proses-otomasyonu"]:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_enosad_proses_all_members()
            st.success("The scan is complete!")
        
if url in ["https://enosad.org.tr/tr/fabrika-otomasyonu"]:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_enosad_fabrika_all_members()
            st.success("The scan is complete!")

if url in ["https://enosad.org.tr/tr/robotik-ve-mekatronik"]:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_enosad_robotik_all_members()
            st.success("The scan is complete!")

if url in ["https://enosad.org.tr/tr/sanayide-dijital-donusum"]:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_enosad_sanayi_all_members()
            st.success("The scan is complete!")

if url in ["https://uyeler.roboder.org.tr/"]:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_roboder_all_members()
            st.success("The scan is complete!")

if url in ["https://www.innotrans.de/en/visit/exhibitor-directory#/search/f=h-entity_orga;v_sg=0;v_fg=0;v_fpa=FUTURE"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_innotrans()
            st.success("The scan is complete!")

if url in ["https://exhibitors.hvacr-world.com/hvacr-world-2025/Exhibitor"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                scrape_hvacr_world(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://www.advancedengineeringuk.com/exhibitors/", "https://www.advancedengineeringuk.com/exhibitors"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                st.session_state['function_name'] = 'advanced_engineering'
                scrape_advanced_engineering(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://sps.mesago.com/nuernberg/en/exhibitor-search.html?page=1&pagesize=30", "https://sps.mesago.com/nuernberg/en/exhibitor-search.html"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                st.session_state['function_name'] = 'mesago'
                scrape_mesago(int(sayfa_sayisi))
            st.success("The scan is complete!")

if url in ["https://exhibitors.gitexafrica.com/gitex-africa-2025/Exhibitor"]:
    scroll_sayisi = st.text_input("How many times do you want to scroll?")
    if scroll_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                st.session_state['function_name'] = 'gitex_africa_morocco'
                scrape_gitex_africa_morocco(int(scroll_sayisi))
            st.success("The scan is complete!")


if url in ["https://www.yasad.org.tr/uyelerimiz/", "https://www.yasad.org.tr/uyelerimiz"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                st.session_state['function_name'] = 'yasad_uyeler'
                scrape_yasad_uyeler(int(sayfa_sayisi))
            st.success("The scan is complete!")

st.text('© Baran Çakı 2025')