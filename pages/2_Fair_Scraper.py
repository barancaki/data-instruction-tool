from Scrape.tuyap_sablon1 import scrape_burtarim_fair,scrape_replast_all_pages,scrape_pencere_all_pages,scrape_smtech_eurasia_all_pages,scrape_iplikfuari_all_pages,scrape_maktek_all_pages
from Scrape.deustche_messe_sablon import scrape_win_eurasia_all_pages,scrape_how_all_pages,scrape_sodex_all_pages,scrape_automechanika_all_pages
from Scrape.tuyap_sablon2 import scrape_packaging_fair,scrape_plast_eurasia_all_pages,scrape_intermob_all_pages,scrape_woodtech_all_pages,scrape_texhibitionist_all_pages,scrape_bauma_all_exhibitors
from Scrape.bagimsiz_sablonlar import scrape_evchargeshow,scrape_atechfuari,scrape_hvacr_world
from Scrape.ai_created import scrape_advanced_engineering, scrape_mesago, scrape_gitex_africa_morocco, scrape_yasad_uyeler, scrape_logimat, scrape_acrex_india, scrape_aquatherm_tashkent, scrape_ifat_exhibitors, scrape_ahri_members, scrape_warsaw_hvac_expo, scrape_ptc_asia, scrape_mca_world_fair, scrape_logimotion, scrape_gitex

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

if url in ["https://www.logimat-messe.de/en/fair/exhibitor-directory", "https://www.logimat-messe.de/en/fair/exhibitor-directory#/search/f=h-entity_orga;v_sg=0;v_fg=0;v_fpa=FUTURE"]:
    show_more_sayisi = st.text_input("How many times do you want to click 'Show More'?")
    if show_more_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                st.session_state['function_name'] = 'logimat'
                scrape_logimat(int(show_more_sayisi))
            st.success("The scan is complete!")

if url in ["https://acrex.in/ExhibitorList-2026"]:
    if st.button("Scan"):
        with st.spinner("Pages are being scanned..."):
            if 'scraping_in_progress' not in st.session_state:
                st.session_state['scraping_in_progress'] = False
            if not st.session_state['scraping_in_progress']:
                st.session_state['scraping_in_progress'] = True
                st.session_state['function_name'] = 'acrex_india'
                scrape_acrex_india()
                st.session_state['scraping_in_progress'] = False
            else:
                st.warning("Scraping already in progress...")
        st.success("The scan is complete!")

if "aquatherm-tashkent.uz" in url:
    st.info("Target: Aquatherm Tashkent Exhibitor List")
    
    # Kullanıcıdan sayfa sayısı alma
    page_count_input = st.text_input("How many pages do you want to scan?", value="5")
    
    if page_count_input:
        if st.button("Start Scan"):
            with st.spinner("Scanning pages and extracting emails..."):
                st.session_state['function_name'] = 'aquatherm'
                
                # Fonksiyonu çağır (Tam sayıya çevirmeyi unutma)
                scrape_aquatherm_tashkent(int(page_count_input))
            
            st.balloons() # İşlem bitince konfeti (Opsiyonel)

if "ifat.de" in url:
    st.info("Target: IFAT Munich Exhibitor List (Load More Structure)")
    
    # Kullanıcıdan 'Load More' sayısı alma
    load_more_input = st.number_input("How many times should 'Show More' be clicked? (Approx. 20 companies per click)", min_value=0, value=5, step=1)
    
    if st.button("Start Scan"):
        with st.spinner("Initializing scraper..."):
            st.session_state['function_name'] = 'ifat'
            
            # Fonksiyonu çalıştır
            scrape_ifat_exhibitors(int(load_more_input))
        
        st.balloons()

if "ahrinet.org" in url:
    st.info("Target: AHRI Members List")
    
    if st.button("Start Scan"):
        with st.spinner("Scanning pages and extracting emails..."):
            st.session_state['function_name'] = 'ahri_members'
            scrape_ahri_members()
        
        st.balloons()

if "warsawhvacexpo.com" in url:
    st.info("Target: Warsaw HVAC Expo Exhibitors Catalog")
    
    # Kullanıcıdan 'Load More' sayısı alma
    load_more_input = st.number_input("How many times should 'Load More' be clicked?", min_value=0, value=5, step=1)
    
    if st.button("Start Scan"):
        with st.spinner("Scanning pages and extracting company data..."):
            st.session_state['function_name'] = 'warsaw_hvac_expo'
            scrape_warsaw_hvac_expo(int(load_more_input))
        
        st.balloons()

if "ptc-asia.com" in url or "service.ptc-asia.com" in url:
    st.info("Target: PTC Asia (Power Transmission and Control) Exhibitors")
    
    # Kullanıcıdan sayfa sayısı alma
    page_count = st.number_input("How many pages to scrape?", min_value=1, value=5, step=1)
    
    if st.button("Start Scan"):
        with st.spinner("Scanning pages and extracting company data..."):
            st.session_state['function_name'] = 'ptc_asia'
            scrape_ptc_asia(int(page_count))
        
        st.balloons()

if "mcaworldfair.com" in url:
    st.info("Target: MCA World Fair Katılımcılar")
    
    if st.button("Start Scan"):
        with st.spinner("Extracting company names and searching for emails through their websites..."):
            st.session_state['function_name'] = 'mca_world_fair'
            scrape_mca_world_fair()
        
        st.balloons()

if url in ["https://logimotion.ae.messefrankfurt.com/dubai/en/exhibitor-search.html?page=1&pagesize=30", "https://logimotion.ae.messefrankfurt.com/dubai/en/exhibitor-search.html"]:
    sayfa_sayisi = st.text_input("How many pages do you want to scrape?")
    if sayfa_sayisi:
        if st.button("Scan"):
            with st.spinner("Pages are being scanned..."):
                st.session_state['function_name'] = 'logimotion'
                scrape_logimotion(int(sayfa_sayisi))
            st.success("The scan is complete!")

if "exhibitors.gitex.com" in url:
    st.info("Target: Gitex Global Dubai")
    scroll_count = st.number_input("How many times to scroll (load more)?", min_value=1, value=5)
    
    if st.button("Scan"):
        with st.spinner("Scanning..."):
            st.session_state['function_name'] = 'gitex_global'
            scrape_gitex(scroll_count)
        st.success("Scan complete!")
        st.balloons()

st.text('© Baran Çakı 2025')
