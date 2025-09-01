# 3_AI_Scraper.py

import os
import asyncio
import streamlit as st
# --- 3_AI_Scraper.py (ek import) ---
from ai_scripts.auto_exhibitor_crawler import AutoExhibitorCrawler  # NEW

# Ollama (lokal) tarafı
from ai_scripts.ollama_parser import (
    get_clean_texts_from_urls as get_clean_texts_ollama,
    parse_with_ollama,
)

# Google (Gemini) tarafı
from ai_scripts.google_parser import (
    get_clean_texts_from_urls as get_clean_texts_google,
    parse_with_google,
)

from auth_helper import check_authentication, get_user_info, show_user_info_sidebar


# -------------------------
# Yardımcı: Streamlit'te güvenli async çalıştırma
# -------------------------
def run_async(coro):
    """
    Streamlit çalışırken bazen asyncio.run() mevcut event loop nedeniyle hata verebilir.
    Bu yardımcı, güvenli şekilde coroutine'i çalıştırır.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# -------------------------
# Authentication
# -------------------------
check_authentication()
user_info = get_user_info()
show_user_info_sidebar()

# -------------------------
# Sidebar Bilgi
# -------------------------
st.sidebar.header("Welcome to the AI Scraper tool!")
st.sidebar.markdown(
    '''## 🤖 Website Analysis with AI

Analyze one or more web pages and extract exactly what you need with AI.

### Engines:
- **Local (Ollama)** → Model: *gpt-oss:20b* (runs locally)
- **Google API (Gemini)** → Models: *gemini-1.5-flash* (faster), *gemini-1.5-pro* (more accurate)

### Steps:
1. Enter one or more page URLs (comma-separated).  
2. Enter a request in English in the analysis section.  
3. Choose the AI Engine (Local or Google API).  
4. Click **Analyze**.

**Examples**  
- List all company names and countries on the page.  
- Extract all contact email addresses.  
- Summarize the product categories.  

ℹ️ The analysis prompt must be written in English only.
'''
)
st.sidebar.text('© Baran Çakı 2025')

# -------------------------
# Main UI
# -------------------------
st.header("Website Analysis with AI")

with st.expander("Click to analyze with AI", expanded=True):
    # Motor seçimi
    engine = st.radio(
        "Choose AI Engine:",
        options=["Local (Ollama)", "Google API (Gemini)"],
        horizontal=True,
        key="engine_choice",
    )

    # Google seçildiyse ek alanlar
    google_api_key_default = ""
    try:
        google_api_key_default = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        google_api_key_default = os.getenv("GOOGLE_API_KEY", "")

    google_api_key = None
    gemini_model = None
    insecure_hosts_input = ""

    if engine == "Google API (Gemini)":
        google_api_key = st.text_input(
            "Google API Key",
            value=google_api_key_default,
            type="password",
            help="You can also set it via st.secrets['GOOGLE_API_KEY'] or env var GOOGLE_API_KEY.",
            key="google_api_key",
        )
        gemini_model = st.selectbox(
            "Gemini Model",
            options=["gemini-1.5-flash", "gemini-1.5-pro"],
            index=0,
            help="Select 'flash' for speed or 'pro' for accuracy.",
            key="gemini_model",
        )
        insecure_hosts_input = st.text_input(
            "Optional: Domains to bypass SSL verification (comma-separated)",
            value="",
            help=(
                "Use only if a site has broken/invalid SSL chain but you trust it."
                " Example: packagingfair.com,www.packagingfair.com"
            ),
            key="insecure_hosts",
        )

    # URL ve prompt alanları
    ai_urls = st.text_area(
        "Enter the URL(s) of the website you want to review (comma separated):",
        key="ai_urls",
        placeholder="https://example.com/page1, https://example.com/page2",
    )
    parse_description = st.text_area(
        "What do you want AI to analyze? (e.g., “List all company names on the page”)",
        key="parse_description",
        placeholder="List all product names and their prices if available.",
        height=120,
    )

    # Analyze butonu
    if st.button("Analyze with AI", type="primary"):
        # Basit doğrulamalar
        if not ai_urls or not parse_description:
            st.warning("Please enter both the URL(s) and the analysis request.")
            st.stop()

        if engine == "Google API (Gemini)" and not (google_api_key or google_api_key_default):
            st.warning("Please enter your Google API Key or set it as GOOGLE_API_KEY in secrets/env.")
            st.stop()

        # URL'leri hazırla
        urls = [u.strip() for u in ai_urls.split(",") if u.strip()]

        # Google için insecure hosts setini hazırla
        insecure_hosts = set()
        if engine == "Google API (Gemini)" and insecure_hosts_input.strip():
            insecure_hosts = {h.strip().lower() for h in insecure_hosts_input.split(",") if h.strip()}

        with st.spinner(f"Analyzing {len(urls)} page(s) with {engine}..."):
            try:
                if engine == "Local (Ollama)":
                    # HTML fetch & temizleme
                    cleaned_texts = run_async(get_clean_texts_ollama(urls))
                    # AI parse
                    result = parse_with_ollama(cleaned_texts, parse_description)

                else:
                    # Google: fetch (güvenli + opsiyonel domain bazlı bypass)
                    effective_key = google_api_key or google_api_key_default
                    cleaned_texts = run_async(get_clean_texts_google(urls, insecure_hosts=insecure_hosts))
                    # AI parse
                    result = parse_with_google(
                        cleaned_texts=cleaned_texts,
                        parse_description=parse_description,
                        api_key=effective_key,
                        model_name=gemini_model or "gemini-1.5-flash",
                    )

                st.success("The analysis is complete!")
                st.subheader("Result")
                st.text_area("AI Answer:", result, height=320)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.stop()

# 🧪 BETA: Auto Exhibitor Crawler
with st.expander("🧪 Beta: Auto Exhibitor Crawler", expanded=False):

    st.markdown("""
**Amaç:** Bir *listing* sayfasından katılımcı/marka/kart detaylarını otomatik toplayıp (gerekirse browser fallback ile),
elde edilen **temiz metinleri** aynı sayfada **AI ile analiz etmek**.
    """)

    # Crawler parametreleri
    listing_url = st.text_input("Listing URL", key="aec_listing_url", placeholder="https://example.com/katilimci-listesi")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_items = st.number_input("Max items", min_value=1, value=50, step=1, key="aec_max_items")
        concurrent_limit = st.number_input("Concurrent limit", min_value=1, value=8, step=1, key="aec_concurrency")
    with col2:
        per_request_timeout = st.number_input("Per-request timeout (sec)", min_value=5, value=30, step=5, key="aec_timeout")
        delay_between_requests = st.number_input("Delay between requests (sec)", min_value=0.0, value=0.0, step=0.1, key="aec_delay")
    with col3:
        headless = st.checkbox("Headless browser", value=True, key="aec_headless")
        scroll_times = st.number_input("Scroll times (fallback)", min_value=0, value=4, step=1, key="aec_scroll_times")

    col4, col5 = st.columns(2)
    with col4:
        scroll_wait = st.number_input("Scroll wait (sec)", min_value=0.0, value=0.8, step=0.1, key="aec_scroll_wait")
    with col5:
        wait_after_click = st.number_input("Wait after click (sec)", min_value=0.0, value=0.6, step=0.1, key="aec_wait_after_click")

    insecure_hosts_raw = st.text_input(
        "SSL doğrulamasını bypass edeceğin domainler (virgül ile)",
        value="",
        help="Sadece zinciri bozuk ama güvendiğin siteler için. Örn: packagingfair.com,www.packagingfair.com",
        key="aec_insecure_hosts",
    )
    insecure_hosts = {h.strip().lower() for h in insecure_hosts_raw.split(",") if h.strip()}

    st.divider()

    # Çalıştırma modu
    mode = st.radio(
        "Çalıştırma modu",
        options=["Sadece Crawler", "Crawler + AI Analizi"],
        horizontal=True,
        key="aec_mode",
    )

    # AI seçenekleri (Bu beta bölüme özel - bağımsız)
    if mode == "Crawler + AI Analizi":
        engine_beta = st.radio(
            "AI Engine (Beta bölüm)",
            options=["Local (Ollama)", "Google API (Gemini)"],
            horizontal=True,
            key="aec_engine_choice",
        )
        parse_description_beta = st.text_area(
            "AI'ya neyi çıkarmasını istiyorsun? (İngilizce yaz)",
            key="aec_parse_description",
            placeholder="List all exhibitor company names and their countries, one per line.",
            height=120,
        )

        google_api_key_default_beta = ""
        try:
            google_api_key_default_beta = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            google_api_key_default_beta = os.getenv("GOOGLE_API_KEY", "")

        google_api_key_beta = None
        gemini_model_beta = None
        if engine_beta == "Google API (Gemini)":
            google_api_key_beta = st.text_input(
                "Google API Key (Beta)",
                value=google_api_key_default_beta,
                type="password",
                help="st.secrets['GOOGLE_API_KEY'] veya env GOOGLE_API_KEY ile de çalışır.",
                key="aec_google_api_key",
            )
            gemini_model_beta = st.selectbox(
                "Gemini Model (Beta)",
                options=["gemini-1.5-flash", "gemini-1.5-pro"],
                index=0,
                help="Hız için 'flash', doğruluk için 'pro'.",
                key="aec_gemini_model",
            )

    # Çalıştır butonları
    col_run1, col_run2 = st.columns([1, 2])
    with col_run1:
        run_btn = st.button("ÇALIŞTIR", type="primary", key="aec_run")

    # Sonuç alanları
    if run_btn:
        if not listing_url:
            st.warning("Lütfen listing URL gir.")
            st.stop()

        with st.spinner("Crawler çalışıyor..."):
            try:
                crawler = AutoExhibitorCrawler(
                    max_items=max_items,
                    per_request_timeout=per_request_timeout,
                    delay_between_requests=delay_between_requests,
                    concurrent_limit=concurrent_limit,
                    insecure_hosts=insecure_hosts,
                    scroll_times=scroll_times,
                    scroll_wait=scroll_wait,
                    wait_after_click=wait_after_click,
                    headless=headless,
                )
                texts, meta = run_async(crawler.run(listing_url))
            except Exception as e:
                st.error(f"Crawler sırasında hata: {e}")
                st.stop()

        st.success(f"Crawler tamamlandı. Toplanan kayıt: {len(texts)}")
        # Metinler + meta gösterimi
        if meta:
            try:
                import pandas as pd  # sadece tablo için
                df_meta = pd.DataFrame(meta)
                st.dataframe(df_meta, use_container_width=True, height=300)
            except Exception:
                st.write(meta)

        # Metinleri inceleme
        with st.expander("Toplanan Metinler", expanded=False):
            for i, t in enumerate(texts, start=1):
                st.text_area(f"#{i} metin", t, height=200)

        # İndirme (JSON)
        try:
            import json
            payload = {"source": listing_url, "count": len(texts), "meta": meta, "texts": texts}
            st.download_button(
                "JSON indir",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name="crawler_output.json",
                mime="application/json",
            )
        except Exception:
            pass

        # İsteğe bağlı: AI Analizi
        if mode == "Crawler + AI Analizi":
            if not parse_description_beta:
                st.warning("AI analizi için bir açıklama (prompt) gir.")
                st.stop()

            with st.spinner(f"AI analizi çalışıyor ({engine_beta})..."):
                try:
                    if engine_beta == "Local (Ollama)":
                        # Ollama: doğrudan metinleri kullan
                        result_beta = parse_with_ollama(texts, parse_description_beta)
                    else:
                        effective_key_beta = google_api_key_beta or google_api_key_default_beta
                        if not effective_key_beta:
                            st.warning("Google API Key gerekli.")
                            st.stop()
                        result_beta = parse_with_google(
                            cleaned_texts=texts,
                            parse_description=parse_description_beta,
                            api_key=effective_key_beta,
                            model_name=gemini_model_beta or "gemini-1.5-flash",
                        )
                    st.subheader("AI Sonucu (Beta)")
                    st.text_area("AI Answer:", result_beta, height=320)
                except Exception as e:
                    st.error(f"AI analizi sırasında hata: {e}")

st.text('© Baran Çakı 2025')