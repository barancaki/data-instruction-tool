from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import streamlit as st
import time
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from webdriver_manager.chrome import ChromeDriverManager
from table_to_sql import save_to_sqlite,create_mysql_dump_from_sqlite

def kutuphane():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import pandas as pd
    import streamlit as st
    import time
    import plotly.express as px
    import requests
    from bs4 import BeautifulSoup
    import re
    import urllib.parse
    from webdriver_manager.chrome import ChromeDriverManager
    from table_to_sql import save_to_sqlite,create_mysql_dump_from_sqlite
    from Scrape.scrape import bing_ilk_link_al,site_icerisinden_email_bul

def google_ilk_link_manual(firma_adi):
    query = urllib.parse.quote(firma_adi)
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers, timeout=10)

    # Debug için dosyaya yaz
    with open("google_result.html", "w", encoding="utf-8") as f:
        f.write(response.text)

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.startswith("/url?q="):
            clean_link = href.split("/url?q=")[1].split("&")[0]
            print(f"🔗 Bulunan URL: {clean_link}")
            return clean_link

    print("❌ Hiçbir uygun link bulunamadı.")
    return None

def bing_ilk_link_al(firma_adi):
    query = urllib.parse.quote(f"{firma_adi} resmi web sitesi")
    url = f"https://www.bing.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    print(f"🔍 Bing araması yapılıyor: {firma_adi}")
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Sonuçları gez
    for result in soup.select("li.b_algo h2 a"):
        link = result.get("href")
        if not link or not link.startswith("http"):
            continue

        # ❌ Sosyal medya linklerini geç
        yasakli = ["facebook.com", "linkedin.com", "instagram.com", "twitter.com", "trendyol", "hepsiburada", "amazon"]
        if any(kelime in link for kelime in yasakli):
            continue

        print(f"🌐 Seçilen link: {link}")
        return link

    print("❌ Bing sonucu bulunamadı.")
    return None

def site_icerisinden_email_bul(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("window-size=1200,800")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        print(f"🌍 Siteye gidiliyor: {url}")
        driver.get(url)
        time.sleep(2)

        page_text = driver.page_source
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        email_list = re.findall(email_pattern, page_text)

        benzersiz_mailler = list(set(email_list))
        print(f"📬 Bulunan E-postalar: {benzersiz_mailler}")
        return benzersiz_mailler

    except Exception as e:
        print("❌ Site içeriği alınamadı:", e)
        return []

    finally:
        driver.quit()

    # 🔁 Kullanım:
    # firma = "2D Kimya"
    # url = google_ilk_link_manual(firma)
    # if url:
    #     site_icerisinden_email_bul(url)