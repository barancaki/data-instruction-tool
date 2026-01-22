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
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException

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
    query = urllib.parse.quote(f"{firma_adi} official website")
    url = f"https://www.bing.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    print(f"🔍 Bing araması yapılıyor: {firma_adi}")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        # Farklı selector'ları dene
        selectors = [
            "li.b_algo h2 a",
            "div.b_algo h2 a",
            "a[href^='http']",
            "cite"
        ]

        for selector in selectors:
            for result in soup.select(selector):
                link = result.get("href")
                if result.name == "cite":
                    continue

                if not link or not link.startswith("http"):
                    continue

                # ❌ Sosyal medya linklerini geç
                yasakli = ["facebook.com", "linkedin.com", "instagram.com", "twitter.com", "x.com", "trendyol", "hepsiburada", "amazon", "ebay", "youtube.com", "tiktok.com"]
                if any(kelime in link.lower() for kelime in yasakli):
                    continue

                # ❌ Bing ve Microsoft linklerini geç
                if "bing.com" in link.lower() or "microsoft.com" in link.lower():
                    continue

                print(f"🌐 Seçilen link: {link}")
                return link

        print("❌ Bing sonucu bulunamadı.")
        return None
    except Exception as e:
        print(f"❌ Bing arama hatası: {e}")
        return None

def google_ilk_link_al(firma_adi):
    query = urllib.parse.quote(f"{firma_adi} official website")
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    print(f"🔍 Google araması yapılıyor: {firma_adi}")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        # Google sonuçları için selector'lar
        selectors = [
            "div.g a",
            "a[href]",
            "cite"
        ]

        for selector in selectors:
            for result in soup.select(selector):
                link = result.get("href")
                if result.name == "cite":
                    continue

                if not link or not link.startswith("http"):
                    continue

                # ❌ Google linklerini geç
                if "google.com" in link.lower() or "google.co" in link.lower():
                    continue

                # ❌ Sosyal medya linklerini geç
                yasakli = ["facebook.com", "linkedin.com", "instagram.com", "twitter.com", "x.com", "trendyol", "hepsiburada", "amazon", "ebay", "youtube.com", "tiktok.com"]
                if any(kelime in link.lower() for kelime in yasakli):
                    continue

                print(f"🌐 Seçilen link: {link}")
                return link

        print("❌ Google sonucu bulunamadı.")
        return None
    except Exception as e:
        print(f"❌ Google arama hatası: {e}")
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

    # --- YARDIMCI FONKSİYONLAR (DuckDuckGo ve Gelişmiş Email Arama) ---

def duckduckgo_search_selenium(driver, firma_adi):
    """DuckDuckGo'da firma adını arar ve ilk organik linki döner."""
    search_query = f"{firma_adi} official website"
    print(f"🦆 DDG aranıyor: {search_query}")
    
    try:
        driver.get("https://duckduckgo.com/")
        
        # Arama kutusunu bekle ve bul
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        search_box.clear()
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)
        
        # Sonuçların yüklenmesini bekle
        time.sleep(3)
        
        # Sonuç linklerini bul (DDG'nin yapısına göre selector)
        # Genellikle sonuçlar [data-testid="result-title-a"] içindedir
        results = driver.find_elements(By.CSS_SELECTOR, "a[data-testid='result-title-a']")

        yasakli = ["facebook.com", "linkedin.com", "instagram.com", "twitter.com", 
                   "x.com", "trendyol", "hepsiburada", "amazon", "ebay", 
                   "youtube.com", "tiktok.com", "duckduckgo.com", "indiamart", "tradeindia"]

        for res in results:
            try:
                link = res.get_attribute("href")
                if not link or not link.startswith("http"): continue
                
                # Yasaklı kelime kontrolü
                if any(y in link.lower() for y in yasakli):
                    continue
                
                print(f"🌐 Bulunan Link: {link}")
                return link
            except:
                continue
                
        print("❌ DDG sonucu bulunamadı.")
        return None
    except Exception as e:
        print(f"❌ DDG Arama Hatası: {e}")
        return None

def extract_emails_from_source(page_source):
    """Verilen HTML kaynağındaki email benzeri metinleri Regex ile bulur."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, page_source)
    
    valid_emails = []
    for mail in list(set(emails)):
        # Resim veya script dosyalarını email sanmasını engelle
        if mail.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.js', '.css', '.woff')):
            continue
        valid_emails.append(mail)
    return valid_emails

def find_email_advanced(driver, url):
    """
    Verilen URL'e gider, ana sayfayı tarar. 
    Bulamazsa 'Contact' veya 'About' sayfasına gidip orayı da tarar.
    """
    if not url: return ""
    
    found_emails = []
    
    try:
        print(f"🏠 Ana siteye gidiliyor: {url}")
        driver.set_page_load_timeout(20) # 20 saniye zaman aşımı
        
        try:
            driver.get(url)
        except TimeoutException:
            print("⚠️ Site çok yavaş, yükleme durduruldu. Mevcut içerik taranacak.")
            driver.execute_script("window.stop();")
        except Exception as e:
            print(f"❌ Siteye erişilemedi: {e}")
            return ""

        time.sleep(3) # Sayfanın kendine gelmesi için bekle
        
        # 1. AŞAMA: Ana Sayfa Taraması
        found_emails.extend(extract_emails_from_source(driver.page_source))
        
        if found_emails:
            print(f"📬 Ana sayfada bulundu: {found_emails[0]}")
            return found_emails[0]

        # 2. AŞAMA: İletişim Sayfası Arama
        print("🔎 Ana sayfada bulunamadı, 'Contact' linki aranıyor...")
        contact_keywords = ["contact", "iletişim", "about", "hakkımızda", "get in touch"]
        
        links = driver.find_elements(By.TAG_NAME, "a")
        contact_url = None
        
        for link in links:
            try:
                text = link.text.lower()
                href = link.get_attribute("href")
                if href and any(keyword in text for keyword in contact_keywords):
                    # Aynı sayfa içi linkleri (#) veya javascript linklerini ele
                    if "#" in href or "javascript" in href: continue
                    contact_url = href
                    break
            except:
                continue
                
        if contact_url:
            print(f"👉 İletişim sayfası bulundu, gidiliyor: {contact_url}")
            try:
                driver.get(contact_url)
                time.sleep(3)
                found_emails.extend(extract_emails_from_source(driver.page_source))
            except TimeoutException:
                 print("⚠️ İletişim sayfası zaman aşımı, mevcut içerik taranacak.")
                 driver.execute_script("window.stop();")
                 found_emails.extend(extract_emails_from_source(driver.page_source))
            except Exception as e:
                print(f"❌ İletişim sayfasına gidilemedi: {e}")

        if found_emails:
            # Tekrarları temizle ve ilkini döndür
            final_email = list(set(found_emails))[0]
            print(f"📬 İletişim sayfasında bulundu: {final_email}")
            return final_email
            
        print("❌ Email bulunamadı.")
        return ""
        
    except Exception as e:
        print(f"❌ Genel email tarama hatası: {e}")
        return ""