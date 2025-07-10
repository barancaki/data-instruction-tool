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
import platform
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from webdriver_manager.chrome import ChromeDriverManager


def scrape_replast_all_pages(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    tablo = []
    page_num = 1

    while True:
        url = f"https://replasteurasia.com/katilimci-listesi?page={page_num}"
        driver.get(url)
        time.sleep(3)

        # Sayfadaki firma bloklarını al
        table = driver.find_elements(By.CLASS_NAME, "filter-list__item")

        # Sayfa boşsa döngüyü bitir
        if len(table) == 0:
            print(f"Veri bitti. Son sayfa: {page_num-1}")
            break

        print(f"{page_num}. sayfa işleniyor...")

        for item in table:
            try:
                firma_adi = item.find_element(By.XPATH, ".//div[@class='table-block-content'][1]").text
            except:
                firma_adi = " "

            try:
                adres = item.find_element(By.XPATH, ".//div[@class='table-block-content'][2]").text
                parcalar = adres.split("/")
                ulke = parcalar[-1].strip()
            except:
                adres = " "
                ulke = " "
            try:
                telefon = item.find_element(By.XPATH, ".//a[starts-with(@href, 'tel:')]").text
            except:
                telefon = " "
            try:
                site = item.find_element(By.XPATH, ".//a[starts-with(@href, 'http')]").get_attribute("href")
            except:
                site = " "
            try:
                # Butona tıkla
                detay_buton = item.find_element(By.CLASS_NAME, "js-open-table-detail")
                driver.execute_script("arguments[0].click();", detay_buton)
                time.sleep(0.5)  # açılma süresi

                # Ürün gruplarını listele
                urun_gruplari_liste = item.find_elements(By.CLASS_NAME, "table-detail-wrapper__list-item")
                urun_gruplari = ", ".join([li.text for li in urun_gruplari_liste])
            except:
                urun_gruplari = " "

            tablo.append({
                "Firma": firma_adi,
                "Adres": adres,
                "Ülke":ulke,
                "Telefon": telefon,
                "Web adresi": site,
                "Ürün Grupları": urun_gruplari,
                "Company Mail":"",
                "Company Zip-Code":""            
            })

        page_num += 1

    driver.quit()

    df = pd.DataFrame(tablo)

    if st:
        st.dataframe(df)
        ulke_sayilari = df["Ülke"].value_counts().reset_index()
        ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
        fig = px.bar(ulke_sayilari, x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
        st.plotly_chart(fig)
    else:
        print(df.head())



def scrape_win_eurasia_all_pages(url, sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)

    base_url = "https://platform.win-eurasia.com"
    tablo = []

    for page_num in range(1, sayfa_sayisi + 1):
        print(f"🔄 {page_num}. sayfa yükleniyor...")
        driver.get(f"{base_url}/participants?page={page_num}")
        time.sleep(2)

        # Tüm detay linklerini topla
        detay_linkleri = []
        firma_adi_listesi = []
        ulke_listesi = []

        firma_kartlari = driver.find_elements(By.CSS_SELECTOR, "div.cell.small-12")

        for kart in firma_kartlari:
            try:
                link_element = kart.find_element(By.CSS_SELECTOR, "a.o.link.as-block.fx.dropshadow.for-child")
                href = link_element.get_attribute("href")
                if href:
                    detay_link = href if href.startswith("http") else base_url + href
                else:
                    continue

                firma_adi = kart.find_element(By.CLASS_NAME, "search-snippet-name").get_attribute("innerText").strip()                
                ulke = kart.find_element(By.CLASS_NAME, "search-snippet-description").text.upper().strip()

                detay_linkleri.append(detay_link)
                firma_adi_listesi.append(firma_adi)
                ulke_listesi.append(ulke)

            except Exception as e:
                print(f"❌ Link/firma bilgisi alınamadı: {e}")
                continue

        # Her detay sayfasına gir ve veriyi çek
        for i, detay_link in enumerate(detay_linkleri):
            firma_adi = firma_adi_listesi[i]
            ulke = ulke_listesi[i]

            try:
                driver.get(detay_link)
                time.sleep(2)

                # Adres
                try:
                    adres_listesi = driver.find_elements(By.CSS_SELECTOR, "ul.t.set-250-regular.as-copy li")
                    adres = " ".join([li.text.strip() for li in adres_listesi])
                except:
                    adres = ""

                # Telefon
                try:
                    telefonlar = driver.find_elements(By.CSS_SELECTOR, "ul.t.set-250-regular.as-copy li a")
                    telefon = ""
                    for tel in telefonlar:
                        if "Telefon" in tel.text:
                            telefon = tel.text.replace("Telefon:", "").strip()
                            break
                except:
                    telefon = ""

                # Company Mail
                try:
                    mail_element = driver.find_element(By.CSS_SELECTOR, "a[href^='mailto:']")
                    email = mail_element.get_attribute("href").replace("mailto:", "").strip()
                except:
                    email = ""

                tablo.append({
                    "Firma": firma_adi,
                    "Ülke": ulke,
                    "Adres": adres,
                    "Telefon": telefon,
                    "Company Mail": email
                })

                print(f"✅ {firma_adi} eklendi.")

            except Exception as e:
                print(f"❌ {firma_adi} için detay sayfasına gidilemedi: {e}")
                continue

    driver.quit()

    df = pd.DataFrame(tablo)
    print("\n🎯 TOPLAM FİRMA SAYISI:", len(df))
    print(df.head())
    if st:
        st.dataframe(df)

def scrape_packaging_fair(sayfa_sayisi):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    base_url = "https://packagingfair.com/katilimci-listesi"
    full_url_prefix = "https://packagingfair.com"
    tablo = []

    for page_num in range(1, sayfa_sayisi + 1):
        page_url = f"{base_url}?page={page_num}"
        print(f"\n🔄 {page_num}. sayfa yükleniyor: {page_url}")
        driver.get(page_url)
        time.sleep(2)

        try:
            brand_elements = driver.find_elements(By.CSS_SELECTOR, "div.brand-item.mt-30.active a.brand-link")
            firma_linkleri = []
            for a_tag in brand_elements:
                href = a_tag.get_attribute("href")
                if href:
                    full_link = href if href.startswith("http") else full_url_prefix + "/" + href.lstrip("/")
                    firma_linkleri.append(full_link)
                    print(f"🔗 Firma linki bulundu: {full_link}")
        except Exception as e:
            print(f"❌ Sayfa {page_num} linkleri çekilemedi: {e}")
            continue

        # Her firma detayına gir
        for link in firma_linkleri:
            print(f"  🔍 Firma detayına giriliyor: {link}")
            try:
                driver.get(link)
                time.sleep(2)

                # Firma adı
                try:
                    firma_adi = driver.find_element(By.CSS_SELECTOR, "h1.company-title").text.strip()
                except:
                    firma_adi = ""

                # Ülke bilgisi
                try:
                    ulke_ikon = driver.find_element(By.CSS_SELECTOR, "i.fa.fa-globe")
                    ulke = ulke_ikon.find_element(By.XPATH, "..").text.strip()
                except:
                    ulke = ""

                # Telefon, Adres, Web Sitesi
                telefon, adres, website = "", "", ""
                try:
                    bilgiler = driver.find_elements(By.CSS_SELECTOR, "div.schedule-list ul li")
                    for li in bilgiler:
                        icon_html = li.get_attribute("innerHTML")

                        if "fa-phone" in icon_html:
                            telefon = li.text.strip()
                        elif "fa-location-dot" in icon_html:
                            adres = li.text.strip()
                        elif "fa-globe" in icon_html:
                            try:
                                website = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                            except:
                                website = li.text.strip()
                except:
                    pass
                if website:
                    try:
                    # Mail çekmek için şirketin websitesine otomatik giden program
                        firma_mail = site_icerisinden_email_bul(website)
                        if firma_mail == "team@packagingfair.com":
                            firma_mail = "Bu websitesi artık geçerli değildir."
                    except:
                        firma_mail = ""
                else:   
                    try:
                        # Mail çekmek için google üzerinden ilk search şirketin websitesine otomatik giden program
                            firmanin_url = bing_ilk_link_al(firma_adi)
                            firma_mail = site_icerisinden_email_bul(firmanin_url)
                    except:
                            firma_mail = ""

                tablo.append({
                    "Firma Adı": firma_adi,
                    "Ülke": ulke.upper(),
                    "Telefon": telefon,
                    "Adres": adres,
                    "Web Sitesi": website,
                    "Firma Mail": firma_mail                
                    })

                print(f"  ✅ Eklendi: {firma_adi}")

            except Exception as e:
                print(f"  ❌ Firma detay alınamadı: {e}")
                continue

    driver.quit()

    df = pd.DataFrame(tablo)
    if st:
        st.dataframe(df)
        ulke_sayilari = df["Ülke"].value_counts().reset_index()
        ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
        fig = px.bar(ulke_sayilari, x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
        st.plotly_chart(fig)
    else:
        print(f"\n🎯 Toplam çekilen firma sayısı: {len(df)}")


def scrape_burtarim_fair(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    tablo = []
    page_num = 1

    while True:
        url = f"https://www.burtarim.com/katilimci-listesi?page={page_num}"
        driver.get(url)
        time.sleep(3)

        # Sayfadaki firma bloklarını al
        table = driver.find_elements(By.CLASS_NAME, "filter-list__item")

        # Sayfa boşsa döngüyü bitir
        if len(table) == 0:
            print(f"Veri bitti. Son sayfa: {page_num-1}")
            break

        print(f"{page_num}. sayfa işleniyor...")

        for item in table:
            try:
                firma_adi = item.find_element(By.XPATH, ".//div[@class='table-block-content'][1]").text
            except:
                firma_adi = " "

            try:
                adres = item.find_element(By.XPATH, ".//div[@class='table-block-content'][2]").text
                parcalar = adres.split("/")
                ulke = parcalar[-1].strip()
            except:
                adres = " "
                ulke = " "
            try:
                telefon = item.find_element(By.XPATH, ".//a[starts-with(@href, 'tel:')]").text
            except:
                telefon = " "
            try:
                site = item.find_element(By.XPATH, ".//a[starts-with(@href, 'http')]").get_attribute("href")
            except:
                site = " "
            try:
                # Butona tıkla
                detay_buton = item.find_element(By.CLASS_NAME, "js-open-table-detail")
                driver.execute_script("arguments[0].click();", detay_buton)
                time.sleep(0.5)  # açılma süresi

                # Ürün gruplarını listele
                urun_gruplari_liste = item.find_elements(By.CLASS_NAME, "table-detail-wrapper__list-item")
                urun_gruplari = ", ".join([li.text for li in urun_gruplari_liste])
            except:
                urun_gruplari = " "

            tablo.append({
                "Firma": firma_adi,
                "Adres": adres,
                "Ülke":ulke,
                "Telefon": telefon,
                "Web adresi": site,
                "Ürün Grupları": urun_gruplari,
                "Company Mail":"",
                "Company Zip-Code":"",          
            })

        page_num += 1

    driver.quit()

    df = pd.DataFrame(tablo)

    if st:
        st.dataframe(df)
        ulke_sayilari = df["Ülke"].value_counts().reset_index()
        ulke_sayilari.columns = ["Ülke", "Firma Sayısı"]
        fig = px.bar(ulke_sayilari, x="Ülke", y="Firma Sayısı", title="Ülkelere Göre Firma Dağılımı")
        st.plotly_chart(fig)
    else:
        print(df.head())
    
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
    query = urllib.parse.quote(firma_adi)
    url = f"https://www.bing.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    print(f"🔍 Bing araması yapılıyor: {firma_adi}")
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Arama sonuçlarını çek (organik ilk link genelde h2 > a içinde)
    for result in soup.select("li.b_algo h2 a"):
        link = result.get("href")
        if link and link.startswith("http"):
            print(f"🌐 Bulunan ilk link: {link}")
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