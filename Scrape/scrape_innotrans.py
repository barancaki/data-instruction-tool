import time
import pandas as pd
import sqlite3
import io
import os
import streamlit as st # Streamlit'in kurulu olduğunu varsayıyoruz

# Selenium ve ilgili kütüphaneler
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager

# --- YARDIMCI FONKSİYONLAR (Değişiklik yok) ---

DB_NAME = "fuar_data.db"
TABLE_NAME = "firmalar"
SQL_DUMP_NAME = "fuar_data.sql"

def save_to_sqlite(df, db_name=DB_NAME, table_name=TABLE_NAME):
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Veri başarıyla {db_name} içerisindeki {table_name} tablosuna kaydedildi.")
    except Exception as e:
        print(f"SQLite'a kaydetme hatası: {e}")
        if st:
            st.error(f"SQLite'a kaydetme hatası: {e}")

def create_mysql_dump_from_sqlite(db_name=DB_NAME, sql_dump_name=SQL_DUMP_NAME):
    try:
        conn = sqlite3.connect(db_name)
        dump_buffer = io.StringIO()
        for line in conn.iterdump():
            dump_buffer.write(f"{line}\n")
        
        conn.close()
        dump_content = dump_buffer.getvalue()
        dump_buffer.close()

        dump_content = dump_content.replace("BEGIN TRANSACTION;", "START TRANSACTION;")
        dump_content = dump_content.replace("PRAGMA foreign_keys=OFF;", "")

        with open(sql_dump_name, "w", encoding="utf-8") as f:
            f.write(dump_content)
            
        print(f"MySQL uyumlu SQL dökümü {sql_dump_name} dosyasına oluşturuldu.")

    except sqlite3.OperationalError as e:
        print(f"Veritabanı dökümü oluşturulurken hata: {e}")
        if st:
            st.warning("SQL dökümü oluşturulamadı. Önce veritabanını oluşturun.")
    except Exception as e:
        print(f"SQL dökümü oluşturma hatası: {e}")
        if st:
            st.error(f"SQL dökümü oluşturma hatası: {e}")


# --- GÜNCELLENMİŞ scrape_innotrans FONKSİYONU ---

def scrape_innotrans():
    """
    InnoTrans fuarı katılımcılarını web sitesinden kazır.
    
    Yavaş siteler için optimize edilmiştir:
    1. Görselleri yüklemez (daha hızlı yükleme için).
    2. Zaman aşımları 60 saniyeye çıkarıldı.
    3. Başarısız olursa 3 kez yeniden deneme (retry) mekanizması eklendi.
    4. "Show More" ve "Retry" bekleme süreleri artırıldı.
    """
    
    # --- YENİ: Zaman Aşımları ve Ayarlar ---
    COOKIE_TIMEOUT = 30 # Cookie butonu bekleme süresi
    SHOW_MORE_TIMEOUT = 20 # "Show More" butonu bekleme süresi
    SHOW_MORE_SLEEP = 3.0 # "Show More" sonrası JS yüklemesi için bekleme (YENİ)
    
    DETAIL_PAGE_TIMEOUT = 60 # Detay sayfası yükleme zaman aşımı (YENİ)
    MAX_RETRIES = 3 # Başarısız link için yeniden deneme sayısı
    RETRY_WAIT = 5 # Denemeler arası bekleme süresi (YENİ)
    
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox") 
    options.add_argument("--disable-dev-shm-usage") 
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Görselleri kapat (Hız için)
    options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
    options.add_argument('--blink-settings=imagesEnabled=false')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(DETAIL_PAGE_TIMEOUT + 10) # Sayfanın kendisine de bir zaman aşımı ver
    
    base_url = "https://www.innotrans.de/en/visit/exhibitor-directory#/search/f=h-entity_orga;v_sg=0;v_fg=0;v_fpa=FUTURE"
    driver.get(base_url)
    
    # 1. ADIM: Cookie pop-up'ını kabul et
    try:
        cookie_button = WebDriverWait(driver, COOKIE_TIMEOUT).until(
            EC.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"))
        )
        cookie_button.click()
        print("Cookie 'Accept all' butonuna tıklandı.")
        time.sleep(2) 
    except TimeoutException:
        print("Cookie pop-up'ı bulunamadı veya zaman aşımına uğradı, devam ediliyor...")
    except Exception as e:
        print(f"Cookie butonuna tıklarken hata: {e}")

    # 2. ADIM: "Show more" butonuna sonuna kadar bas
    print(f"Sayfa yükleniyor ve 'Show more' butonlarına (her {SHOW_MORE_SLEEP} saniyede bir) basılıyor...")
    if st:
        st.info(f"Sayfa yükleniyor ve 'Show more' butonlarına basılıyor... Bu işlem {int(2940/25)} tıklama gerektirebilir, lütfen bekleyin.") # Toplam firma sayısına göre tahmini tıklama

    while True:
        try:
            show_more_button = WebDriverWait(driver, SHOW_MORE_TIMEOUT).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".EWP5KKC-u-a.EWP5KKC-u-d"))
            )
            
            driver.execute_script("arguments[0].scrollIntoView(true);", show_more_button)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", show_more_button)
            
            time.sleep(SHOW_MORE_SLEEP) 
            
        except (TimeoutException, NoSuchElementException):
            print("'Show more' butonu bulunamadı veya tüm firmalar yüklendi.")
            break
        except Exception as e:
            print(f"Beklenmedik bir hata (Show more): {e}")
            break

    print("Tüm firma kartları yüklendi. Detay linkleri toplanıyor...")
    
    # 3. ADIM: Güvenli yöntemle detay linklerini topla
    exhibitor_cards = driver.find_elements(By.CSS_SELECTOR, ".EWP5KKC-w-J")
    detail_links = []
    
    for card in exhibitor_cards:
        found_link = False
        try:
            all_links_in_card = card.find_elements(By.TAG_NAME, "a")
            for link_element in all_links_in_card:
                href = link_element.get_attribute("href")
                if href and "/detail/" in href:
                    detail_links.append(href)
                    found_link = True
                    break 
            
            if not found_link:
                print("Bir kartta '/detail/' linki bulunamadı, atlanıyor.")
                
        except Exception as e:
            print(f"Link toplarken bir kartta hata: {e}")
            continue

    total_firms = len(detail_links)
    print(f"Toplam {total_firms} adet firma detayı kazınacak.")
    if st:
        st.info(f"Tüm firmalar yüklendi. {total_firms} adet firma detayı çekiliyor...")
        if total_firms == 0:
            st.warning("Hiç firma linki bulunamadı. Site yapısı değişmiş olabilir.")
            driver.quit()
            return
        progress_bar = st.progress(0)

    tablo = []
    
    # 4. ADIM: Her bir detay linkine gidip bilgileri çek
    for i, link in enumerate(detail_links):
        if not link:
            continue
        
        success = False 
        
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Firma {i+1}/{total_firms} | DENEME {attempt + 1}/{MAX_RETRIES}: Sayfa yükleniyor (Max {DETAIL_PAGE_TIMEOUT}sn)... {link}")
                driver.get(link)
                
                # Detay sayfasındaki iletişim bloğunun yüklenmesini bekle (60 saniye)
                WebDriverWait(driver, DETAIL_PAGE_TIMEOUT).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".EWP5KKC-B-s")) 
                )
                print("... Sayfa yüklendi, veriler çekiliyor.")
                
                # Sayfa yüklendi, verileri çek
                try:
                    firma_adi = driver.find_element(By.CSS_SELECTOR, ".EWP5KKC-B-A").text.strip()
                except NoSuchElementException:
                    firma_adi = ""

                phone = ""
                website = ""
                email = ""

                info_blocks = driver.find_elements(By.CSS_SELECTOR, ".EWP5KKC-Q-b")
                for block in info_blocks:
                    try:
                        icon = block.find_element(By.CSS_SELECTOR, ".EWP5KKC-Q-f").get_attribute("iconcode")
                        
                        if icon == "": # Telefon
                            phone = block.find_element(By.CSS_SELECTOR, ".EWP5KKC-Q-j a").get_attribute("href").replace("tel:", "").strip()
                        elif icon == "": # Website
                            website = block.find_element(By.CSS_SELECTOR, ".EWP5KKC-Q-j a").get_attribute("href").strip()
                        elif icon == "": # Email
                            email = block.find_element(By.CSS_SELECTOR, ".EWP5KKC-Q-j a").get_attribute("href").replace("mailto:", "").strip()
                    except:
                        continue 
                        
                tablo.append({
                    "Firma": firma_adi,
                    "Web adresi": website,
                    "Mail": email,
                    "Telefon": phone
                })
                
                success = True 
                print(f"... Başarıyla çekildi: {firma_adi}")

                if st:
                    progress_bar.progress((i + 1) / total_firms)

                break # Yeniden deneme döngüsünden çık
                
            except TimeoutException:
                print(f"... DENEME {attempt + 1}/{MAX_RETRIES}: ZAMAN AŞIMI (Timeout) {link}. {RETRY_WAIT}sn sonra tekrar denenecek.")
                time.sleep(RETRY_WAIT)
            except Exception as e:
                print(f"... DENEME {attempt + 1}/{MAX_RETRIES}: KRİTİK HATA {link}: {e}. {RETRY_WAIT}sn sonra tekrar denenecek.")
                time.sleep(RETRY_WAIT)

        if not success:
            print(f"!!! FİRMA ATLANDI: {link} {MAX_RETRIES} denemeden sonra (her biri {DETAIL_PAGE_TIMEOUT}sn) alınamadı.")
            if st:
                st.warning(f"Bir firma {MAX_RETRIES} denemeden sonra alınamadı: {link}")


    driver.quit()
    
    if not tablo:
        print("Hiçbir firma bilgisi çekilemedi.")
        if st:
            st.warning("Hiçbir firma bilgisi çekilemedi. Site yapısı değişmiş olabilir.")
        return

    # 5. ADIM: Veriyi temizle, kaydet ve göster
    df = pd.DataFrame(tablo)
    df.replace("", pd.NA, inplace=True) 
    df = df.dropna(how='all', subset=['Firma', 'Web adresi', 'Mail', 'Telefon']) 
    df = df.drop_duplicates(subset=['Firma', 'Web adresi']) 
    df = df.fillna(" ") 

    print(f"\n🎯 Toplam çekilen ve filtrelenen firma sayısı: {len(df)}")
    
    save_to_sqlite(df, db_name=DB_NAME, table_name=TABLE_NAME)
    create_mysql_dump_from_sqlite(db_name=DB_NAME, sql_dump_name=SQL_DUMP_NAME)

    if st:
        st.success(f"Başarıyla {len(df)} adet firma bilgisi çekildi!")
        st.dataframe(df)

        try:
            with open(DB_NAME, "rb") as f:
                st.download_button(
                    label="📥 Database (.db) İndir",
                    data=f,
                    file_name=DB_NAME,
                    mime="application/octet-stream"
                )
        except FileNotFoundError:
            st.error(".db dosyası bulunamadı.")
            
        try:
            with open(SQL_DUMP_NAME, "rb") as f:
                st.download_button(
                    label="📥 SQL (.sql) İndir",
                    data=f,
                    file_name=SQL_DUMP_NAME,
                    mime="application/sql"
                )
        except FileNotFoundError:
            st.error(".sql dosyası bulunamadı.")
            
    else:
        print("\n--- Veri Önizleme ---")
        print(df.head())