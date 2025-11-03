# ai_scripts/google_parser.py

"""
Google Generative AI (Gemini) ile sayfa içeriklerini chunk'lara bölüp
verilen parse_description'a göre çıktı döner.

Kullanım:
    from ai_scripts.google_parser import get_clean_texts_from_urls, parse_with_google

    cleaned_texts = asyncio.run(get_clean_texts_from_urls(
        urls=["https://..."],
        insecure_hosts={"packagingfair.com"}  # İsteğe bağlı
    ))
    result = parse_with_google(cleaned_texts, "List all product names...", api_key="YOUR_KEY")
"""

import os
import ssl
import certifi
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from aiohttp.client_exceptions import ClientSSLError
from ssl import SSLCertVerificationError
from yarl import URL
from bs4 import BeautifulSoup
import google.generativeai as genai


# -------------------------
# Text Chunking (kelime bazlı)
# -------------------------
def chunk_text(text, max_length=1000):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i+max_length])


# -------------------------
# Robust HTML Fetcher (SSL güvenli + domain bazlı bypass opsiyonu)
# -------------------------
_DEFAULT_HEADERS = {
    # Bazı siteler User-Agent istemeden içerik vermez, güvenli bir UA gönderelim:
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36 AI-Scraper/1.0"
    )
}

async def _fetch_one(session: aiohttp.ClientSession, url: str, ssl_context: ssl.SSLContext, insecure_hosts: set | None):
    """
    Tek bir URL'i getirir.
    - Varsayılan: sertifika doğrulaması açık (connector üzerinden).
    - Eğer host insecure_hosts içinde ise: bu istekte ssl=False (doğrulama kapalı).
    - Eğer doğrulama hatası alırsak: bir kereye mahsus ssl=False ile tekrar dener (log yazar).
    """
    insecure_hosts = insecure_hosts or set()
    host = (URL(url).host or "").lower()

    # Domain insecure listede ise bu istek için ssl=False
    per_request_ssl = False if host in insecure_hosts else None  # None => connector'daki SSL context kullanılır

    try:
        async with session.get(
            url,
            timeout=ClientTimeout(total=30),
            ssl=per_request_ssl,
            headers=_DEFAULT_HEADERS
        ) as resp:
            html = await resp.text()
    except (ClientSSLError, SSLCertVerificationError) as e:
        # Domain insecure listede değilse bir defaya mahsus insecure retry yap
        if host not in insecure_hosts:
            print(f"⚠️ SSL doğrulama hatası: {url} -> {e}. Tek seferlik insecure retry deniyorum...")
            try:
                async with session.get(
                    url,
                    timeout=ClientTimeout(total=30),
                    ssl=False,  # doğrulamayı kapat
                    headers=_DEFAULT_HEADERS
                ) as resp:
                    html = await resp.text()
            except Exception as e2:
                print(f"❌ Insecure retry da başarısız: {url} -> {e2}")
                return ""
        else:
            print(f"❌ SSL hatası (insecure host listesinde ama yine de başarısız): {url} -> {e}")
            return ""
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return ""

    # HTML'i parse et
    soup = BeautifulSoup(html, "html.parser")

    # Mümkün olduğunca ana içerik alanını hedefleyelim:
    main_content = (
        soup.select_one("main")
        or soup.select_one(".content")
        or soup.select_one("article")
    )

    if main_content:
        return main_content.get_text(separator="\n", strip=True)
    else:
        # Yine de sayfadaki tüm görünen metni döndür
        return soup.get_text(separator="\n", strip=True)


async def get_clean_texts_from_urls(urls: list[str], insecure_hosts: set[str] | None = None):
    """
    URL'leri getirir ve temiz metinleri döndürür.
    - insecure_hosts: { "packagingfair.com", "www.packagingfair.com" } gibi, doğrulamayı
      kapatmamız gereken (bilinçli) domain'leri buraya ekleyebilirsin.
    """
    # ✅ Certifi CA ile güvenli SSL context
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_one(session, url, ssl_context, insecure_hosts) for url in urls]
        return await asyncio.gather(*tasks)


# -------------------------
# Google Gemini ile AI Parsing
# -------------------------
def parse_with_google(cleaned_texts, parse_description, api_key=None, model_name="gemini-2.5-flash"):
    """
    cleaned_texts: fetch edilmiş ve temizlenmiş sayfa metinlerinin listesi
    parse_description: istenen çıkarım promptu (İngilizce önerilir)
    api_key: Google API Key (parametre ile gelmezse ortam değişkenlerinden okunur)
    model_name: gemini modeli (örn. 'gemini-2.5-flash' veya 'gemini-2.5-pro')
    """
    # API key alma (parametre > env > yoksa hata)
    effective_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not effective_api_key:
        raise ValueError(
            "Google API Key bulunamadı. Lütfen parse_with_google(api_key=...) ile geçin "
            "veya ortam değişkeni GOOGLE_API_KEY olarak ayarlayın."
        )

    # Google GenAI yapılandırma
    genai.configure(api_key=effective_api_key)
    model = genai.GenerativeModel(model_name)

    template = (
        "Extract the specific information from this text: {dom_content}\n"
        "Follow these rules:\n"
        "1. Only output what matches the description: {parse_description}\n"
        "2. No extra text or comments.\n"
        "3. If nothing matches, output empty string.\n"
    )

    parsed_results = []

    for page_idx, cleaned_text in enumerate(cleaned_texts, start=1):
        chunks = list(chunk_text(cleaned_text, max_length=1000))

        for i, chunk in enumerate(chunks, start=1):
            prompt = template.format(dom_content=chunk, parse_description=parse_description)
            try:
                resp = model.generate_content(prompt)
                text = getattr(resp, "text", "") or ""
            except Exception as e:
                print(f"❌ Google GenAI çağrısında hata (Sayfa {page_idx}, Chunk {i}): {e}")
                text = ""

            print(f"📄 [Google] Page {page_idx} - Parsed chunk {i}/{len(chunks)}")
            parsed_results.append(text)

    return "\n".join(parsed_results)