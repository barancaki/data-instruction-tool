# ai_scripts/auto_exhibitor_crawler.py
# -*- coding: utf-8 -*-
"""
AutoExhibitorCrawler
--------------------
Bir *listing* sayfasından otomatik olarak:
 1) Detay linklerini statik olarak keşfetmeye çalışır
 2) Yetersiz kalırsa Playwright ile (headless/visible) kartlara tıklar (modal/drawer/yenisayfa)
 3) Toplanan metinleri temizleyip geri döndürür (mevcut LLM parser zincirinle uyumlu)

Kullanım (örnek):
    from ai_scripts.auto_exhibitor_crawler import AutoExhibitorCrawler
    import asyncio

    crawler = AutoExhibitorCrawler(
        max_items=200,
        insecure_hosts={"packagingfair.com", "www.packagingfair.com"}  # SSL'i bilerek bypass edeceğin alanlar
    )
    texts, meta = asyncio.run(crawler.run("https://packagingfair.com/katilimci-listesi"))

Dönüş:
    texts: List[str]  -> Detaylardan çıkarılan temiz metinler
    meta:  List[dict] -> Her kayıt için {source, url (detay/veya listing), method, title?}

Gereksinimler:
    pip install aiohttp beautifulsoup4 certifi playwright
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import re
import ssl
from typing import List, Optional, Set, Tuple, Dict
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
import certifi
from yarl import URL  # 🔧 aiohttp.helpers.Url yerine doğrusu

# ----------------------------------------------------------------------
# Sabitler & yardımcılar
# ----------------------------------------------------------------------

DEFAULT_HEADERS = {
    # Bazı siteler User-Agent istemeden içerik vermez; makul ve sabit bir UA kullanalım:
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36 AI-Scraper/1.2"
    )
}

# EN + TR anahtarlar (detay linkini puanlamak için)
DETAIL_KEYWORDS = re.compile(
    r"(exhibitor|brand|company|firma|katilimci|katılımcı|participant|vendor|"
    r"profile|profil|detay|detail|details?)",
    re.I,
)

# Kart benzeri sınıf ipuçları
CARD_HINT_CLASSES = [
    "card",
    "exhibitor",
    "brand",
    "company",
    "firma",
    "participant",
    "grid-item",
    "item",
    "tile",
    "result",
    "listing",
    "list-item",
]


def _is_same_domain(base_url: str, href: str) -> bool:
    """href aynı domain'de mi? (boş netloc => aynı sayılır)"""
    try:
        b = urlparse(base_url)
        h = urlparse(href)
        return h.netloc == "" or h.netloc == b.netloc
    except Exception:
        return True


def _abs_url(base: str, href: str) -> str:
    """href'i mutlak URL'e çevir."""
    return urljoin(base, href)


def _clean_text(text: str) -> str:
    """Boşluk/sekme temizliği + satır başlarını normalize et."""
    text = text or ""
    text = re.sub(r"[ \t]+", " ", text)
    # art arda gelen boş satırları sadeleştir
    text = re.sub(r"\n{3,}", "\n\n", text)
    # satır sonlarında fazlalıkları kırp
    text = "\n".join([ln.strip() for ln in text.splitlines()])
    return text.strip()


# ----------------------------------------------------------------------
# Crawler
# ----------------------------------------------------------------------
class AutoExhibitorCrawler:
    def __init__(
        self,
        max_items: Optional[int] = None,
        per_request_timeout: int = 30,
        delay_between_requests: float = 0.0,
        concurrent_limit: int = 8,
        insecure_hosts: Optional[Set[str]] = None,
        scroll_times: int = 4,
        scroll_wait: float = 0.8,
        wait_after_click: float = 0.6,
        headless: bool = True,
    ):
        """
        Parametreler:
            max_items: Toplanacak maksimum detay kartı/adet (None: sınırsız)
            per_request_timeout: HTTP timeout (saniye)
            delay_between_requests: Nazik tarama için istekler arası bekleme
            concurrent_limit: Statik isteklerde eşzamanlı istek sınırı
            insecure_hosts: SSL doğrulamasını kapatmak istediğin domain'ler (bilinçli)
            scroll_times: Browser fallback’te sayfayı kaç kez aşağı kaydır
            scroll_wait: Her scroll sonrası bekleme (sn)
            wait_after_click: Kart tıklandıktan sonra bekleme (sn)
            headless: Playwright tarayıcısı headless mı açılsın
        """
        self.max_items = max_items
        self.per_request_timeout = per_request_timeout
        self.delay_between_requests = delay_between_requests
        self.concurrent_limit = concurrent_limit
        self.insecure_hosts = {h.lower() for h in (insecure_hosts or set())}
        self.scroll_times = scroll_times
        self.scroll_wait = scroll_wait
        self.wait_after_click = wait_after_click
        self.headless = headless

        # SSL (certifi)
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    # ------------------------------- HTTP -------------------------------

    async def _get_html(self, session: aiohttp.ClientSession, url: str) -> str:
        """Tek bir URL'i getir, gerekirse bu istekte SSL doğrulamasını kapat."""
        host = (URL(url).host or "").lower()
        ssl_flag = False if host in self.insecure_hosts else None  # None => connector context'i kullan
        async with session.get(
            url,
            timeout=ClientTimeout(total=self.per_request_timeout),
            ssl=ssl_flag,
            headers=DEFAULT_HEADERS,
        ) as resp:
            resp.raise_for_status()
            # Bazı sayfalarda encoding problemleri olabilir; errors="ignore" ile yumuşatıyoruz.
            return await resp.text(errors="ignore")

    async def _fetch_many(self, urls: List[str]) -> List[Tuple[str, str]]:
        """
        Birden çok URL'i indir.
        Dönüş: [(url, html), ...] (Hata durumunda html = "")
        """
        sem = asyncio.Semaphore(self.concurrent_limit)
        out: List[Tuple[str, str]] = []

        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:

            async def _worker(u: str):
                async with sem:
                    try:
                        html = await self._get_html(session, u)
                        out.append((u, html))
                    except Exception as e:
                        print(f"❌ fetch error: {u} -> {e}")
                        out.append((u, ""))
                    if self.delay_between_requests > 0:
                        await asyncio.sleep(self.delay_between_requests)

            await asyncio.gather(*[_worker(u) for u in urls])

        return out

    # ---------------------------- DISCOVERY -----------------------------

    def _discover_detail_links(self, listing_url: str, html: str) -> List[str]:
        """
        Listing içerisinden "detay" sayfalarını sezgisel olarak keşfet.
        """
        soup = BeautifulSoup(html, "html.parser")

        # 1) Tüm <a> elemanlarını puanlayalım
        candidates: List[Tuple[str, int]] = []
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if not href or href.startswith("#") or href.lower().startswith("javascript"):
                continue

            absu = _abs_url(listing_url, href)
            if not _is_same_domain(listing_url, absu):
                continue

            score = 0

            # URL pattern + anchor text + class + data-* ipuçları
            if DETAIL_KEYWORDS.search(href):
                score += 3
            anchor_text = (a.get_text() or "").strip()
            if DETAIL_KEYWORDS.search(anchor_text):
                score += 2
            class_txt = " ".join(a.get("class") or [])
            if any(cls in class_txt.lower() for cls in CARD_HINT_CLASSES):
                score += 1
            data_url = a.get("data-url") or a.get("data-href") or ""
            if DETAIL_KEYWORDS.search(data_url):
                score += 2

            # Rel/role ipuçları
            rel = " ".join(a.get("rel") or [])
            role = a.get("role") or ""
            if "detail" in rel.lower() or "dialog" in role.lower():
                score += 1

            if score > 0:
                candidates.append((absu, score))

        # 2) En çok tekrar eden/benzer path’leri öne al (örn. /brand/..., /exhibitor/...)
        def _norm(u: str) -> str:
            p = urlparse(u)
            return f"{p.scheme}://{p.netloc}{p.path}"  # query/fragment at

        freq: Dict[str, int] = {}
        for u, _ in candidates:
            freq[_norm(u)] = freq.get(_norm(u), 0) + 1

        ranked = sorted(
            candidates,
            key=lambda t: (t[1], freq.get(_norm(t[0]), 1)),
            reverse=True,
        )

        # 3) Sırayı koruyarak tekilleştir
        seen = set()
        links: List[str] = []
        for u, _ in ranked:
            nu = _norm(u)
            if nu in seen:
                continue
            seen.add(nu)
            links.append(u)
            if self.max_items and len(links) >= self.max_items:
                break

        return links

    def _detect_card_density(self, html: str) -> int:
        """Kart benzeri öğe sayısını kaba taslak bulur."""
        soup = BeautifulSoup(html, "html.parser")
        count = 0
        # Genel kart ipuçları
        for cls in CARD_HINT_CLASSES:
            count += len(soup.select(f".{cls}"))
        # Çok gevşek bir ölçü: >~5 ise kayda değer bir yoğunluk var denebilir
        return count

    # ------------------------------ PARSE -------------------------------

    def _extract_clean_text(self, html: str) -> str:
        """Ana içerik alanını hedefleyip temiz metin elde et."""
        soup = BeautifulSoup(html, "html.parser")
        # Görünür olmayan/gereksiz etiketleri at
        for bad in soup(["script", "style", "noscript"]):
            bad.decompose()

        main = soup.select_one("main") or soup.select_one(".content") or soup.select_one("article")
        text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
        return _clean_text(text)

    # ------------------------- BROWSER FALLBACK -------------------------

    async def _browser_click_collect(self, url: str) -> List[Tuple[str, str]]:
        """
        Playwright ile:
         - sayfayı aç
         - gerektiği kadar scroll et
         - kartları bul (heuristic)
         - sırayla tıkla ve detay/metin çek
        Dönen: [(method, text)] -> method = "browser"
        """
        from playwright.async_api import async_playwright

        results: List[Tuple[str, str]] = []

        # Olası kart seçicileri (öncelik sırası)
        CARD_SELECTORS = [
            ".exhibitor-card",
            ".brand-card",
            ".company-card",
            ".card",
            ".exhibitor",
            ".brand",
            ".company",
            ".firma",
            ".grid-item",
            ".item",
            ".tile",
            ".listing .item",
            "[data-card]",
            "[role=listitem]",
        ]

        # Olası detay container seçicileri
        DETAIL_CONTAINERS = [
            ".modal.show",
            ".modal-dialog",
            ".modal-content",
            ".modal-body",
            "[role=dialog]",
            ".dialog",
            ".drawer.open",
            ".drawer-content",
            ".lightbox.open",
            ".popup.open",
            ".offcanvas.show",
            "main",
            ".content",
            "article",
        ]

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"])
            page = await context.new_page()

            try:
                await page.goto(url, wait_until="networkidle", timeout=60_000)

                # Sonsuz liste olasılığına karşı scroll
                for _ in range(self.scroll_times):
                    await page.mouse.wheel(0, 20000)
                    await page.wait_for_timeout(int(self.scroll_wait * 1000))

                # Kartları sırayla dene
                card_handles = []
                for sel in CARD_SELECTORS:
                    items = await page.query_selector_all(sel)
                    if items:
                        card_handles = items
                        break

                if not card_handles:
                    # Son çare: linklere körlemesine tıklamak riskli; vazgeçiyoruz.
                    return results

                if self.max_items:
                    card_handles = card_handles[: self.max_items]

                for idx, card in enumerate(card_handles, start=1):
                    try:
                        # Kart içinde link/buton varsa onu hedef al
                        link = await card.query_selector("a, [role=button], button, [onclick], [data-href], [data-url]")
                        target = link or card

                        # Click
                        await target.click()
                        await page.wait_for_timeout(int(self.wait_after_click * 1000))

                        # Detay metni topla (öncelik: modal/drawer)
                        text = ""
                        for dsel in DETAIL_CONTAINERS:
                            elem = await page.query_selector(dsel)
                            if elem:
                                inner = await elem.inner_text()
                                if inner and inner.strip():
                                    text = inner.strip()
                                    break

                        # Eğer modal yok ve yeni sayfaya geçildiyse: body'den metin al
                        if not text:
                            body = await page.query_selector("body")
                            if body:
                                text = (await body.inner_text() or "").strip()

                        results.append(("browser", _clean_text(text) if text else ""))

                        # Modal/Drawer varsa kapatmayı dene
                        closed = False
                        for close_sel in [".modal .btn-close", ".modal .close", ".drawer .close", ".popup .close", "[data-dismiss=modal]"]:
                            btn = await page.query_selector(close_sel)
                            if btn:
                                await btn.click()
                                await page.wait_for_timeout(200)
                                closed = True
                                break
                        if not closed:
                            # ESC ile kapatmayı dene
                            await page.keyboard.press("Escape")
                            await page.wait_for_timeout(150)

                    except Exception as e:
                        print(f"⚠️ Browser click error (#{idx}): {e}")
                        results.append(("browser", ""))
                        try:
                            await page.keyboard.press("Escape")
                        except Exception:
                            pass

            finally:
                await context.close()
                await browser.close()

        return results

    # -------------------------------- RUN -------------------------------

    async def run(self, listing_url: str) -> Tuple[List[str], List[Dict]]:
        """
        Yürüt ve sonuçları döndür.
        Dönüş:
            texts: Detaylardan çıkarılan metinler
            meta:  Her kayıt için {source, url (detay/veya listing), method, title?}
        """
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 1) Liste sayfasını çek
            try:
                listing_html = await self._get_html(session, listing_url)
            except Exception as e:
                print(f"❌ Listing fetch error: {listing_url} -> {e}")
                return [], []

        # 2) Statik: detay linklerini keşfet
        links = self._discover_detail_links(listing_url, listing_html)

        # Basit karar: yeterli sayıda link varsa statik yol
        # Eşik: >=5 link veya (>=3 link ve kart yoğunluğu >=5)
        card_density = self._detect_card_density(listing_html)
        static_ok = len(links) >= 5 or (len(links) >= 3 and card_density >= 5)

        texts: List[str] = []
        meta: List[Dict] = []

        if static_ok and links:
            # 3) Detay sayfalarını indir
            if self.max_items:
                links = links[: self.max_items]
            fetched = await self._fetch_many(links)
            for url, html in fetched:
                text = self._extract_clean_text(html) if html else ""
                texts.append(text)
                meta.append({"source": listing_url, "url": url, "method": "static"})
        else:
            # 4) Browser fallback: kartlara tıklayıp modal/expand vs. topla
            browser_results = await self._browser_click_collect(listing_url)
            for method, text in browser_results:
                texts.append(text)
                meta.append({"source": listing_url, "url": listing_url, "method": method})

        # 🔚 Her akışta garanti dönüş
        return texts, meta


__all__ = ["AutoExhibitorCrawler"]