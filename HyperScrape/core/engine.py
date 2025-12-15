"""
Main HyperScrape Engine.
Coordinates template selection, scraping execution, and result aggregation.
"""

import time
import logging
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..models.fair import Company, FairConfig, ScrapeResult
from ..config.settings import Settings
from .driver_pool import DriverPool
from .cache import ScrapeCache


# Import all templates
from ..templates.tuyap_list import TuyapListTemplate
from ..templates.link_detail import LinkDetailTemplate
from ..templates.deutsche_platform import DeutschePlatformTemplate
from ..templates.modal_popup import ModalPopupTemplate
from ..templates.card_list import CardListTemplate
from ..templates.member_cards import MemberCardsTemplate
from ..templates.search_results import SearchResultsTemplate
from ..templates.scroll_load import ScrollLoadTemplate
from ..templates.xpath_detail import XPathDetailTemplate
from ..templates.text_list import TextListTemplate
from ..templates.image_gallery import ImageGalleryTemplate
from ..templates.blog_cards import BlogCardsTemplate


logger = logging.getLogger(__name__)


class HyperScrapeEngine:
    """Main scraping engine that coordinates all operations."""
    
    # Template registry
    TEMPLATES = {
        'tuyap_list': TuyapListTemplate,
        'link_detail': LinkDetailTemplate,
        'deutsche_platform': DeutschePlatformTemplate,
        'modal_popup': ModalPopupTemplate,
        'card_list': CardListTemplate,
        'member_cards': MemberCardsTemplate,
        'search_results': SearchResultsTemplate,
        'scroll_load': ScrollLoadTemplate,
        'xpath_detail': XPathDetailTemplate,
        'text_list': TextListTemplate,
        'image_gallery': ImageGalleryTemplate,
        'blog_cards': BlogCardsTemplate,
    }
    
    def __init__(self, config: FairConfig):
        """
        Initialize engine with fair configuration.
        
        Args:
            config: Fair configuration object
        """
        self.config = config
        self.driver_pool = DriverPool()
        self.cache = ScrapeCache()
        self.template = self._get_template()
        
    def _get_template(self):
        """Get appropriate template based on config."""
        template_class = self.TEMPLATES.get(self.config.template_type)
        
        if not template_class:
            raise ValueError(f"Unknown template type: {self.config.template_type}")
        
        return template_class(self.config, self.driver_pool, self.cache)
    
    def scrape(
        self,
        page_count: Optional[int] = None,
        parallel: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> ScrapeResult:
        """
        Scrape the fair website.
        
        Args:
            page_count: Number of pages to scrape (None = auto-detect)
            parallel: Whether to use parallel processing
            progress_callback: Callback function(progress, status) for updates
            
        Returns:
            ScrapeResult with companies and statistics
        """
        start_time = time.time()
        all_companies = []
        success_count = 0
        error_count = 0
        
        logger.info(f"Starting scrape for {self.config.name}")
        
        try:
            # Get page URLs to scrape
            if self.config.pagination_type == 'none':
                page_urls = [self.config.base_url]
            elif self.config.pagination_type == 'scroll' or self.config.pagination_type == 'load_more':
                # These templates handle pagination internally
                page_urls = [self.config.base_url]
            else:
                # Generate page URLs
                if page_count is None:
                    page_count = self._auto_detect_page_count()
                page_urls = [self.config.get_page_url(i) for i in range(1, page_count + 1)]
            
            logger.info(f"Will scrape {len(page_urls)} page(s)")
            
            # Scrape pages
            if parallel and len(page_urls) > 1:
                companies = self._scrape_parallel(page_urls, progress_callback)
            else:
                companies = self._scrape_sequential(page_urls, progress_callback)
            
            all_companies = companies
            success_count = len(companies)
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}", exc_info=True)
            error_count += 1
        
        finally:
            # Cleanup
            self.driver_pool.close_all()
        
        duration = time.time() - start_time
        
        result = ScrapeResult(
            companies=all_companies,
            total_pages=len(page_urls),
            success_count=success_count,
            error_count=error_count,
            duration_seconds=duration,
            config_name=self.config.name
        )
        
        logger.info(f"Scraping completed: {success_count} companies in {duration:.2f}s")
        
        return result
    
    def _scrape_sequential(
        self,
        page_urls: List[str],
        progress_callback: Optional[Callable] = None
    ) -> List[Company]:
        """Scrape pages sequentially."""
        all_companies = []
        
        for idx, url in enumerate(page_urls):
            if progress_callback:
                progress = (idx + 1) / len(page_urls)
                progress_callback(progress, f"Scraping page {idx + 1}/{len(page_urls)}")
            
            try:
                companies = self.template.scrape_page(url)
                all_companies.extend(companies)
                logger.info(f"Page {idx + 1}: Found {len(companies)} companies")
                
                # Delay between requests
                time.sleep(Settings.REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"Error scraping page {idx + 1}: {e}")
        
        return all_companies
    
    def _scrape_parallel(
        self,
        page_urls: List[str],
        progress_callback: Optional[Callable] = None
    ) -> List[Company]:
        """Scrape pages in parallel using thread pool."""
        all_companies = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=Settings.MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(self.template.scrape_page, url): url
                for url in page_urls
            }
            
            for future in as_completed(future_to_url):
                completed += 1
                url = future_to_url[future]
                
                if progress_callback:
                    progress = completed / len(page_urls)
                    progress_callback(progress, f"Completed {completed}/{len(page_urls)} pages")
                
                try:
                    companies = future.result()
                    all_companies.extend(companies)
                    logger.info(f"Page completed: Found {len(companies)} companies")
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
        
        return all_companies
    
    def _auto_detect_page_count(self) -> int:
        """
        Auto-detect the number of pages.
        Returns a reasonable default if detection fails.
        """
        if not self.config.max_page_auto_detect:
            return 1
        
        # Try to detect from first page
        try:
            # This is a simple implementation
            # Each template can override this logic
            return 1  # Default to 1 page for safety (user can specify more)
        except:
            return 1
