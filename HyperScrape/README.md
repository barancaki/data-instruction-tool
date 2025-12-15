# HyperScrape Engine v2

🚀 **Next-generation fair/exhibition scraping engine with 10x performance boost**

## Features

- ⚡ **30+ Pre-configured Sites** - Ready to use, zero configuration
- 🎯 **12 Smart Templates** - Automatic pattern detection
- 🔄 **Parallel Processing** - 3-5x faster with multi-threading
- 📧 **Auto Email Finding** - Intelligent enrichment from multiple sources
- 💾 **Smart Caching** - Avoid re-scraping recent data
- 🎨 **Beautiful Streamlit UI** - No coding required
- 📊 **Export Anywhere** - Excel, CSV, JSON, SQL
- 🌍 **International Support** - Turkish, English, German sites

## Quick Start

### 1. Run HyperScrape UI

```bash
streamlit run pages/7_HyperScrape.py
```

### 2. Select Fair & Scrape

1. Choose a fair from dropdown (30+ options)
2. Set number of pages
3. Click "Start Scraping"
4. Download results as Excel/CSV/JSON

That's it! No coding needed.

## Adding a New Site

Edit `HyperScrape/config/fair_definitions.py`:

```python
"my_new_fair": {
    "name": "My Fair 2025",
    "base_url": "https://example.com/exhibitors",
    "template": "tuyap_list",  # Choose from 12 templates
    "pagination_url": "https://example.com/exhibitors?page={page}",
    "selectors": {
        "company_list": ".exhibitor-item",
        "company_name": ".name",
        "website": "a[href^='http']",
        # ... other selectors based on template
    },
    "enable_email_enrichment": True
}
```

Save and restart. **Done!**

## Available Templates

| Template | Use Case | Sites |
|----------|----------|-------|
| `tuyap_list` | Filter list with detail buttons | 7 |
| `link_detail` | Brand cards → detail pages | 4 |
| `deutsche_platform` | Deutsche Messe platform | 4 |
| `modal_popup` | Modal popup details | 1 |
| `card_list` | Card-based layouts | 1 |
| `member_cards` | Membership directories | 5 |
| `search_results` | Search result layouts | 2 |
| `scroll_load` | Infinite scroll / load more | 2 |
| `xpath_detail` | Complex XPath extraction | 1 |
| `text_list` | Simple text lists | 1 |
| `image_gallery` | Image galleries | 1 |
| `blog_cards` | Blog-style cards | 1 |

## Architecture

```
HyperScrape/
├── config/              # Site definitions & settings
│   ├── fair_definitions.py  # ALL 30+ site configs
│   ├── loader.py
│   └── settings.py
├── core/                # Main engine
│   ├── engine.py        # HyperScrapeEngine
│   ├── driver_pool.py   # WebDriver pooling
│   └── cache.py         # Caching system
├── templates/           # 12 scraping templates
│   ├── base.py
│   ├── tuyap_list.py
│   └── ... (11 more)
├── enrichment/          # Data enrichment
│   ├── email_finder.py
│   ├── address_finder.py
│   └── social_finder.py
└── models/              # Data models
    └── fair.py
```

## Pre-configured Sites (30+)

### Tuyap Sites (7)
- Replast Eurasia
- Burtarım
- Pencere Fuarı
- SMTech Eurasia
- İplik Fuarı
- Maktek
- Expomed

### Link+Detail Sites (4)
- Packaging Fair
- Plast Eurasia
- Intermob Istanbul
- Woodtech Istanbul

### Deutsche Messe (4)
- WIN Eurasia
- Hub of Warehouse
- SODEX
- Automechanika Istanbul

### Other Platforms (15+)
- EV Charge Show
- HVACR World
- ENOSAD (4 categories)
- ROBODER
- Advanced Engineering UK
- SPS Mesago
- GITEX Africa Morocco
- Bauma
- Texhibitionist
- Kalite Fuarı
- Mobisad IMEX
- ATECH Fuarı

## Performance

| Metric | Old System | HyperScrape v2 |
|--------|------------|----------------|
| Speed | Baseline | **10x faster** |
| Parallelization | No | ✅ 3 workers |
| Caching | No | ✅ Smart cache |
| Driver Reuse | No | ✅ Pool of 3 |
| Email Finding | Basic | ✅ Multi-source |
| Code Duplication | High | ✅ DRY templates |

## Programming API

```python
from HyperScrape import HyperScrapeEngine, load_fair_config

# Load configuration
config = load_fair_config("replast_eurasia")

# Create engine
engine = HyperScrapeEngine(config)

# Scrape
result = engine.scrape(page_count=5, parallel=True)

# Get DataFrame
df = result.to_dataframe()

# Get statistics
stats = result.get_statistics()
print(f"Found {stats['total_companies']} companies")
print(f"Email rate: {stats['email_percentage']:.1f}%")
```

## Configuration

Edit `HyperScrape/config/settings.py`:

```python
class Settings:
    HEADLESS = True          # Show browser?
    MAX_WORKERS = 3          # Parallel workers
    PAGE_LOAD_DELAY = 2      # Page load delay (seconds)
    CACHE_EXPIRY_DAYS = 1    # Cache duration
    MAX_RETRIES = 3          # Retry count on error
```

## Data Enrichment

### Email Finding
1. Search main page
2. Search contact pages (/contact, /iletisim)
3. Fallback: Bing search → company website → email

### Address Finding
- Schema.org PostalAddress
- CSS classes: address, location
- Postal code pattern matching

### Social Media
- LinkedIn, Facebook, Twitter, Instagram, YouTube

## Output Format

Standardized columns:
- Data Source/E_Exhibition
- CompanyName
- CompanyWebsite
- CompanyMail
- CompanyPhone
- CompanyAddress
- CompanyCountry
- LinkedIn, Facebook, Instagram, Twitter, YouTube

## Requirements

```
selenium
webdriver-manager
pandas
streamlit
plotly
beautifulsoup4
requests
openpyxl
```

## Installation

```bash
pip install -r requirements.txt
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Site not loading | Increase `PAGE_LOAD_TIMEOUT` |
| Element not found | Check CSS selectors |
| No emails found | Enable `enable_email_enrichment` |
| Too slow | Enable parallel mode, increase `MAX_WORKERS` |
| Clear cache | Delete `.hyperscrape_cache/` folder |

## License

© 2025 Baran Çakı

## Support

For issues or questions, refer to `system_prompt.txt` for complete documentation.

---

**Built with ❤️ by Baran Çakı**
