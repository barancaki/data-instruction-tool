"""
Fair website definitions.
Each fair is defined with its template type, URL patterns, and selectors.

ADDING A NEW FAIR:
==================
1. Choose the appropriate template based on the site structure:
   - 'tuyap_list': For sites with .filter-list__item structure
   - 'link_detail': For sites with brand cards that link to detail pages
   - 'deutsche_platform': For Deutsche Messe platform sites
   - 'modal_popup': For sites using modal popups
   - 'card_list': For sites with card-based layouts
   - 'member_cards': For membership directory sites
   - 'search_results': For sites with search result layouts
   - 'scroll_load': For infinite scroll sites
   - 'text_list': For sites with simple text lists
   - 'image_gallery': For sites using image galleries
   - 'xpath_detail': For sites requiring complex XPath extraction

2. Define selectors based on the template requirements
3. Set pagination type and parameters
4. Enable/disable email/address enrichment as needed

Example:
--------
"my_fair": {
    "name": "My Fair 2025",
    "base_url": "https://example.com/exhibitors",
    "template": "tuyap_list",
    "pagination_type": "page",
    "pagination_url": "https://example.com/exhibitors?page={page}",
    "selectors": {
        "company_list": ".exhibitor-item",
        "company_name": ".name",
        "address": ".address",
        ...
    },
    "enable_email_enrichment": True
}
"""

FAIR_DEFINITIONS = {
    # ========== TUYAP LIST PATTERN ==========
    # Sites using .filter-list__item structure with detail buttons
    
    "replast_eurasia": {
        "name": "Replast Eurasia",
        "base_url": "https://replasteurasia.com/katilimci-listesi",
        "template": "tuyap_list",
        "pagination_type": "page",
        "pagination_url": "https://replasteurasia.com/katilimci-listesi?page={page}",
        "selectors": {
            "company_list": ".filter-list__item",
            "company_name": ".table-block-content:nth-of-type(1)",
            "address": ".table-block-content:nth-of-type(2)",
            "phone": "a[href^='tel:']",
            "website": "a[href^='http']",
            "detail_button": ".js-open-table-detail",
            "product_groups": ".table-detail-wrapper__list-item"
        },
        "enable_email_enrichment": True
    },
    
    "burtarim": {
        "name": "Konya Tarım Fuarı",
        "base_url": "https://www.konyatarimfuari.com/katilimci-listesi",
        "template": "tuyap_list",
        "pagination_type": "page",
        "pagination_url": "https://www.konyatarimfuari.com/katilimci-listesi?page={page}",
        "selectors": {
            "company_list": ".filter-list__item",
            "company_name": ".table-block-content:nth-of-type(1)",
            "address": ".table-block-content:nth-of-type(2)",
            "phone": "a[href^='tel:']",
            "website": "a[href^='http']",
            "detail_button": ".js-open-table-detail",
            "product_groups": ".table-detail-wrapper__list-item"
        },
        "enable_email_enrichment": True
    },
    
    "pencere_fuari": {
        "name": "Avrasya Pencere Fuarı",
        "base_url": "https://www.avrasyapencerefuari.com/katilimci-listesi",
        "template": "tuyap_list",
        "pagination_type": "page",
        "pagination_url": "https://www.avrasyapencerefuari.com/katilimci-listesi?page={page}",
        "selectors": {
            "company_list": ".filter-list__item",
            "company_name": ".table-block-content:nth-of-type(1)",
            "address": ".table-block-content:nth-of-type(2)",
            "phone": "a[href^='tel:']",
            "website": "a[href^='http']",
            "detail_button": ".js-open-table-detail",
            "product_groups": ".table-detail-wrapper__list-item"
        },
        "enable_email_enrichment": True
    },
    
    "smtech_eurasia": {
        "name": "SMTech Eurasia",
        "base_url": "https://smtech-eurasia.com/katilimci-listesi",
        "template": "tuyap_list",
        "pagination_type": "page",
        "pagination_url": "https://smtech-eurasia.com/katilimci-listesi?page={page}",
        "selectors": {
            "company_list": ".filter-list__item",
            "company_name": ".table-block-content:nth-of-type(1)",
            "address": ".table-block-content:nth-of-type(2)",
            "phone": "a[href^='tel:']",
            "website": "a[href^='http']",
            "detail_button": ".js-open-table-detail",
            "product_groups": ".table-detail-wrapper__list-item"
        },
        "enable_email_enrichment": True
    },
    
    "iplik_fuari": {
        "name": "İplik Fuarı",
        "base_url": "https://iplikfuari.com/katilimci-listesi",
        "template": "tuyap_list",
        "pagination_type": "page",
        "pagination_url": "https://iplikfuari.com/katilimci-listesi?page={page}",
        "selectors": {
            "company_list": ".filter-list__item",
            "company_name": ".table-block-content:nth-of-type(1)",
            "address": ".table-block-content:nth-of-type(2)",
            "phone": "a[href^='tel:']",
            "website": "a[href^='http']",
            "detail_button": ".js-open-table-detail",
            "product_groups": ".table-detail-wrapper__list-item"
        },
        "enable_email_enrichment": True
    },
    
    "maktek": {
        "name": "Maktek Fuarı",
        "base_url": "https://maktekfuari.com/katilimci-listesi",
        "template": "tuyap_list",
        "pagination_type": "page",
        "pagination_url": "https://maktekfuari.com/katilimci-listesi?page={page}",
        "selectors": {
            "company_list": ".filter-list__item",
            "company_name": ".table-block-content:nth-of-type(1)",
            "address": ".table-block-content:nth-of-type(2)",
            "phone": "a[href^='tel:']",
            "website": "a[href^='http']",
            "detail_button": ".js-open-table-detail",
            "product_groups": ".table-detail-wrapper__list-item"
        },
        "enable_email_enrichment": True
    },
    
    "expomed": {
        "name": "Expomed Istanbul",
        "base_url": "https://expomedistanbul.com/katilimci-listesi",
        "template": "tuyap_list",
        "pagination_type": "page",
        "pagination_url": "https://expomedistanbul.com/katilimci-listesi?page={page}",
        "selectors": {
            "company_list": ".filter-list__item",
            "company_name": ".table-block-content:nth-of-type(1)",
            "address": ".table-block-content:nth-of-type(2)",
            "phone": "a[href^='tel:']",
            "website": "a[href^='http']",
            "detail_button": ".js-open-table-detail",
            "product_groups": ".table-detail-wrapper__list-item"
        },
        "enable_email_enrichment": True
    },
    
    # ========== LINK + DETAIL PAGE PATTERN ==========
    # Sites with brand items that link to detail pages
    
    "packaging_fair": {
        "name": "Packaging Fair",
        "base_url": "https://packagingfair.com/katilimci-listesi",
        "template": "link_detail",
        "pagination_type": "page",
        "pagination_url": "https://packagingfair.com/katilimci-listesi?page={page}",
        "selectors": {
            "brand_cards": "div.brand-item.mt-30.active a.brand-link",
            "company_name": "h1.company-title",
            "country_icon": "i.fa.fa-globe",
            "info_list": "div.schedule-list ul li",
            "phone_icon": "fa-phone",
            "address_icon": "fa-location-dot",
            "website_icon": "fa-globe"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://packagingfair.com"
        },
        "enable_email_enrichment": True
    },
    
    "plast_eurasia": {
        "name": "Plast Eurasia",
        "base_url": "https://plasteurasia.com/katilimci-listesi",
        "template": "link_detail",
        "pagination_type": "page",
        "pagination_url": "https://plasteurasia.com/katilimci-listesi?page={page}",
        "selectors": {
            "brand_cards": "div.brand-item.mt-30.active a.brand-link",
            "company_name": "h1.company-title",
            "country_icon": "i.fa.fa-globe",
            "info_list": "div.schedule-list ul li",
            "phone_icon": "fa-phone",
            "address_icon": "fa-location-dot",
            "website_icon": "fa-globe"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://plasteurasia.com"
        },
        "enable_email_enrichment": True
    },
    
    "intermob": {
        "name": "Intermob Istanbul",
        "base_url": "https://www.intermobistanbul.com/katilimci-listesi",
        "template": "link_detail",
        "pagination_type": "page",
        "pagination_url": "https://www.intermobistanbul.com/katilimci-listesi?page={page}",
        "selectors": {
            "brand_cards": "div.brand-item.mt-30.active a.brand-link",
            "company_name": "h1.company-title",
            "country_icon": "i.fa.fa-globe",
            "info_list": "div.schedule-list ul li",
            "phone_icon": "fa-phone",
            "address_icon": "fa-location-dot",
            "website_icon": "fa-globe"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://www.intermobistanbul.com"
        },
        "enable_email_enrichment": True
    },
    
    "woodtech": {
        "name": "Woodtech Istanbul",
        "base_url": "https://www.woodtechistanbul.com/katilimci-listesi",
        "template": "link_detail",
        "pagination_type": "page",
        "pagination_url": "https://www.woodtechistanbul.com/katilimci-listesi?page={page}",
        "selectors": {
            "brand_cards": "div.brand-item.mt-30.active a.brand-link",
            "company_name": "h1.company-title",
            "country_icon": "i.fa.fa-globe",
            "info_list": "div.schedule-list ul li",
            "phone_icon": "fa-phone",
            "address_icon": "fa-location-dot",
            "website_icon": "fa-globe"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://www.woodtechistanbul.com"
        },
        "enable_email_enrichment": True
    },
    
    # ========== DEUTSCHE MESSE PLATFORM PATTERN ==========
    # Sites using Deutsche Messe's platform
    
    "win_eurasia": {
        "name": "WIN Eurasia",
        "base_url": "https://platform.win-eurasia.com/participants",
        "template": "deutsche_platform",
        "pagination_type": "page",
        "pagination_url": "https://platform.win-eurasia.com/participants?page={page}",
        "selectors": {
            "company_card": "div.cell.small-12",
            "detail_link": "a.o.link.as-block.fx.dropshadow.for-child",
            "company_name": ".search-snippet-name",
            "country": ".search-snippet-description",
            "address_list": "ul.t.set-250-regular.as-copy li",
            "email": "a[href^='mailto:']",
            "phone": "a"  # Contains "Telefon:" text
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://platform.win-eurasia.com"
        },
        "enable_email_enrichment": False  # Already has emails
    },
    
    "hub_of_warehouse": {
        "name": "Hub of Warehouse",
        "base_url": "https://platform.hubofwarehouse.com/participants",
        "template": "deutsche_platform",
        "pagination_type": "page",
        "pagination_url": "https://platform.hubofwarehouse.com/participants?page={page}",
        "selectors": {
            "company_card": "div.cell.small-12",
            "detail_link": "a.o.link.as-block.fx.dropshadow.for-child",
            "company_name": ".search-snippet-name",
            "country": ".search-snippet-description",
            "address_list": "ul.t.set-250-regular.as-copy li",
            "email": "a[href^='mailto:']",
            "phone": "a"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://platform.hubofwarehouse.com"
        },
        "enable_email_enrichment": False
    },
    
    "sodex": {
        "name": "SODEX",
        "base_url": "https://platform.sodex.com.tr/participants",
        "template": "deutsche_platform",
        "pagination_type": "page",
        "pagination_url": "https://platform.sodex.com.tr/participants?page={page}",
        "selectors": {
            "company_card": "div.cell.small-12",
            "detail_link": "a.o.link.as-block.fx.dropshadow.for-child",
            "company_name": ".search-snippet-name",
            "country": ".search-snippet-description",
            "address_list": "ul.t.set-250-regular.as-copy li",
            "email": "a[href^='mailto:']",
            "phone": "a"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://platform.sodex.com.tr"
        },
        "enable_email_enrichment": False
    },
    
    "automechanika": {
        "name": "Automechanika Istanbul",
        "base_url": "https://automechanikaistanbulplus.com/participants",
        "template": "deutsche_platform",
        "pagination_type": "page",
        "pagination_url": "https://automechanikaistanbulplus.com/participants?page={page}",
        "selectors": {
            "company_list": "div.list__item",
            "detail_link": ".list__company a",
            "company_name": ".list__company a",
            "address": ".location__pin",
            "website_links": ".field__link"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://automechanikaistanbulplus.com",
            "variant": "automechanika"  # Slightly different structure
        },
        "enable_email_enrichment": True
    },
    
    # ========== MODAL POPUP PATTERN ==========
    # Sites using modal popups for details
    
    "evchargeshow": {
        "name": "EV Charge Show",
        "base_url": "https://www.evchargeshow.com/exhibitor",
        "template": "modal_popup",
        "pagination_type": "scroll",  # Infinite scroll
        "selectors": {
            "company_cards": "#exhibitorsList .col-lg-4.col-md-6.col-sm-12",
            "company_name": "h5",
            "country": ".text-muted",
            "detail_button": "button",
            "modal": ".modal.show",
            "website_in_modal": ".modal.show a[href^='http']",
            "close_button": ".modal.show button.btn-close"
        },
        "template_config": {
            "requires_scroll": True,
            "country_index": 1  # Second .text-muted element
        },
        "enable_email_enrichment": True
    },
    
    # ========== CARD LIST PATTERN ==========
    # Sites using card-based layouts
    
    "hvacr_world": {
        "name": "HVACR World 2025",
        "base_url": "https://exhibitors.hvacr-world.com/hvacr-world-2025/Exhibitor",
        "template": "card_list",
        "pagination_type": "offset",  # Uses JavaScript pagination
        "pagination_param": "offset",
        "selectors": {
            "company_cards": "div.card.h-100",
            "detail_link": "h5.card-title a",
            "company_name": "h1.company-title",
            "stand_info": "h6",
            "category": "span.badge.bg-secondary",
            "info_elements": "div.company-info div",
            "social_links": "div.social-links a"
        },
        "template_config": {
            "needs_detail_page": True,
            "items_per_page": 24,
            "pagination_function": "searchFilter"  # JavaScript function name
        },
        "enable_email_enrichment": True,
        "enable_social_enrichment": True
    },
    
    # ========== MEMBER CARDS PATTERN ==========
    # Membership directory sites
    
    "enosad_proses": {
        "name": "ENOSAD Proses Otomasyonu",
        "base_url": "https://enosad.org.tr/tr/proses-otomasyonu",
        "template": "member_cards",
        "pagination_type": "none",
        "selectors": {
            "member_cards": "div.grid.grid-cols-1.md\\:grid-cols-2 a[href^='/tr/']",
            "company_name_xpath": "//h3[text()='Üye Kurumsal Firma Ünvanı']/following-sibling::div",
            "phone_xpath": "//h3[contains(text(),'Telefon')]/following-sibling::div",
            "email_xpath": "//h3[contains(text(),'Kontakt e-posta')]/following-sibling::div",
            "address_xpath": "//h3[contains(text(),'Adresi')]/following-sibling::div",
            "website_xpath": "//h3[contains(text(),'Web sitesi')]/following-sibling::a"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_link": "https://enosad.org.tr"
        },
        "enable_email_enrichment": False  # Already has emails
    },
    
    "enosad_fabrika": {
        "name": "ENOSAD Fabrika Otomasyonu",
        "base_url": "https://enosad.org.tr/tr/fabrika-otomasyonu",
        "template": "member_cards",
        "pagination_type": "none",
        "selectors": {
            "member_cards": "div.grid.grid-cols-1.md\\:grid-cols-2 a[href^='/tr/']",
            "company_name_xpath": "//h3[text()='Üye Kurumsal Firma Ünvanı']/following-sibling::div",
            "phone_xpath": "//h3[contains(text(),'Telefon')]/following-sibling::div",
            "email_xpath": "//h3[contains(text(),'Kontakt e-posta')]/following-sibling::div",
            "address_xpath": "//h3[contains(text(),'Adresi')]/following-sibling::div",
            "website_xpath": "//h3[contains(text(),'Web sitesi')]/following-sibling::a"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_link": "https://enosad.org.tr"
        },
        "enable_email_enrichment": False
    },
    
    "enosad_robotik": {
        "name": "ENOSAD Robotik ve Mekatronik",
        "base_url": "https://enosad.org.tr/tr/robotik-ve-mekatronik",
        "template": "member_cards",
        "pagination_type": "none",
        "selectors": {
            "member_cards": "div.grid.grid-cols-1.md\\:grid-cols-2 a[href^='/tr/']",
            "company_name_xpath": "//h3[text()='Üye Kurumsal Firma Ünvanı']/following-sibling::div",
            "phone_xpath": "//h3[contains(text(),'Telefon')]/following-sibling::div",
            "email_xpath": "//h3[contains(text(),'Kontakt e-posta')]/following-sibling::div",
            "address_xpath": "//h3[contains(text(),'Adresi')]/following-sibling::div",
            "website_xpath": "//h3[contains(text(),'Web sitesi')]/following-sibling::a"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_link": "https://enosad.org.tr"
        },
        "enable_email_enrichment": False
    },
    
    "enosad_sanayi": {
        "name": "ENOSAD Sanayide Dijital Dönüşüm",
        "base_url": "https://enosad.org.tr/tr/sanayide-dijital-donusum",
        "template": "member_cards",
        "pagination_type": "none",
        "selectors": {
            "member_cards": "div.grid.grid-cols-1.md\\:grid-cols-2 a[href^='/tr/']",
            "company_name_xpath": "//h3[text()='Üye Kurumsal Firma Ünvanı']/following-sibling::div",
            "phone_xpath": "//h3[contains(text(),'Telefon')]/following-sibling::div",
            "email_xpath": "//h3[contains(text(),'Kontakt e-posta')]/following-sibling::div",
            "address_xpath": "//h3[contains(text(),'Adresi')]/following-sibling::div",
            "website_xpath": "//h3[contains(text(),'Web sitesi')]/following-sibling::a"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_link": "https://enosad.org.tr"
        },
        "enable_email_enrichment": False
    },
    
    "roboder": {
        "name": "ROBODER",
        "base_url": "https://uyeler.roboder.org.tr/",
        "template": "member_cards",
        "pagination_type": "load_more",  # Uses load more button
        "selectors": {
            "load_more_button": "div.jet-filters-pagination__link",
            "member_cards": "div.elementor-widget-container a[href^='https://uyeler.roboder.org.tr/firma/']",
            "company_name": "h2.elementor-heading-title",
            "website_xpath": "//span[contains(text(),'.com') or contains(text(),'.net') or contains(text(),'.org')]",
            "email_xpath": "//span[contains(text(),'@')]",
            "phone_xpath": "//span[contains(text(),'0')]",
            "address_xpath": "//span[contains(text(),'/')]"
        },
        "template_config": {
            "needs_detail_page": True,
            "requires_load_more": True
        },
        "enable_email_enrichment": False
    },
    
    # ========== SEARCH RESULTS PATTERN ==========
    # Sites with search result-style layouts
    
    "advanced_engineering": {
        "name": "Advanced Engineering UK",
        "base_url": "https://www.advancedengineeringuk.com/exhibitors/",
        "template": "search_results",
        "pagination_type": "page",
        "pagination_url": "https://www.advancedengineeringuk.com/exhibitors/?stands%5Bpage%5D={page}",
        "selectors": {
            "company_cards": "li.ais-Hits-item",
            "detail_link": "a.card__link",
            "company_name": "h1.stand-details__title",
            "website": ".stand-details__info-line-content a",
            "address": ".contact-info-card__info-line-content",
            "products": "div.stand-details__info-line-content div.stand-details__category-pill"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://www.advancedengineeringuk.com"
        },
        "enable_email_enrichment": True
    },
    
    "mesago": {
        "name": "SPS Mesago",
        "base_url": "https://sps.mesago.com/nuernberg/en/exhibitor-search.html",
        "template": "search_results",
        "pagination_type": "page",
        "pagination_url": "https://sps.mesago.com/nuernberg/en/exhibitor-search.html?page={page}&pagesize=30",
        "selectors": {
            "company_cards": "div.ex-exhibitor-search-results-container a.a-link--no-focus",
            "company_name": "h1.ex-exhibitor-detail__title-headline",
            "phone": "a.ex-contact-box__address-field-tel-number",
            "website": "a.ex-contact-box__website-link",
            "email": "a.ex-contact-box__contact-btn",  # mailto link
            "products": "div.ex-keyword-list__container ul li.ex-keyword-list__keyword"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://sps.mesago.com"
        },
        "enable_email_enrichment": True
    },
    
    # ========== SCROLL LOAD PATTERN ==========
    # Sites with infinite scroll
    
    "gitex_africa": {
        "name": "GITEX Africa Morocco",
        "base_url": "https://exhibitors.gitexafrica.com/gitex-africa-2025/Exhibitor",
        "template": "scroll_load",
        "pagination_type": "scroll",
        "selectors": {
            "company_cards": "div.item.col-12.list-group-item",
            "detail_button": "div.button_block a.btn",
            "company_name": "h4.group.card-title.inner.list-group-item-heading",
            "stand_info": "p",  # Contains "Stand No"
            "country": "span[style*='float:left']",
            "website": "li.social_website a",
            "products": "ul.sector_block li"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://exhibitors-dwtc.exhibitoronlinemanual.com",
            "scroll_count": 10  # Default scroll times
        },
        "enable_email_enrichment": True,
        "enable_social_enrichment": True
    },
    
    "bauma": {
        "name": "Bauma Exhibitors",
        "base_url": "https://exhibitors.bauma.de/en/exhibitors-and-products/exhibitors-brand-names",
        "template": "scroll_load",
        "pagination_type": "load_more",
        "selectors": {
            "company_rows": "td.content_company",
            "load_more_button": "tr.lazymore td",
            "company_name": "td.content_company",
            "country": "td.content_country"
        },
        "template_config": {
            "requires_load_more": True,
            "base_url_for_join": "https://exhibitors.bauma.de"
        },
        "enable_email_enrichment": True
    },
    
    # ========== XPATH DETAIL PATTERN ==========
    # Sites requiring complex XPath extraction
    
    "texhibitionist": {
        "name": "Texhibitionist",
        "base_url": "https://www.texhibitionist.com/katilimcilar",
        "template": "xpath_detail",
        "pagination_type": "page",
        "pagination_url": "https://www.texhibitionist.com/katilimcilar?page={page}",
        "selectors": {
            "company_table": "div.row.row-cols-2.row-cols-lg-3.g-2.g-lg-4.gy-5",
            "company_links": "div.col a",
            "company_name": "div.title",
            "email_xpath": "//div[@class='key'][contains(text(), 'E-mail')]/following-sibling::*[1]",
            "phone_xpath": "//div[@class='key'][contains(text(), 'Telefon')]/following-sibling::*[1]",
            "website_xpath": "//div[@class='key'][contains(text(), 'Web Site')]/following-sibling::*[1]",
            "address": "div.address"
        },
        "template_config": {
            "needs_detail_page": True,
            "base_url_for_join": "https://www.texhibitionist.com"
        },
        "enable_email_enrichment": False  # Already has emails
    },
    
    # ========== TEXT LIST PATTERN ==========
    # Simple text list sites
    
    "kalite_fuari": {
        "name": "Kalite Fuarı",
        "base_url": "https://kalitefuari.com/katilimci-listesi/",
        "template": "text_list",
        "pagination_type": "none",
        "selectors": {
            "list_container": "div.wpb_wrapper h4"
        },
        "template_config": {
            "separator": "<br>",  # HTML separator
            "clean_newlines": True
        },
        "enable_email_enrichment": True
    },
    
    # ========== IMAGE GALLERY PATTERN ==========
    # Image gallery-based sites
    
    "mobisadimex": {
        "name": "Mobisad IMEX",
        "base_url": "https://www.mobisadimex.com/2024-katilimci-listesi/",
        "template": "image_gallery",
        "pagination_type": "none",
        "selectors": {
            "gallery_id": "#gallery-1",
            "gallery_items": "figure.gallery-item",
            "image": "img"
        },
        "template_config": {
            "extract_name_from": "src",  # Extract from image src
            "name_pattern": r"/([^/]+)-logo",  # Regex pattern
            "name_cleanup": ["-", " "],  # Replace - with space
            "titlecase": True
        },
        "enable_email_enrichment": True
    },
    
    # ========== INNOTRANS (SPECIAL CASE) ==========
    "innotrans": {
        "name": "InnoTrans",
        "base_url": "https://www.innotrans.de/en/visit/exhibitor-directory",
        "template": "custom",  # Uses custom implementation
        "pagination_type": "none",
        "template_config": {
            "custom_function": "scrape_innotrans"  # Points to specific function
        },
        "enable_email_enrichment": True
    },
    
    # ========== ATECH FUARI (BLOG CARDS) ==========
    "atech_fuari": {
        "name": "ATECH Fuarı",
        "base_url": "https://atechfuari.com/firmalar/",
        "template": "blog_cards",
        "pagination_type": "none",
        "selectors": {
            "company_cards": ".blog-standard-content.row .news-block-one.col-lg-4",
            "company_name": "h3 a",
            "website": "a.btn.btn-sm.btn-secondary"
        },
        "enable_email_enrichment": True
    }
}
