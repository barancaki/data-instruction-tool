"""
Data models for companies, fair configurations, and scrape results.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class Company:
    """Represents a company with all its information."""
    
    # Core info
    name: str = ""
    website: str = ""
    email: str = ""
    email2: str = ""
    phone: str = ""
    address: str = ""
    city: str = ""
    country: str = ""
    zip_code: str = ""
    
    # Additional info
    products: str = ""
    data_source: str = ""
    
    # Social media (enriched data)
    linkedin: str = ""
    facebook: str = ""
    instagram: str = ""
    twitter: str = ""
    youtube: str = ""
    
    # Other fields
    stand_no: str = ""
    detail_link: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "Data Source/E_Exhibition": self.data_source,
            "Product": self.products,
            "CompanyName": self.name,
            "CompanyWebsite": self.website,
            "CompanyMail": self.email,
            "CompanyMail2": self.email2,
            "CompanyPhone": self.phone,
            "CompanyAddress": self.address,
            "CompanyZipCode": self.zip_code,
            "CompanyCity": self.city,
            "CompanyCountry": self.country,
            "LinkedIn": self.linkedin,
            "Facebook": self.facebook,
            "Instagram": self.instagram,
            "Twitter": self.twitter,
            "YouTube": self.youtube,
            "Stand No": self.stand_no,
            "Detail Link": self.detail_link,
        }


@dataclass
class FairConfig:
    """Configuration for a fair/exhibition website."""
    
    # Basic info
    name: str
    base_url: str
    template_type: str
    
    # Pagination
    pagination_type: str = "page"  # 'page', 'offset', 'scroll', 'load_more', 'none'
    pagination_param: str = "page"  # URL parameter name
    pagination_url: str = ""  # Full URL pattern with {page} placeholder
    max_page_auto_detect: bool = True
    
    # Selectors
    selectors: Dict[str, str] = field(default_factory=dict)
    
    # Template-specific config
    template_config: Dict[str, Any] = field(default_factory=dict)
    
    # Enrichment
    enable_email_enrichment: bool = True
    enable_address_enrichment: bool = False
    enable_social_enrichment: bool = False
    
    def get_page_url(self, page_num: int) -> str:
        """Get URL for a specific page number."""
        if self.pagination_type == "none":
            return self.base_url
        elif self.pagination_url:
            return self.pagination_url.format(page=page_num)
        elif self.pagination_type == "page":
            separator = "&" if "?" in self.base_url else "?"
            return f"{self.base_url}{separator}{self.pagination_param}={page_num}"
        else:
            return self.base_url


@dataclass
class ScrapeResult:
    """Result of a scraping operation."""
    
    companies: List[Company]
    total_pages: int 
    success_count: int
    error_count: int
    duration_seconds: float
    config_name: str = ""
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert companies to pandas DataFrame."""
        if not self.companies:
            return pd.DataFrame()
        
        data = [company.to_dict() for company in self.companies]
        return pd.DataFrame(data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the scraping result."""
        df = self.to_dataframe()
        
        stats = {
            "total_companies": len(self.companies),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "duration_seconds": self.duration_seconds,
            "companies_per_second": len(self.companies) / self.duration_seconds if self.duration_seconds > 0 else 0,
        }
        
        if not df.empty:
            stats["email_found"] = df["CompanyMail"].notna().sum()
            stats["email_percentage"] = (stats["email_found"] / len(df)) * 100
            stats["website_found"] = df["CompanyWebsite"].notna().sum()
            stats["countries"] = df["CompanyCountry"].nunique()
            
        return stats
