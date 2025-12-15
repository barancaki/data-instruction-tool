"""
Fair configuration loader.
Loads fair definitions and creates FairConfig objects.
"""

from typing import Dict, Optional
from ..models.fair import FairConfig
from .fair_definitions import FAIR_DEFINITIONS


def load_fair_config(fair_key: str) -> Optional[FairConfig]:
    """
    Load configuration for a specific fair.
    
    Args:
        fair_key: The unique key for the fair (e.g., 'replast_eurasia')
        
    Returns:
        FairConfig object or None if not found
    """
    if fair_key not in FAIR_DEFINITIONS:
        return None
    
    fair_def = FAIR_DEFINITIONS[fair_key]
    
    return FairConfig(
        name=fair_def["name"],
        base_url=fair_def["base_url"],
        template_type=fair_def["template"],
        pagination_type=fair_def.get("pagination_type", "page"),
        pagination_param=fair_def.get("pagination_param", "page"),
        pagination_url=fair_def.get("pagination_url", ""),
        max_page_auto_detect=fair_def.get("max_page_auto_detect", True),
        selectors=fair_def.get("selectors", {}),
        template_config=fair_def.get("template_config", {}),
        enable_email_enrichment=fair_def.get("enable_email_enrichment", True),
        enable_address_enrichment=fair_def.get("enable_address_enrichment", False),
        enable_social_enrichment=fair_def.get("enable_social_enrichment", False),
    )


def load_all_configs() -> Dict[str, FairConfig]:
    """
    Load all fair configurations.
    
    Returns:
        Dictionary mapping fair keys to FairConfig objects
    """
    return {
        fair_key: load_fair_config(fair_key)
        for fair_key in FAIR_DEFINITIONS.keys()
    }


def get_fair_list() -> list[Dict[str, str]]:
    """
    Get list of all available fairs with their basic info.
    
    Returns:
        List of dictionaries with 'key', 'name', 'template' fields
    """
    return [
        {
            "key": fair_key,
            "name": fair_def["name"],
            "template": fair_def["template"],
            "url": fair_def["base_url"]
        }
        for fair_key, fair_def in FAIR_DEFINITIONS.items()
    ]
