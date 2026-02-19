---
name: fair-scraper-generator
description: Generate and integrate new fair exhibition scrapers in this codebase using the standard Selenium + Streamlit patterns. Use when the user wants a new fair/exhibitor scraper added to `Scrape/ai_created.py` and wired into `pages/2_Fair_Scraper.py`.
---

# Fair Scraper Generator Skill

This skill automates the creation of new fair exhibition scrapers by following standardized patterns in the codebase.

## Objective
Generate a Selenium-based scraper function in `Scrape/ai_created.py` and integrate it into the Streamlit UI in `pages/2_Fair_Scraper.py`.

## Pre-requisites
Before generating code, ask the user for:
1. **Target URL**: The exhibition list URL.
2. **Pagination Type**: (Page-based URL, "Load More" button, Infinite Scroll, or Single Page).
3. **List Selectors**: CSS selector for the container of each company in the list.
4. **Field Selectors**: CSS selectors for Company Name, Website, and any other relevant fields.

## Implementation Steps

### 1. Generate Scraper Function
Add a new function to `Scrape/ai_created.py`.
- **Use the Template**: Follow the structure in `.codex/skills/fair-scraper-generator/references/scraper_template.py`.
- **Helpers**: Always use `from Scrape.scrape import find_email_advanced, handle_cookie_consent_final`.
- **Data Schema**: Every scraper MUST include the following columns in the final DataFrame:
    - `Data Source/ExhibitionName`
    - `ExhibitionProductGroup`
    - `CompanyName`
    - `CompanyWebsite`
    - `CompanyMail`
    - `CompanyMail2`
    - `CompanyPhone`
    - `CompanyAddress`
    - `CompanyZipCode`
    - `CompanyCity`
    - `CompanyCountry`
    - `CompanyBusinessType`
- **Download Logic**: Ensure the function includes the Streamlit code to display the DataFrame and provide a CSV download button.

### 2. Update Streamlit UI
Modify `pages/2_Fair_Scraper.py`.
- **Import**: Add the new function name to the import list from `Scrape.ai_created` at the top of the file.
- **UI Block**: Append a new `if "keyword" in url:` block following the pattern in `.codex/skills/fair-scraper-generator/references/ui_template.py`.
- **Inputs**: Use appropriate input fields (e.g., `st.number_input` for page counts).

## Verification
After generation, verify:
- No duplicate imports.
- Selenium driver is initialized with headless options and a custom user-agent.
- The `find_email_advanced` helper is used if a website URL is extracted.
- The UI block correctly triggers the new function.
