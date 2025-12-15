"""
HyperScrape Engine v2 - Streamlit Interface
Next-generation fair scraping with 10x performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import io
import time
from HyperScrape import HyperScrapeEngine, load_all_configs, load_fair_config
from HyperScrape.config.loader import get_fair_list
from auth_helper import check_authentication, get_user_info, show_user_info_sidebar

# Page config
st.set_page_config(
    page_title="HyperScrape Engine v2",
    page_icon="🚀",
    layout="wide"
)

# Authentication
check_authentication()
user_info = get_user_info()
show_user_info_sidebar()

# Sidebar
with st.sidebar:
    st.header("🚀 HyperScrape v2")
    st.markdown("*Next-gen scraping engine*")
    st.markdown("---")
    
    st.markdown("### ⚡ Features")
    st.markdown("""
    - ✅ 30+ Pre-configured Sites
    - ✅ 10x Faster Performance  
    - ✅ Auto Email Finding
    - ✅ Smart Caching
    - ✅ Parallel Processing
    """)
    
    st.markdown("---")
    st.text('© Baran Çakı 2025')

# Main content
st.title("🚀 HyperScrape Engine v2")
st.markdown("**The next generation fair scraper** - 10x faster, infinitely scalable")

# Get all available fairs
fair_list = get_fair_list()

# Create selection interface
col1, col2 = st.columns([2, 1])

with col1:
    # Fair selection
    fair_options = {f"{fair['name']} ({fair['template']})": fair['key'] for fair in fair_list}
    selected_fair_display = st.selectbox(
        "📌 Select Fair/Exhibition",
        options=list(fair_options.keys()),
        help="Choose from 30+ pre-configured exhibition sites"
    )
    selected_fair_key = fair_options[selected_fair_display]

with col2:
    # Page count
    page_count = st.number_input(
        "📄 Number of Pages",
        min_value=1,
        max_value=100,
        value=5,
        help="Number of pages to scrape (leave blank for auto-detect)"
    )

# Advanced settings
with st.expander("⚙️ Advanced Settings"):
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        parallel_mode = st.checkbox("Parallel Processing", value=True, help="3-5x faster scraping")
    
    with col_b:
        enable_email_enrichment = st.checkbox("Email Enrichment", value=True, help="Find emails from websites")
    
    with col_c:
        use_cache = st.checkbox("Use Cache", value=True, help="Avoid re-scraping recent data")

# Load configuration for selected fair
config = load_fair_config(selected_fair_key)

if config:
    # Display configuration info
    with st.expander("ℹ️ Configuration Details"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Template Type", config.template_type)
        with col2:
            st.metric("Pagination", config.pagination_type)
        with col3:
            st.metric("Email Enrichment", "✅" if config.enable_email_enrichment else "❌")

# Scrape button
if st.button("🚀 Start Scraping", type="primary", use_container_width=True):
    if not config:
        st.error("Configuration not found for selected fair!")
    else:
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_placeholder = st.empty()
        
        # Apply settings to config
        config.enable_email_enrichment = enable_email_enrichment
        
        # Initialize engine
        engine = HyperScrapeEngine(config)
        
        # Progress callback
        def update_progress(progress: float, status: str):
            progress_bar.progress(progress)
            status_text.text(f"🔄 {status}")
        
        try:
            # Start scraping
            start_time = time.time()
            status_text.text("🔄 Initializing scraper...")
            
            result = engine.scrape(
                page_count=page_count if page_count else None,
                parallel=parallel_mode,
                progress_callback=update_progress
            )
            
            duration = time.time() - start_time
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Success message
            st.success(f"✅ Scraping completed! Found {len(result.companies)} companies in {duration:.2f} seconds")
            
            # Get statistics
            stats = result.get_statistics()
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Companies", stats['total_companies'])
            with col2:
                st.metric("Emails Found", f"{stats.get('email_found', 0)} ({stats.get('email_percentage', 0):.1f}%)")
            with col3:
                st.metric("Duration", f"{stats['duration_seconds']:.2f}s")
            with col4:
                st.metric("Speed", f"{stats['companies_per_second']:.1f} co/s")
            
            # Convert to DataFrame
            df = result.to_dataframe()
            
            if not df.empty:
                # Display data
                st.subheader("📊 Scraped Data")
                st.dataframe(df, use_container_width=True, height=400)
                
                # Download options
                st.subheader("📥 Download Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Excel download
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_buffer,
                        file_name=f"{selected_fair_key}_{int(time.time())}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # CSV download
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{selected_fair_key}_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # JSON download
                    json_str = df.to_json(orient='records', force_ascii=False, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"{selected_fair_key}_{int(time.time())}.json",
                        mime="application/json"
                    )
                
                # Visualizations
                st.subheader("📈 Visualizations")
                
                # Country distribution
                if "CompanyCountry" in df.columns and df["CompanyCountry"].notna().sum() > 0:
                    country_counts = df["CompanyCountry"].value_counts().head(15).reset_index()
                    country_counts.columns = ["Country", "Count"]
                    
                    fig1 = px.bar(
                        country_counts,
                        x="Country",
                        y="Count",
                        title="Top 15 Countries by Company Count",
                        color="Count",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Email availability
                email_stats = pd.DataFrame({
                    'Status': ['Has Email', 'No Email'],
                    'Count': [
                        df['CompanyMail'].notna().sum(),
                        df['CompanyMail'].isna().sum()
                    ]
                })
                
                fig2 = px.pie(
                    email_stats,
                    values='Count',
                    names='Status',
                    title='Email Availability',
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            else:
                st.warning("No companies found. Try adjusting the page count or check the website.")
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

# Information section
st.markdown("---")
st.markdown("### 📚 How to Add a New Fair")
st.code("""
# 1. Open: HyperScrape/config/fair_definitions.py
# 2. Add your fair definition:

"my_fair": {
    "name": "My Fair 2025",
    "base_url": "https://example.com/exhibitors",
    "template": "tuyap_list",  # Choose appropriate template
    "pagination_url": "https://example.com/exhibitors?page={page}",
    "selectors": {
        "company_list": ".exhibitor-item",
        "company_name": ".name",
        # ... other selectors
    }
}

# 3. Save and restart - that's it!
""", language="python")

st.markdown("### 🎯 Available Templates")
templates_info = pd.DataFrame([
    {"Template": "tuyap_list", "Use Case": "Filter list items with detail buttons", "Sites": "7"},
    {"Template": "link_detail", "Use Case": "Brand cards → detail pages", "Sites": "4"},
    {"Template": "deutsche_platform", "Use Case": "Deutsche Messe platform sites", "Sites": "4"},
    {"Template": "modal_popup", "Use Case": "Modal popup details", "Sites": "1"},
    {"Template": "card_list", "Use Case": "Card-based layouts", "Sites": "1"},
    {"Template": "member_cards", "Use Case": "Membership directories", "Sites": "5"},
    {"Template": "search_results", "Use Case": "Search result layouts", "Sites": "2"},
    {"Template": "scroll_load", "Use Case": "Infinite scroll sites", "Sites": "2"},
    {"Template": "xpath_detail", "Use Case": "Complex XPath extraction", "Sites": "1"},
    {"Template": "text_list", "Use Case": "Simple text lists", "Sites": "1"},
    {"Template": "image_gallery", "Use Case": "Image galleries", "Sites": "1"},
    {"Template": "blog_cards", "Use Case": "Blog-style cards", "Sites": "1"},
])

st.dataframe(templates_info, use_container_width=True, hide_index=True)
