#!/usr/bin/env python3
"""
Streamlit Application for Auto- Segmentation Engine
This application provides a user-friendly GUI for classifying companies based on their website content
using semantic similarity with predefined product categories.
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import re
import time
from urllib.parse import urlparse
import warnings
import io
import sys
import subprocess
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from auth_helper import check_authentication, get_user_info, show_user_info_sidebar

# -------------------------
# Authentication
# -------------------------
check_authentication()
user_info = get_user_info()
show_user_info_sidebar()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Auto- Segmentation Engine",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_semantic_model():
    """
    Load the sentence-transformers model for semantic similarity.
    This function is cached to avoid reloading the model on every run.
    """
    try:
        with st.spinner("Loading semantic model..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading semantic model: {e}")
        return None

def install_required_packages():
    """
    Check for and install required packages if they are not already installed.
    """
    required_packages = [
        'pandas',
        'requests', 
        'beautifulsoup4',
        'sentence-transformers',
        'scipy',
        'openpyxl'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'beautifulsoup4':
                import bs4
            elif package == 'sentence-transformers':
                import sentence_transformers
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.warning(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                st.success(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                st.error(f"✗ Failed to install {package}: {e}")
                return False
    return True

def clean_text(text):
    """
    Clean and preprocess text by removing extra whitespace, 
    non-alphanumeric characters, and common stop words.
    """
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

@st.cache_resource
def create_robust_session():
    """
    Create a robust requests session with retry strategy and proper headers.
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def get_random_user_agent():
    """
    Get a random user agent to avoid detection.
    """
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    return random.choice(user_agents)

def scrape_website_content(url, timeout=15):
    """
    Scrape content from a website URL with robust error handling.
    Returns cleaned text content from the website.
    """
    session = create_robust_session()
    
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = 'https://' + url
        
        # Set comprehensive headers to mimic a real browser
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',  # Remove zstd to avoid compression issues
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Make request with timeout and proper error handling
        response = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        
        # Check for successful response
        if response.status_code == 200:
            # Handle different content encodings
            content = response.content
            
            # Try to decode content properly
            try:
                if response.encoding:
                    text = response.text
                else:
                    # Fallback to UTF-8 if encoding is not specified
                    text = content.decode('utf-8', errors='ignore')
            except UnicodeDecodeError:
                # If UTF-8 fails, try other encodings
                try:
                    text = content.decode('latin-1', errors='ignore')
                except:
                    text = content.decode('utf-8', errors='replace')
            
            # Parse HTML content
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text content
            text_content = soup.get_text()
            
            # Clean the text
            cleaned_text = clean_text(text_content)
            
            # Limit text length to avoid memory issues (first 5000 characters)
            if len(cleaned_text) > 5000:
                cleaned_text = cleaned_text[:5000]
            
            return cleaned_text
            
        else:
            # Handle specific HTTP status codes
            if response.status_code == 403:
                st.warning(f"Access forbidden for {url} (403) - Website may be blocking automated requests")
            elif response.status_code == 429:
                st.warning(f"Rate limited for {url} (429) - Too many requests")
            else:
                st.warning(f"HTTP error {response.status_code} for {url}")
            return ""
        
    except requests.exceptions.Timeout:
        st.warning(f"Timeout error for {url} - Request took too long")
        return ""
    except requests.exceptions.ConnectionError:
        st.warning(f"Connection error for {url} - Unable to connect")
        return ""
    except requests.exceptions.TooManyRedirects:
        st.warning(f"Too many redirects for {url}")
        return ""
    except requests.exceptions.RequestException as e:
        st.warning(f"Request error for {url}: {str(e)}")
        return ""
    except Exception as e:
        st.warning(f"Unexpected error scraping {url}: {str(e)}")
        return ""

def process_product_categories(df):
    """
    Process product categories from uploaded CSV file.
    Combines all columns into a unique list of categories.
    """
    try:
        # Combine all columns into a single list
        all_categories = []
        for column in df.columns:
            categories = df[column].dropna().astype(str).tolist()
            all_categories.extend(categories)
        
        # Remove duplicates and empty strings
        unique_categories = list(set([cat.strip() for cat in all_categories if cat.strip()]))
        
        return unique_categories
    except Exception as e:
        st.error(f"Error processing product categories: {e}")
        return []

def detect_file_type(file):
    """
    Detect the actual file type by reading the first few bytes.
    """
    try:
        file.seek(0)
        first_bytes = file.read(4)
        file.seek(0)
        
        # Check for Excel file signature (PK header for ZIP-based formats)
        if first_bytes.startswith(b'PK'):
            return 'excel'
        # Check for CSV-like content (text with commas)
        elif b',' in first_bytes or b'\n' in first_bytes:
            return 'csv'
        else:
            return 'unknown'
    except:
        return 'unknown'

def validate_excel_file(file):
    """
    Validate that the uploaded file is a proper Excel file.
    """
    try:
        # First, detect the actual file type
        file_type = detect_file_type(file)
        
        if file_type == 'csv':
            # If it's actually a CSV file, read it as CSV
            df = pd.read_csv(file)
            return df, "Warning: File appears to be CSV format, not Excel. Reading as CSV instead."
        
        # Try to read the file as Excel
        df = pd.read_excel(file, engine='openpyxl')
        return df, None
        
    except Exception as e:
        # If Excel reading fails, try to read as CSV as fallback
        try:
            file.seek(0)  # Reset file pointer
            df = pd.read_csv(file)
            return df, f"Warning: Excel reading failed, reading as CSV instead. Original error: {str(e)}"
        except Exception as csv_error:
            return None, f"Error reading file: {str(e)}. CSV fallback also failed: {str(csv_error)}"

def validate_csv_file(file):
    """
    Validate that the uploaded file is a proper CSV file.
    """
    try:
        df = pd.read_csv(file)
        return df, None
    except Exception as e:
        return None, f"Error reading CSV file: {str(e)}"

def calculate_category_embeddings(model, categories):
    """
    Calculate embeddings for all product categories.
    Returns a dictionary mapping categories to their embeddings.
    """
    category_embeddings = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, category in enumerate(categories):
        embedding = model.encode([category])
        category_embeddings[category] = embedding[0]
        
        progress = (i + 1) / len(categories)
        progress_bar.progress(progress)
        status_text.text(f"Calculating embeddings: {i + 1}/{len(categories)} categories")
    
    progress_bar.empty()
    status_text.empty()
    
    return category_embeddings

def find_best_category_match(company_embedding, category_embeddings):
    """
    Find the best matching category for a company based on cosine similarity.
    Returns the best category and its similarity score.
    """
    best_category = ""
    best_score = -1
    
    for category, category_embedding in category_embeddings.items():
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(company_embedding, category_embedding)
        
        if similarity > best_score:
            best_score = similarity
            best_category = category
    
    return best_category, best_score

def try_alternative_scraping(url, session):
    """
    Try alternative scraping methods for problematic websites.
    """
    try:
        # Try with different headers for problematic sites
        alternative_headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        response = session.get(url, headers=alternative_headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text_content = soup.get_text()
            return clean_text(text_content)[:5000]
    except:
        pass
    
    return ""

def classify_companies(companies_df, model, category_embeddings):
    """
    Classify all companies based on their website content.
    Returns a list of results.
    """
    results = []
    total_companies = len(companies_df)
    
    # Create progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, row in companies_df.iterrows():
        company_name = row['firma_adi']
        website_url = row['website']
        
        # Update progress
        progress = (index + 1) / total_companies
        progress_bar.progress(progress)
        status_text.text(f"Processing: {company_name} ({index + 1}/{total_companies})")
        
        # Scrape website content
        website_content = scrape_website_content(website_url)
        
        # If primary scraping fails, try alternative method
        if not website_content:
            session = create_robust_session()
            website_content = try_alternative_scraping(website_url, session)
        
        if not website_content:
            results.append({
                'Firma Adı': company_name,
                'Website URL': website_url,
                'Tahmin Edilen Kategori': 'N/A',
                'Benzerlik Skoru': 0.0
            })
            continue
        
        # Calculate company embedding
        company_embedding = model.encode([website_content])[0]
        
        # Find best matching category
        best_category, similarity_score = find_best_category_match(company_embedding, category_embeddings)
        
        results.append({
            'Firma Adı': company_name,
            'Website URL': website_url,
            'Tahmin Edilen Kategori': best_category,
            'Benzerlik Skoru': round(similarity_score, 4)
        })
        
        # Add a random delay to be respectful to websites and avoid detection
        delay = random.uniform(0.5, 2.0)  # Random delay between 0.5-2 seconds
        time.sleep(delay)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results

def create_excel_download(results):
    """
    Create an Excel file in memory for download.
    Returns the Excel file as bytes.
    """
    df_results = pd.DataFrame(results)
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Classification Results')
    
    output.seek(0)
    return output.getvalue()

def main():
    """
    Main Streamlit application function.
    """
    # Main header
    st.markdown('<h1 class="main-header">Auto- Segmentation Engine</h1>', unsafe_allow_html=True)
    
    # Sidebar for file uploads
    st.sidebar.markdown('<div class="sidebar-header">📁 File Upload</div>', unsafe_allow_html=True)
    
    # File uploaders
    product_groups_file = st.sidebar.file_uploader(
        "Upload Product Groups CSV",
        type=['csv'],
        help="Upload the WIN Ana & Alt Ürün Grupları Listesi(EN).csv file"
    )
    
    companies_file = st.sidebar.file_uploader(
        "Upload Company List Excel",
        type=['xlsx', 'csv'],
        help="Upload the firmalar.xlsx file with company names and websites. CSV files are also accepted."
    )
    
    # Add helpful information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - **Excel files**: Must have columns `firma_adi` and `website`
    - **CSV files**: Will be automatically detected and processed
    - **File formats**: The app can handle both proper Excel files and CSV files with .xlsx extension
    """)
    
    # Check if packages are installed
    if not install_required_packages():
        st.error("Failed to install required packages. Please check your Python environment.")
        return
    
    # Load semantic model
    model = load_semantic_model()
    if model is None:
        st.error("Failed to load semantic model. Please try again.")
        return
    
    # Main content area
    if product_groups_file is None or companies_file is None:
        st.warning("⚠️ Please upload both files to start the classification process.")
        
        # Show file format requirements
        st.markdown("### 📋 File Requirements")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Product Groups CSV File:**
            - File format: `.csv`
            - Should contain product categories in multiple columns
            - All columns will be combined into a single list of categories
            """)
        
        with col2:
            st.markdown("""
            **Company List Excel File:**
            - File format: `.xlsx`
            - Required columns: `firma_adi`, `website`
            - `firma_adi`: Company names
            - `website`: Company website URLs
            """)
        
        return
    
    # Process uploaded files
    try:
        # Process product groups
        st.info("📊 Processing product groups...")
        product_groups_df, csv_warning = validate_csv_file(product_groups_file)
        
        if product_groups_df is None:
            st.error(f"Failed to read CSV file: {csv_warning}")
            return
        
        if csv_warning:
            st.warning(csv_warning)
        
        product_categories = process_product_categories(product_groups_df)
        
        if not product_categories:
            st.error("No product categories found in the uploaded CSV file.")
            return
        
        st.success(f"✓ Loaded {len(product_categories)} unique product categories")
        
        # Process companies
        st.info("🏢 Processing company list...")
        companies_df, excel_warning = validate_excel_file(companies_file)
        
        if companies_df is None:
            st.error(f"Failed to read Excel file: {excel_warning}")
            return
        
        if excel_warning:
            st.warning(excel_warning)
        
        # Validate required columns
        if 'firma_adi' not in companies_df.columns or 'website' not in companies_df.columns:
            st.error("Required columns 'firma_adi' and 'website' not found in the Excel file.")
            st.error(f"Available columns: {list(companies_df.columns)}")
            return
        
        st.success(f"✓ Loaded {len(companies_df)} companies")
        
        # Show preview of data
        st.markdown("### 📋 Data Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Product Categories (first 10):**")
            st.write(pd.DataFrame({'Categories': product_categories[:10]}))
        
        with col2:
            st.markdown("**Companies (first 5):**")
            st.write(companies_df.head())
        
        # Run classification button
        st.markdown("---")
        if st.button("🚀 Run Classification", type="primary", use_container_width=True):
            
            # Calculate category embeddings
            st.info("🧠 Calculating product category embeddings...")
            category_embeddings = calculate_category_embeddings(model, product_categories)
            
            # Classify companies
            st.info("🌐 Scraping websites and classifying companies...")
            results = classify_companies(companies_df, model, category_embeddings)
            
            # Display results
            st.success("✅ Classification completed successfully!")
            
            # Results section
            st.markdown("### 📊 Classification Results")
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Display interactive table
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics
            st.markdown("### 📈 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Companies", len(results))
            
            with col2:
                successful = len([r for r in results if r['Tahmin Edilen Kategori'] != 'N/A'])
                st.metric("Successfully Classified", successful)
            
            with col3:
                avg_score = sum([r['Benzerlik Skoru'] for r in results if r['Benzerlik Skoru'] > 0])
                avg_score = avg_score / len([r for r in results if r['Benzerlik Skoru'] > 0]) if len([r for r in results if r['Benzerlik Skoru'] > 0]) > 0 else 0
                st.metric("Average Similarity Score", f"{avg_score:.3f}")
            
            with col4:
                failed = len([r for r in results if r['Tahmin Edilen Kategori'] == 'N/A'])
                st.metric("Failed Classifications", failed)
            
            # Download button
            st.markdown("### 💾 Download Results")
            
            # Create Excel file for download
            excel_data = create_excel_download(results)
            
            st.download_button(
                label="📥 Download Excel File",
                data=excel_data,
                file_name="sonuc.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Show top categories
            st.markdown("### 🏆 Top Predicted Categories")
            category_counts = results_df['Tahmin Edilen Kategori'].value_counts().head(10)
            st.bar_chart(category_counts)
    
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        st.error("Please check your file formats and try again.")

if __name__ == "__main__":
    main()
