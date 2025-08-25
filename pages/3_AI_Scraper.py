import streamlit as st
import asyncio
from ai_scripts.ollama_parser import get_clean_texts_from_urls, parse_with_ollama
from auth_helper import check_authentication, get_user_info, show_user_info_sidebar

# -------------------------
# Authentication
# -------------------------
check_authentication()
user_info = get_user_info()
show_user_info_sidebar()

st.sidebar.header("Welcome to the AI Scraper tool!")
st.sidebar.markdown('''## 🤖 Website Analysis with AI

With the help of artificial intelligence, you can extract the information you want from multiple trade fair website pages.

🔧 Artificial Intelligence Model Used:
• Model: gpt-oss:20b  
• Infrastructure: Runs locally via Ollama  
• Working Principle: The page content is divided into parts and analyzed. Filtered information is returned based on the prompt.

➤ Steps:
1. Enter one or more page URLs (comma-separated).  
2. Enter a request in English in the analysis section.  
3. Click the “Analyze” button.  
4. The AI model analyzes all pages and lists the results.

Examples:      
- List all company names and countries on the page.  
- Extract all contact email addresses.  
- Summarize the product categories.  

ℹ️ This prompt feature must be written in English only. ! 
''')
st.sidebar.text('© Baran Çakı 2025')

st.header("Website Analysis with AI")

with st.expander("Click to analyze with AI"):
    ai_urls = st.text_area(
        "Enter the URL(s) of the website you want to review (comma separated):",
        key="ai_urls"
    )
    parse_description = st.text_area(
        "What do you want AI to analyze? (e.g., “List all company names on the page”)",
        key="parse_description"
    )

    if st.button("Analyze with AI"):
        if not ai_urls or not parse_description:
            st.warning("Please enter both the URL(s) and the analysis request.")
        else:
            urls = [u.strip() for u in ai_urls.split(",") if u.strip()]
            with st.spinner(f"Analyzing {len(urls)} page(s) with AI..."):
                cleaned_texts = asyncio.run(get_clean_texts_from_urls(urls))
                result = parse_with_ollama(cleaned_texts, parse_description)

            st.success("The analysis is complete!")
            st.subheader("Result")
            st.text_area("AI Answer:", result, height=300)

st.text('© Baran Çakı 2025')
