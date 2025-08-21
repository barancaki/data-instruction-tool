import streamlit as st
from ai_scripts.ollama_parser import get_clean_text_from_url, parse_with_ollama
from auth_helper import check_authentication,get_user_info,show_user_info_sidebar
# Authentication kontrolü
check_authentication()

# Kullanıcı bilgilerini al
user_info = get_user_info()

# Sidebar'da kullanıcı bilgilerini göster
show_user_info_sidebar()

st.sidebar.header("Welcome to the AI Scraper tool!")
st.sidebar.markdown('''## 🤖 Website Analysis with AI

With the help of artificial intelligence, you can extract the information you want from the trade fair website in natural language.

🔧 Artificial Intelligence Model Used:
•    Model: llama3:8b

•    Infrastructure: Runs locally via Ollama

•    Working Principle: The page content is divided into parts and analyzed. Filtered information is returned based on the prompt.

➤ Steps:
1.    Go to the “AI Analysis” page from the left menu.
            
2.    Enter the full URL of the web page you want to analyze.
            
3.    Enter a request in natural language in the “What do you want to ask the AI?” section:
            
4.    Click the “Analyze” button.
            
5. The AI model analyzes the page and lists the results below.            

            
Examples:      
List all company names and countries on the page.
Extract all contact email addresses.
Summarize the product categories.


ℹ️ This prompt feature must be written in English only. ! ''')
st.sidebar.text('© Baran Çakı 2025')
st.header("Website Analysis with AI")

with st.expander("Click to analyze with AI"):
    ai_url = st.text_input("Enter the URL of the website you want to review:", key="ai_url")
    parse_description = st.text_area("What do you want AI to analyze? (e.g., “List all company names on the page”)')", key="parse_description")

    if st.button("Analyze with AI"):
        if not ai_url or not parse_description:
            st.warning("Please enter both the URL and the analysis request.")
        else:
            with st.spinner("Being analyzed with AI..."):
                cleaned_text = get_clean_text_from_url(ai_url)
                result = parse_with_ollama(cleaned_text, parse_description)

            st.success("The analysis is complete!")
            st.subheader("Result")
            st.text_area("AI Answer:", result, height=300)

st.text('© Baran Çakı 2025')