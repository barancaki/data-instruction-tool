import streamlit as st

st.set_page_config(page_title="User Manual", layout="centered")

st.title("📘 User Manual")

st.markdown('''
This tool was developed to collect company and participant information from websites in the exhibition industry and analyze this information using an artificial intelligence model to produce meaningful results.
 
📌 The structure is modular and can be easily integrated with different exhibition websites in the future.

---

## 🔍 1. Scanning the Participant List

### ➤ Steps:

1. Go to the **“AI Web Scraper”** page from the left menu.
2. Paste the following example URL into the box:

https://replasteurasia.com/katilimci-listesi

3. Click the “Scan” button.
4. The system automatically collects participant information from all pages.

⚠️ Note: If the scanned site has more than 15 pages, you must enter the number of sites to be scanned.

✔️ Retrieved Information:
• Company name
• Company address
• Company country
• Company phone number
• Company website
	• Company product groups (may not be available for every site)

✔️ In the extracted table:
    You can take advantage of a variety of features such as download search and indexing.
✔️ You can also request a graphical output. Example graph: Country - Number of companies

---

## 🤖 2. Website Analysis with AI

With AI support, you can extract the information you want from the trade show website in natural language.

🔧 Artificial Intelligence Model Used:
•    Model: llama3:8b

•    Infrastructure: Runs locally via Ollama

•    Working Principle: The page content is divided into parts and analyzed. Filtered information is returned based on the prompt.

➤ Steps:
1.    Go to the “Website Analysis with AI” page from the left menu.

2.    Enter the full URL of the trade fair website you want to analyze.

3.    Enter a request in natural language in the “What would you like to ask the AI?” section:
            
4.    Click the “Analyze” button.
            
5.    The AI model analyzes the page and lists the results below.            

            
Examples:      
List all company names and countries on the page.
Extract all contact email addresses.
Summarize the product categories.


ℹ️ This prompt feature must be written in English only!

---
## ❓ FAQ

### Q: When I enter some websites, the Scan button does not appear. Why is that?
A: The Scraper field is designed for special functions. Functions will be added and the model will be developed according to the websites to be used.

### Q: Why does AI miss some information?  
A: If the content of web pages is dynamically loaded (with JavaScript) or the content is very long and complex, AI may skip some sections. Writing the prompt more clearly usually yields better results.

### Q: How reliable is AI?  
A: LLMs are quite successful in text generation, but 100% accuracy should not be expected. Manual verification is recommended for critical information.

### Q: It's running very slowly, what should I do?  
A: You can enter fewer pages from the website you linked to, but this is related to the CPU-GPU relationship. If the computer is fast, the automation is fast.

## 🧑‍💻 Developer's Note

This tool was developed to save time for industry professionals who want to collect and interpret data.
It is open source and will be continuously developed based on your feedback.

Happy analyzing! 🚀''')
st.sidebar.text('© Baran Çakı 2025')
st.text('© Baran Çakı 2025')
