from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Langchain LLM Setup
template = (
    "Extract the specific information from this text: {dom_content}\n"
    "Follow these rules:\n"
    "1. Only output what matches the description: {parse_description}\n"
    "2. No extra text or comments.\n"
    "3. If nothing matches, output empty string.\n"
)

model = OllamaLLM(model="llama3:8b", max_tokens=512, temperature=0.3)

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


# Kullanışlı yardımcı fonksiyonlar
def get_clean_text_from_url(url):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    driver.get(url)
    html_source = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html_source, "html.parser")

    # İsteğe bağlı: Sadece ana içerik alanlarını çek
    main_content = (
        soup.select_one("main") or
        soup.select_one(".content") or
        soup.select_one("article")
    )

    if main_content:
        return main_content.get_text(separator="\n", strip=True)
    else:
        return soup.get_text(separator="\n", strip=True)


def chunk_text(text, max_length=1000):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i+max_length])


def parse_with_ollama(cleaned_text, parse_description):
    parsed_results = []
    chunks = list(chunk_text(cleaned_text, max_length=1000))

    for i, chunk in enumerate(chunks, start=1):
        response = chain.invoke({
            "dom_content": chunk,
            "parse_description": parse_description
        })
        print(f"Parsed chunk {i}/{len(chunks)}")
        parsed_results.append(response)

    return "\n".join(parsed_results)


# Örnek kullanım
if __name__ == "__main__":
    url = "https://example.com"
    parse_description = "List all product names and their prices if available."

    print(f"📥 Fetching and parsing HTML from: {url}")
    cleaned_text = get_clean_text_from_url(url)

    print("🧠 Passing to Ollama for parsing...")
    result = parse_with_ollama(cleaned_text, parse_description)

    print("\n🎯 Final Result:\n")
    print(result)