import asyncio
import aiohttp
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# LangChain LLM Setup
# -------------------------
template = (
    "Extract the specific information from this text: {dom_content}\n"
    "Follow these rules:\n"
    "1. Only output what matches the description: {parse_description}\n"
    "2. No extra text or comments.\n"
    "3. If nothing matches, output empty string.\n"
)

model = OllamaLLM(model="gpt-oss:20b")   # Default model
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


# -------------------------
# Async HTML Fetcher
# -------------------------
async def fetch(session, url):
    try:
        async with session.get(url, timeout=20) as response:
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")

            # Sadece ana içerik alanını çek
            main_content = (
                soup.select_one("main") or
                soup.select_one(".content") or
                soup.select_one("article")
            )

            if main_content:
                return main_content.get_text(separator="\n", strip=True)
            else:
                return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return ""


async def get_clean_texts_from_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)


# -------------------------
# Text Chunking
# -------------------------
def chunk_text(text, max_length=1000):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i+max_length])


# -------------------------
# AI Parsing
# -------------------------
def parse_with_ollama(cleaned_texts, parse_description):
    parsed_results = []

    for page_idx, cleaned_text in enumerate(cleaned_texts, start=1):
        chunks = list(chunk_text(cleaned_text, max_length=1000))

        for i, chunk in enumerate(chunks, start=1):
            response = chain.invoke({
                "dom_content": chunk,
                "parse_description": parse_description
            })
            print(f"📄 Page {page_idx} - Parsed chunk {i}/{len(chunks)}")
            parsed_results.append(response)

    return "\n".join(parsed_results)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3"
    ]
    parse_description = "List all product names and their prices if available."

    print(f"📥 Fetching and parsing HTML from {len(urls)} pages...")
    cleaned_texts = asyncio.run(get_clean_texts_from_urls(urls))

    print("🧠 Passing to Ollama for parsing...")
    result = parse_with_ollama(cleaned_texts, parse_description)

    print("\n🎯 Final Result:\n")
    print(result)
