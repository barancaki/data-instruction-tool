import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urlparse

def extract_emails_from_text(text):
    """Extracts emails from a text string using regex."""
    # Regex for finding email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return set(re.findall(email_pattern, text))

def get_emails_from_url(url):
    """Scrapes a URL and extracts emails from its content."""
    try:
        # Ensure URL has a scheme
        if not url.startswith('http'):
            url = 'http://' + url

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Scanning: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract from visible text
        text_emails = extract_emails_from_text(soup.get_text())
        
        # Extract from mailto links
        mailto_emails = set()
        for a in soup.find_all('a', href=True):
            if a['href'].startswith('mailto:'):
                email = a['href'].replace('mailto:', '').split('?')[0]
                mailto_emails.add(email)
                
        all_emails = text_emails.union(mailto_emails)
        return list(all_emails)
        
    except Exception as e:
        print(f"Error scanning {url}: {e}")
        return []

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'Saha Expo WebsiteList.xlsx')
    output_file = os.path.join(script_dir, 'email_results.xlsx')
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    try:
        print("Reading Excel file...")
        df = pd.read_excel(input_file)
        
        if 'Web Sitesi' not in df.columns:
            print("Error: Column 'Web Sitesi' not found in the Excel file.")
            return
            
        results = []
        
        total_sites = len(df)
        for index, row in df.iterrows():
            website = row['Web Sitesi']
            if pd.isna(website):
                continue
                
            website = str(website).strip()
            print(f"[{index + 1}/{total_sites}] Processing {website}...")
            
            emails = get_emails_from_url(website)
            
            if emails:
                print(f"  Found emails: {', '.join(emails)}")
            else:
                print("  No emails found.")
                
            results.append({
                'Website': website,
                'Emails': ', '.join(emails) if emails else ''
            })
            
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        print(f"Saving results to {output_file}...")
        output_df.to_excel(output_file, index=False)
        print("Done!")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
