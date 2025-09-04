import os
import codecs
import json
import base64 

import time 
import pandas as pd 
from selenium import webdriver 
from selenium.webdriver import Chrome 
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import requests
import hashlib

print("Initializing Chrome WebDriver...")

# Define the Chrome webdriver options
options = webdriver.ChromeOptions() 
options.add_argument("--headless") # Set the Chrome webdriver to run in headless mode for scalability
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# By default, Selenium waits for all resources to download before taking actions.
# However, we don't need it as the page is populated with dynamically generated JavaScript code.
options.page_load_strategy = "none"

try:
    # Use webdriver manager to automatically download and manage chromedriver
    service = Service(ChromeDriverManager().install())
    driver = Chrome(service=service, options=options)
    
    # Remove automation indicators
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    # Set an implicit wait of 5 seconds to allow time for elements to appear before throwing an exception
    driver.implicitly_wait(5)
    print("Chrome WebDriver initialized successfully.")
except WebDriverException as e:
    print(f"Failed to initialize Chrome WebDriver: {e}")
    print("Please ensure Chrome is installed on your system.")
    exit(1)
except Exception as e:
    print(f"Unexpected error initializing WebDriver: {e}")
    exit(1)

urls = set()
urlsFileName = os.path.join("../data/htmlpages", "urls.txt")

# Ensure directories exist
os.makedirs("../data/htmlpages", exist_ok=True)
os.makedirs(os.path.join("../data/htmlpages", "knowledge"), exist_ok=True)
os.makedirs(os.path.join("../data/htmlpages", "knowledge", "official"), exist_ok=True)
print("Created necessary directories.")

def getFileName(url):
    string_bytes = url.encode("ascii") 
    
    base64_bytes = base64.b64encode(string_bytes) 
    base64_string = base64_bytes.decode("ascii")
    return base64_string
def get_all_website_links(url):
    links = set()
    urls_queue = set([url])

    links.add(url)
    processed_count = 0
    while urls_queue:
        current_url = urls_queue.pop()
        processed_count += 1
        print(f"[{processed_count}] Processing: {current_url}")
        try:
            driver.get(current_url)
            time.sleep(10)  # Wait longer for anti-bot checks
            
            # Check if we're on a "Just a moment" or similar page
            if "just a moment" in driver.title.lower() or "please wait" in driver.title.lower():
                print("    Waiting for anti-bot protection to clear...")
                time.sleep(15)  # Wait longer for Cloudflare or similar

            try:
                article = driver.find_element(By.CSS_SELECTOR, "article[class*='article']")
                articlehead = article.find_element(By.CSS_SELECTOR, "h1[class*='article']")
                articlebody = article.find_element(By.CSS_SELECTOR, "div[class*='article-body']")
                article_json = {}

                urls.add(current_url)
                #article_json["source"] = current_url
                article_json["title"] = articlehead.text
                article_json["body"] = articlebody.text
               
                filename = getFileName(current_url)
                n=os.path.join("../data/htmlpages", "knowledge", "official",filename)
                with open(n, 'w') as f:
                    json.dump(article_json, f)
                json_data = json.dumps(article_json)
                print(f"    ✓ Saved article: {article_json['title'][:60]}...")            
            except NoSuchElementException:
                print("    No article found in this webpage.")
                # Let's debug what we can find on the page
                print(f"    Page title: {driver.title}")
                # Check if we can find any content at all
                try:
                    body = driver.find_element(By.TAG_NAME, "body")
                    print(f"    Page has content: {len(body.text)} characters")
                except:
                    print("    Could not find page body")

            items = driver.find_elements(By.CSS_SELECTOR, "li[class*='blocks-item']")
            if (len(items) == 0):
              items = driver.find_elements(By.CSS_SELECTOR, "li[class*='article-list-item']")
            nextpage = driver.find_elements(By.CSS_SELECTOR, "li[class*='pagination-next']")
            
            print(f"    Found {len(items)} content items, {len(nextpage)} pagination items")
            alllinks = items + nextpage

            #print(alllinks)
            for link in alllinks:
                href = link.find_element(By.TAG_NAME, "a").get_attribute("href")
                if href == "" or href is None:
                    continue

                links.add(href)
                urls_queue.add(href)
            #driver.quit()

        except requests.exceptions.RequestException as e:
            print(f"Request failed for {current_url}: {e}")

    return links

try:
    print("Starting website scraping...")
    links = get_all_website_links("https://support.medirecords.com/hc/en-us")
    
    print(f"\n✓ Scraping completed! Found {len(urls)} articles.")
    print(f"✓ Total links discovered: {len(links)}")
    
    #print(links)
    with open(urlsFileName,'w') as f:
       f.write(str(urls))
    print(f"✓ URLs saved to {urlsFileName}")
    
except Exception as e:
    print(f"Error during scraping: {e}")
finally:
    # Always close the webdriver
    try:
        driver.quit()
        print("✓ WebDriver closed.")
    except:
        pass