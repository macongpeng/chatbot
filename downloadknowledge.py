import os
import codecs
import json

import time 
import pandas as pd 
from selenium import webdriver 
from selenium.webdriver import Firefox 
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By 
import requests
import hashlib

# Define the Firefox webdriver options
options = webdriver.FirefoxOptions() 
options.add_argument("--headless") # Set the Chrome webdriver to run in headless mode for scalability

# By default, Selenium waits for all resources to download before taking actions.
# However, we don't need it as the page is populated with dynamically generated JavaScript code.
options.page_load_strategy = "none"

# Pass the defined options objects to initialize the web driver 
driver = Firefox(options=options) 
# Set an implicit wait of 5 seconds to allow time for elements to appear before throwing an exception
driver.implicitly_wait(5)

def get_all_website_links(url):
    links = set()
    urls_queue = set([url])

    links.add(url)
    while urls_queue:
        current_url = urls_queue.pop()
        print(current_url)
        try:
            driver.get(current_url)
            time.sleep(5)

            try:
                article = driver.find_element(By.CSS_SELECTOR, "article[class*='article'")
                articlehead = article.find_element(By.CSS_SELECTOR, "h1[class*='article'")
                articlebody = article.find_element(By.CSS_SELECTOR, "div[class*='article-body'")
                article_json = {}
                article_json["source"] = current_url
                article_json["title"] = articlehead.text
                article_json["body"] = articlebody.text
               
                filename = hashlib.md5(current_url.encode()).hexdigest() + ".json"
                n=os.path.join("htmlpages", "knowledge", "official",filename)
                with open(n, 'w') as f:
                    json.dump(article_json, f)
                json_data = json.dumps(article_json)             
            except NoSuchElementException:
                print("No article found in this webpage.")

            items = driver.find_elements(By.CSS_SELECTOR, "li[class*='blocks-item'")
            if (len(items) == 0):
              items = driver.find_elements(By.CSS_SELECTOR, "li[class*='article-list-item'")
            nextpage = driver.find_elements(By.CSS_SELECTOR, "li[class*='pagination-next'")
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

links = get_all_website_links("https://support.medirecords.com/hc/en-us")
print(links)