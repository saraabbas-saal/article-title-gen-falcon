
from bs4 import BeautifulSoup as soup
import requests
import os
import random
import json
from collections import deque

# Dict to keep track of visited URLs and their children
urls = {}
base_url = "/wiki/Red_panda"
random.seed(22)
visited_urls = set()

def make_soup(url):
    resp = requests.get(url)
    s = soup(resp.content, "lxml")
    return s

def scrape_urls(s, num=5):
    urls = []
    selector = r"#mw-content-text > div.mw-content-ltr.mw-parser-output > p > a"
    elements = s.select(selector)
    random.shuffle(elements)
    count = 0
    
    for element in elements:
        href = element.get('href')
        if href is not None and href.startswith('/wiki/') and href not in visited_urls:
            urls.append(href)
            count += 1
            if count >= num:
                break
                
    return urls

def scrape(url):
    full_url = "https://en.wikipedia.org" + url
    s = make_soup(full_url)
    title_tag = s.select_one("#firstHeading > *")
    title = title_tag.text.strip() if title_tag else "No Title"
    print(f"Page title: {title}")
    urls = scrape_urls(s)
    
    
    my_page = {}
    references={}
    my_selector = r"#mw-content-text > div.mw-content-ltr.mw-parser-output *"
    header= title
    my_page[header]=[]
    
    reference_selector=r"#mw-content-text > div.mw-content-ltr.mw-parser-output > div.reflist > div > ol > li > span.reference-text"
        
    for element in s.select(my_selector):
        if element.name not in ["p", "h2", "h3","cite","a"]:
            continue
        
        if element.name == "h2":
            # print(element)
            header = element.select_one('span').text.strip()
            my_page[header] = []
            
        elif element.name in ["p","h3","cite"]:
            try:
                # print(header)
                # if header not in ["External links","Notes"]:
                my_page[header].append(element.text)
                # print("aloooooooooooo")
            except NameError:
                my_page.setdefault(title, []).append(element.text)
        
        if "References" in header:
            i = 0
            for element in s.select(reference_selector):
                
                i += 1
                references[i]= {}
                reference_text = element.text.strip()
                
                try:
                    hrefs=[]
                    a_tag= element.find_all('a')
                    for e in a_tag:
                        hrefs.append(e.get('href'))
                    
                    references[i]['text']= reference_text
                    references[i]['links']= hrefs
                    
                except Exception:
                    references[i]['text']= reference_text
                                
    
    # Writing html file 
    parent_path="/Users/sa.abbas/Documents/internship_lessons/Wikipedia scraping"
    folder_path= os.path.join(parent_path, title)        
    if not os.path.isdir(folder_path):
        # print(folder_path)
        os.mkdir(folder_path)
        
    with open(f"{folder_path}/{title}.html", "w", encoding="utf-8") as f:
            f.write(s.prettify())
            
    for header in my_page :
        
        if "References" in header:
            # with open(f"{folder_path}/references.txt", "w", encoding="utf-8") as f:
            out_file = open(f"{folder_path}/{header}.json", "w",encoding="utf-8")
            json.dump(references, out_file, indent = 4, ensure_ascii=False)
            out_file.close()
            
            # f.write('\n'.join(my_page[header]))
            
        
        # if header in ["External links", "Notes"]:
        #     continue
        with open(f"{folder_path}/{header}.txt", "w", encoding="utf-8") as f:
            f.write('\n'.join(my_page[header]))

    return urls

def bfs_scrape(base_url, max_urls=500):
    queue = deque([base_url])
    visited = set()
    scraped_urls = []

    while queue and len(scraped_urls) < max_urls:
        current_url = queue.popleft()
        if current_url in visited:
            continue

        visited.add(current_url)
        scraped_urls.append(current_url)
        print(f"Scraped: {current_url} - {len(scraped_urls)}/{max_urls}")

        child_urls = scrape(current_url)
        for url in child_urls:
            if len(scraped_urls) >= max_urls:
                break
            if url not in visited:
                queue.append(url)
                
    return scraped_urls

def main():
    max_urls = 500
    scraped_urls = bfs_scrape(base_url, max_urls)
    print(f"Total scraped URLs: {len(scraped_urls)}")

if __name__ == "__main__":
    main()
