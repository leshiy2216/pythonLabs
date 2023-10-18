from bs4 import BeautifulSoup
import requests
import os

def create_dir(img_class):
    if not os.path.isdir("dataset"):
       os.mkdir("dataset")
    os.mkdir(f"dataset/{img_class}")

def img_download(img_class):
    create_dir(img_class)

    url = f"https://www.bing.com/images/search?q={img_class}"
    headers = headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.google.com/",
    "Accept-Encoding": "gzip, deflate, br",
}
    response = requests.get(url, headers).text
    soup = BeautifulSoup(response, 'lxml')
    img_tags = soup.find_all('img', {"src": True}, class_='mimg')
    image_urls = [img['src'] for img in img_tags]
    k = 0

    for img_url in image_urls:
        if k >= 1000:
            break

        response = requests.get(img_url)
        filename = f"dataset/{img_class}/{k:04}.jpg"
            
        with open(filename, 'wb') as f:
            f.write(response.content)
            k += 1

img_download("cats")
img_download("dogs")