from bs4 import BeautifulSoup
import requests
import os
import json

def createDir(img_class):
    if not os.path.isdir("dataset"):
       os.mkdir("dataset")
    os.mkdir(f"dataset/{img_class}")

def yandexSearch(img_class):
    createDir(img_class)

    url = f"https://yandex.ru/images/search?text={img_class}&p="
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    pageNumb = 1
    countImg = 0

    while countImg < 1000:
        current_url = f"{url}{pageNumb}"
        response = requests.get(current_url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            image_tags = soup.find_all('img')

            for img in image_tags:
                img_url = img.get('src')
                if img_url:
                    img_response = requests.get(img_url)

                    if img_response.status_code == 200:
                        filename = os.path.join('dataset', img_class, f'{countImg}.jpg')
                        with open(filename, 'wb') as img_file:
                            img_file.write(img_response.content)
                        countImg += 1
        
    pageNumb += 1
        
yandexSearch("cats")
yandexSearch("dogs")