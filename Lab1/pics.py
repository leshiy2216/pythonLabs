from bs4 import BeautifulSoup
import requests
import os
import json

def createDir(img_class):
    if not os.path.isdir("dataset"):
       os.mkdir("dataset")
    os.mkdir(f"dataset/{img_class}")

def yandexSearch(img_class):
    url = "https://yandex.ru/images/search?text={img_class}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    image_tags = soup.find_all('img')

    image_urls = [img['src'] for img in image_tags]
    print(image_urls)

yandexSearch("cats")