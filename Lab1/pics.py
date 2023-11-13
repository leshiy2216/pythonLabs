from bs4 import BeautifulSoup
import requests
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dir(img_class):
    try:
        folder_path = os.path.join("dataset", img_class)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except Exception as e:
        logger.exception("Can't create a folder")


def img_download(img_class):
    create_dir(img_class)

    page = 1
    img_num = 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.google.com/",
        "Accept-Encoding": "gzip, deflate, br",
        }
    while img_num < 1000:
        url = f"https://www.bing.com/images/search?q={img_class}&first={page}"
        try:
            response = requests.get(url, headers)
            soup = BeautifulSoup(response.text, 'lxml')
            img_tags = soup.find_all('img', {"src": True}, class_='mimg')
            image_urls = [img['src'] for img in img_tags]

            for img_url in image_urls:
                try:
                    response = requests.get(img_url)
                    filename = os.path.join("dataset", img_class, f"{img_num:04}.jpg")

                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    img_num += 1
                except Exception as e:
                    logger.exception("Error downloading image")
                page += 1
        except Exception as e:
            logger.exception("Error fetching data")


if __name__=="__main__":
    img_download("cats")
    img_download("dogs")