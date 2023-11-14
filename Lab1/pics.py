from bs4 import BeautifulSoup
import requests
import os
import logging
import argparse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dir(folder_path, subfolder_path):
    try:
        subfolder_path = os.path.join(folder_path, subfolder_path)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    except Exception as e:
        logger.exception("Can't create a folder")


def img_download(subfolder_path, folder_path, num_images):
    create_dir(folder_path, subfolder_path)

    page = 1
    k = 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.google.com/",
        "Accept-Encoding": "gzip, deflate, br",
    }

    while k < num_images:
        url = f"https://www.bing.com/images/search?q={subfolder_path}&first={page}"
        try:
            response = requests.get(url, headers)
            soup = BeautifulSoup(response.text, 'lxml')
            img_tags = soup.find_all('img', {"src": True}, class_='mimg')
            image_urls = [img['src'] for img in img_tags]

            for img_url in image_urls:
                try:
                    response = requests.get(img_url)
                    filename = os.path.join(folder_path, subfolder_path, f"{k:04}.jpg")

                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    k += 1
                except Exception as e:
                    logger.exception("Ошибка при загрузке изображения")
                page += 1
        except Exception as e:
            logger.exception("Ошибка при получении данных")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Download images from Bing Images')
    parser.add_argument('subfolder_path', type = str, help = 'Name of the image class')
    parser.add_argument('folder_path', type = str, help = 'Output directory for saving images')
    parser.add_argument('num_images', type = int, help = 'Number of images to download')

    args = parser.parse_args()
    img_download(args.subfolder_path, args.folder_path, args.num_images)