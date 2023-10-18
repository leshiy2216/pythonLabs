from bs4 import BeautifulSoup
import requests
import os

def createDir():
    if not os.path.isdir("dataset"):
        os.mkdir("dataset")
    os.mkdir("dataset/cats")
    os.mkdir("dataset/dogs")

createDir()