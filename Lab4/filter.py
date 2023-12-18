import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(image_path):
    image = cv2.imread(image_path)

    b, g, r = cv2.split(image)

    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    hist_b = hist_b / hist_b.sum()
    hist_g = hist_g / hist_g.sum()
    hist_r = hist_r / hist_r.sum()

    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.plot(hist_b, color='b'), plt.title('Blue Histogram')
    plt.subplot(132), plt.plot(hist_g, color='g'), plt.title('Green Histogram')
    plt.subplot(133), plt.plot(hist_r, color='r'), plt.title('Red Histogram')
    plt.show()

    return hist_b, hist_g, hist_r