import numpy as np

from scipy.spatial import distance
from PIL import Image
import imagehash
import cv2

hash1 = str(imagehash.phash(Image.open('car1.png')))
hash2 = str(imagehash.phash(Image.open('car2.png')))
hash3 = str(imagehash.phash(Image.open('car3.png')))


def hamming_distance(h1, h2):
    counter = 0
    for i in range(len(h1)):
        if h1[i] != h2[i]:
            counter += 1
    return counter 

print('hash1-hash2', hamming_distance(hash1,hash2))
print('hash1-hash3', hamming_distance(hash1,hash3))
print('hash2-hash3', hamming_distance(hash2,hash3))
print('hash2-hash2', hamming_distance(hash2,hash2))