from preprocessing import *
from segmentation import *
import os
from datetime import datetime
import skimage.io as io


enumerated_dict = {
    'ا': 1,
    'ب': 2,
    'ت': 3,
    'ث': 4,
    'ج': 5,
    'ح': 6,
    'خ': 7,
    'د': 8,
    'ذ': 9,
    'ر': 10,
    'ز': 11,
    'س': 12,
    'ش': 13,
    'ص': 14,
    'ض': 15,
    'ط': 16,
    'ظ': 17,
    'ع': 18,
    'غ': 19,
    'ف': 20,
    'ق': 21,
    'ك': 22,
    'ل': 23,
    'م': 24,
    'ن': 25,
    'ه': 26,
    'و': 27,
    'ي': 28,
    'لا': 29
}

def map_char(char):
    return enumerated_dict[char]


def get_char_images(imgs_path, txt_path, size):
    imgs = os.listdir(imgs_path)
    txts = os.listdir(txt_path)
    imgs.sort()
    txts.sort()

    labels = []
    data = []
    was = datetime.now()
    for i in range(size):
        path = os.path.join(imgs_path, imgs[i])
        img = io.imread(path)
        lines = get_lines(img)
        for line in lines:
            words = extract_words_one_line(line)
            length = len(words)
            for i in range(length):
                chars = []
                # get chars of words[length - i - 1]
                # for char in chars:
                # data.append(map_char(char))
        path = os.path.join(txt_path, txts[i])
        with open(path, 'r') as f:
            words = f.read().split(' ')
            for word in words:
                for char in word:
                    labels.append(char)
    print(f'got {size} data in: {datetime.now() - was}')
    return data, labels
