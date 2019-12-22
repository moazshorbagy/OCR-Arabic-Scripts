import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate

# ========== #
# De-skewing #
# ========== #

def deskew_1(img):
    img_blur = cv2.medianBlur(img,5).astype('uint8')
    thresh = cv2.threshold(cv2.bitwise_not(img_blur), 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)   
    return rotated

def deskew_2(img):
    image = rgb2gray(img)

    #threshold to get rid of extraneous noise
    thresh = threshold_otsu(image)
    normalize = image > thresh

    # gaussian blur
    blur = gaussian(normalize, 3)

    # canny edges in scikit-image
    edges = canny(blur)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1)/(x2 - x1) if (x2-x1) else 0 for (x1,y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)
    
    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90-rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    return (rotate(img,rotation_number,cval=1) * 255).astype(np.uint8)

def deskew(img):
    return deskew_2(deskew_1(img))

# ============ #
# Thresholding #
# ============ #

def thresholding(img, thresh=None):
    binary = np.copy(img)

    thresh = thresh or threshold_otsu(img)
    binary[img < thresh] = 1
    binary[img >= thresh] = 0
    return binary

# ============= #
# Fetching Data #
# ============= #

from segmentation import get_lines, extract_words_one_line, get_char_from_word
from time import time
import skimage.io as io
import os

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


def get_char_images(imgs_path='scanned', txt_path='text', start=0, end=1000):
    imgs = os.listdir(imgs_path)
    txts = os.listdir(txt_path)
    imgs.sort()
    txts.sort()

    segErrors = []
    labelWords = []
    data = []
    labels = []
    was = time()
                
    for i in range(start, end):        
        # Getting labels
        path = os.path.join(txt_path, txts[i])
        with open(path, 'r') as f:
            words = f.read().split(' ')
            for word in words:
                labelWords.append(word)
        
        # Getting images
        path = os.path.join(imgs_path, imgs[i])

        original = io.imread(path)

        deskewed = deskew(original)

        lines = get_lines(deskewed)
        
        thresholded_lines = []
        for line in lines:
            thresholded_lines.append(thresholding(line))

        linesWithWords = []
        lengthOfWords = 0
        for line in thresholded_lines:
            wordsFromLine = extract_words_one_line(line)
            linesWithWords.append(wordsFromLine)
            lengthOfWords += len(wordsFromLine)

        # Check for word segmentation error
        if(lengthOfWords != len(labelWords)):
            print(f'skipping {path}')
            continue
        
        currLabelIndex = -1
        for i in range(len(linesWithWords)): # looping on lines
            for j in range(len(linesWithWords[i])): # looping on words in specific line
                currLabelIndex += 1
                chars = get_char_from_word(linesWithWords[i][j], thresholded_lines[i], True)

                # Check for character segmentation error
                if(len(chars) != len(labelWords[currLabelIndex])):
                    segErrors.append((labelWords[currLabelIndex], i, j))
                    continue
            
                for k in range(len(chars)):
                    labels.append(chars[k])
                    data.append(labelWords[currLabelIndex][k])

    print(f'got {end-start} images in: {int(time() - was)} sec')
    return data, labels, segErrors
