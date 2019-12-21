import cv2
import numpy as np
from scipy.ndimage import label
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# =================== #
# Structural features #
# =================== #


def f_get_holes(word):
    contours, _ = cv2.findContours(
        word, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, n = label(word)
    return max(0, len(contours) - n)


def f_get_dots(word):
    contours, _ = cv2.findContours(
        word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for contour in contours:
        contour = np.unique(contour, axis=0)
        if(contour.shape[0] < 4):
            count += 1
        elif(contour.shape[0] < 6):
            count += 2
        elif(contour.shape[0] < 7):
            count += 3
    return count


def f_strokes_count(word):
    pass


def f_width():
    pass


def f_height():
    pass


def f_max_y():
    pass


def f_min_y():
    pass

# ====================== #
# Global transformations #
# ====================== #

# works with grayscale images only
# a boolean parameter indicates whether the image is in grayscale or binarized
# word images should be of fixed size (28x20) in the paper

def get_circular_lbp(word, is_binarized=False):
    if is_binarized:
        # This filter transforms a binarized image into a gray scal one
        filter = np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])
        word = np.array(convolve2d(word * 255, filter)).astype(int)

    word = np.array(word)
    lbps = []
    powers_of_2 = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    for i in range(1, word.shape[0] - 1):
        for j in range(1, word.shape[1] - 1):
            roi = word.copy()[i - 1: i + 2, j - 1: j + 2]
            roi[roi <= roi[1, 1]] = 0
            roi[roi > roi[1, 1]] = 1
            value = roi[1, 0] * powers_of_2[7] + roi[2, 0] * powers_of_2[6] + \
                roi[2, 1] * powers_of_2[5] + \
                roi[2, 2] * powers_of_2[4] + roi[1, 2] * powers_of_2[3] + roi[0, 2] * \
                powers_of_2[2] + roi[0, 1] * powers_of_2[1] + \
                roi[0, 0] * powers_of_2[0]
            lbps.append(value)

    # Get a histogram of 59 bins
    lbps = np.array(lbps)
    hist, _ = np.histogram(lbps, bins=10)
    return hist

def f_multi_lbp(word, is_binarized=False):
    #get full word image lbp
    fullWordHist = get_circular_lbp(word, is_binarized)

    #compute lbp for 4 quarters of the image
    w, h = word.shape
    upperLeft = word[0 : w // 2, 0 : h // 2]
    upperRight = word[w // 2 : w, 0 : h // 2]
    lowerLeft= word[0 : w // 2, h // 2 : h]
    lowerRight = word[w // 2 : w, h // 2 : h]

    upperLeftHist = get_circular_lbp(upperLeft, is_binarized)
    upperRighttHist = get_circular_lbp(upperRight, is_binarized)
    lowerLeftHist = get_circular_lbp(lowerLeft, is_binarized)
    lowerRightHist = get_circular_lbp(lowerRight, is_binarized)
    # Return a feature vector of 295 dimensions
    return np.concatenate((fullWordHist, upperLeftHist, upperRighttHist, lowerLeftHist, lowerRightHist))

def f_ft():
    pass

# ==================== #
# Statistical features #
# ==================== #


def f_norm_vertical_hist(word):
    pass


def f_norm_horizontal_hist(word):
    pass


def f_zoning():
    pass


def f_vertical_crossings():
    pass


def f_horizontal_crossings():
    pass
