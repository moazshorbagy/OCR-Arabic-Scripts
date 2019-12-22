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

def get_circular_lbp(word, nbins, is_binarized=False):
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
    hist, _ = np.histogram(lbps, bins=nbins)
    return hist

def f_multi_lbp(word, is_binarized=False, nbins=10):
    #get full word image lbp
    fullWordHist = get_circular_lbp(word, nbins, is_binarized)
    #compute lbp for 4 quarters of the image
    w, h = word.shape
    upperLeft = word[0 : w // 2, 0 : h // 2]
    upperRight = word[w // 2 : w, 0 : h // 2]
    lowerLeft= word[0 : w // 2, h // 2 : h]
    lowerRight = word[w // 2 : w, h // 2 : h]

    upperLeftHist = get_circular_lbp(upperLeft, nbins, is_binarized)
    upperRighttHist = get_circular_lbp(upperRight, nbins, is_binarized)
    lowerLeftHist = get_circular_lbp(lowerLeft, nbins, is_binarized)
    lowerRightHist = get_circular_lbp(lowerRight, nbins, is_binarized)
    # Return a feature vector of 295 dimensions
    return np.concatenate((fullWordHist, upperLeftHist, upperRighttHist, lowerLeftHist, lowerRightHist))

#assumes the input img is binarized
def f_ft(img):
    filter = np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])
    word = np.array(convolve2d(img * 255, filter)).astype(int)
    ft = np.fft.fft2(word)
    ft = np.fft.fftshift(ft)
    h, w = ft.shape
    w = w // 2
    h = h // 2
    ft = ft[h - 2 : h + 2, w - 2 : w + 2]
    h, w = ft.shape
    ft = np.reshape(ft, (h * w,))
    # To reconstruct the signal:
    # ift = np.fft.ifft2(ft).real
    return ft

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
	
	
#############################
##     YAMINA PAPER        ##
#############################

def cut_extra_height(img):
    histo = np.sum(img, axis=1)
    count = 0
    count1 = 0
    for i in range(len(histo)):
        if histo[i] == 0:
            count +=1
        else:
            break

    for i in range(len(histo)-1, 0,-1):
        if histo[i] == 0:
            count1 +=1
        else: 
            break
    return img[count : img.shape[0]-count1 , :]



def f_w_over_h(img):
    new  = cut_extra_height(img)
    return new.shape[1]/new.shape[0]


def f_bpixels_over_wpixels(img):
    num_zeros = np.count_nonzero(img == 0)
    num_ones = np.count_nonzero(img == 1)
    return num_zeros/num_ones 

def f_horizontal_transitions(img):
    count = 0
    for i in range(img.shape[0]):
        prev = img[i,0]
        for  j in range(img.shape[1]):
            if img[i,j] != prev:
                prev = img[i,j]
                count+=1
    return count 


def f_vertical_transitions(img):
    count = 0
    for i in range(img.shape[1]):
        prev = img[0,i]
        for  j in range(img.shape[0]):
            if img[j,i] != prev:
                prev = img[j,i]
                count+=1
    return count 


# ==================== #
#   Get all features   #
# ==================== #

def get_features(char, use_ft_lbp=False):
    holes = np.array([f_get_holes(char)])
    dots = np.array([f_get_dots(char)])
    features = np.concatenate((holes, dots))
    if use_ft_lbp:
        lbp = np.array(f_multi_lbp(char, is_binarized=True, nbins=2))
        # ft = np.array(f_ft(char))
        features = np.concatenate((features, lbp))
    return features