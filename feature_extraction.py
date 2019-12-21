import cv2
import numpy as np
from scipy.ndimage import label

# =================== #
# Structural features #
# =================== #

def f_get_holes(word):
    contours, _ = cv2.findContours(word, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, n = label(word)
    return max(0, len(contours) - n)

def f_get_dots(word):
    contours, _ = cv2.findContours(word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

def f_multi_lbp():
    pass

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
