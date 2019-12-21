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