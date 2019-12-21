from commonfunctions import *
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu
from skimage import data
from skimage.filters import threshold_otsu


#image processing resources
from skimage.io import imread, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate
import math
#testing 
import numpy as np
import os
from skimage import transform as tf
from skimage.color import rgb2gray


def thresholding(img):
    
    # thresh = threshold_otsu(img)
    
    binary = np.copy(img)

    binary[img < 185] = 1
    binary[img >= 185] = 0
    return binary

def vertical_histogram(img):
    return np.sum(img, axis=0)

def horizontal_histogram(img):
    return np.sum(img, axis=1)

#############################################################################

def get_width_line(line):
    v_histo = vertical_histogram(line)
    
    start =  0
    end = 0
    for i in range(len(v_histo)):
        if  v_histo[i] != 0:
            start  = i
            break            
    for i in range(len(v_histo)-1,0,-1):
        
        if  v_histo[i] != 0:
            end  = i
            break

    return start , end+1

##############################################################################

def get_mean_space_length(hist):
    spacesLength = []

    flag = False
    start = 0
    end = 0
    mean = 0
    
    for i in range(hist.shape[0]):
        
        if  hist[i] == 0 and not flag:
            flag = True
            start = i
            continue
        
        if  hist[i] != 0 and hist[i-1] == 0 and flag:
            flag = False
            end = i-1
            spacesLength.append(end - start)

    spacesLength = np.asarray(spacesLength)

    mean = int(spacesLength.sum() / spacesLength.shape[0])
    
    return mean

#########################################################################

def get_lines(image):
    lines = []

    hist = horizontal_histogram(image)
    indices = []
    line_start = False
    empty_line = image.shape[1] * 255
    for i in range(hist.shape[0]):
        if not line_start and hist[i] != empty_line:
            indices.append(i)
            line_start = True
            continue

        if line_start and hist[i] == empty_line:
            indices.append(i)
            line_start = False

    for i in range(0, len(indices), 2):
        lines.append(image[indices[i]:indices[i+1], :])

    return lines

###########################################################################

def extract_words_one_line(line):
    line = thresholding(line)

    hist = vertical_histogram(line)

    mean = get_mean_space_length(hist)

    in_word = False
    word_start = 0
    word_end = 0
    temp = 0
    words = []
    
    i = 0
    while i < hist.shape[0]:

        if  not in_word and hist[i] != 0:
            word_start = i
            in_word = True    
        
        elif in_word and hist[i] == 0:
            count = 0
            j = i 
            temp = i - 1

            while j < hist.shape[0] and hist[j] == 0 :
                count += 1
                j += 1
            
            if count > mean:
                in_word = False
                word_end = temp +1
                words.append(line[:, word_start:word_end])                  
            
            i = j - 1
        
        i += 1
       
    return words
###########################################################################


def deskew(img):
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

    return rotation_number

###########################################################################



if __name__=='__main__':

    img = io.imread('scanned/capr3.png', as_gray=True)

    rotation_angle = deskew(img)

    print('Rotation angle is {0}'.format(rotation_angle) + '\n')

    image = tf.rotate(img,rotation_angle,cval=1)

    show_images([image,img],['Rotated','Original'])

    #check if the function rotate returns the matrix between 0 and 1 or 0 and 255 

    lines = get_lines(image)

    words = extract_words_one_line(lines[0])

    # THE WORDS IN THE LINE COMES IN REVERSE ORDER (LEFT TO RIGHT)

    for word in words:
        # io.imshow(word, cmap=plt.cm.gray)
        io.imshow(skeletonize(word))
        io.show()
