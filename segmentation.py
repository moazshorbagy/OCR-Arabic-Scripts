from commonfunctions import *
%matplotlib inline
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu


def thresholding(img):
    
    thresh = threshold_otsu(img)
    
    binary = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] < thresh:
                binary[i,j] = 1
            else:
                binary[i,j] = 0
    return binary
def vertical_histogram(img):
    histo = np.zeros(img.shape[1])
    for i in range (img.shape[1]):
        histo[i] = np.sum(img[:,i])
    return histo

def horizontal_histogram(img):
    histo = np.zeros(img.shape[0])
    for i in range (img.shape[0]):
        histo[i] = np.sum(img[i,:])
    return histo

#############################################################################      
def get_width_line(image):
    line = thresholding(image)
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
def get_meanLength_spaces(v_hist):
    spacesLength = []
    
    flag = False
    start = 0
    end = 0
    mean = 0
    
    for i in range(len(v_hist)):
        
        if  v_hist[i] == 0 and flag == False :
            flag =  True
            start = i
        
        if  v_hist[i] != 0 and v_hist[i-1] == 0  and flag == True :
            flag = False
            end = i-1
            spacesLength.append(end - start )

    spacesLength =  np.asarray(spacesLength)
    spacesLength.sort()
    
    mean = int(spacesLength.sum() / len(spacesLength)) 
    
    return mean
    

#########################################################################

def extract_lines(image):
    
    img = thresholding(image)
        
    lines = extract_words_one_line(image.transpose())

    LinesList = []
    
    for i in range(len(lines)-1):
        lines[i] = lines[i].transpose()
        
        
        s,e = get_width_line(lines[i])
        
        
        LinesList.append(lines[i][:,s:e])
    
    LinesList =  np.asarray(LinesList)
       
    return LinesList

###########################################################################

def extract_words_one_line(line):
    img = thresholding(line)
    
    v_hist = vertical_histogram(img)
    
    mean = get_meanLength_spaces(v_hist)
    

    in_word = False
    word_start = 0
    word_end = 0
    temp = 0
    wordStartings = []
    
    i = 0
    while i < len(v_hist):

        if  v_hist[i] != 0 and in_word == False:

            word_start = i
            
            in_word = True    
        
        elif v_hist[i] == 0 and in_word == True:
            
            count = 0
            j = i 
            temp = i - 1
            while j<len(v_hist) and v_hist[j] ==0 :
                count += 1
                j += 1
            
            if count > mean:
                in_word = False
                word_end = temp +1
                i = j - 1
                
                wordStartings.append(line[:,word_start:word_end])  
               
                
            else: 
                i = j - 1
        
        i+=1
    
    word_end = len(v_hist) 
    
    wordStartings.append(line[: , word_start:word_end])  
    
    

    return wordStartings
        

    
    
    
image = rgb2gray(io.imread('Dataset/capr2.png'))

image2 =np.copy(image)

Lines =  extract_lines(image)

for i in range(len(Lines)):
    show_images([Lines[i] ],[''])

Words = extract_words_one_line(Lines[2])

#THE WORDS IN THE LINE COMES IN REVERSE ORDER (LEFT TO RIGHT)

for i in range(len(Words)):
    show_images([Words[i] ],[''])


