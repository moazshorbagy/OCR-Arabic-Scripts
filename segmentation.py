from commonfunctions import *
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

def get_lines(image):
    lines = []

    hist = horizontal_histogram(image)
    indices = []
    line_start = False
    empty_line = image.shape[1] * 255
    for i in range(image.shape[0]):
        if not line_start and hist[i] != empty_line:
            indices.append(i)
            line_start = True

        if line_start and hist[i] == empty_line:
            indices.append(i)
            line_start = False

    for i in range(0, len(indices), 2):
        lines.append(image[indices[i]:indices[i+1], :])

    return lines

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



if __name__=='__main__':

    image = io.imread('scanned/capr2.png', as_gray=True)

    lines = get_lines(image)

    words = extract_words_one_line(lines[0])

    # THE WORDS IN THE LINE COMES IN REVERSE ORDER (LEFT TO RIGHT)

    for word in words:
        io.imshow(word)
        io.show()
