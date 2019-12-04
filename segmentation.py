from commonfunctions import *
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu


def thresholding(img):
    
    thresh = threshold_otsu(img)
    
    binary = np.copy(img)

    binary[img < thresh] = 1
    binary[img >= thresh] = 0
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
    wordStartings = []
    empty_line = line.shape[0]
    
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
                wordStartings.append(line[:, word_start:word_end])                  
            
            i = j - 1
        
        i += 1
       
    return wordStartings



if __name__=='__main__':

    image = io.imread('scanned/capr3.png', as_gray=True)

    lines = get_lines(image)

    words = extract_words_one_line(lines[0])

    # THE WORDS IN THE LINE COMES IN REVERSE ORDER (LEFT TO RIGHT)

    for word in words:
        # io.imshow(word, cmap=plt.cm.gray)
        io.imshow(skeletonize(word))
        io.show()
