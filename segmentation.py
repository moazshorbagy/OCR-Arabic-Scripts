from utility import vertical_histogram, horizontal_histogram
from feature_extraction import cut_extra_height, cut_extra_width
import numpy as np

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
    for i in range(hist.shape[0] - 2):
        if not line_start and hist[i] != empty_line:
            indices.append(i)
            line_start = True
            continue

        if line_start and hist[i] == empty_line and hist[i+1] == empty_line and hist[i+2] == empty_line:
            indices.append(i)
            line_start = False

    for i in range(0, len(indices), 2):
        lines.append(image[indices[i]:indices[i+1], :])

    return lines

###########################################################################

# line must be negative (and thresholded)
def extract_words_one_line(line):

    hist = vertical_histogram(line)

    # mean = get_mean_space_length(hist)

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

            if count >= 3:
                in_word = False
                word_end = temp +1
                words.append(line[:, word_start:word_end])

            i = j - 1

        i += 1

    words.reverse()
    return words 


# ------------------------- Char Segmentation ----------------------------------
class SR:
    def __init__(self):
        startIndex=None
        endIndex=None
        cutIndex=None

def baseLine(img):
    return np.argmax(horizontal_histogram(img))

def maximumTransition(img,baseIndex):
    height,width=img.shape
    max=0
    mtIndex=None
    for i in range(0,baseIndex+1):
        counter=0
        flag=False
        for j in range(width):
            if img[i,j]==1 and flag==False:
                counter+=1
                flag=True
            elif img[i,j]==0 and flag==True:
                flag=False
        if counter>max:
            mtIndex=i
            max=counter
    return mtIndex

def segment_word(word, MTI, line, MFV, baseIndex):
    images, locations, characters = detect_seen(word)
    
    
    chars = []
    for i in range(len(images)):

        if(i in locations):
            chars.append(images[i])
        else:
            chars += cutPoints(images[i],MTI,line,MFV,baseIndex)
    
    return chars


def cutPoints(word,MTI,line,MFV,baseIndex):
    Flag=True
    cuts=[]
    hist=vertical_histogram(word)
    temp=word.shape[1]-1
    while(word[MTI,temp]==0):
        if(temp == 0):
            break
        temp-=1


    for i in range(temp,0,-1):
        if word[MTI,i]==1 and Flag==False:
            sr.startIndex=i
            midIndex=int((sr.endIndex+sr.startIndex)/2)

            if hist[midIndex]==0:
                sr.cutIndex=midIndex
                cuts.insert(0,sr)

            elif 0 in hist[sr.startIndex:sr.endIndex+1]:
                a1=hist[midIndex-1:sr.startIndex-1:-1]
                a2=hist[midIndex+1:sr.endIndex+1]
                min1=min2=0
                while(min1<len(a1) and a1[min1]!=0):
                    min1+=1
                while(min2<len(a2) and a2[min2]!=0):
                    min2+=1

                if min1<min2 and min1!=len(a1):
                    sr.cutIndex=midIndex-min1-1
                else:
                    sr.cutIndex=midIndex+min2+1


                cuts.insert(0,sr)


            elif hist[midIndex]==MFV:
                sr.cutIndex=midIndex
                cuts.insert(0,sr)

            elif len(hist[sr.startIndex:sr.endIndex+1][hist[sr.startIndex:sr.endIndex+1]<=MFV])!=0 :
                a1=hist[midIndex:sr.startIndex-1:-1]
                a2=hist[midIndex+1:sr.endIndex+1]
                min1=min2=0
                while(min1<len(a1) and a1[min1]>MFV):
                    min1+=1
                while(min2<len(a2) and a2[min2]>MFV):
                    min2+=1

                if min1<min2 and min1!=len(a1):
                    sr.cutIndex=midIndex-min1
                else:
                    sr.cutIndex=midIndex+min2+1


                cuts.insert(0,sr)

            elif  len(hist[sr.startIndex:midIndex+1][hist[sr.startIndex:midIndex+1]<=MFV])!=0 :
                a1=hist[midIndex-1:sr.startIndex:-1]
                min1=0
                while(min1<len(a1) and a1[min1]>MFV):
                    min1+=1
                sr.cutIndex=midIndex-min1-1
                cuts.insert(0,sr)

            else:
                sr.cutIndex=midIndex
                cuts.insert(0,sr)

            Flag=True

        elif word[MTI,i]==0 and Flag==True:
            sr=SR()
            sr.endIndex=i
            Flag=False

    cuts=filteration(word,cuts,baseIndex,MTI,MFV,hist)
    chars=[]
    starting=0
    for sr in cuts:
        chars.append(word[:,starting:sr.cutIndex+1])
        starting=sr.cutIndex+1
    chars.append(word[:,starting:])
    chars = remove_strokes(chars, baseIndex)
    return chars

def remove_strokes(chars, baseIndex):
    filtered = []
    for char in chars:
        char = cut_extra_width(char)
        if(char.shape[1] > 14):
            v_hist = vertical_histogram(char[:,4:-4])
            min_value = np.min(v_hist)
            half = v_hist.shape[0]//2
            left = np.argwhere(v_hist[:half] == min_value)
            right = np.argwhere(v_hist[half:] == min_value)
            _left=-1; _right=-1
            if(len(left)): _left = half - left[0, 0]
            if(len(right)): _right = right[0, 0]

            if(_right == -1):
                cutAt = left[0, 0] + 4
            elif(_left == -1):
                cutAt = half + right[0, 0] + 4
            elif(_left < _right):
                cutAt = left[0, 0] + 4
            else:
                cutAt = half + right[0, 0] + 4
            
            if(np.sum(char[:, :cutAt]) != 0):
                filtered.append(char[:, :cutAt])
            if(np.sum(char[:, cutAt:]) != 0):
                filtered.append(char[:, cutAt:])
        
        elif(np.any(char[baseIndex+2:, :]) or np.sum(char) > 16 or np.any(char[:baseIndex-5, :]) or char.shape[1] > 7):
            filtered.append(char)

    return filtered

def filteration(word,SRL,BaselineIndex,MaxTransitionsIndex,MFV,hist):

    filterdCut_1=filterdCut_2=[]
    sr1=SR()
    sr2=SR()
    sr1.cutIndex=0
    sr2.cutIndex=word.shape[1]-1

    SRL.insert(0,sr1)
    SRL.append(sr2)

    i=(len(SRL)-2)

    while(i>0):
        horizontal=horizontal_histogram(word[:,SRL[i].startIndex:SRL[i].endIndex+1])
        if hist[SRL[i].cutIndex]==0:
            filterdCut_1.insert(0,SRL[i])
            i-=1

        elif np.sum(word[BaselineIndex,SRL[i].startIndex+1:SRL[i].endIndex])==0:
            if np.sum(horizontal[0:BaselineIndex])<np.sum(horizontal[BaselineIndex+1:]):
                i-=1
            elif hist[SRL[i].cutIndex]<=MFV:
                filterdCut_1.insert(0,SRL[i])
                i-=1
            else:
                i-=1

        elif np.sum(word[0:MaxTransitionsIndex+1,SRL[i].cutIndex])!=0:
            i-=1

        else:
            filterdCut_1.insert(0,SRL[i])
            i-=1

    return filterdCut_1

def get_char_from_word(word, line, isThresholded=False):
    if(not isThresholded):
        line = thresholding(line)

    baseIndex = baseLine(line)
    hist = vertical_histogram(line)
    MFV = np.bincount(hist[hist!=0].astype('int64')).argmax()
    MTI = maximumTransition(line, baseIndex)
    chars = segment_word(word, MTI, line, MFV, baseIndex)
    chars.reverse()
    return chars
	
############################################################


def check_horizontal(new):
    

    histo = np.sum(new, axis=1)

    
    first = False
    second = False
    counter = 0
    
    valid = False
    
    for i in range(len(histo)):
        
        if first == False and histo[i] != 0 :
            first = True
            counter += 1
        
        if first == True and histo[i] == 0 :
            second = True
        
        if second == True and histo[i] != 0 and histo[i-1] == 0:
            valid = True
            counter +=1
    

    return valid 



def detect_seen(img ):

    new = np.copy(img)

    endofWord_SE = [
        [1,0,0,0,1,0,0,0,1],
        [1,1,0,1,1,1,0,0,1],
        [1,1,1,1,0,1,1,1,0],
        [1,0,0,0,0,0,0,0,0],

    ] 

    middleofWord_SE = [

        [1,1,1,0,0,1,1,0,0,1,1],
        [1,0,1,1,1,0,1,1,1,0,0],

    ]
    
    middleofWord_SE2 = [
  
        [1,1,1,0,0,1,1,0,0,1,1],
        [1,0,1,1,1,0,0,1,1,0,0],


    ]

    middleofWord_SE3 = [
 
        [1,1,1,0,0,1,1,0,0,1,0],
        [1,0,1,1,1,0,1,1,1,1,0],

    ]
    middleofWord_SE4 = [

        [1,1,1,0,0,1,1,0,0,1,1],
        [1,0,1,1,1,0,1,1,1,1,0],

    ]
    middleofWord_SE5 = [
  
        [1,1,1,0,0,1,1,0,0,1,0],
        [1,0,1,1,1,0,1,1,1,1,1],

    ]
    
    middleofWord_SE=np.asarray(middleofWord_SE)
    middleofWord_SE2=np.asarray(middleofWord_SE2)
    middleofWord_SE3=np.asarray(middleofWord_SE3)
    middleofWord_SE4=np.asarray(middleofWord_SE4)
    middleofWord_SE5=np.asarray(middleofWord_SE5)
    endofWord_SE = np.asarray(endofWord_SE)

    start = []
    end = []
    valid = False
    count = 0 

    for i in range(8,new.shape[0]-middleofWord_SE.shape[0]+1):
        for j in range(new.shape[1]-middleofWord_SE.shape[1]+1):

            check1 = np.array_equal(new[ i : i+middleofWord_SE.shape[0] , j :   j+middleofWord_SE.shape[1]], middleofWord_SE)
            check3 = np.array_equal(new[ i : i+middleofWord_SE3.shape[0] , j :   j+middleofWord_SE3.shape[1]], middleofWord_SE3)
            check2 = np.array_equal(new[ i : i+middleofWord_SE2.shape[0] , j :   j+middleofWord_SE2.shape[1]], middleofWord_SE2)
            check4 = np.array_equal(new[ i : i+middleofWord_SE4.shape[0] , j :   j+middleofWord_SE4.shape[1]], middleofWord_SE4)
            check5 = np.array_equal(new[ i : i+middleofWord_SE5.shape[0] , j :   j+middleofWord_SE5.shape[1]], middleofWord_SE5)
            checkend = np.array_equal(new[ i : i+endofWord_SE.shape[0] , j :   j+endofWord_SE.shape[1]], endofWord_SE)
            if check3 == True or check2 == True or check1 == True or check4 == True or check5 == True:

                start.append(j)
                end.append(j + middleofWord_SE.shape[1])
                count+=1
                valid = True
                check1 = False
                check2 = False
                check3 = False
                check4 = False
                check5 = False
            
            elif checkend ==True:
                
                start.append(j-8)
                end.append(j + middleofWord_SE.shape[1]-1)
                count+=1
                valid = True
                checkend = False


    locations  = [] 
    images = []
    characters = []
    if valid == True:

        loc = 0
        
        start.sort()
        end.sort()


        if start[0] > 1:
            images.append(np.copy(new[:,0:start[0]-1]))
            loc+=1


        for i in range(len(start)):

                images.append(np.copy(new[:,start[i]:end[i]]))
                valid  = check_horizontal(new[:,start[i]:end[i]])
                if valid == True:
                    characters.append (1)
                else:
                    characters.append(0)

                locations.append(loc)
                loc+=1

                if i < len(start)-1 and  ((start[i+1]-1) - (end[i]+1)) > 1  :
                
                    if start[i+1] != end[i]+1:
                        images.append(np.copy(new[:,end[i]+1:start[i+1]-1]))
                        loc+=1

        if end[len(end)-1] < new.shape[1]-2 :
            images.append(np.copy(new[:,end[len(end)-1]:new.shape[1]]))
        
    else:
        images.append(new)



    return images , locations , characters 
	#images list of images cut if found seen else the image itself
	#locations contaion locations of seen or sheen in the images list
	#characters contain the type of seen or sheen 
	# seen  => 0
	# sheen => 1

#################################################################	