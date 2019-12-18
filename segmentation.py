from commonfunctions import *
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu,gaussian
from skimage.draw import line as skiLine
import cv2



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

def show_vertical_projection(img):
    plt.figure()
    hist=vertical_histogram(img)
    bar(np.arange(0,len(hist)).astype(np.uint8), hist, width=0.8, align='center')
    plt.show()

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

def extract_words_one_line(line,hist):

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

###########################################################################

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

def cutPoints(word,MTI,line,mvf,baseIndex):
    Flag=True
    cuts=[]
    MTI=maximumTransition(word,baseIndex)
    temp=word.copy().astype('uint8')
    rr,cc=skiLine(baseIndex,0,baseIndex,(word.shape[1])-1)
    temp[rr,cc]=1
    io.imshow(temp)
    io.show()
    hist=vertical_histogram(word)
    temp=word.shape[1]-1
    while(word[MTI,temp]==0):
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


            elif hist[midIndex]==mvf:
                sr.cutIndex=midIndex
                cuts.insert(0,sr)

            elif len(hist[sr.startIndex:sr.endIndex+1][hist[sr.startIndex:sr.endIndex+1]<=mvf])!=0 :
                a1=hist[midIndex:sr.startIndex-1:-1]
                a2=hist[midIndex+1:sr.endIndex+1]
                min1=min2=0
                while(min1<len(a1) and a1[min1]>mvf):
                    min1+=1
                while(min2<len(a2) and a2[min2]>mvf):
                    min2+=1

                if min1<min2 and min1!=len(a1):
                    sr.cutIndex=midIndex-min1
                else:
                    sr.cutIndex=midIndex+min2+1


                cuts.insert(0,sr)

            elif  len(hist[sr.startIndex:midIndex+1][hist[sr.startIndex:midIndex+1]<=mvf])!=0 :
                a1=hist[midIndex-1:sr.startIndex:-1]
                min1=0
                while(min1<len(a1) and a1[min1]>mvf):
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


    io.imshow(word)
    io.show()

    cuts=filteration(word,cuts,baseIndex,MTI,mvf,hist)
    starting=0
    for sr in cuts:
        io.imshow(word[:,starting:sr.cutIndex+1])
        io.show()
        starting=sr.cutIndex+1



def filteration(word,SRL,BaselineIndex,MaxTransitionsIndex,MVF,hist):

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
            print("1")
            filterdCut_1.insert(0,SRL[i])
            i-=1

        elif np.sum(word[BaselineIndex,SRL[i].startIndex+1:SRL[i].endIndex])==0:
            print("3")
            if np.sum(horizontal[0:BaselineIndex])<np.sum(horizontal[BaselineIndex+1:]):
                i-=1
            elif hist[SRL[i].cutIndex]<=MVF:
                filterdCut_1.insert(0,SRL[i])
                i-=1
            else:
                i-=1

        elif np.sum(word[0:MaxTransitionsIndex+1,SRL[i].cutIndex])!=0:
            print("2")
            i-=1

        else:
            filterdCut_1.insert(0,SRL[i])
            i-=1

    return filterdCut_1
    for i in range(len(filterdCut_1)):
        if i==(len(filterdCut_1)-2):
            region1=word[:BaselineIndex,filterdCut_1[i].cutIndex:]
            region2=word[BaselineIndex+1:,filterdCut_1[i].cutIndex:]
        else:
            region1=word[:BaselineIndex,filterdCut_1[i-1].cutIndex:filterdCut_1[i].cutIndex+1]
            region2=word[BaselineIndex+1:,filterdCut_1[i-1].cutIndex:filterdCut_1[i].cutIndex+1]

        if check_Stroke(region1,region2):
            print("in-1")
            region1=region2=[]
            if i>3:
                region1=word[:BaselineIndex+1,filterdCut_1[i-3].cutIndex:filterdCut_1[i-2].cutIndex+1]
                region2=word[BaselineIndex+1:,filterdCut_1[i-3].cutIndex:filterdCut_1[i-2].cutIndex+1]
                if check_Stroke(region1,region2):
                    i-=2
                    filterdCut_2.insert(0,filterdCut_1[i])
                    print("in-2")
                else:
                    i-=1
                    filterdCut_2.insert(0,filterdCut_1[i])

            else:
                i-=1
        else:
            filterdCut_2.insert(0,filterdCut_1[i])
            i-=1


    return filterdCut_2


def check_Stroke(region1,region2):
    try:
        _,label1=label(region1,return_num=True)
        _,label2=label(region2,return_num=True)
    except:
        return False
    if label1<2 and label2<2:
        return True

    return False



if __name__=='__main__':
    image = io.imread('../Dataset/scanned/capr2.png', as_gray=True)
    io.imshow(image)
    io.show()

    lines = get_lines(image)

    lines[2] =thresholding(lines[2])                        # -------------- 1

    baseIndex=baseLine(lines[2])                                    # -------------- 2
    maximumTransitionIndex=maximumTransition(lines[2],baseIndex)    # -------------- 3

    hist=vertical_histogram(lines[2])

    words = extract_words_one_line(lines[2],hist)                   # -------------- 4

    mvf=np.bincount(hist[hist!=0].astype('int64')).argmax()


    # THE WORDS IN THE LINE COMES IN REVERSE ORDER (LEFT TO RIGHT)

    for word in words:
        # io.imshow(word, cmap=plt.cm.gray)
        cutPoints(word,maximumTransitionIndex,lines[2],mvf,baseIndex)

        #io.imshow(skeletonize(word))
        #io.show()
        #show_vertical_projection(word)
