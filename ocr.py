from preprocessing import deskew, thresholding
from segmentation import *
from seg_accuracy import get_total_img_word_seg_acc
from feature_extraction import f_get_holes, f_get_dots
from utility import vertical_histogram
import skimage.io as io
import numpy as np

if __name__=='__main__':

    # Testing word segmentation for all images

    # acc, errors_in = get_total_img_word_seg_acc(0, 1500, 'scanned', 'text')

    # print(acc)

    # Testing feature extraction

    original = io.imread('../Dataset/scanned/capr1.png', as_gray=True)

    deskewed = deskew(original)

    lines = get_lines(deskewed)

    words = []
    for line in lines:
        line = thresholding(line)
        words += extract_words_one_line(line)

    io.imshow(words[0])
    io.show()

    io.imshow(lines[0])
    io.show()

    lines[0]=thresholding(lines[0])
    baseIndex = baseLine(lines[0])
    hist = vertical_histogram(lines[0])
    MFV = np.bincount(hist[hist!=0].astype('int64')).argmax()
    MTI= maximumTransition(lines[0],baseIndex)
    cuts = cutPoints(words[0], MTI, lines[0], MFV, baseIndex)
    for cut in cuts:
        io.imshow(cut)
        io.show()


    words1 = []
    for line in lines:
        line = thresholding(line, 210)
        words1 += extract_words_one_line(line)

    words2 = []
    for line in lines:
        line = thresholding(line, 125)
        words2 += extract_words_one_line(line)

    # THE WORDS IN THE LINE COMES IN REVERSE ORDER (LEFT TO RIGHT)
    for i in range(len(words)):
        print(f'Number of dots = {f_get_dots(words2[i])}')
        print(f'Number of holes = {f_get_holes(words1[i])}')
        io.imshow(words[i]/1.0)
        io.show()
