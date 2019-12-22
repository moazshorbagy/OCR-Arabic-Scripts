from preprocessing import deskew, thresholding, get_char_images
from segmentation import get_lines, extract_words_one_line, get_char_from_word, remove_strokes, baseLine
from seg_accuracy import get_total_img_word_seg_acc
from feature_extraction import f_get_holes, f_get_dots
from utility import vertical_histogram
import skimage.io as io
import numpy as np

if __name__=='__main__':

    # Testing word segmentation for all images

    # acc, errors_in = get_total_img_word_seg_acc(0, 1500, 'scanned', 'text')

    # print(acc)

    # Testing character segmentation

    data, labels, errors = get_char_images('scanned', 'text', 1, 2)
    print(len(labels), len(errors))
    seg_accuracy = (100 * len(labels)) // (len(labels) + len(errors))
    print(f'Segmentation Accuracy: {seg_accuracy}')
    
    for error, line, column in errors:
        print(error, line, column)

    for i in range(len(data)):
        print(data[i])
        io.imshow(labels[i])
        io.show()

    # Testing feature extraction

    original = io.imread('scanned/capr10.png', as_gray=True)

    deskewed = deskew(original)

    lines = get_lines(deskewed)

    thresholded_lines = []
    for line in lines:
        thresholded_lines.append(thresholding(line))
    
    words = []
    for line in thresholded_lines:
        words += extract_words_one_line(line)
    
    baseIndex = baseLine(lines[0])
    print(baseIndex)
    for i in range(15):
        chars = get_char_from_word(words[i], thresholded_lines[0], True)
        chars = remove_strokes(chars, baseIndex)
        for char in chars:
            io.imshow(char)
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
