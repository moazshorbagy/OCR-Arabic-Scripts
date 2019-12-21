from segmentation import extract_words_one_line, get_lines, deskew
import skimage.io as io
from skimage import transform as tf

def get_single_img_word_seg_acc(img, txtFileName):
    num_words = 0
    with open(txtFileName, 'r') as f:
        num_words = len(f.read().split(' '))

    lines = get_lines(img)

    num_extracted_words = 0
    for line in lines:
        num_extracted_words += len(extract_words_one_line(line))

    return (1 - abs(num_words - num_extracted_words) / num_words) * 100
    
# change img_sub_nam
img_sub_name = lambda path, x: '%s/capr%d.png'%(path, (x+1))
txt_sub_name = lambda path, x: '%s/capr%d.txt'%(path, (x+1))
# calculates accuracy over the dataset from startIdx + 1 to endIdx

def get_total_img_word_seg_acc(startIdx, endIdx, dataset_path, labels_path):
    errors_in = []
    total_acc = 0.0
    for i in range(startIdx, endIdx):
        name = img_sub_name(dataset_path, i)
        txt_name = txt_sub_name(labels_path, i)
        img = io.imread(name, as_gray=True)
        rotation_angle = deskew(img)
        deskewed_img = tf.rotate(img,rotation_angle,cval=1)
        deskewed_img *= 255
        sub_acc = get_single_img_word_seg_acc(deskewed_img, txt_name)
        total_acc += sub_acc
        if sub_acc < 100.0:
            print(f'accuracy of capr#{i+1}: {sub_acc}')
            errors_in.append(i + 1)
    total_acc /= (endIdx - startIdx)
    return total_acc, errors_in

