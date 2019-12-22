from preprocessing import *
from segmentation import get_lines, extract_words_one_line, get_char_from_word, remove_strokes, baseLine
from seg_accuracy import get_total_img_word_seg_acc
from feature_extraction import f_get_holes, f_get_dots, f_center_of_mass
from utility import vertical_histogram
import skimage.io as io
import numpy as np
from classification import *

if __name__=='__main__':

    flag=False
    modelPath="zaki.sav"
    image="verification/scanned/capr6.png"
    if flag:
        model=load_model(modelPath)
        chars = get_char_images_pred(image)
        X_test=[]
        for char in data:
            X_test.append(get_features(char, False))
        
    else:
        data, labels, errors = get_char_images('verification/scanned', 'verification/text', 1, 2)
        features = []
        for char in data:
            features.append(get_features(char, False))
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)
        model = build_model(X_train,y_train)
        save_model(model,modelPath)

    predictions = model.predict(X_test)
    
    save_predictions(predictions,"test/capr6.txt")
