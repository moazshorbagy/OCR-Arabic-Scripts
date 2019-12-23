from preprocessing import *
from segmentation import get_lines, extract_words_one_line, get_char_from_word, remove_strokes, baseLine
from seg_accuracy import get_total_img_word_seg_acc
from feature_extraction import f_get_holes, f_get_dots, f_center_of_mass
from utility import vertical_histogram
import skimage.io as io
import numpy as np
from classification import *
from time import time

if __name__=='__main__':

    flag=False
    modelPath="zaki200.sav"
    image="verification/scanned/capr1.png"
    was = time()
    if flag:
        model=load_model(modelPath)
        words = get_char_images_pred(image)
        X_test=[]
        for word in words:
            for char in word:
                X_test.append(get_features(char, False))
        
    else:
        data, labels, errors, words_lengths = get_char_images('verification/scanned', 'verification/text', 0, 200)
        features = []
        for char in data:
            features.append(get_features(char, False))
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)
        # data = np.asarray(data)
        # labels = np.asarray(labels)
        # print(data.shape)
        # print(labels.shape)
        model = build_model(features,labels)
        save_model(model,modelPath)


    predictions = model.predict(X_test)
    now = int(time() - was)
    print(now)
    print(predictions)
    new_pred = []
    current = 0
    for i in range(len(words)):
        for j in range(len(words[i])):
            new_pred.append(predictions[current])
            current += 1
        new_pred.append(0)
    # X_test = np.asarray(X_test)
    save_predictions(new_pred,"test/capr1.txt")
