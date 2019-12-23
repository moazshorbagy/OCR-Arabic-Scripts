from classification import build_model, load_model
from feature_extraction import get_features
import os
import skimage.io as io
from preprocessing import get_char_images_pred, save_predictions

def test_data(folder_path, model_path):
    data = os.listdir(folder_path)
    model = load_model(model_path)

    for element in data:
        img_path = os.path.join(folder_path, element)
        image = io.imread(img_path)
        words = get_char_images_pred(image)

        test_points = []
        for word in words:
            for char in word:
                test_points.append(get_features(char, False))

        predictions = model.predict(test_points)
        new_pred = []
        current = 0
        for i in range(len(words)):
            for _ in range(len(words[i])):
                new_pred.append(predictions[current])
                current += 1
            new_pred.append(0)
        element = element.split('.')[0]
        element = element + '.txt'
        path = os.path.join("test", element)
        save_predictions(new_pred, path)
