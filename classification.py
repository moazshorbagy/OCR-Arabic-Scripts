# ===================== #
# Getting Training Data #
# ===================== #
from preprocessing import *
import numpy as np
from sklearn.preprocessing import LabelEncoder

# data, labels, errors = get_char_images('verification/scanned', 'verification/text', 1, 2)
# save_labels(labels)
# features = []
# for char in data:
#     features.append(get_features(char, False))
# # save_features(features)

# X, Y = load_dataset('features.txt', 'labels.txt')

# def shape_labels(Y, num_classes=29):
#     labels = []
#     for i in range(len(Y)):
#         label = np.zeros(num_classes)
#         label[Y[i] - 1] += 1
#         labels.append(label)
#     labels = np.asarray(labels)
#     return labels
# #Y = shape_labels(LabelEncoder(Y))
# encode=LabelEncoder()
# encode.fit(Y)
# Y = np_utils.to_categorical(encode.transform(Y))

# chars = []

# X = []
# y = []

# for char in chars:
#     X.append(get_features(char))

# ============ #
# Preproessing #
# ============ #

from sklearn.decomposition import PCA
pca = PCA(n_components = 20)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)


from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# X_train = X[0:4000]
# X_test = X[4000:]
# y_train = Y[0:4000]
# y_test = Y[4000:]
# =========================== #
# Building Model and Training #
# =========================== #

from sklearn.linear_model import LogisticRegression
clfLR = LogisticRegression(solver='lbfgs', max_iter=5000)
# clfLR.fit(X_train_vectorized, y_train)
# predicted_labels = clfLR.predict(test_data)

from sklearn import naive_bayes
clfrNB = naive_bayes.GaussianNB()
# clfrNB.fit(train_data, train_labels)
# predicted_labels = clfrNB.predict(test_data)

from sklearn.neural_network import MLPRegressor
clfMLP = MLPRegressor(hidden_layer_sizes=(6, 6))
# clfMLP.fit(X_train, Y_train)
# predicted_labels = clfMLP.predict(test_data)


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
def build_model(X_train, y_train):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,15), random_state=1, max_iter=20000)
    clf.fit(X_train, y_train)
    return clf

def save_model(model,filename):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    return pickle.load(open(filename, 'rb'))
