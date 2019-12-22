# ===================== #
# Getting Training Data #
# ===================== #
from preprocessing import *
import numpy as np
# from keras.utils import np_utils
# from sklearn.preprocessing import LabelEncoder

# data, labels, errors = get_char_images('verification/scanned', 'verification/text', 0, 1)
# save_char_imgs(labels, 'chars')
# save_labels(data)
# save_features('chars', 0, 5628)

X, Y = load_dataset('features.txt', 'labels.txt')

# def shape_labels(Y, num_classes=29):
#     labels = []
#     for i in range(len(Y)):
#         label = np.zeros(num_classes)
#         label[Y[i] - 1] += 1
#         labels.append(label)
#     labels = np.asarray(labels)
#     return labels
# Y = shape_labels(Y)
# Y = np_utils.to_categorical(Y)
# print(Y[0,:])

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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.35, random_state = 0)

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
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,15), random_state=1, max_iter=10000)
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)
print(y_predicted[0])
print(y_test[0])
print(accuracy_score(y_test, y_predicted))