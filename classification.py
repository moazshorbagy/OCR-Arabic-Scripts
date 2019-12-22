# ===================== #
# Getting Training Data #
# ===================== #

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
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

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
X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)

X_pred = [
    [0., 0.],
    [1., 2.],
    [1., 1.]
]
print(clf.predict(X_pred))