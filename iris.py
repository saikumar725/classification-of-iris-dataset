# -*- coding: utf-8 -*-

import pickle
from sklearn.datasets import load_iris
iris = load_iris()


from sklearn.model_selection import StratifiedKFold

skf=StratifiedKFold(n_splits=5)

skf.get_n_splits(iris.data,iris.target)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)





filename = 'classification-model.pkl'
pickle.dump(model, open(filename, 'wb'))