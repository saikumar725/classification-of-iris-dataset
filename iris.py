# -*- coding: utf-8 -*-

import pickle
from sklearn.datasets import load_iris
iris = load_iris()



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3,random_state=0)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=110)
model.fit(x_train, y_train)

pred=model.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, pred))


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))   

filename = 'classification-model.pkl'
pickle.dump(model, open(filename, 'wb'))