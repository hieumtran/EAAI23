from sklearn import svm
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn import tree

train = pd.read_csv('./data/train_svm.csv')
test = pd.read_csv('./data/test_svm.csv')
test_model = pd.read_csv('./data/output_41.csv')

x_train = train.loc[:50000, ['val', 'ars']]
y_train = train.loc[:50000, ['class']]

# breakpoint()

x_test = test.loc[:, ['val', 'ars']]
x_test_model = test_model.loc[:, ['val', 'ars']]
y_test = test.loc[:, ['class']].to_numpy()

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x_train, y_train)
predict = clf.predict(x_test)
y_test = np.reshape(y_test, (-1,))
print(y_test)
print(np.count_nonzero(predict == y_test) / len(y_test))
joblib.dump(clf, 'svm.pkl')

clf = joblib.load('svm.pkl')
predict_model  = clf.predict(x_test_model)
predict = clf.predict(x_test)
y_test = np.reshape(y_test, (-1,))
cnt = 0
cnt_model = 0
for i in range(len(predict)):
    if predict_model[i] == y_test[i]:
        cnt_model += 1
    if predict[i] == y_test[i]:
        cnt += 1
print(f'Accuracy:{cnt/len(predict)}')
print(f'Accuracy:{cnt_model/len(predict)}')

