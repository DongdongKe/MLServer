# prepare a data set for classification with the decision tree algorithm
import pickle
from scipy.sparse import data
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
# from tensorflow import keras
# import tensorflow as tf
import numpy as np
import sklearn

# sess = tf.compat.v1.Session()

AMUSEMENTIDX = 1
BaseLine = 0
Stress = 2
file_path = "C:/Users/KeDon/Downloads/dataset/classification"
file_name = "wesad-chest-combined-classification-eda.csv"
test_path = os.path.join(file_path,file_name)
data_set = pd.read_csv(os.path.join(file_path,file_name))
#print(data_set)
y_v1 =[]
y_v2 =[]
y_v3 =[]
# for i in range(1,1000):
#         if not data_set.iloc[i,-1] == AMUSEMENTIDX:
#                 filtered_data.append(data_set.iloc[i,:])
print(data_set)
# X = data_set.iloc[:, :6].values # all rows, first six column
# y = data_set.iloc[:, -1].values  # all rows, last column
# for i in y:
#         if i == BaseLine:
#                 y_v1.append(i)
#         if i == AMUSEMENTIDX:
#                 y_v2.append(i)
#         if i == Stress:
#                 y_v3.append(i) 


# clf = SVC(C=2, gamma=1)
# ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# Cross Validation
# scores = cross_val_score(clf, X, y, cv=5)
# print("Cross validation - %0.2f accuracy with a standard deviation of %0.6f" % (scores.mean(), scores.std()))
# clf = clf.fit(X_train, y_train)
# clf = tf.keras.applications.MobileNet()
# print(clf)
# create a TF model with the same architecture
# tf_model = tf.keras.models.Sequential()
# tf_model.add(tf.keras.Input(shape=(X_train.shape)))
# tf_model.add(tf.keras.layers.Dense(1))
# # assign the parameters from sklearn to the TF model
# tf_model.layers[0].weights[0].assign(clf.coef)
# tf_model.layers[0].bias.assign(clf.intercept_)
# # verify the models do the same prediction
# assert np.all((tf_model(X_train) > 0)[:, 0].numpy() == clf.predict(X_train))

# from keras.regularizers import l2
# from keras.models import Sequential
# from keras.layers import Dense

# model = Sequential()
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1), kernel_regularizer=l2(0.01))
# model.add(activation('softmax'))
# model.compile(loss='squared_hinge',
#               optimizer='adadelta',
#               metrics=['accuracy'])
# model.fit(X, Y_labels)

# guesses = clf.predict(X_test)
# for i in range(len(guesses)):
#         print(guesses[i])
# tf.saved_model.save(clf,'/VC')
# save the model to disk
# filename = 'hope.h5'
# pickle.dump(clf, open(filename, 'wb'))
# tf.saved_model.save(clf,test_path)


# # tf svm
# example_id = np.array(['%d' % i for i in range(len(y_train))])

# x_column_name = 'x'
# example_id_column_name = 'example_id'

# train_input_fn = tf.estimator.inputs.pandas_input_fn(
#     x={x_column_name: X_train, example_id_column_name: example_id},
#     y=y_train,
#     num_epochs=None,
#     shuffle=True)

# svm = tf.contrib.learn.SVM(
#     example_id_column=example_id_column_name,
#     feature_columns=(tf.contrib.layers.real_valued_column(
#         column_name=x_column_name, dimension=128),),
#     l2_regularization=0.1)

# svm.fit(input_fn=train_input_fn, steps=10)
# metrics = svm.evaluate(input_fn=train_input_fn, steps=1)
# print("Loss", metrics['loss'], "\nAccuracy", metrics['accuracy'])
# end of tf svm

# correct = 0
# incorrect = 0
# for i in range(len(guesses)):
#         if guesses[i] == y_test[i]:
#                 correct+=1
#         else:
#                 incorrect+=1
# acc = correct/len(guesses)
# print( "accuracy:" +str(acc))

#service for RT analysis:
#1 make sure your data is in the correct format (meean, max, min, range, KURT, skew)
#2 recieve those values (WS?)
#3 pickle.load(clf.pickle) --> Clf.predict(incoming data) in python service
#4 send vaues back to phone

