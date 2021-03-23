import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from joblib import parallel_backend

with open('age_images.pkl', 'rb') as f:
  images = pickle.load(f)


with open('age_list.pkl', 'rb') as f:
  age_list = pickle.load(f)

filedir = "/home/azureuser/server/crop_part1"

import numpy as np
images1 = np.squeeze(images)
# print(images1.shape)
images_f=np.array(images1)
ages_f=np.array(age_list)
np.save(filedir+'image.npy',images_f)
np.save(filedir+'age.npy',ages_f)


labels=[]
i=0
while i<len(age_list):
  label=age_list[i]
  labels.append(label)
  i+=1
# print(labels)
classes = []
for i in labels:
    i = int(i)
    if i <= 15:
        classes.append(0)
    if (i>15) and (i<=30):
        classes.append(1)
    if (i>30) and (i<60):
        classes.append(2)
    if i>=60:
        classes.append(3)
# print(classes)

from keras.utils.np_utils import to_categorical

images_f = images_f.astype('float32')
images_f_2=images_f/255
categorical_labels = to_categorical(classes, num_classes=4)

labels_f=np.array(classes)

images_f_2.shape

import tensorflow as tf
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(images_f_2, categorical_labels,test_size=0.15)



def Convolution(input_tensor,filters):
  x = Conv2D(filters=filters,kernel_size=(2, 2),padding = 'same',strides=(1, 1),kernel_regularizer=l2(0.001))(input_tensor)
  x = Dropout(0.1)(x)
  x= Activation('relu')(x)
  return x

def build_model(innerlayers = 5, activationf = "sigmoid"):
  print("\nBuilding model with args : " + str(innerlayers) + " " + str(activationf) + "\n")
  input_shape = (48,48,3)
  inputs = Input((input_shape))
  k = 32
  conv= Convolution(inputs,k)
  maxp = MaxPooling2D(pool_size = (2,2)) (conv)
  for i in range(innerlayers-1):
    k *= 2
    conv = Convolution(maxp,k)
    maxp = MaxPooling2D(pool_size = (2, 2)) (conv)
  flatten= Flatten() (maxp)
  dense_1= Dense(k,activation='relu')(flatten)
  drop_1=Dropout(0.2)(dense_1)
  output= Dense(4,activation=activationf)(drop_1)
  model = Model(inputs=[inputs], outputs=[output])
  model.compile(loss="categorical_crossentropy", optimizer="Adam",	metrics=["accuracy"])
  return model

# Model=build_model()
# Model.summary()

model = KerasClassifier(build_fn=build_model, verbose=0)
# define the grid search parameters
#batch_size = [10, 20, 40, 60, 80, 100]
activationf = ["sigmoid","relu","softmax"]
innerlayers = [2,3,4,5]
epochs = [1,10,20]

param_grid = dict(epochs=epochs, activationf = activationf, innerlayers = innerlayers)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=10)


# fle_s='Emotion_detection.h5'
# checkpointer = ModelCheckpoint(fle_s, monitor='loss',verbose=1,save_best_only=True,save_weights_only=False, mode='auto',save_freq='epoch')
# callback_list=[checkpointer]
print("i came here")
#History = None
#with parallel_backend('threading'):
History = grid.fit(X_train,Y_train)#,batch_size=32,validation_data=(X_test,Y_test),epochs=10)#,callbacks=[callback_list])
# score = grid.evaluate(X_train, Y_train)
# score = grid.evaluate(X_test, Y_test)
# Pred=grid.predict(X_test)
Pred=model.predict(X_test)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
#plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
          #              wspace=0.35)
plt.savefig('g5.png')
plt.close()

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
#plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                       #wspace=0.35)
plt.savefig('g6.png')

best_parameters = grid.best_params_
print(best_parameters)

cv_results = grid.cv_results_
print(cv_results["params"])
print(cv_results["mean_test_score"])

