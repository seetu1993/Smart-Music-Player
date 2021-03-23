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

with open('emotion_images.pkl', 'rb') as f:
  images = pickle.load(f)

with open('emotion_labels.pkl', 'rb') as f:
  labels = pickle.load(f)

images_f=np.array(images)
labels_f=np.array(labels)
images_f_2=images_f/255

# corresponding to 7 emotions there are 7 classes
num_of_classes=7
labels_encoded=tf.keras.utils.to_categorical(labels_f,num_classes=num_of_classes)

#splitting the dataset into training and test data set
X_train, X_test, Y_train, Y_test= train_test_split(images_f_2, labels_encoded,test_size=0.25)


def Convolution(input_tensor,filters):
  x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1),kernel_regularizer=l2(0.001))(input_tensor)
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
  output= Dense(7,activation=activationf)(drop_1)
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
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=10)


# fle_s='Emotion_detection.h5'
# checkpointer = ModelCheckpoint(fle_s, monitor='loss',verbose=1,save_best_only=True,save_weights_only=False, mode='auto',save_freq='epoch')
# callback_list=[checkpointer]
print("i came here")
History=grid.fit(X_train,Y_train)

Pred=Model.predict(X_test)
import matplotlib.pyplot as plt
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
plt.savefig('graph1.png')
plt.close()
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
#plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
#                       wspace=0.35)
plt.savefig('graph2.png')

best_parameters = grid.best_params_
print(best_parameters)

cv_results = grid.cv_results_
print(cv_results["params"])
print(cv_results["mean_test_score"])

