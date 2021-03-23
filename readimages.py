
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# file path where the dataset resides
filedir="/home/azureuser/server/CK+48"

# print the name of the folders present in the dataset

files=os.listdir(filedir)
print(files)

# storing the emotions in the list
emotion =['happy', 'disgust', 'contempt', 'sadness', 'fear', 'surprise', 'anger']

# Read each image using opencv , resize it to 48x48
# append the image on images list and  label in the labels list


i=0
images=[]
labels=[]
for file in files:
  idx=emotion.index(file)
  label=idx
  full_path=filedir+'/'+file
  files_exp= os.listdir(full_path)
  counter = 0

  for file_2 in files_exp:
    file_main=full_path+'/'+file_2
    image= cv2.imread(file_main)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image= cv2.resize(image,(48,48))
    images.append(image)
    labels.append(label)
    i+=1

#saving images , labels  by pickle

with open('emotion_images.pkl', 'wb') as f:
  pickle.dump(images, f)
with open('emotion_labels.pkl', 'wb') as f:
  pickle.dump(labels, f)
