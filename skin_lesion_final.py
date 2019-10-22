
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle

!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

!unzip skin-cancer-mnist-ham10000.zip

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.callbacks import *
from torchvision.models import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import keras
import shutil
import keras
from keras.models import Model,Sequential
from keras.layers import *
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

path = Path('')
labels = pd.read_csv('HAM10000_metadata.csv', sep=',')
labels.head()

imageid = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(path, '*', '*.jpg'))}

labels['path'] = labels['image_id'].map(imageid.get)
labels['path'] = labels['path'].str[9:]
labels.sample(5)

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
labels["dx"] =labels["dx"].map({'nv': 'Melanocytic nevi','mel': 'Melanoma', 
                                'bkl': 'Benign keratosis ','bcc': 'Basal cell carcinoma',
                                'akiec': 'Actinic keratoses','vasc': 'Vascular lesions',
                                'df': 'Dermatofibroma'})

y = labels['dx']
y1 = pd.get_dummies(y)
path = labels['path']

y1.columns

x_train=np.empty((10015,100,100,3),np.uint8)
for i in range(10015):
  im = cv2.imread('HAM1000'+path[i])
  x_train[i]=cv2.resize(im,(100,100))
  
  
y = labels['dx']
y1 = pd.get_dummies(y)

img = x_train[3456]
cv2.imwrite('file.jpg',img)

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_train = (x_train - x_train_mean)/x_train_std

from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(x_train, y1, test_size=0.1, random_state=42)

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=5,  
        zoom_range = 0.1,  
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=False,  
        vertical_flip=False)  
datagen.fit(x_tr)

input_shape = (100, 100, 3)
num_classes = 7
weight_decay = 0.01

model = Sequential()
model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y),
                                                 y)

epochs = 50 
batch_size = 20
history = model.fit_generator(
    datagen.flow(x_tr,y_tr, batch_size=batch_size),
    steps_per_epoch=x_tr.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(x_te,y_te),
    validation_steps=x_te.shape[0] // batch_size
    ,callbacks=[learning_rate_reduction],
    class_weight = class_weights
)

model.save('sigmoid_new.h5')

