

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





from keras.utils import Sequence
import math
class datagen(Sequence):
  def __init__(self,labels,batch_size,image_size):
    self.target=labels
    self.batch_length=batch_size
    self.image_length=image_size
    self.size=len(labels)
  def __getitem__(self,i):
    start=i*self.batch_length
    end=min((i+1)*self.batch_length,self.size)
    train_img=np.empty(((end-start)*6,self.image_length,self.image_length,3),'uint8')
    a=0
    for x in range(start,end):
      img = x_tr[x]
      num_x = np.random.randint(0,30)
      num_y = np.random.randint(0,30)
      img_ran = img[num_x:num_x+128, num_y:num_y+128]
      rot_ran = np.random.randint(1,10)
      rot_ran_neg = np.random.randint(1,10)
      rot_img_pos = ndimage.rotate(img_ran, rot_ran)
      rot_img_neg = ndimage.rotate(img_ran, -rot_ran_neg)
      rot_pos = cv2.resize(rot_img_pos,(128,128))
      rot_neg = cv2.resize(rot_img_neg,(128,128))
      rows,cols, depth = img_ran.shape
      ran_shift_x = np.random.randint(1,10)
      ran_shift_y = np.random.randint(1,10)
      M = np.float32([[1,0,ran_shift_x],[0,1,ran_shift_y]])
      img_shift = cv2.warpAffine(img,M,(cols,rows))
      ran_shift_x_new = np.random.randint(1,10)
      ran_shift_y_new = np.random.randint(1,10)
      img_sh_base = cv2.resize(img,(128,128))
      M = np.float32([[1,0,ran_shift_x],[0,1,ran_shift_y]])
      img_shift_new = cv2.warpAffine(img_sh_base,M,(cols,rows))
      horizontal_img = cv2.flip(img_ran, 1 )
      bright50 = increase_brightness(img_ran, value=50)
      bright70 = increase_brightness(img_ran, value=70)
      bright_num= np.random.randint(10,40)
      bright_ran = increase_brightness(img_ran, value=bright_num)
        
      
      train_img[a] = img_ran
      train_img[a + end-start] = rot_pos
      train_img[a + 2*(end-start)] = rot_neg
      train_img[a + 3*(end-start)] = img_shift
      
      train_img[a + 4*(end-start)] = horizontal_img
      
      train_img[a + 5*(end-start)] = bright50
      
      
      a=a+1
      
    Y_train=self.target[start:end]
    y_train=np.concatenate((Y_train,Y_train,Y_train,Y_train,Y_train,Y_train),axis=0)
    return (train_img,y_train)
  def __len__(self):
    return math.ceil(self.size/self.batch_length)

from keras.utils import Sequence
import math
class datagen1(Sequence):
    
  def __init__(self,labels,batch_size,image_size):
    self.target=labels
    self.batch_length=batch_size
    self.image_length=image_size
    self.size=len(labels)
  def __getitem__(self,i):
    start=i*self.batch_length
    end=min((i+1)*self.batch_length,self.size)
    train_img=np.empty((end-start,self.image_length,self.image_length,3),'uint8')
    a=0
    for x in range(start,end):
      img = x_te[x]
      img = cv2.resize(img, (128,128))
      train_img[a]=img
      a=a+1
    y_train=self.target[start:end]
    
    return (train_img,y_train)
  def __len__(self):
    return math.ceil(self.size/self.batch_length)

from keras.utils import Sequence
import math
class datagen_tr(Sequence):
    
  def __init__(self,labels,batch_size,image_size):
    self.target=labels
    self.batch_length=batch_size
    self.image_length=image_size
    self.size=len(labels)
  def __getitem__(self,i):
    start=i*self.batch_length
    end=min((i+1)*self.batch_length,self.size)
    train_img=np.empty((end-start,self.image_length,self.image_length,3),'uint8')
    a=0
    for x in range(start,end):
      img = cv2.imread('HAM1000'+path[x])
      img = cv2.resize(img, (224,224))
      train_img[a]=img/255.0
      a=a+1
    y_train=self.target[start:end]
    
    return (train_img,y_train)
  def __len__(self):
    return math.ceil(self.size/self.batch_length)

from keras.utils import Sequence
import math
class datagen_val(Sequence):
    
  def __init__(self,labels,batch_size,image_size):
    self.target=labels
    self.batch_length=batch_size
    self.image_length=image_size
    self.size=len(labels)
  def __getitem__(self,i):
    start=i*self.batch_length
    end=min((i+1)*self.batch_length,self.size)
    train_img=np.empty((end-start,self.image_length,self.image_length,3),'uint8')
    a=0
    for x in range(start,end):
      img = cv2.imread('HAM1000'+path[9050+x])
      img = cv2.resize(img, (224,224))
      train_img[a]=img/255.0
      a=a+1
    y_train=self.target[start:end]
    
    return (train_img,y_train)
  def __len__(self):
    return math.ceil(self.size/self.batch_length)

traingen=datagen_tr(y1[:9050],16,224)
valgen=datagen_val(y1[9050:],16,224)

##traingen=datagen(y_tr,16,128)
##valgen=datagen1(y_te,16,128)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.1, 
                                            min_lr=0.000000001)











weight_decay = 1e-2
base = keras.applications.xception.Xception(alpha=1.0, include_top=False, weights='imagenet')
x = base.output

x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
predictions = Dense(7, activation='softmax',kernel_regularizer=regularizers.l2(weight_decay))(x)

model_xcep = Model(inputs=base.input, outputs=predictions)

for layer in base.layers:
    layer.trainable = True


adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_xcep.compile(loss='categorical_crossentropy',optimizer=adam ,metrics=['acc'])

model_xcep.summary

model_xcep.fit_generator(traingen,
                   validation_data=valgen,
                   epochs=5,
                   shuffle=True,
                   verbose=1,
                   callbacks=[learning_rate_reduction],
                   ##class_weight = class_weights
                   )


model_xcep.save('xcep_skin.h5')