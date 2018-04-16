
# coding: utf-8

# In[1]:


import os
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
import sklearn


# In[2]:


samples = []
with open('D:/learning/udacity/CarND/term1/data/P3_data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

del samples[0]


# In[3]:


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[4]:


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                center_current_path ='D:/learning/udacity/CarND/term1/data/P3_data/data/IMG/'+filename
                
                source_path = batch_sample[1]
                filename = source_path.split('/')[-1]
                left_current_path ='D:/learning/udacity/CarND/term1/data/P3_data/data/IMG/'+filename
                
                source_path = batch_sample[2]
                filename = source_path.split('/')[-1]
                right_current_path ='D:/learning/udacity/CarND/term1/data/P3_data/data/IMG/'+filename
                
                center_image = cv2.imread(center_current_path) 
                center_angle = float(batch_sample[3])
                center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
                
                left_image = cv2.imread(left_current_path)
                left_angle = float(batch_sample[3])+0.2
                left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                
                
                right_image = cv2.imread(right_current_path)
                right_angle = float(batch_sample[3])-0.2
                right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                
                images.append(center_image)
                angles.append(center_angle)
                
                images.append(left_image)
                angles.append(left_angle)
                
                images.append(right_image)
                angles.append(right_angle)
             
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)
            
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
            
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[5]:


model = Sequential()
model.add(Lambda(lambda x:(x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
#model.add(Dropout(0.8))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#model.add(Dropout(0.8))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
#model.add(Dropout(0.8))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Dropout(0.8))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(0.8))
model.add(Activation('relu'))
model.add(Dense(50))
#model.add(Dropout(0.8))
model.add(Activation('relu'))
model.add(Dense(10))
#model.add(Dropout(0.8))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')

