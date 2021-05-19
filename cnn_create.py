from PIL import Image
from numpy import array
import pandas as pd
import numpy as np
from numpy import array
import os
import cv2
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder


def image_array(full_zip):
    '''This loads in all the images and their respective label and stores it as a nested list'''
    global img_size
    image_array = []
    for zip_name in full_zip:
        file_list = os.listdir(zip_name[1])
        for img_file in file_list:
            label = zip_name[0]
            file_name = (zip_name[1] + '/' + img_file)
            print(file_name)
            img = Image.open(file_name)
            open_cv_image = array(img)
            # Convert RGB to BGR
            # open_cv_image = open_cv_image[:, :, ::-1].copy()
            resized_arr = cv2.resize(open_cv_image, [int(img_size[0]), int(img_size[1])])
            image_array.append([resized_arr, label])
    return np.array(image_array)

# The resized image size, needed for memory size
img_size = (202, 226)

# Finding folder - file structure
subdirs = [x[0].replace('./', '') for x in os.walk('./dataset')]
max_depth = max([x.count('/') for x in subdirs])

# Get file names
final_dirs = [x for x in subdirs if x.count('/') == max_depth]

# Convert directory into note name
labels = [x.replace('/dataset/cleaned/', '').replace('/', ' ') for x in final_dirs]

# Combined file names with class names
combined = zip(labels, final_dirs)

# Get images array
final_images_labelled = image_array(combined)

# Generate training x and y variables
x_train = []
y_train = []
for x in final_images_labelled:
    x_train.append(x[0])
    y_train.append(x[1])

# Normalize the data
x_train = np.array(x_train)/255

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
y_train_dummy = np_utils.to_categorical(encoded_Y)

rosetta_stone = pd.DataFrame([y_train, encoded_Y])
rosetta_stone.to_csv("Rosetta Stone.csv")

# Generate Keras CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size[1], img_size[0], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(12, activation="softmax"))

# Print model summary
model.summary()

# Compile model
model.compile(optimizer = Adam() , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# Fit Model
history = model.fit(x_train, y_train_dummy, epochs = 10)

# Save model
model.save('my_model')
