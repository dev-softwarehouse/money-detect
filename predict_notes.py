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
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
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
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            resized_arr = cv2.resize(open_cv_image, [int(img_size[0]), int(img_size[1])])
            image_array.append([resized_arr, label])
    return np.array(image_array)

# The resized image size, needed for memory size
img_size = (403, 226)

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

# Convert classes to not take into account left or right (needed for final note count)
new_labels = [x.replace('dataset cleaned left ', '').replace('dataset cleaned right ', '') for x in labels]

# Load saved CNN model
model = keras.models.load_model('my_model')

# Create list with all images to be predicted
x_predict = []
for x in final_images_labelled:
    x_predict.append(x[0])

# Normalize the data
x_predict = np.array(x_predict)/255

# Generate predictions and convert probabilities to predictions
predictions = model.predict(x_predict)
classes = np.argmax(predictions, axis = 1)

# Convert classes number to note
pred_notes = []
for i in classes:
    pred_notes.append(new_labels[i])

# Convert predicted notes into dictionary with note count (divided by 2 because of left/right)
count_notes = {}
for x in new_labels:
    count_notes.update({x: pred_notes.count(x)/2})

# Save predictions to CSV
df = pd.DataFrame(list(count_notes.items()),columns = ['Note','Count'])
df.to_csv("Note Count.csv")
