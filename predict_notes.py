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
    image_names = []
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
            image_names.append(img_file)
    return np.array(image_array), np.array(image_names)

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
final_images, file_names = image_array(combined)

# Load saved CNN model
model = keras.models.load_model('my_model')

# Create list with all images to be predicted
x_predict = []
for x in final_images:
    x_predict.append(x[0])

# Normalize the data
x_predict = np.array(x_predict)/255

# Generate predictions and combine half notes into single data frame
predictions = model.predict(x_predict)
pred_df = pd.DataFrame(predictions)

# Replace column names with those from the Rosetta Stone (removing the preface)
col_names = [str(x).replace("dataset cleaned left ", "").replace("dataset cleaned right ", "") for x in rosetta_stone["Note"]]
pred_df.columns = col_names

# Insert the file names for later referencing
pred_df.insert(0, "File Names", file_names)

# Read in the Rosetta Stone to Convert Class to Note
rosetta_stone = pd.read_csv("Rosetta Stone.csv")
rosetta_stone = rosetta_stone.iloc[:, 1:].transpose()
rosetta_stone.columns = ["Note", "Class"]

# Drop Duplicates to get Unique
rosetta_stone.drop_duplicates(inplace=True)

# Convert class to numeric to sort (so that columns are in the correct order)
rosetta_stone["Class"] = pd.to_numeric(rosetta_stone["Class"])
rosetta_stone = rosetta_stone.sort_values("Class")

# Sort and Group Half Note Predictions into 1 Prediction per Full Note
pred_df = pred_df.sort_values("File Names")
pred_df = pred_df.groupby("File Names").sum()
pred_df = pred_df.groupby(level=0, axis=1).sum()
pred_df = pred_df/2

# Find Most Probable Note and Probability it Exists
prob_note = pred_df.max(axis=1)
classes = pred_df.idxmax(axis=1)

# Save to CSV
file_names = np.unique(file_names)
final_pred_df = pd.DataFrame({"Image": file_names,
                              "Note Predicted": classes,
                              "Probability": prob_note})
final_pred_df.to_csv("Final Predictions.csv")


# Save Note Counts to CSV
df = final_pred_df.value_counts("Note Predicted")
df.to_csv("Note Count.csv")
