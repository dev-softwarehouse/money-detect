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

def split_left_right(image):
    # Split the notes into left halves and right halves
    width, height = image.size
    split_left = image.crop((0, 0, width/2, height))
    split_right = image.crop((width/2, 0, width, height))
    return array(split_left), array(split_right)

def image_array(file_names):
    '''This loads in all the images and their respective label and stores it as a nested list'''
    global crop_img_size
    image_array = []
    img = Image.open(f"{file_names}")
    img_size = img.size
    if img_size[0] < img_size[1]:
        img = img.rotate(90, expand=True)
    split_left, split_right = split_left_right(img)
    def resize_img(resize_img):
        return(cv2.resize(resize_img, [int(crop_img_size[0]), int(crop_img_size[1])]))
        # arr = arr[:, :, ::-1].copy()
    split_left = resize_img(split_left)
    split_right = resize_img(split_right)
    return([split_left, split_right])

crop_img_size = (202, 226)

image_file = input("Please enter image file name: ")

final_images = image_array(image_file)

# Load saved CNN model
model = keras.models.load_model('my_model')

x_predict = []
for x in final_images:
    x_predict.append(x)

# Normalize the data
x_predict = np.array(x_predict)/255

# Generate predictions and combine half notes into single data frame
predictions = model.predict(x_predict)
pred_df = pd.DataFrame(predictions)

# Read in the Rosetta Stone to Convert Class to Note
rosetta_stone = pd.read_csv("Rosetta Stone.csv")
rosetta_stone = rosetta_stone.iloc[:, 1:].transpose()
rosetta_stone.columns = ["Note", "Class"]

# Drop Duplicates to get Unique
rosetta_stone.drop_duplicates(inplace=True)

# Convert class to numeric to sort (so that columns are in the correct order)
rosetta_stone["Class"] = pd.to_numeric(rosetta_stone["Class"])
rosetta_stone = rosetta_stone.sort_values("Class")


# Replace column names with those from the Rosetta Stone (removing the preface)
col_names = [str(x).replace("dataset cleaned left ", "").replace("dataset cleaned right ", "") for x in rosetta_stone["Note"]]
pred_df.columns = col_names

# Convert class to numeric to sort (so that columns are in the correct order)
rosetta_stone["Class"] = pd.to_numeric(rosetta_stone["Class"])
rosetta_stone = rosetta_stone.sort_values("Class")

# Insert the file names for later referencing
pred_df.insert(0, "File Names", image_file)

# Sort and Group Half Note Predictions into 1 Prediction per Full Note
pred_df = pred_df.groupby("File Names").sum()
pred_df = pred_df.groupby(level=0, axis=1).sum()
pred_df = pred_df/2

# Find Most Probable Note and Probability it Exists
prob_note = pred_df.max(axis = 1)
classes = pred_df.idxmax(axis = 1)

final_pred_df = pd.DataFrame({"Note Predicted": classes,
                              "Probability": prob_note})

print(final_pred_df)
final_pred_df.to_csv("Note Prediction.csv")
