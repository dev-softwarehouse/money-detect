from PIL import Image
from numpy import array
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot
import cv2
import os


def image_array(file_names):
    '''This loads in all the images and their respective label and stores it as a nested list'''
    image_array = []
    for i in file_names.iterrows():
        img_file = i[1][0]
        label = i[1][1]
        print(img_file)
        img = Image.open(f"dataset/{img_file}")
        img_size = img.size
        if img_size[0] < img_size[1]:
            img = img.rotate(90, expand=True)
        arr = array(img)
        image_array.append([arr, label])
    return(image_array)

def partition_notes(images_array, notes):
    '''This creates a dictionary with each denomination as a key'''
    ind_notes = []
    for z in images_array:
        if z[1] == notes:
            ind_notes.append(z[0])
    return(ind_notes)

def flip_correct(images):
    '''This checks if the bills are all the same way around. If they're not, it flips them'''
    for i in range(0, len(images)-1):
        # Converts numpy to an open CV image
        imageA = images[i][:, :, ::-1].copy()
        imageB = images[i+1][:, :, ::-1].copy()
        # convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        # Tests to see the similarity
        (score, diff) = ssim(grayA, grayB, full=True)
        # If the similarity is below a certain score, assume it's flipped the wrong way
        # and flip it back
        if score < 0.6:
            temp = Image.fromarray(images[i])
            temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
            temp = temp.transpose(Image.FLIP_TOP_BOTTOM)
            images[i] = np.asarray(temp)
    return(images)

def split_left_right(images):
    # Split the notes into left halves and right halves
    split_left = []
    split_right = []
    for image in images:
        image = Image.fromarray(image)
        width, height = image.size
        temp_left = image.crop((0, 0, width/2, height))
        temp_right = image.crop((width/2, 0, width, height))
        split_left.append(np.asarray(temp_left))
        split_right.append(np.asarray(temp_right))
    return split_left, split_right

def save_images(images, bill, side):
    # Function to save all the data
    count = 0
    if not os.path.exists(f"dataset/cleaned/{side}/{bill}"):
        os.makedirs(f"dataset/cleaned/{side}/{bill}")
    for image in images:
        count = count + 1
        image = Image.fromarray(image)
        image.save(f"dataset/cleaned/{side}/{bill}/{str(count)}.jpg")



# Load in all the files using the file name <-> denomination CSV
file_list = pd.read_csv("dataset_file_names_labelled.csv")
images = image_array(file_list)

# Get all the unique denominations
unique_labels = pd.unique(file_list.iloc[:, 1])

# Create a dictionary where the images of denominations of the same type
# are all linked to the key (the key being that denomination)
separated_notes = dict()
for note in unique_labels:
    temp_dict = {str(note): partition_notes(images, note)}
    separated_notes.update(temp_dict)

# Checks that all the notes are the same way around
for note in unique_labels:
    note = str(note)
    if len(separated_notes[note]) > 1:
        temp_dict = {note: flip_correct(separated_notes[note])}
        separated_notes.update(temp_dict)

# Split in to left and right halves
separated_notes_left = separated_notes.copy()
separated_notes_right = separated_notes.copy()
for note in unique_labels:
    note = str(note)
    if len(separated_notes[note]) > 1:
        notes_left, notes_right = split_left_right(separated_notes[note])
        temp_dict_left = {note: notes_left}
        temp_dict_right = {note: notes_right}
        separated_notes_left.update(temp_dict_left)
        separated_notes_right.update(temp_dict_right)

# Save images
for note in unique_labels:
    note = str(note)
    if len(separated_notes[note]) > 1:
        save_images(separated_notes_left[note], note, 'left')
        save_images(separated_notes_right[note], note, 'right')
