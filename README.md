# Money Detect
Requires modules:
  numpy
  pandas
  tensorflow
  keras
  opencv
  scikit-learn
  pillow

# Script Order
image_manipulate.py: This ensures that all notes are rotated and flipped so they are the same way around. They are then split in to left & right halves.
cnn_create.py: This loads the cleaned and halved images, resizes (for memory purposes) the images, converts into a format for Keras, and then runs the CNN
predict_notes.py: Loads in the prediction data set and does the necessary transformations. Loads in the CNN model. Creates predictions and then saves the count of the notes in to a CSV file
