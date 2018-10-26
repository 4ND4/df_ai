import cv2
import pickle
import os.path
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

import config
from helpers import resize_to_fit


directory_path = os.path.expanduser(config.directory_path)
MODEL_LABELS_FILENAME = config.MODEL_LABELS_FILENAME


# initialize the data and labels
data = []
labels = []
batch_size = 64
epochs = 100
image_size = 200
model_name = 'data_aug_{}_13k.h5'.format(image_size)

# loop over the input images

for root, dirs, files in os.walk(directory_path):
    for filename in files:

        file_name_no_extension, file_extension = os.path.splitext(filename)

        if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.png':
            full_path = os.path.join(root, filename)

            image = cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize the letter so it fits in a 20x20 pixel box
            #image = resize_to_fit(image, 20, 20)
            image = resize_to_fit(image, image_size, image_size)

            # Add a third channel dimension to the image to make Keras happy
            image = np.expand_dims(image, axis=2)

            label = os.path.basename(root)

            # Add the letter image and it's label to our training data
            data.append(image)
            labels.append(label)


# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.3, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(image_size, image_size, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(5, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network

callbacks = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
# autosave best Model
best_model_file = model_name
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=2, save_best_only=True)

# In[16]:

train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

val_datagen = ImageDataGenerator()

history = model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size), nb_epoch=epochs,
                                    samples_per_epoch=len(X_train), validation_data=val_datagen.flow(
        X_test, Y_test, batch_size=64, shuffle=False), nb_val_samples=len(X_test), callbacks=[callbacks, best_model])

#history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
#                    validation_data=(X_test, Y_test), shuffle=True, callbacks=[callbacks, best_model])

print('done')
