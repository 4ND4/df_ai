import cv2
import pickle
import os.path
import numpy as np
from keras import Model
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Flatten, Dense

import config
from helpers import resize_to_fit

directory_path = os.path.expanduser(config.directory_path)
MODEL_LABELS_FILENAME = config.MODEL_LABELS_FILENAME
MODEL_FILENAME = config.MODEL_FILENAME

# initialize the data and labels
data = []
labels = []

epochs = 100
image_size = 254
batch_size = 100

# loop over the input images

for root, dirs, files in os.walk(directory_path):
    for filename in files:

        file_name_no_extension, file_extension = os.path.splitext(filename)

        if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.png':
            full_path = os.path.join(root, filename)

            image = cv2.imread(full_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize the letter so it fits in a 20x20 pixel box
            #image = resize_to_fit(image, 20, 20)
            image = resize_to_fit(image, image_size, image_size)

            # Add a third channel dimension to the image to make Keras happy
            #image = np.expand_dims(image, axis=2)

            label = int(os.path.basename(root))

            # Add the letter image and it's label to our training data
            data.append(image)
            labels.append(label)


# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.3, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with

Y_train = to_categorical(LabelEncoder().fit_transform(Y_train))

Y_test = to_categorical(LabelEncoder().fit_transform(Y_test))


vgg16_base = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(image_size, image_size, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding new layers...')
output = vgg16_base.get_layer(index=-1).output
output = Flatten()(output)
# let's add a fully-connected layer
output = Dense(4096, activation="relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(512, activation="relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
# and a logistic layer -- let's say we have 200 classes
output = Dense(17, activation='softmax')(output)


vgg16_model = Model(vgg16_base.input, output)
#InceptionV3_model.summary()


# In[ ]:


for layer in vgg16_model.layers[:19]:
    layer.trainable = False


vgg16_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#model.fit(X_train, Y_train, batch_size=128, epochs=epochs, verbose=1)

# Train the neural network
#model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs, verbose=1)


callbacks = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
# autosave best Model
best_model_file = "data_augmented_weights.h5"
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


history = vgg16_model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size), nb_epoch=epochs,
                                    samples_per_epoch=len(X_train), validation_data=val_datagen.flow(
        X_test, Y_test, batch_size=64, shuffle=False), nb_val_samples=len(X_test), callbacks=[callbacks, best_model])

#history = vgg16_model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
#                          validation_data=(X_test, Y_test), shuffle=True, callbacks=[callbacks, best_model])


# Save the trained model to disk
#vgg16_model.save(MODEL_FILENAME)

print('done')