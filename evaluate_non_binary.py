import os
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import SGD
import numpy as np

from helpers import resize_to_fit

image_size = 150
img_width, img_height = image_size, image_size
validation_directory = os.path.expanduser('~/euro13k/validation/testing_split/')
#validation_directory = os.path.expanduser('D:/Documents/research/evaluation/ninos/cropped_aligned/')
model_path = os.path.expanduser('data_augmented_euro13k_vgg16.h5')

# load the model we saved
model = load_model(model_path)

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

for i in range(1,26):

    predict_image_directory = os.path.expanduser(validation_directory + str(i))

    files_ = [f for f in os.listdir(predict_image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    list_images_labels = []

    with open('eval.csv', "w") as csv:
        columnTitleRow = "real age, estimated\n"
        csv.write(columnTitleRow)

        for f in files_:
            data = []

            real_age = i

#            img = image.load_img(os.path.join(predict_image_directory, f), target_size=(img_width, img_height))

            image = cv2.imread(os.path.join(predict_image_directory, f))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = resize_to_fit(image, image_size, image_size)
            image = np.expand_dims(image, axis=2)

            #data = np.array(data, dtype="float") / 255.0

#            x = image.img_to_array(img)
#            x = np.expand_dims(x, axis=0)

            value = model.predict_classes(image)

            print(value)
            #print(value)

            #list_images_labels.append((f, real_age, value[0]))

            row = str(real_age) + "," + str(value[0]) + "\n"
            csv.write(row)

            print(real_age, value[0])
