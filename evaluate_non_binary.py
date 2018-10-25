import os

from keras.models import load_model
from keras.preprocessing import image
import numpy as np


img_width, img_height = 254, 254
validation_directory = os.path.expanduser('~/euro13k/euro13k/')
model_path = os.path.expanduser('visage_model_vgg16_100_600s.h5')

# load the model we saved
model = load_model(model_path)

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
            real_age = i

            img = image.load_img(os.path.join(predict_image_directory, f), target_size=(img_width, img_height))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            value = model.predict_classes(x)

            list_images_labels.append((f, real_age, value[0]))

            row = str(real_age) + "," + str(value[0]) + "\n"
            csv.write(row)

            print(real_age, value[0])