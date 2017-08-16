import os
import csv
import cv2 
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D

# load training data from disk
images = []
measurements = []
datadir = "data"
for dir in os.listdir(datadir):
    csvpath = os.path.join(datadir, dir, "driving_log.csv")
    print(csvpath)
    if os.path.exists(csvpath):
        print("Load image in: " + csvpath)
        with open(csvpath) as csvfile:
            dictreader = csv.DictReader(csvfile)
            for row in dictreader:
                source_path = row['center']
                filename = source_path.split('/')[-1]
                current_path = os.path.join(datadir, dir, 'IMG', filename)
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(row['steering'])
                measurements.append(measurement)
                # add flipped image
                images.append(np.fliplr(image))
                measurements.append(-measurement)

X_train = np.array(images)
y_train = np.array(measurements)


# define neural network structure
# model = Sequential()
# model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# model.add(Lambda(lambda x: x / 255.0 - 0.5))
# model.add(Conv2D(6, (5, 5), activation="relu"))
# model.add(MaxPooling2D())
# model.add(Conv2D(6, (5, 5), activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# nvidia self driving car network architeture
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
# model.add(MaxPooling2D())
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
# model.add(MaxPooling2D())
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# train neural network
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

# save weights to file
model.save('model.h5')
