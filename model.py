import csv 
import cv2 
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

# load training data from disk
images = []
measurements = []
with open("data/driving_log.csv") as csvfile: 
    dictreader = csv.DictReader(csvfile)
    for row in dictreader:
        source_path = row['center']
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(row['steering'])
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


# define neural network structure
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

# train neural network
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)

# save weights to file
model.save('model.h5')