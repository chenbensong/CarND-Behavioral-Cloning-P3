import os
import csv
import cv2 
import numpy as np
import argparse

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default = "",
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()
    # load training data from disk
    images = []
    measurements = []
    datadir = "data"
    correction = 0.2

    def load_image(source, datadir, dir):
        filename = source.split('/')[-1]
        current_path = os.path.join(datadir, dir, 'IMG', filename)
        # image = cv2.imread(current_path)
        return current_path

    def add_image(image, measurement):
        images.append(image)
        measurements.append(measurement)


    for dir in os.listdir(datadir):
        csvpath = os.path.join(datadir, dir, "driving_log.csv")
        print(csvpath)
        if os.path.exists(csvpath):
            print("Load image in: " + csvpath)
            with open(csvpath) as csvfile:
                dictreader = csv.DictReader(csvfile)
                for row in dictreader:
                    image = load_image(row['center'], datadir, dir)
                    measurement = float(row['steering'])
                    add_image(image, measurement)

                    image = load_image(row['left'], datadir, dir)
                    add_image(image, measurement+correction)
                    image = load_image(row['right'], datadir, dir)
                    add_image(image, measurement - correction)


    data = list(zip(images, measurements))

    print(len(data))

    train_data, validation_data = train_test_split(data, test_size=0.2)
    print(len(train_data))
    print(len(validation_data))

    def generator(samples, batch_size=32):
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                imgs = []
                angles = []
                for batch_sample in batch_samples:
                    name = batch_sample[0]
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[1])
                    imgs.append(image)
                    angles.append(angle)
                    # add flipped image
                    imgs.append(np.fliplr(image))
                    angles.append(-measurement)

                # trim image to only see section with road
                X_train = np.array(imgs)
                y_train = np.array(angles)
                # print(X_train.shape)
                # print(y_train.shape)
                yield sklearn.utils.shuffle(X_train, y_train)

    # X_train = np.array(images)
    # y_train = np.array(measurements)


    # define neural network structure

    model = None
    if args.model != "":
        model = load_model(args.model)
    else:
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
        model.add(Dense(100, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1))

    # train neural network
    model.compile(loss='mse', optimizer='adam')

    batch_size = int(80/2)
    print(batch_size)
    # compile and train the model using the generator function
    train_generator = generator(train_data, batch_size=batch_size)
    validation_generator = generator(validation_data, batch_size=batch_size)

    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
    model.fit_generator(train_generator,
                        steps_per_epoch=int(len(train_data)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=int(len(validation_data)/batch_size),
                        nb_epoch=10)

    # save weights to file
    model.save('model.h5')
