import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
import pickle


def get_data(output_array, path_folder, img_size, validation = False):

    if validation:
        path_folder = path_folder + "/validation"
    else:
        path_folder = path_folder + "/train"

    for i in Classes:
        path = os.path.join(path_folder,i)
        class_num = Classes.index(i)
        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            output_array.append([new_array, class_num])

def shuffle_data(raw_data):
    random.shuffle(raw_data)

def save_data(data_to_be_saved, train_or_test = "train"):

    x = []
    y = []

    for i in range(len(data_to_be_saved)):
        x.append(data_to_be_saved[i][0])
        y.append(data_to_be_saved[i][1])

    X = []

    X = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    if train_or_test == "train":
        save_file_X = "X_" + "train" + ".pickle"
        save_file_y = "y_" + "train" + ".pickle"
    else:
        save_file_X = "X_" + "test" + ".pickle"
        save_file_y = "y_" + "test" + ".pickle"

    pickle_out = open(save_file_X,"wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(save_file_y,"wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

def load_data(train_or_test = "train"):

    if train_or_test == "train":
        save_file_X = "X_" + "train" + ".pickle"
        save_file_y = "y_" + "train" + ".pickle"
    else:
        save_file_X = "X_" + "test" + ".pickle"
        save_file_y = "y_" + "test" + ".pickle"

    pickle_in = open(save_file_X,"rb")
    X = pickle.load(pickle_in)

    pickle_in = open(save_file_y,"rb")
    y = pickle.load(pickle_in)

    X = X/255.0

    return X, y

IMG_SIZE = 130
Classes = ["horses", "humans"]
Path = "C:/Users/yasht/Desktop/ML_Challenge"

raw_train_data = []
raw_test_data = []

get_data(raw_test_data, Path, IMG_SIZE, validation = True)

get_data(raw_train_data, Path, IMG_SIZE)

shuffle_data(raw_test_data)

shuffle_data(raw_train_data)

save_data(raw_test_data, train_or_test = "test")

save_data(raw_train_data, train_or_test = "train")

X_train, y_train = load_data(train_or_test = "train")

X_test, y_test = load_data(train_or_test = "test")

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, y_train, validation_data=(X_test,y_test),  batch_size=64, epochs=40)
loss, acc = model.evaluate(X_test,y_test, verbose = 0)
print("Accuracy is",acc * 100)

import matplotlib.pyplot as plt
#print(history.history.keys())
# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()


# Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.show()
#tensorboard --logdir==training:C:\Users\yasht\Desktop\horse-or-human\logs --host=127.0.0.1
#try to use tensorboard but its not working
