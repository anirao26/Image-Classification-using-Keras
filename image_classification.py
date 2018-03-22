import pandas as pd
import numpy as np
from keras.models import Sequential, Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D as Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image
plt.style.use('ggplot')


def read_train_data():
    image_arr = []
    with (open('train-x.txt','r')) as master_data:
        for line in master_data:
            img = Image.open("train/"+line.strip())
            image_arr.append(np.array(img.resize((28, 28), PIL.Image.ADAPTIVE)))
    return np.array(image_arr)


def read_train_labels():
    labels_data = []
    with (open('train-y.txt','r')) as master_data:
        for y in master_data:
            labels_data.append(y.strip())
    return(np.array(labels_data))


def read_test_data():
    image_arr = []
    with (open('test-x.txt','r')) as master_data:
        for line in master_data:
            img = Image.open("test/"+line.strip())
            image_arr.append(np.array(img.resize((28, 28), PIL.Image.ADAPTIVE)))
    return np.array(image_arr)


def shuffle_data(image_data, labels_data):
    X_train, X_val, y_train, y_val = train_test_split(image_data, labels_data, test_size=0.2, random_state=1)
    return X_train, X_val, y_train, y_val


def encode_label_values(labels_data, num_classes):
    return(np_utils.to_categorical(labels_data, num_classes))


def build_keras_model(learning_rate, epoch):
    # print(learning_rate)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    decay = learning_rate/epoch
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def tune_hyper_parameters(train_data, dev_data, train_label_values, dev_label_values, epoch, learning_rate, batch_size):

    train_label_values = encode_label_values(train_label_values, 2)
    dev_label_values = encode_label_values(dev_label_values, 2)
    model = build_keras_model(learning_rate, epoch)
    history = model.fit(train_data, train_label_values,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1,
          validation_data=(dev_data, dev_label_values),
          shuffle=True
          )


    test_data_values = read_test_data()
    print(model.predict_classes(test_data_values))

    #Plot the Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig("Loss_Curves.png")

    #Plot the Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.savefig("Accuracy_Curves.png")


def main():
    image_data = read_train_data()
    labels_data = read_train_labels()
    image_data_test = read_test_data()
    train_data, dev_data, train_label_values, dev_label_values = shuffle_data(image_data, labels_data)
    tune_hyper_parameters(train_data, dev_data, train_label_values, dev_label_values, 1000, 0.0001, 50)

    
if __name__ == '__main__':
    main()

