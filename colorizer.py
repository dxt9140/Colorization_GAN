import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import keras
import pickle
from cnn import CNN
from definitions import *
import cv2


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict


def create_new_data_split(file):
    batch_1 = unpickle(DATA_DIR + 'data_batch_1')
    batch_2 = unpickle(DATA_DIR + 'data_batch_2')
    batch_3 = unpickle(DATA_DIR + 'data_batch_3')
    batch_4 = unpickle(DATA_DIR + 'data_batch_4')
    batch_5 = unpickle(DATA_DIR + 'data_batch_5')
    test = unpickle(DATA_DIR + 'test_batch')

    all_data_flat = batch_1[b'data']
    all_data_flat = np.append(all_data_flat, batch_2[b'data'], axis=0)
    all_data_flat = np.append(all_data_flat, batch_3[b'data'], axis=0)
    all_data_flat = np.append(all_data_flat, batch_4[b'data'], axis=0)
    all_data_flat = np.append(all_data_flat, batch_5[b'data'], axis=0)

    all_data = all_data_flat.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    all_labels = batch_1[b'labels']
    all_labels = np.append(all_labels, batch_2[b'labels'], axis=0)
    all_labels = np.append(all_labels, batch_3[b'labels'], axis=0)
    all_labels = np.append(all_labels, batch_4[b'labels'], axis=0)
    all_labels = np.append(all_labels, batch_5[b'labels'], axis=0)

    test_data = test[b'data']
    test_data = test_data.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    test_labels = test[b'labels']

    full = dict()
    full['x_train'] = all_data
    full['y_train'] = all_labels
    full['x_test'] = test_data
    full['y_test'] = test_labels

    with open(file, 'wb') as fl:
        pickle.dump(full, fl)

    return full


def load_data_split(file):
    if not os.path.exists(file):
        return None
    else:
        return unpickle(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="new_split",
                        action='store_true', help="Force a new data split to be created", default=False)
    parser.add_argument("-i", dest="iterations", type=int, nargs=1, help="Number of iterations to run", default=50)
    parser.add_argument("-f", dest="infile", type=str, help="Use a file as input", default="input.pkl")
    parser.add_argument("-e", dest="epochs", type=int, help="Specify a number of epochs to run", default=100)
    parser.add_argument("-b", dest="batch_size", type=int,
                        help="Specify the size of each batch to run through an epoch", default=100)
    argl = parser.parse_args()
    print(argl)

    if argl.new_split or not os.path.exists(INPUT_DIR + argl.infile):
        print("Making new data split: ", argl.infile)
        data_split = create_new_data_split(INPUT_DIR + argl.infile)
    else:
        print("Loading data split: ", argl.infile)
        data_split = load_data_split(INPUT_DIR + argl.infile)

    print(data_split.keys())
    data = data_split['x_train']
    print(data.shape)

    model = CNN.build()
    model.compile(optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9), loss='mean_squared_error')

    gray_data = list()
    for d in data:
        gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        gray = np.reshape(gray, (gray.shape[0], gray.shape[1], 1))
        gray_data.append(gray)
    gray_data = np.array(gray_data)

    num_samples = data.shape[0]

    for i in range(argl.iterations):
        for e in range(argl.epochs):
            sample_idc = np.random.randint(0, num_samples, argl.batch_size)
            sampled_images = gray_data[sample_idc]
            real_images = data[sample_idc]

            model.train_on_batch(sampled_images, real_images)

    cv2.namedWindow("Compare")
    cv2.resizeWindow("Compare", 300, 400)

    while 1:

        sample = np.random.randint(0, num_samples, 1)

        gray = gray_data[sample]
        img = data[sample]

        predict = model.predict(gray)
        predict = np.reshape(predict, (32, 32, 3))

        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        all = np.hstack((img, gray, predict))

        cv2.imshow("Compare", all)
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    main()
