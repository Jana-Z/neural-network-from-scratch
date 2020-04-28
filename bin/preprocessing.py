import matplotlib.pyplot as plt
from mnist import MNIST
import numpy as np

def load_and_split_mnist(mnist_path='./mnist-data', test_split= 0.05):
    mndata = MNIST(mnist_path)
    X, y = mndata.load_training()   # already shuffled
    threshhold_index = int(test_split * len(X))
    y = hot_one_encode(y, 10)
    X_test, y_test = X[:threshhold_index], y[:threshhold_index]
    X_train, y_train = X[threshhold_index:], y[threshhold_index:]
    X_train, y_train,  X_test, y_test = (
        np.squeeze(np.array([X_train]), axis=0) / 255,
        np.squeeze(np.array([y_train]), axis=0),
        np.squeeze(np.array([X_test]), axis=0) / 255,
        np.squeeze(np.array([y_test]), axis=0))
    print(f'shape of:\nX_train: {X_train.shape}\ny_train: {y_train.shape}\nX_test: {X_test.shape}\ny_test: {y_test.shape}')
    return X_train, y_train, X_test, y_test

def plot_image(image, label):
    plt.title(f'This is a {label}')
    reshaped_image = np.array(image).reshape(28, 28)
    plt.imshow(reshaped_image, cmap='gray')
    plt.show()

def hot_one_encode(data, nb_classes):
    '''Convert an iterable of indices to one-hot encoded labels.'''
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
