import numpy as np
from matplotlib import pyplot as plt


def imshow(arr, vmin=0.0, vmax=1.0):
    # add an empty dimension to display 2-channel images
    if arr.ndim == 3 and arr.shape[-1] == 2:
        w, h, _ = arr.shape
        arr = np.dstack((arr, np.zeros((w, h, 1))))

    plt.imshow(arr, interpolation='nearest', vmin=vmin, vmax=vmax, cmap=plt.get_cmap('gray'))
    plt.show()


def numpy_all_the_way(list_of_arrays):
    shape = list(list_of_arrays[0].shape)
    shape[:0] = [len(list_of_arrays)]
    arr = np.concatenate(list_of_arrays).reshape(shape)
    return arr
