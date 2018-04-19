import numpy as np
from matplotlib import pyplot as plt


def imshow(numpy_array, vmin=0.0, vmax=1.0):
    plt.imshow(numpy_array, interpolation='nearest', vmin=vmin, vmax=vmax, cmap=plt.get_cmap('gray'))
    plt.show()


def numpy_all_the_way(list_of_arrays):
    shape = list(list_of_arrays[0].shape)
    shape[:0] = [len(list_of_arrays)]
    arr = np.concatenate(list_of_arrays).reshape(shape)
    return arr
