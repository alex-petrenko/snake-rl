from matplotlib import pyplot as plt


def imshow(numpy_array, vmin=0.0, vmax=1.0):
    plt.imshow(numpy_array, interpolation='nearest', vmin=vmin, vmax=vmax, cmap=plt.get_cmap('gray'))
    plt.show()
