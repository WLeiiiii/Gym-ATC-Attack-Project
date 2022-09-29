import numpy as np
from matplotlib import pyplot as plt


def display_plt(list_name, x_label, y_label, title_name):
    plt.figure()
    plt.plot(np.arange(len(list_name)), list_name)
    plt.ylabel('{}'.format(y_label))
    plt.xlabel('{}'.format(x_label))
    plt.title('{}'.format(title_name))
    plt.show()
    pass
