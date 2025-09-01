# coding: utf-8

import numpy as np
from itertools import product

if __name__ == '__main__':

    img = np.zeros((1000, 1000))

    for x, y in product(range(1000), repeat=2):
        img[x, y] = np.sin((x + y) / (4 * np.pi))

    np.save('./lines.npy', img)

    for x, y in product(range(1000), repeat=2):
        img[x, y] = y / 999 * np.sin((x + y) / (4 * np.pi))

    np.save('./lines_intensity.npy', img)

    for x, y in product(range(1000), repeat=2):
        img[x, y] = max(np.sin((x + y) / (4 * np.pi)),
                        np.sin((x - y) / (4 * np.pi)))

    np.save('./cross.npy', img)

    for x, y in product(range(1000), repeat=2):
        img[x, y] = np.sin(np.sqrt((x - 499) ** 2 +
                                   (y - 499) ** 2) / (4 * np.pi))

    np.save('./circles.npy', img)

    for x, y in product(range(1000), repeat=2):
        img[x, y] = np.sin((x + 8 * np.sin(y / (4 * np.pi))) / (4 * np.pi))

    np.save('./waves.npy', img)
