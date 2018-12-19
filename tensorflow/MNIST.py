import struct
import numpy as np


def decode_idx3_ubyte(idx3_ubyte_file, saveflag=False):

    status = 'save/'

    with open(idx3_ubyte_file, 'rb') as f:
        buf = f.train_image.read()

    index = 0
    magic, imagenum, rows, cols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    image = np.empty((imagenum, rows, cols))
    image_size = rows * cols

    # define how many numbers(size of photo) to get for one time
    fmt = '>' + str(image_size) + 'B'

    for i in range(100):
        im = struct.unpack_from(fmt, buf, index)
        im = np.reshape(im, [rows, cols])
        image[i] = np.array(im)

        if saveflag:
            im = image.from_array(np.unit8(image[i]))
            im.save(status + str(i) + '.png')

        index += struct.calcsize(fmt)

    return image, imagenum


def decode_idx1_ubyte(idx1_ubyte_file):

    with open(idx1_ubyte_file, 'rb') as f:
        buf = f.read()

    index = 0
    magic, labelnum = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    labels = np.zeros(labelnum)

    for i in range(labelnum):
        labels[i] = np.array(struct.unpack_from('>B', buf, index))
        index += struct.calcsize('>B')

    return labels, labelnum

