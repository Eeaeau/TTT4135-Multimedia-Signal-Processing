import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt


def dct2(block):
    return fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")


data_block = np.array(
    [
        124,
        125,
        122,
        120,
        122,
        119,
        117,
        118,
        121,
        121,
        120,
        119,
        119,
        120,
        120,
        118,
        126,
        124,
        123,
        122,
        121,
        121,
        120,
        120,
        124,
        124,
        125,
        125,
        126,
        125,
        124,
        124,
        127,
        127,
        128,
        129,
        130,
        128,
        127,
        125,
        143,
        142,
        143,
        142,
        140,
        139,
        139,
        139,
        150,
        148,
        152,
        152,
        152,
        152,
        150,
        151,
        156,
        159,
        158,
        155,
        158,
        158,
        157,
        156,
    ]
)
data_block = data_block.reshape((8, 8))

print(data_block)

data_block_DCT2 = dct2(data_block)

print(data_block_DCT2)
np.savetxt("data_block_DCT2.csv", (data_block_DCT2), fmt="%d")

plt.imshow(np.log(np.abs(data_block_DCT2)))
plt.show()
