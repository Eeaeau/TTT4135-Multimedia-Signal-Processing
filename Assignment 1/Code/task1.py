import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt


def dct2(block):
    return fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")


# implement 2D IDCT
def idct2(a):
    return fftpack.idct(fftpack.idct(a.T, norm="ortho").T, norm="ortho")


# def compute_psnr(original, img2):
#     img1 = img1.astype(np.float64) / 255.0
#     img2 = img2.astype(np.float64) / 255.0
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return "Same Image"
#     return 20 * np.log10(1.0 / mse)


def compute_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return "Same Image"
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


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

# plt.imshow(data_block)
# plt.show()


q_table = np.array(
    [
        16,
        11,
        10,
        16,
        24,
        40,
        51,
        61,
        12,
        12,
        14,
        19,
        26,
        58,
        60,
        55,
        14,
        13,
        16,
        24,
        40,
        57,
        69,
        56,
        14,
        17,
        22,
        29,
        51,
        87,
        80,
        62,
        18,
        22,
        37,
        56,
        68,
        109,
        103,
        77,
        24,
        35,
        55,
        64,
        81,
        104,
        113,
        92,
        49,
        64,
        78,
        87,
        103,
        121,
        120,
        101,
        72,
        92,
        95,
        98,
        112,
        100,
        103,
        99,
    ]
)
q_table = q_table.reshape((8, 8))

# -------------------- a --------------------- #
data_block_DCT2 = dct2(data_block)

print(data_block_DCT2)
np.savetxt(
    "data_block_DCT2.csv", (data_block_DCT2), fmt="%d"
)  # used for tabular in latex with generator

# plt.imshow(np.log(np.abs(data_block_DCT2)))
# plt.show()

# -------------------- b --------------------- #

DCT2_quantized = np.floor(data_block_DCT2 / q_table + 0.5)
print(DCT2_quantized)
np.savetxt(
    "DCT2_quantized.csv", (DCT2_quantized), fmt="%d"
)  # used for tabular in latex with generator

# plt.imshow(DCT2_quantized)
# plt.show()

# -------------------- c --------------------- #

dequantized = q_table * DCT2_quantized

data_block_IDCT2 = idct2(dequantized)

print(data_block_IDCT2)
# plt.imshow(data_block_IDCT2)
# plt.show()

np.savetxt(
    "data_block_IDCT2.csv", (data_block_IDCT2), fmt="%d"
)  # used for tabular in latex with generator

PSNR = compute_psnr(data_block, data_block_IDCT2)
print("psnr: ", PSNR)
