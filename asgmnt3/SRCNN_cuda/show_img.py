from PIL import Image
import numpy as np
import pandas as pd

df0 = pd.read_csv("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/input_image/img0.csv", header=None)
data = df0.to_numpy()
data *= 255.
data = np.reshape(data, (3, 288, 352)).astype(np.uint8)
# data = np.reshape(data, (3, 288, 352))
data = data.transpose(1, 2, 0)
img = Image.fromarray(data, 'RGB')
img.show()

df = pd.read_csv("/home/chengyih001/Documents/ECE1718/A3/asgmnt3_local/SRCNN_cpp_simple/output_image/img0_res.csv", header=None)
data2 = df.to_numpy()
# data2 = np.abs(data2)
data2 *= 255.
data2 = np.reshape(data2, (3, 288, 352)).astype(np.uint8)
# data2 = np.reshape(data2, (3, 288, 352))
data2 = data2.transpose(1, 2, 0)
img2 = Image.fromarray(data2, 'RGB')
img2.show()