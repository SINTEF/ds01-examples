"""
Prints whether you have access to a CUDA device with Tensorflow.

Quickstart:
$ mamba ceate --name tf python=3.11
$ mamba activate tf
$ pip install tensorflow

References:
- https://www.geeksforgeeks.org/how-to-check-if-tensorflow-is-using-gpu/
- https://pytorch.org/get-started/locally/
"""

import tensorflow as tf
from tensorflow.python.client import device_lib

print("")
print("*** Available Physical Devices")
print(tf.config.list_physical_devices("GPU"))

print("")
print("*** Full List of Local Devices")
print(device_lib.list_local_devices())
