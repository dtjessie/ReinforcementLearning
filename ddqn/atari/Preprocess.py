import numpy as np


def preprocess(obs):
    # This comes from Geron's "Hands-On Machine Learning"
    # Original obs has shape (210, 160, 3)
    # Cut this down to (88, 80, 1), convert to gray-scale
    # Also we need to worry about size of the images
    # since we want to put as many in memory as possible.
    # Store them as int8 rather than normalizing and saving as floats
    img = obs[25:200:2, ::2]  # crop and downsize
    img = img.mean(axis=2)  # to grayscale
    img = (img - 128).astype(np.int8)
    return img.reshape(88, 80, 1)
