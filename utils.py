from matplotlib import pyplot as plt
import numpy as np

def show_img(img):
    plt.imshow(img, cmap='gray')

def idx2char(idx):
    if idx < 10:
        return str(idx)
    else:
        return chr(97 + idx - 10)
    
def softmax(z):
    a = np.exp(z)
    return a / np.sum(a, axis=-1, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))