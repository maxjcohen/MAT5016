from matplotlib import pyplot as plt
import numpy as np

def show_img(img):
    plt.imshow(img, cmap='gray')
    
def softmax(z):
    a = np.exp(z)
    return a / np.sum(a, axis=-1, keepdims=True)