from matplotlib import pyplot as plt

def show_img(img):
    img = img.reshape((28, 28))
    plt.imshow(img, cmap='gray')