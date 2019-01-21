import numpy as np
from matplotlib import pyplot as plt

from RBM import RBM
from DBN import DBN
from utils import softmax

def sigmoid(z):
    return 1 / (1+np.exp(-z))

class DNN(DBN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.L = len(self.layers)
        
    def forward(self, X):
        # Return last layer output after sotfmax
        return softmax(super().forward(X))
    
    def forward_full(self, X):
        # Returns outputs for all hidden layers. Apply softmax on last layer.
        self.A = [X]
        for idx, layer in enumerate(self.layers[:-1]):
            self.A.append(sigmoid(layer.forward(self.A[idx])))
        self.A.append(softmax(self.layers[-1].forward(self.A[-1])))
        
        return self.A[-1]
    
    def backpropagation(self, X, Y, lr=1e-3):
        m = X.shape[-1]
        
        # Propagate
        self.forward_full(X)
        
        # Compute dWs and dbs
        dWs = []
        dbs = []
        for l in reversed(range(1, len(self.layers)+1)):
            if l == self.L:
                dZ = self.A[-1] - Y
            else:
                dZ = np.dot(dZ, self.layers[l].W) * self.A[l] * (1 - self.A[l])
            
            dWs.append( 1/m * np.dot(dZ.T, self.A[l-1]) )
            dbs.append( 1/m * np.sum(dZ, axis=0, keepdims=True) )
            
        # Reverse dWs and dbs
        dWs = dWs[::-1]
        dbs = dbs[::-1]
        
        # Update weights
        for l, layer in enumerate(self.layers):
            layer.W += - lr * dWs[l]
            layer.b += - lr * dbs[l]
    
    def train_supervised(self, X, Y, epochs, batch_size=32, lr=1e-3):
        m = X.shape[0]
        
        hist_loss = []
        
        num_batches = (m + batch_size - 1) // batch_size
        idx_batches = [(i * batch_size, min(m-1, (i + 1) * batch_size)) for i in range(num_batches)]
        
        for e in range(epochs):
            for idx_b, idx_e in idx_batches:
                self.backpropagation(X[idx_b:idx_e, ...], Y[idx_b:idx_e, ...], lr=lr)

            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            acc = self.compute_acc(Y, Y_hat)
            if not e % 10:
                print(f'Loss: {np.round(loss, 3)}\tAcc: {np.round(acc, 3)}')
            hist_loss.append(loss)
        plt.plot(hist_loss)
        
    def compute_loss(self, y_true, y_pred):
        return np.sum( -y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred) ) / y_true.shape[0]
    
    def compute_acc(self, y_true, y_pred):
        return np.sum( np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1) ) * 100 / y_true.shape[0]
        