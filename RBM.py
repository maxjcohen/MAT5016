import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import softmax, sigmoid

class RBM():
    def __init__(self, q, n, activation="sigmoid"):
        self.W = np.random.normal(scale=0.01, size=(q, n)) * 0.01
        self.a = np.zeros((1, n))
        self.b = np.zeros((1, q))
        
        if activation == "sigmoid":
            self.activation = sigmoid
        elif activation == "softmax":
            self.activation = softmax
        else:
            raise NameError(f"Activation function {activation} not recognized.")
        
    def forward(self, X):
        return self.activation(np.dot(X, self.W.T) + self.b)
    
    def backward(self, H):
        return self.activation(np.dot(H, self.W) + self.a)
    
    def train(self, X, epochs=5, lr=1e-3):
        m = len(X)
        losses = np.zeros(epochs)
        
        v_0 = X
        
        for epoch in tqdm(range(epochs)):

            # Sample h_0
            p_h_v_0 = self.forward(v_0)
            h_0 = p_h_v_0 >= np.random.random(p_h_v_0.shape)

            # Sample v_1
            p_v_h_0 = self.backward(h_0)
            v_1 = p_v_h_0 >= np.random.random(p_v_h_0.shape)
            
            # Loss
            losses[epoch] = ((v_0 - v_1) ** 2).mean()

            p_h_v_1 = self.forward(v_1)

            # Compute deltas
            dw = np.dot(v_0.T, p_h_v_0) - np.dot(v_1.T, p_h_v_1)
            da = np.sum(v_0 - v_1, axis=0)
            db = np.sum(p_h_v_0 - p_h_v_1, axis=0)

            # Update weights
            self.W += lr * dw.T
            self.a += lr * da.T
            self.b += lr * db.T
            
        plt.plot(losses)