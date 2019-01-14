from RBM import RBM

class DBN():
    def __init__(self, layers=3, n_h=[10, 16, 32, 784]):
        self.layers = [RBM(n_h[i], n_h[i+1]) for i in range(len(n_h)-1)]
        
    def train(self, X, epochs=5, lr=1e-3):
        for layer in reversed(self.layers):
            layer.train(X, epochs=epochs, lr=lr)
            X = layer.forward(X)
            
    def forward(self, X):
        Y = X
        for layer in reversed(self.layers):
            Y = layer.forward(Y)
        return Y
    
    def backward(self, H):
        A = H
        for layer in self.layers:
            A = layer.backward(A)
        return A