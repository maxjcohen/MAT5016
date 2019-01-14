from RBM import RBM

class DBN():
    def __init__(self, n_h):
        self.layers = [RBM(n_h[i+1], n_h[i]) for i in range(len(n_h)-1)]
        
    def train(self, X, epochs=5, lr=1e-3):
        for layer in self.layers:
            layer.train(X, epochs=epochs, lr=lr)
            X = layer.forward(X)
            
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, H):
        for layer in reversed(self.layers):
            H = layer.backward(H)
        return H