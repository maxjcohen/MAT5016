from RBM import RBM

class DBN():
    def __init__(self, n_h):
        self.layers = [RBM(n_h[i+1], n_h[i]) for i in range(len(n_h)-1)]
        
        self.L = len(self.layers)
        
    def train(self, X, epochs=5, lr=1e-3, except_last=False):
        for l, layer in enumerate(self.layers):
            
            if except_last and l == self.L-1:
                return
            
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