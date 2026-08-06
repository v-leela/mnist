import numpy as np

GREEN = (0, 234, 0)
WHITE = (255, 255, 255)


class NN:
    def __init__(self, screen, para):
        self.para = para
        self.screen = screen

    def Relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forwardprop(self, X):
        W1, b1, W2, b2, W3, b3 = self.para
        Z1 = W1 @ X + b1
        A1 = self.Relu(Z1)

        Z2 = W2 @ A1 + b2
        A2 = self.Relu(Z2)

        Z3 = W3 @ A2 + b3
        A3 = self.softmax(Z3)

        return A3

    def shape(self):
        a = []
        for para in self.para:
            a.append(para.shape)
        print(a)

    def predict(self, X):
        y_pred = self.forwardprop(X)
        prediction = np.argsort(y_pred, axis=0)[::-1]
        pred_prob = np.sort(y_pred, axis=0)[::-1]

        return prediction, pred_prob
