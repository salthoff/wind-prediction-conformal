"""Contains class for performing standard conformal prediction

"""

import numpy as np
from crepes import ConformalRegressor
from crepes.fillings import sigma_knn


class Conformal_std():
    def __init__(self, window_length = np.inf):
        self.window_length = window_length
        self.residuals = None
        self.sigmas = None
        self.system = None
        self.input = None



    def train(self,data, forecast, label):
        if self.residuals != None:
            self.residuals = np.r_[self.residuals, (label -forecast)]
            self.sigmas = np.r_[self.sigmas, sigma_knn(X = data, residuals = (label -forecast))]
            self.input = np.r_[self.input, data]
        else:
            self.residuals = label -forecast
            self.sigmas = sigma_knn(X=data, residuals=self.residuals)
            self.input = data
        
        self.system = ConformalRegressor.fit(residuals=self.residuals, sigmas=self.sigmas)

    def predict(self, data,forecast,confidence = 0.95):
        sigmas_test = sigma_knn(X = self.data, residuals=self.residuals, X_test = data)
        pred = self.system.predict(y_hat = forecast, sigmas = self.sigmas, confidence = confidence)

        return pred
