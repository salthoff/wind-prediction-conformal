"""Contains class for performing standard conformal prediction

"""

import numpy as np
from crepes import ConformalRegressor
from crepes.fillings import sigma_knn


class Conformal_std():
    def __init__(self, window_length = np.inf):
        self.window_length = window_length
        self.residuals = np.array([])
        self.sigmas = None
        self.system = ConformalRegressor()
        self.input = np.array([])



    def calibrate(self, data, forecast, label):
        if self.residuals.size != 0:
            self.residuals = np.r_[self.residuals, (label -forecast)]
            self.input = np.r_[self.input, data]
            #self.sigmas = sigma_knn(X = self.input, residuals=self.residuals)
        else:
            self.residuals = label - forecast
            #self.sigmas = sigma_knn(X=data, residuals=self.residuals)
            self.input = data
        
        if len(self.residuals) > self.window_length:
            self.residuals=self.residuals[-self.window_length:]
            #self.sigmas = self.sigmas[-self.window_length:]
            self.input = self.input[-self.window_length:,:]
        
        self.sigmas = sigma_knn(X = self.input, residuals=self.residuals)

        self.system.fit(residuals=np.squeeze(self.residuals), sigmas=self.sigmas)
        

    def predict(self, data,forecast,confidence = 0.95):
        sigmas_test = sigma_knn(X = self.input, residuals=self.residuals, X_test = data.reshape(1, -1))
        pred = self.system.predict(y_hat = forecast, sigmas = sigmas_test, confidence = confidence)
        return np.squeeze(pred)
