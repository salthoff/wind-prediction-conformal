"""Contains class for performing standard (normalized) conformal prediction

"""

import numpy as np
from crepes import ConformalRegressor
from crepes import ConformalPredictiveSystem
from crepes.fillings import sigma_knn


class Conformal_std():
    def __init__(self, window_length = np.inf):
        self.window_length = window_length
        self.residuals = np.array([])
        self.sigmas = None
        self.system = ConformalPredictiveSystem()
        self.input = np.array([])



    def calibrate(self, data, forecast, label):
        if self.residuals.size != 0:
            self.residuals = np.r_[self.residuals, (label -forecast)]
            self.input = np.r_[self.input, data]
        else:
            self.residuals = label - forecast
            self.input = data
        
        if len(self.residuals) > self.window_length:
            self.residuals=self.residuals[-self.window_length:]
            self.input = self.input[-self.window_length:,:]
        
        self.sigmas = sigma_knn(X = self.input, residuals=self.residuals)

        self.system.fit(residuals=np.squeeze(self.residuals), sigmas=self.sigmas)
        

    def predict(self, data, forecast, length_distr = 200, ymin = 0, ymax = 100):
        sigmas_test = sigma_knn(X = self.input, residuals=self.residuals, X_test = data.reshape(1, -1))
        pred = np.squeeze(self.system.predict(y_hat = forecast, sigmas = sigmas_test, return_cpds=True))
        if pred[0]< ymin:
            pred = pred[pred > ymin]
            pred = np.r_[ymin, pred, ymax]
            pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        else:
            pred = np.r_[ymin, pred, ymax]
            pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        return np.squeeze(pred)
