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
        #pred = np.squeeze(self.system.predict(y_hat = forecast, sigmas = sigmas_test, y_min = ymin, y_max = ymax, lower_percentiles=np.linspace(0,50,num=length_distr//2 , endpoint=False),higher_percentiles=np.linspace(50,100,num=length_distr//2 )))
        #print(pred.shape)
        #print(pred[pred >= ymax][:-1])
        #print(np.linspace(pred[-(pred[pred >= ymax].size+1)],ymax,num=pred[pred >= ymax].size+1)[1:-1])
        #if pred[pred >= ymax-1].size > 1:
        #    print('yes')
        #    pred[pred >= ymax][:-1] = np.linspace(pred[-(pred[pred >= ymax].size+1)],ymax,num=pred[pred >= ymax].size+1)[1:-1]
         #   pred = pred[pred > ymin]
          #  pred = np.r_[ymin, pred, ymax]
           # pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        #else:
         #   pred = np.r_[ymin, pred, ymax]
          #  pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        if pred[0]< ymin:
        
            pred[pred < ymin] = ymin
        
            pred = np.r_[pred, ymax]
            pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        else:
            pred = np.r_[ymin, pred, ymax]
            if len(pred) != length_distr:
                pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        return np.squeeze(pred)
