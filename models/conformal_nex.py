"""Containing class for doing conformal prediction with weighted residuals

"""

import numpy as np

class Conformal_nex():
    """
    Class for using the non-exchangeable conformal predictor

    Attributes
    ---------
    ff: float
        Forgetting factor
    weights:
        weights associated with each example
    residuals:
        saved residuals from point predictions
    
    Methods
    -------
    calibrate(data, forecast, label)
        Calibrates the system
    predict(data, forecast, length_distr = 200, ymin = 0, ymax = 100)
        Predicts the CDF from data and forecast
    """

    def __init__(self, forget_factor):
        self.ff = forget_factor
        self.weigths = np.array([])
        self.residuals = np.array([])
    
    def calibrate(self,data, forecast, label):
        if self.weigths.size != 0:
            self.weigths = np.r_[np.power(self.ff, range(len(forecast)+len(self.weigths)-1,len(self.weigths)-1,-1)), self.weigths]
            self.residuals = np.r_[self.residuals,np.abs(label-forecast)]
        else:
            self.weigths = np.power(self.ff, range(len(forecast)-1,-1,-1))
            self.residuals = np.abs(label-forecast)


        
        
        
    def predict(self,data, forecast, length_distr = 200, ymin = 0, ymax = 100):
        weights_cal = self.weigths / (np.sum(self.weigths) + 1)
        num_i = np.min([len(weights_cal)//2,length_distr//2])
        pred = forecast
        for i in range(num_i - 1):
            confidence = (i+1)/num_i
            if (np.sum(weights_cal) >= confidence):
                ordR = np.argsort(np.squeeze(self.residuals))
                ind_thres = np.min(np.where(np.cumsum(weights_cal[ordR])>=confidence))         
                cal_thres = np.sort(np.squeeze(self.residuals))[ind_thres]
                pred = np.r_[forecast - cal_thres,pred, forecast + cal_thres]
            else:
                pred = np.r_[ymin,pred, ymax]

        if pred[0]< ymin:
            pred[pred < ymin] = ymin
            pred = np.r_[pred, ymax]
            pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        else:
            pred = np.r_[ymin, pred, ymax]
            if len(pred) != length_distr:
                pred = np.interp(np.linspace(0,1,num=length_distr),np.linspace(0,1,num=len(pred)),pred)
        
        return np.squeeze(pred)