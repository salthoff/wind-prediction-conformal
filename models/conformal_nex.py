"""Containing class for doing conformal prediction with weighted residuals

"""

import numpy as np

class Conformal_nex():
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


        
        
        
    def predict(self,data, modeloutput, confidence = 0.95):
        weights_cal = self.weigths / (np.sum(self.weigths) + 1)
        if (np.sum(weights_cal) >= confidence):
            ordR = np.argsort(np.squeeze(self.residuals))
            ind_thres = np.min(np.where(np.cumsum(weights_cal[ordR])>=confidence))         
            cal_thres = np.sort(np.squeeze(self.residuals))[ind_thres]
        else:
            cal_thres = np.inf
        
        yPI = np.squeeze([modeloutput - cal_thres, modeloutput + cal_thres])
        return yPI